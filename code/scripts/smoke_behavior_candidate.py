"""Review selection without API calls; --execute runs a paid candidate smoke.

Includes the same first 20 Fold4 conversations plus the two known mixed-intent
cases, deduplicated. Saves full per-turn state for qualitative comparison.
"""
import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]

def serialize(v):
    if is_dataclass(v): return serialize(asdict(v))
    if isinstance(v,dict): return {k:serialize(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,set)): return [serialize(x) for x in v]
    return v

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-convs',type=int,default=20)
    ap.add_argument('--execute',action='store_true',help='Make paid LLM calls')
    ap.add_argument('--out-dir',type=Path)
    args=ap.parse_args()
    from eval.autonomous_full_recommendation_eval import configure_fold,load_split,extract_user_turns
    split=configure_fold('fold4')
    all_convs=load_split('test',split_dir=split)
    chosen=all_convs[:args.max_convs]
    ids={str(c['conv_id']) for c in chosen}
    for cid in ['69994','74670']:
        if cid not in ids:
            chosen.append(next(c for c in all_convs if str(c['conv_id'])==cid));ids.add(cid)
    print('Selected conversations:',[str(c['conv_id']) for c in chosen],flush=True)
    print('User turns:',sum(len(extract_user_turns(c)) for c in chosen),flush=True)
    if not args.execute:
        print('Dry run only. No API calls. Add --execute to run the paid smoke test.')
        return
    if not os.environ.get('DEEPSEEK_API_KEY'):raise SystemExit('DEEPSEEK_API_KEY is missing')
    config={
        'AUTONOMOUS_REASONING_BUDGET':'adaptive','AUTONOMOUS_RETRIEVAL_BACKEND':'conserve_stack',
        'AUTONOMOUS_PLAN_POLICY_MODE':'validator','AUTONOMOUS_PLANNER_MODE':'auto',
        'AUTONOMOUS_REFLECTION_MODE':'agent','AUTONOMOUS_REFLECTION_WRITES_INTENT':'1',
        'AUTONOMOUS_CRITIC_MODE':'hybrid','AUTONOMOUS_INTENT_MODE':'lightweight',
        'AUTONOMOUS_EXECUTION_SCOPE':'full','AUTONOMOUS_RESPONSE_MODE':'template',
        'AUTONOMOUS_REFLECTION_AGENT_RULE_SUPPORT':'0',
    }
    for agent in ['PLANNER','REFLECTION','CRITIC']:
        config.update({f'AUTONOMOUS_{agent}_PROVIDER':'deepseek',
                       f'AUTONOMOUS_{agent}_MODEL':'deepseek-v4-flash',
                       f'AUTONOMOUS_{agent}_TIMEOUT':'60'})
    os.environ.update(config)
    out=args.out_dir or ROOT/'eval'/'results'/f'behavior_smoke_{time.strftime("%Y%m%d_%H%M%S")}'
    out.mkdir(parents=True,exist_ok=False)
    manifest={'configuration':config,'conversation_ids':[str(c['conv_id']) for c in chosen],
              'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in ROOT.joinpath('src').rglob('*.py')},
              'notes':'Candidate smoke, includes two known failures; not an unbiased benchmark.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    from src.autonomous.runner import AutonomousRunner
    from src.autonomous.eval_logger import EvalLogger
    failed=0;count=0
    with (out/'turns.jsonl').open('w') as stream:
        for i,c in enumerate(chosen,1):
            cid=str(c['conv_id']);print(f'[{i}/{len(chosen)}] {cid}',flush=True)
            try:
                runner=AutonomousRunner(planner_mode='auto',eval_logger=EvalLogger(enabled=False))
                for turn_index,t in enumerate(extract_user_turns(c),1):
                    user=str(t.get('content') or '').strip()
                    if not user:continue
                    started=time.perf_counter();response,state=runner.run_turn(user)
                    row={'conversation_id':cid,'turn_index':turn_index,'user_message':user,
                         'response':response,'latency_seconds':time.perf_counter()-started,
                         'state':serialize(state)}
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();count+=1
                    print(f'  turn {turn_index}: {response[:100]}',flush=True)
            except Exception as exc:
                failed+=1
                stream.write(json.dumps({'conversation_id':cid,'error':f'{type(exc).__name__}: {exc}'})+'\n')
                stream.flush()
    (out/'summary.json').write_text(json.dumps({'conversations':len(chosen),'failed_conversations':failed,
                                             'completed_turns':count},indent=2))
    print('Saved:',out,flush=True)

if __name__=='__main__':main()
