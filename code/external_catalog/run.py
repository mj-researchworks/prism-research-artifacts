"""Replay PSCon user turns against external products; never use PSCon ID labels."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import importlib.metadata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, SUPPORTED_FOLDS, configure, digest, effective_environment, fold_paths, jsonish, load, save


def response_type(state):
    for obs in reversed(state.get('tool_observations') or []):
        if obs.get('component') == 'response_tool':
            match = re.search(r'\btype=([^;]+)', str(obs.get('summary', '')))
            if match:
                return match.group(1).strip()
    return 'unknown'


def classify_output(state, kind, displayed):
    if displayed:
        return 'new_recommendation'
    if kind == 'acceptance':
        return 'retained_acceptance'
    diag = (state.get('candidate_ledger') or {}).get('diagnostics') or {}
    if diag.get('actual_backend') == 'conserve_stack_failed':
        return 'retrieval_error'
    flags = sum((list(state.get(k) or []) for k in ['control_flags','security_flags']), [])
    if 'step_budget_exhausted' in flags:
        return 'step_budget_exhausted'
    if state.get('critic_decision') == 'ASK_USER':
        return 'asked_user'
    if state.get('critic_decision') in {'SEARCH_AGAIN','RERANK','UPDATE_STATE'}:
        return 'critic_unresolved'
    if state.get('candidate_ledger') and not state.get('verified_candidates'):
        return 'no_verified_candidates'
    return 'no_new_recommendation'


def turn_record(cid, source_turn, user_message, response, state, elapsed):
    kind = response_type(state)
    recommendations = state.get('last_recommendations') or []
    displayed = recommendations if kind == 'recommendation' else []
    retained = recommendations if kind == 'acceptance' else []
    return jsonish({
        'conversation_id':cid, 'turn_index':state.get('turn_id'), 'source_turn_id':source_turn,
        'user_message':user_message, 'response':response, 'status':'completed',
        'latency_seconds':elapsed, 'response_type':kind,
        'outcome':classify_output(state,kind,displayed),
        'newly_displayed_ids':[str(p['product_id']) for p in displayed],
        'retained_ids':[str(p['product_id']) for p in retained],
        'displayed_products':displayed,
        'raw_ids':[str(p['product_id']) for p in state.get('raw_candidates') or []],
        'verified_ids':[str(p['product_id']) for p in state.get('verified_candidates') or []],
        'preferences':state.get('preferences'), 'preference_events':state.get('preference_events'),
        'rejected_products':state.get('rejected_products'), 'rejected_brands':state.get('rejected_brands'),
        'rejected_features':state.get('rejected_features'),
        'search_request':state.get('final_search_request'), 'retrieval_plan':state.get('retrieval_plan'),
        'candidate_ledger':state.get('candidate_ledger'),
        'critic_decision':state.get('critic_decision'), 'critic_notes':state.get('critic_notes'),
        'reflection_notes':state.get('reflection_notes'), 'response_notes':state.get('response_notes'),
        'tool_observations':state.get('tool_observations'),
        'planner_trace':[p for p in state.get('planner_trace') or [] if p.get('turn_id')==state.get('turn_id')],
        'control_flags':state.get('control_flags'),
    })


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-convs',type=int)
    parser.add_argument('--conversation-id', action='append', help='Optional diagnostic subset; IDs may be repeated as flags')
    parser.add_argument('--planner-mode',choices=['auto','llm','rule'],default='auto')
    parser.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    parser.add_argument('--fold', choices=SUPPORTED_FOLDS, default='fold4')
    parser.add_argument('--out-dir',type=Path)
    args=parser.parse_args()
    if args.max_convs is not None and args.max_convs < 1:
        raise ValueError('--max-convs must be positive')
    configure(args.planner_mode,args.device,args.fold)
    if args.planner_mode != 'rule' and not os.environ.get('DEEPSEEK_API_KEY'):
        raise RuntimeError('Set DEEPSEEK_API_KEY in your environment before LLM inference')
    from .validate import validate
    validation=validate(fold=args.fold)
    from src.autonomous.eval_logger import EvalLogger
    from src.autonomous.runner import AutonomousRunner
    from src.agents import reranker
    # Reject silent E5 load fallback before spending any LLM tokens.
    if reranker._load_triplet_state() is None:
        raise RuntimeError(f'E5 failed to load: {reranker.get_triplet_reranker_diagnostics()}')
    products=load(ROOT/'data/product_base_table.json')
    paths=fold_paths(args.fold)
    split_path=paths['split']
    conversations=load(split_path)
    if args.conversation_id:
        requested=set(args.conversation_id)
        conversations=[c for c in conversations if str(c['conv_id']) in requested]
        if {str(c['conv_id']) for c in conversations} != requested:
            raise ValueError(f'Some requested conversation IDs do not occur in {args.fold} test')
    if args.max_convs:
        conversations=conversations[:args.max_convs]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=args.out_dir or ROOT/'results'/f'{args.fold}_external_{stamp}'
    out.mkdir(parents=True,exist_ok=False)
    save(out/'run_manifest.json',{'created_utc':stamp,'fold':args.fold,
         'profile':f'final_{args.fold}' if args.planner_mode!='rule' else 'offline_rule_diagnostic',
         'validation':validation,'effective_environment':effective_environment(),
         'source_snapshot_sha256':digest(ROOT/'source_snapshot.json'),
         'external_manifest_sha256':digest(ROOT/'data/external_manifest.json'),
         'build_manifest_sha256':digest(paths['build_manifest']),
         'integration_code_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'external_experiment').glob('*.py')},
         'package_versions':{name:importlib.metadata.version(name) for name in ['torch','transformers','sentence-transformers','faiss-cpu','numpy']},
         'split_sha256':digest(split_path),'selected_conversation_ids':[c['conv_id'] for c in conversations],
         'evaluation':'External relevance requires independent judgments. PSCon exact-ID GT is not used.',
         'replay_limitation':'User ordinal references/acceptance were collected in another recommendation trajectory.'})
    outcomes=Counter()
    elapsed_convs=[]
    failures=invalid=duplicate=exclusion_hits=displayed_turns=0
    run_started=time.perf_counter()
    with (out/'turns.jsonl').open('w') as turns_file, (out/'conversations.jsonl').open('w') as conv_file, (out/'manual_relevance_audit.jsonl').open('w') as audit_file:
        for index,conv in enumerate(conversations,1):
            cid=str(conv['conv_id'])
            started=time.perf_counter()
            failed=False
            history=[]
            print(f'[{index}/{len(conversations)}] conversation {cid}',flush=True)
            try:
                runner=AutonomousRunner(planner_mode=args.planner_mode,eval_logger=EvalLogger(root=out/'traces',enabled=True))
            except Exception as exc:
                elapsed=time.perf_counter()-started
                elapsed_convs.append(elapsed)
                failures+=1
                outcomes['initialization_error']+=1
                error={'conversation_id':cid,'failed':True,'outcome':'initialization_error',
                       'latency_seconds':elapsed,'error':f'{type(exc).__name__}: {exc}'}
                conv_file.write(json.dumps(error)+'\n')
                conv_file.flush()
                continue
            for turn in conv['conversation']:
                if turn.get('role')!='user' or not str(turn.get('content') or '').strip():
                    continue
                message=str(turn['content']).strip()
                history.append({'role':'user','content':message})
                t0=time.perf_counter()
                try:
                    response,state=runner.run_turn(message)
                    row=turn_record(cid,turn.get('turn_id'),message,response,state,time.perf_counter()-t0)
                    all_ids=row['raw_ids']+row['verified_ids']+row['newly_displayed_ids']+row['retained_ids']
                    invalid+=sum(pid not in products for pid in all_ids)
                    newly=row['newly_displayed_ids']
                    duplicate+=len(newly)-len(set(newly))
                    exclusion_hits+=len(set(newly)&set(map(str,state.get('rejected_products') or [])))
                    displayed_turns+=bool(newly)
                    for rank,p in enumerate(row['displayed_products'],1):
                        audit={'conversation_id':cid,'turn_index':state['turn_id'],'rank':rank,
                               'user_history':history.copy(),'active_preferences':row['preferences'],
                               'search_request':row['search_request'],'product_id':p['product_id'],
                               'product':products[str(p['product_id'])],
                               'relevance_0_1_2':None,'hard_constraint_violation_0_1':None,'notes':''}
                        audit_file.write(json.dumps(audit,ensure_ascii=False)+'\n')
                    history.append({'role':'assistant','content':response})
                except Exception as exc:
                    row={'conversation_id':cid,'source_turn_id':turn.get('turn_id'),'user_message':message,
                         'status':'error','outcome':'error','latency_seconds':time.perf_counter()-t0,
                         'error':f'{type(exc).__name__}: {exc}'}
                    failed=True
                turns_file.write(json.dumps(row,ensure_ascii=False)+'\n')
                turns_file.flush()
                audit_file.flush()
                outcomes[row['outcome']]+=1
                print(f'  turn {turn.get("turn_id")}: {row["outcome"]} ({row["latency_seconds"]:.2f}s)',flush=True)
                if failed:
                    break
            elapsed=time.perf_counter()-started
            elapsed_convs.append(elapsed)
            failures+=failed
            conv_file.write(json.dumps({'conversation_id':cid,'failed':failed,'latency_seconds':elapsed})+'\n')
            conv_file.flush()
    def quantile(values,q):
        values=sorted(values)
        if not values:return None
        pos=(len(values)-1)*q
        lo=int(pos)
        return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(pos-lo)
    save(out/'summary.json',{'selected_conversations':len(conversations),'completed_conversations':len(conversations)-failures,
         'failed_conversations':failures,'conversation_completion_rate':(len(conversations)-failures)/max(1,len(conversations)),
         'outcomes':dict(outcomes),'new_recommendation_turns':displayed_turns,
         'invalid_id_occurrences_all_candidate_lists':invalid,'duplicate_display_occurrences':duplicate,
         'displayed_rejected_id_occurrences':exclusion_hits,
         'conversation_latency_mean_s':sum(elapsed_convs)/max(1,len(elapsed_convs)),
         'conversation_latency_p50_s':quantile(elapsed_convs,.5),'conversation_latency_p95_s':quantile(elapsed_convs,.95),
         'total_wall_seconds':time.perf_counter()-run_started,
         'note':'Completion is execution reliability, not recommendation success. Exclusion audit uses system-detected rejections. Relevance and availability require independent judgments.'})
    print(f'Saved: {out}',flush=True)


if __name__=='__main__':
    main()
