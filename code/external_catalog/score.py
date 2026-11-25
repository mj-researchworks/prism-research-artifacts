"""Score independent human relevance judgments exported by the external runner."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from .common import save


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def key(row):
    return (str(row['conversation_id']), int(row['turn_index']), int(row['rank']), str(row['product_id']))


def score(run_dir, judgments):
    expected=read_jsonl(run_dir/'manual_relevance_audit.jsonl')
    labels=read_jsonl(judgments)
    expected_keys={key(r) for r in expected}
    label_keys=[key(r) for r in labels]
    if len(set(label_keys)) != len(label_keys) or not set(label_keys).issubset(expected_keys):
        raise ValueError('Duplicate judgments or products/turns not displayed by this run')
    mapped={key(r):r for r in labels}
    groups=defaultdict(list)
    reviewed_products=violations=known_constraints=0
    for row in expected:
        label=mapped.get(key(row),{})
        relevance=label.get('relevance_0_1_2')
        violation=label.get('hard_constraint_violation_0_1')
        if relevance not in (None,0,1,2) or violation not in (None,0,1):
            raise ValueError('Expected relevance null/0/1/2 and hard violation null/0/1')
        reviewed_products+=relevance is not None
        if violation is not None:
            known_constraints+=1
            violations+=violation
        groups[(str(row['conversation_id']),int(row['turn_index']))].append(relevance)
    complete=[v for v in groups.values() if all(x is not None for x in v)]
    result={
        'displayed_lists':len(groups),'displayed_products':len(expected),
        'relevance_judged_products':reviewed_products,'fully_judged_lists':len(complete),
        'fully_judged_list_fraction':len(complete)/max(1,len(groups)),
        'mean_relevance_precision_at3_on_fully_judged_lists':sum(sum(x>0 for x in v)/3 for v in complete)/len(complete) if complete else None,
        'at_least_one_relevant_rate_on_fully_judged_lists':sum(any(x>0 for x in v) for v in complete)/len(complete) if complete else None,
        'hard_constraint_judged_products':known_constraints,
        'hard_constraint_unknown_products':len(expected)-known_constraints,
        'hard_constraint_violation_rate_on_judged_products':violations/known_constraints if known_constraints else None,
        'note':'Independent judgments of newly displayed products only. Missing/unreviewed judgments are unknown. These are not PSCon exact-ID Hit/NDCG or exhaustive catalogue recall. Availability and preference-revision event adjudication require separate labels.',
    }
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--judgments',type=Path,required=True)
    args=parser.parse_args()
    result=score(args.run_dir,args.judgments)
    save(args.run_dir/'judged_metrics.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
