"""No-LLM diagnostic of the same ConServe retrieval/reranking stack."""
import argparse
import json

from .common import ROOT, configure, jsonish, load, save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--query',default='alarm clock')
    parser.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    parser.add_argument('--out',default='diagnostics/retrieval_probe.json')
    args=parser.parse_args()
    configure('rule',args.device)
    from .validate import validate
    validation=validate()
    from src.autonomous.tools.conserve_search_tool import ConServeSearchTool
    from src.agents import reranker
    if reranker._load_triplet_state() is None:
        raise RuntimeError(reranker.get_triplet_reranker_diagnostics())
    result=ConServeSearchTool().run({
        'latest_user_message':args.query, 'history':[],
        'final_search_request':{'query_text':args.query,'category':args.query,
                                'hard_constraints':{},'soft_preferences':{},'preferred':{},'exclusions':{}},
        'retrieval_plan':{'query_variants':[args.query],'search_depth':50,'top_k':10},
    })
    patch=result.patch
    diagnostics=(patch.get('candidate_ledger') or {}).get('diagnostics') or {}
    products=load(ROOT/'data/product_base_table.json')
    candidates=(patch.get('raw_candidates') or [])+(patch.get('verified_candidates') or [])
    if diagnostics.get('actual_backend')!='conserve_stack':
        raise RuntimeError(f'Full stack did not execute: {diagnostics}')
    if not candidates:
        raise RuntimeError('Probe found no candidates; inspect catalogue availability and integration')
    if any(str(p.get('product_id')) not in products for p in candidates):
        raise ValueError('Nonexternal product returned by probe')
    report={'validation':validation,'query':args.query,'raw_count':len(patch.get('raw_candidates') or []),
            'verified_count':len(patch.get('verified_candidates') or []),'diagnostics':diagnostics,
            'verified':patch.get('verified_candidates') or [],
            'reranker':reranker.get_triplet_reranker_diagnostics(),
            'note':'No LLM calls. Retrieval execution check, not a relevance benchmark.'}
    save(ROOT/args.out,jsonish(report))
    print(json.dumps({'raw_count':report['raw_count'],'verified_count':report['verified_count'],
                      'backend':diagnostics.get('actual_backend')},indent=2))


if __name__=='__main__':
    main()
