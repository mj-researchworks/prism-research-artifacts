#!/usr/bin/env python3
"""Build a portable provenance index from the deposited run records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "run_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    runs = []
    for path in sorted(RESULTS.rglob("summary.json")):
        if "smoke" in str(path).lower():
            continue
        data = load(path)
        runs.append({
            "artifact": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
            "fold": data.get("fold") or (path.parent.name if path.parent.name.startswith("fold") else None),
            "conversation_count": data.get("n_conversations_total"),
            "ground_truth_count": data.get("n_conversations_with_gt"),
            "failed_conversation_count": data.get("n_conversations_failed"),
            "metrics": {k: data[k] for k in (
                "hit@3", "hit@5", "hit@10", "ndcg@10", "mrr",
                "search_hit_rate", "retrieval_miss_rate", "displayed_semantic_hit_rate"
            ) if k in data},
            "effective_config": data.get("effective_config", {}),
        })

    split_file = ROOT / "data/four_folds/all_folds_summary.json"
    external_manifest = RESULTS / "external_catalog/run_manifest.json"
    code_root = ROOT / "code"
    prompt_sources = []
    for path in sorted(code_root.rglob("*.py")):
        prompt_sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})

    manifest = {
        "manifest_version": 1,
        "purpose": "Provenance index for the manuscript's saved PRISM results; generated from deposited run summaries.",
        "reproduction": {
            "offline_table_export": "python code/reproduce_saved_tables.py",
            "requires_model_or_api_calls": False,
            "python_dependencies": "Python standard library only for saved-table exports",
        },
        "dataset": {
            "name": "PSCon English",
            "source": "https://github.com/JieZouIR/PSCon",
            "license_as_stated_by_maintainers": "CC BY 4.0",
            "data_file": "data/PSCon/conversation_en.json",
            "four_fold_assignments": "data/four_folds/",
            "split_summary_sha256": sha256(split_file),
            "total_conversations": load(split_file).get("total_conversations"),
        },
        "recorded_model_identifiers": {
            "llm_provider": "deepseek where recorded in effective_config",
            "llm_model": "deepseek-v4-flash where recorded in effective_config",
            "dense_retrieval_model": "sentence-transformers/all-MiniLM-L6-v2 in external-catalog run manifest",
            "fine_tuned_e5": "fold-specific E5 paths are recorded in run summaries; weights are not included here",
            "immutable_provider_model_revisions": "Not recorded in the supplied run files",
        },
        "run_records": runs,
        "metric_definitions": {
            "ranking": {
                "implementation": "code/eval/autonomous_full_recommendation_eval.py",
                "Hit@K": "Per eligible conversation, 1 when at least one reference product ID occurs in the evaluated top K; aggregate is the mean over eligible conversations.",
                "NDCG@K": "Per-conversation discounted gain using the reference product IDs and evaluated ranking, normalized by ideal ranking; aggregate is the mean over eligible conversations.",
                "MRR": "Mean reciprocal rank of the first reference product in the evaluated ranking.",
                "Search Hit Rate": "Fraction of eligible conversations where the raw retrieved candidate IDs include a reference product.",
                "Retrieval Miss Rate": "1 minus Search Hit Rate.",
                "Displayed Semantic Hit Rate": "Fraction whose displayed products satisfy the semantic reference-match implementation; see evaluator source for model and threshold details.",
                "four-fold summary": "Arithmetic mean and sample standard deviation across the four fold-level scores, as in ranking_mean_sd.csv.",
            },
            "revision_outcomes": {
                "PRCR": "Compliant event-follow-ups / observed follow-up event pairs.",
                "SPVR": "Stale-preference violation event-follow-ups / observed follow-up event pairs.",
                "Unknown": "Unverifiable event-follow-ups / observed follow-up event pairs.",
                "coverage": "Observed follow-up recommendation events / 370 PRISM-derived reference revision events.",
                "semantic_threshold": 0.5,
                "active_match_fraction_threshold": 0.5,
                "follow_up_rule": "First logged recommendation strictly after the revision turn; retained sets count when logged as recommended.",
                "limitations": "Proxy scoring, not independent human adjudication of all baseline outputs; see the Metrics and interpretation limits section in the root README.",
            },
            "external_catalog_diagnostics": {
                "implementation": "code/external_catalog/diagnose.py; same logic is used by reproduce_saved_tables.py on saved JSONL.",
                "interpretation": "Automatic diagnostics only; not human relevance ground truth.",
            },
        },
        "external_catalog_run_manifest": {
            "path": external_manifest.relative_to(ROOT).as_posix(),
            "sha256": sha256(external_manifest),
            "content": load(external_manifest),
        },
        "prompt_and_runtime_source_hashes": prompt_sources,
        "metric_and_analysis_sources": [
            "code/eval/",
            "code/fixed_workflow_eval.py",
            "code/external_catalog/",
        ],
        "known_provenance_limits": [
            "Main-run package/dependency versions and hardware are not consistently recorded in the saved run summaries.",
            "Provider model revision identifiers and API request-level settings are not present in the supplied logs.",
            "The external-run manifest records package versions and effective environment; see its embedded record.",
            "PRCR/SPVR is a reference-based proxy with thresholds and known limitations documented in the root README; it is not independent human adjudication of every baseline event.",
            "Amazon Reviews 2023-derived catalog snapshots are rights-review pending and must not be redistributed unless cleared.",
        ],
    }
    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} with {len(runs)} run summaries and {len(prompt_sources)} source hashes")


if __name__ == "__main__":
    main()
