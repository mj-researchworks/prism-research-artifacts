#!/usr/bin/env python3
"""Export manuscript tables from saved PRISM artifacts, without model/API calls.

Run from the deposit root:
    python code/reproduce_saved_tables.py

Only Python's standard library is required. The exports intentionally preserve
the supplied finalized audit tables; they do not independently validate human
judgments or recompute semantic scores.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tables"
SPLITS = ROOT / "data" / "four_folds"
RESULTS = ROOT / "results"
AUDIT = RESULTS


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(name: str, rows, fields=None):
    rows = list(rows)
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def copy_csv(src: Path, name: str, filter_fn=lambda r: True):
    rows = read_csv(src)
    write_csv(name, [r for r in rows if filter_fn(r)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Fold-partition counts are derived directly from exact saved split summaries.
    all_splits = read_json(SPLITS / "all_folds_summary.json")
    split_rows = []
    for fold, data in all_splits["folds"].items():
        split_rows.append({
            "Fold": fold,
            "Training conversations": data["train"]["n_conversations"],
            "Validation conversations": data["val"]["n_conversations"],
            "Test conversations": data["test"]["n_conversations"],
            "Test conversations with source recommendation action": data["test"]["recommendation_bearing_conversations"],
        })
    write_csv("table01_fold_partitions_from_split_files.csv", split_rows)

    # Table 2: configuration matrix for the three implemented systems.
    # These labels summarize the compared runtime configurations; the exact
    # executable behavior remains in the runtime source and run records.
    configuration_rows = [
        {"Setting": "Conversational memory", "Full PRISM": "Shared blackboard", "Fixed workflow": "Same blackboard", "Single agent": "Controller-owned structured memory"},
        {"Setting": "Nominal retrieval settings", "Full PRISM": "Matched", "Fixed workflow": "Matched", "Single agent": "Matched"},
        {"Setting": "Separate executive planner", "Full PRISM": "Yes", "Fixed workflow": "No", "Single agent": "No"},
        {"Setting": "Separate reflection and critic components", "Full PRISM": "Yes", "Fixed workflow": "Yes", "Single agent": "No"},
        {"Setting": "Search construction", "Full PRISM": "Registered module", "Fixed workflow": "Same module", "Single agent": "Controller-generated request"},
        {"Setting": "Final recommendation selection", "Full PRISM": "Decision module", "Fixed workflow": "Same module", "Single agent": "First three tool-returned candidates"},
        {"Setting": "Adaptive recovery", "Full PRISM": "Bounded", "Fixed workflow": "Disabled", "Single agent": "Disabled"},
        {"Setting": "Execution control", "Full PRISM": "Planner-guided", "Fixed workflow": "Fixed sequence", "Single agent": "Single LLM controller"},
    ]
    write_csv("table02_system_configurations.csv", configuration_rows)

    # Tables 3-4: finalized fold-level and mean/SD comparison tables.
    cmp_dir = AUDIT / "revisions"
    copy_csv(cmp_dir / "ranking_by_fold.csv", "table03_ranking_by_fold.csv",
             lambda r: r.get("System") == "Full PRISM")
    copy_csv(cmp_dir / "ranking_mean_sd.csv", "table04_ranking_mean_sd.csv")

    # Table 5: pooled revision outcomes from the common 370-event inventory.
    metrics = read_json(cmp_dir / "matched_preference_metrics.json")
    rows = []
    for system, block in metrics.items():
        if not isinstance(block, dict) or "pooled" not in block:
            continue
        p = block["pooled"]
        rows.append({
            "System": system, "Reference revision events": p["events"],
            "Follow-up recommendation events": p["follow_up"],
            "No-follow-up events": p["no_follow_up"],
            "Coverage": p["coverage"], "PRCR": p["PRCR"], "Compliant": p["compliant_count"],
            "SPVR": p["SPVR"], "Stale": p["stale_count"],
            "Unknown rate": p["unknown"], "Unknown": p["unknown_count"],
            "Compliant follow-ups per reference event": p["compliant_count"] / p["events"],
        })
    write_csv("table05_revision_outcomes.csv", rows)

    # Table 6: manuscript Fold 3 component substitutions from saved run summaries.
    ablation_specs = [
        ("Full PRISM", RESULTS / "prism/fold3/summary.json"),
        ("Rule verifier", RESULTS / "fold3_ablations/no_llm_critic/summary.json"),
        ("Rule reflection", RESULTS / "fold3_ablations/no_llm_reflection/summary.json"),
        ("Both rule substitutions", RESULTS / "fold3_ablations/no_llm_reflection_no_llm_critic/summary.json"),
    ]
    ablation_metrics = ["hit@3", "hit@5", "hit@10", "ndcg@10", "mrr", "search_hit_rate", "retrieval_miss_rate", "displayed_semantic_hit_rate"]
    write_csv("table06_fold3_ablations.csv", [
        {"System": label, **{m: read_json(path)[m] for m in ablation_metrics}}
        for label, path in ablation_specs
    ])

    # Table 7: separate critic diagnostic (finalized analysis artifact).
    copy_csv(AUDIT / "critic_diagnostic/fold3_critic_summary.csv", "table07_critic_diagnostic.csv")

    # Table 8: count the 28 human-audited cases by recorded manual label.
    audit_csv = AUDIT / "manual_audit/advisor_manual_audit_sample_prcr_spvr.csv"
    audit_rows = read_csv(audit_csv)
    labels = [r.get("Manual Audit Label", "") for r in audit_rows]
    counts = {}
    for label in labels:
        if label:
            counts[label] = counts.get(label, 0) + 1
    label_map = {
        "correctly handled": "Correctly handled revision",
        "genuine missed revision": "Classified as missed revision",
        "no suitable catalogue match": "No suitable catalog match",
        "catalogue limitation": "Catalog limitation",
        "ambiguous/non-actionable": "Ambiguous or nonactionable statement",
        "unknown/unverifiable": "Unknown or unverifiable",
    }
    audit_out = [{"Audit category": name, "Count": counts.get(key, 0)} for key, name in label_map.items()]
    audit_out.append({"Audit category": "Total", "Count": sum(counts.values())})
    write_csv("table08_manual_audit.csv", audit_out)

    # Tables 9-10: finalized metadata and revision-type summaries.
    tax_dir = AUDIT / "metadata_audit"
    copy_csv(tax_dir / "metadata_targeted_adjudication_summary.csv", "table09_metadata_evidence.csv")
    copy_csv(tax_dir / "revision_type_taxonomy.csv", "table10_revision_type_taxonomy.csv")

    # Table 11: rerun the saved deterministic diagnostic on saved JSONL only.
    run_dir = RESULTS / "external_catalog"
    def read_jsonl(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    turns = read_jsonl(run_dir / "turns.jsonl")
    audits = read_jsonl(run_dir / "manual_relevance_audit.jsonl")
    conversations = read_jsonl(run_dir / "conversations.jsonl")
    displayed = [p for t in turns for p in t.get("displayed_products", [])
                 if t.get("outcome") == "new_recommendation"]
    evidence = [p.get("evidence", {}) for p in displayed]
    def mean(key):
        vals = [p[key] for p in displayed if isinstance(p.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None
    revision_turns = set()
    revision_records = 0
    for row in audits:
        prefs = row.get("active_preferences") or {}
        for value in prefs.values() if isinstance(prefs, dict) else []:
            if isinstance(value, dict) and value.get("previous_value") not in (None, ""):
                revision_records += 1
                revision_turns.add((str(row["conversation_id"]), int(row["turn_index"])))
    by_conv = {}
    for turn in turns:
        if turn.get("outcome") == "new_recommendation":
            by_conv.setdefault(str(turn["conversation_id"]), []).append(turn)
    revision_lists = zero_overlap = repeated = total_revision_products = 0
    for cid, seq in by_conv.items():
        for previous, current in zip(seq, seq[1:]):
            if (cid, int(current["turn_index"])) in revision_turns:
                revision_lists += 1
                prev_ids = set(previous.get("newly_displayed_ids", []))
                cur_ids = set(current.get("newly_displayed_ids", []))
                overlap = len(prev_ids & cur_ids)
                total_revision_products += len(cur_ids)
                zero_overlap += overlap == 0
                repeated += overlap
    table11 = {
        "Product-type match rate": sum(e.get("product_type_match") == 1 for e in evidence) / len(evidence),
        "Category-match rate": sum(e.get("category_match") == 1 for e in evidence) / len(evidence),
        "Mean semantic score": mean("semantic_score"),
        "Mean triplet score": mean("triplet_score"),
        "Mean evidence score": mean("evidence_score"),
        "Preference-history records": revision_records,
        "Revision-list comparisons": revision_lists,
        "Zero-overlap comparisons": f"{zero_overlap}/{revision_lists} ({zero_overlap/revision_lists:.4%})",
        "Revision-product novelty": f"{(total_revision_products-repeated)/total_revision_products:.4%}",
    }
    write_csv("table11_external_catalog_diagnostics.csv",
              [{"Measure": k, "Value": v} for k, v in table11.items()])

    if len(audit_rows) != 28 or sum(counts.values()) != 28:
        raise SystemExit(f"Expected 28 labeled audit rows, found {len(audit_rows)} rows / {sum(counts.values())} labels")
    print(f"Wrote offline table exports to {OUT.relative_to(ROOT)}")
    print("No LLM calls or non-standard Python packages used.")


if __name__ == "__main__":
    main()
