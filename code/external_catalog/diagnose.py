"""Summarize automatic external-catalogue diagnostics for a completed run.

These are model/system diagnostics, not human relevance ground truth.
"""
import argparse
import json
from pathlib import Path

from .common import save


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    run = args.run_dir
    turns = read_jsonl(run / "turns.jsonl")
    audits = read_jsonl(run / "manual_relevance_audit.jsonl")
    conversations = read_jsonl(run / "conversations.jsonl")

    displayed = [p for t in turns for p in t.get("displayed_products", [])
                 if t.get("outcome") == "new_recommendation"]
    evidence = [p.get("evidence", {}) for p in displayed]
    def mean(key):
        vals = [p[key] for p in displayed if isinstance(p.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    revisions = 0
    rejection_events = 0
    stale_flags = 0
    for_revision = []
    for row in audits:
        prefs = row.get("active_preferences") or {}
        row_revision = False
        for value in prefs.values() if isinstance(prefs, dict) else []:
            if isinstance(value, dict) and value.get("previous_value") not in (None, ""):
                revisions += 1
                row_revision = True
        if row_revision:
            for_revision.append((str(row["conversation_id"]), int(row["turn_index"])))
        request = row.get("search_request") or {}
        exclusions = row.get("active_preferences", {})
        if (request.get("exclusions", {}).get("products") or
                request.get("exclusions", {}).get("brands") or
                request.get("exclusions", {}).get("features")):
            rejection_events += 1
        for p in row.get("product", {}).get("product_id", "") and [row.get("product")]:
            if p.get("product_id") in (request.get("exclusions", {}).get("products") or []):
                stale_flags += 1

    # Sequential novelty is a diagnostic for requests such as "other options".
    by_conv = {}
    for t in turns:
        if t.get("outcome") == "new_recommendation":
            by_conv.setdefault(str(t["conversation_id"]), []).append(t)
    revision_lists = 0
    revision_new_product_lists = 0
    repeated_products_after_revision = 0
    total_revision_products = 0
    for cid, seq in by_conv.items():
        for previous, current in zip(seq, seq[1:]):
            prev_ids = set(previous.get("newly_displayed_ids", []))
            cur_ids = set(current.get("newly_displayed_ids", []))
            # A state revision is represented in the exported active preference.
            if (cid, int(current["turn_index"])) in for_revision:
                revision_lists += 1
                overlap = len(prev_ids & cur_ids)
                total_revision_products += len(cur_ids)
                if overlap == 0:
                    revision_new_product_lists += 1
                if overlap:
                    repeated_products_after_revision += overlap

    requested_evidence = [e for e in evidence if e]
    result = {
        "run_dir": str(run),
        "conversations": len(conversations),
        "new_recommendation_turns": sum(t.get("outcome") == "new_recommendation" for t in turns),
        "displayed_products": len(displayed),
        "automatic_semantic": {
            "mean_semantic_score": mean("semantic_score"),
            "mean_triplet_score": mean("triplet_score"),
            "mean_evidence_score": mean("evidence_score"),
            "product_type_match_rate": sum(e.get("product_type_match") == 1 for e in evidence) / len(evidence) if evidence else None,
            "category_match_rate": sum(e.get("category_match") == 1 for e in evidence) / len(evidence) if evidence else None,
            "automatic_constraint_flag_rate": sum(e.get("violates_constraints") is True for e in evidence) / len(evidence) if evidence else None,
            "brand_match_rate_when_checked": sum(e.get("brand_match") == 1 for e in requested_evidence) / len(requested_evidence) if requested_evidence else None,
            "feature_match_rate_when_checked": sum(e.get("feature_match") == 1 for e in requested_evidence) / len(requested_evidence) if requested_evidence else None,
            "color_match_rate_when_checked": sum(e.get("color_match") is True for e in requested_evidence) / len(requested_evidence) if requested_evidence else None,
        },
        "preference_diagnostics": {
            "preference_revision_records": revisions,
            "turns_with_exclusions": rejection_events,
            "automatically_detected_stale_exclusions": stale_flags,
            "revision_recommendation_lists": revision_lists,
            "revision_lists_with_zero_product_overlap": revision_new_product_lists,
            "revision_list_zero_overlap_rate": revision_new_product_lists / revision_lists if revision_lists else None,
            "repeated_products_after_revision": repeated_products_after_revision,
            "revision_product_novelty_rate": ((total_revision_products - repeated_products_after_revision) / total_revision_products
                                               if total_revision_products else None),
        },
        "catalogue_support_diagnostics": {
            "products_with_known_category": sum(bool(p.get("category")) for p in displayed),
            "products_with_known_price": sum(p.get("price") is not None for p in displayed),
            "products_with_known_brand": sum(bool(p.get("brand")) for p in displayed),
            "category_unknown_rate": sum(not bool(p.get("category")) for p in displayed) / len(displayed) if displayed else None,
            "price_unknown_rate": sum(p.get("price") is None for p in displayed) / len(displayed) if displayed else None,
            "brand_unknown_rate": sum(not bool(p.get("brand")) for p in displayed) / len(displayed) if displayed else None,
        },
        "latency_seconds": {
            "mean": sum(c.get("latency_seconds", 0) for c in conversations) / len(conversations) if conversations else None,
            "max": max((c.get("latency_seconds", 0) for c in conversations), default=None),
        },
        "interpretation": "Automatic semantic and compliance signals are diagnostics only; independent human judgments are required for external relevance claims.",
    }
    save(run / "automatic_diagnostics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
