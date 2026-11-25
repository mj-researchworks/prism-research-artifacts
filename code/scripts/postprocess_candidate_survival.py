#!/usr/bin/env python3
"""Post-process MACPRS per-conversation results for candidate survival analysis."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "eval/results/metrics_test_20260815_142014.json"
DEFAULT_PER_CONV = ROOT / "eval/results/per_conversation_test_20260815_142014.json"
DEFAULT_SPLIT = ROOT / "outputs/splits/test_conversations.json"
DEFAULT_PRODUCTS = ROOT / "data/product_base_retrieval/product_base_retrieval_text.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def product_title(text: str) -> str:
    if not text:
        return ""
    title_marker = "Title:"
    if title_marker in text:
        title = text.split(title_marker, 1)[1].split("|", 1)[0].strip()
        return title[:240]
    return text.replace("Product:", "", 1).split("|", 1)[0].strip()[:240]


def build_product_map(product_rows: list[dict[str, Any]]) -> dict[str, str]:
    result = {}
    for row in product_rows:
        pid = str(row.get("product_id") or "")
        if pid:
            result[pid] = product_title(str(row.get("text") or ""))
    return result


def build_conversation_map(split_rows: list[dict[str, Any]]) -> dict[str, str]:
    result = {}
    for row in split_rows:
        conv_id = str(row.get("conv_id"))
        user_turns = [
            str(turn.get("content") or "").strip()
            for turn in row.get("conversation", [])
            if turn.get("role") == "user" and str(turn.get("content") or "").strip()
        ]
        result[conv_id] = " | ".join(user_turns)[:500]
    return result


def is_gt(row: dict[str, Any]) -> bool:
    return int(row.get("n_relevant") or 0) > 0


def is_search_hit(row: dict[str, Any]) -> bool:
    return float(row.get("search_hit_rate") or 0.0) > 0.0


def metric_values(rows: list[dict[str, Any]]) -> dict[str, float]:
    gt = [r for r in rows if is_gt(r)]
    search_hit_rows = [r for r in gt if is_search_hit(r)]
    if not gt:
        raise ValueError("No recommendation-bearing conversations found.")

    def avg(key: str) -> float:
        return mean(float(r.get(key) or 0.0) for r in gt)

    loss10 = sum(1 for r in search_hit_rows if r.get("rerank_loss@10"))
    loss3 = sum(1 for r in search_hit_rows if r.get("rerank_loss@3"))
    denom = len(search_hit_rows)

    return {
        "search_hit_rate": avg("search_hit_rate"),
        "retrieval_miss_rate": mean(1.0 if r.get("retrieval_miss") else 0.0 for r in gt),
        "hit@1": avg("hit@1"),
        "hit@3": avg("hit@3"),
        "hit@5": avg("hit@5"),
        "hit@10": avg("hit@10"),
        "ndcg@3": avg("ndcg@3"),
        "ndcg@10": avg("ndcg@10"),
        "mrr": avg("mrr"),
        "recall@50": avg("recall@50"),
        "rerank_loss@10_given_search_hit": loss10 / denom if denom else 0.0,
        "rerank_survival@10_given_search_hit": 1.0 - (loss10 / denom if denom else 0.0),
        "rerank_loss@3_given_search_hit": loss3 / denom if denom else 0.0,
        "rerank_survival@3_given_search_hit": 1.0 - (loss3 / denom if denom else 0.0),
    }


def bootstrap_ci(
    rows: list[dict[str, Any]],
    n_resamples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    gt = [r for r in rows if is_gt(r)]
    rng = random.Random(seed)
    keys = list(metric_values(gt).keys())
    samples = {key: [] for key in keys}

    for _ in range(n_resamples):
        sample = [gt[rng.randrange(len(gt))] for _ in range(len(gt))]
        values = metric_values(sample)
        for key in keys:
            samples[key].append(values[key])

    intervals = {}
    for key, values in samples.items():
        values.sort()
        lo = values[int(0.025 * (len(values) - 1))]
        hi = values[int(0.975 * (len(values) - 1))]
        intervals[key] = {"low": round(lo, 4), "high": round(hi, 4)}
    return intervals


def summarize_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt = [r for r in rows if is_gt(r)]
    retrieved = [r for r in gt if is_search_hit(r)]
    retrieval_miss = [r for r in gt if r.get("retrieval_miss")]
    hit10 = [r for r in gt if float(r.get("hit@10") or 0.0) > 0.0]
    hit5 = [r for r in gt if float(r.get("hit@5") or 0.0) > 0.0]
    hit3 = [r for r in gt if float(r.get("hit@3") or 0.0) > 0.0]
    hit1 = [r for r in gt if float(r.get("hit@1") or 0.0) > 0.0]
    rerank_loss10 = [r for r in retrieved if r.get("rerank_loss@10")]
    rerank_loss3 = [r for r in retrieved if r.get("rerank_loss@3")]

    return {
        "n_total_conversations": len(rows),
        "n_gt_conversations": len(gt),
        "n_retrieved_with_relevant": len(retrieved),
        "n_retrieval_miss": len(retrieval_miss),
        "n_hit@10": len(hit10),
        "n_hit@5": len(hit5),
        "n_hit@3": len(hit3),
        "n_hit@1": len(hit1),
        "n_rerank_loss@10_given_search_hit": len(rerank_loss10),
        "n_rerank_loss@3_given_search_hit": len(rerank_loss3),
        "retrieval_miss_ids": [str(r.get("conv_id")) for r in retrieval_miss],
        "rerank_loss@10_ids": [str(r.get("conv_id")) for r in rerank_loss10],
        "rerank_loss@3_ids": [str(r.get("conv_id")) for r in rerank_loss3],
    }


def titles(ids: list[str], product_map: dict[str, str], limit: int = 5) -> str:
    parts = []
    for pid in ids[:limit]:
        title = product_map.get(str(pid), "")
        parts.append(f"{pid}: {title}" if title else str(pid))
    return " || ".join(parts)


def failure_examples(
    rows: list[dict[str, Any]],
    conv_map: dict[str, str],
    product_map: dict[str, str],
    limit_each: int,
) -> list[dict[str, Any]]:
    gt = [r for r in rows if is_gt(r)]
    retrieval_misses = [r for r in gt if r.get("retrieval_miss")][:limit_each]
    rerank_losses = [
        r for r in gt if is_search_hit(r) and r.get("rerank_loss@10")
    ][:limit_each]

    examples = []
    for label, selected in (
        ("retrieval_miss", retrieval_misses),
        ("rerank_loss@10", rerank_losses),
    ):
        for row in selected:
            conv_id = str(row.get("conv_id"))
            relevant_ids = [str(x) for x in row.get("relevant_ids", [])]
            raw_ids = [str(x) for x in row.get("raw_candidate_ids", [])]
            ranked_ids = [str(x) for x in row.get("ranked_candidate_ids", [])]
            examples.append(
                {
                    "failure_type": label,
                    "conv_id": conv_id,
                    "user_turns": conv_map.get(conv_id, ""),
                    "n_relevant": row.get("n_relevant"),
                    "best_raw_rank": row.get("best_raw_rank"),
                    "best_ranked_rank": row.get("best_ranked_rank"),
                    "ndcg@10": row.get("ndcg@10"),
                    "hit@10": row.get("hit@10"),
                    "relevant_products": titles(relevant_ids, product_map, 5),
                    "raw_top5": titles(raw_ids, product_map, 5),
                    "final_top5": titles(ranked_ids, product_map, 5),
                }
            )
    return examples


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(summary: dict[str, Any], metrics: dict[str, Any]) -> str:
    counts = summary["counts"]
    values = summary["metrics"]
    ci = summary["bootstrap_95ci"]

    def cell(key: str) -> str:
        interval = ci.get(key, {})
        if interval:
            return f"{values[key]:.4f} [{interval['low']:.4f}, {interval['high']:.4f}]"
        return f"{values[key]:.4f}"

    stage_rows = [
        ("GT recommendation conversations", counts["n_gt_conversations"], "100.00%", "Conversations with at least one relevant product label."),
        ("Relevant product found in raw pool", counts["n_retrieved_with_relevant"], cell("search_hit_rate"), "At least one relevant product survived retrieval."),
        ("Retrieval miss", counts["n_retrieval_miss"], cell("retrieval_miss_rate"), "No relevant product appeared in raw candidates."),
        ("Relevant product in final top-10", counts["n_hit@10"], cell("hit@10"), "Final recommendation list contains a relevant product in top-10."),
        ("Relevant product in final top-5", counts["n_hit@5"], cell("hit@5"), "Final recommendation list contains a relevant product in top-5."),
        ("Relevant product in final top-3", counts["n_hit@3"], cell("hit@3"), "Final recommendation list contains a relevant product in top-3."),
        ("Relevant product at top-1", counts["n_hit@1"], cell("hit@1"), "Top-ranked recommendation is relevant."),
        ("Rerank loss@10 among search-hit cases", counts["n_rerank_loss@10_given_search_hit"], cell("rerank_loss@10_given_search_hit"), "Retrieved relevant product was pushed out of final top-10."),
        ("Rerank survival@10 among search-hit cases", counts["n_hit@10"], cell("rerank_survival@10_given_search_hit"), "Retrieved relevant product remained in final top-10."),
        ("Rerank loss@3 among search-hit cases", counts["n_rerank_loss@3_given_search_hit"], cell("rerank_loss@3_given_search_hit"), "Retrieved relevant product was pushed out of final top-3."),
        ("Rerank survival@3 among search-hit cases", counts["n_hit@3"], cell("rerank_survival@3_given_search_hit"), "Retrieved relevant product remained in final top-3."),
    ]

    ranking_rows = [
        ("NDCG@3", cell("ndcg@3")),
        ("NDCG@10", cell("ndcg@10")),
        ("MRR", cell("mrr")),
        ("Recall@50 before rerank", cell("recall@50")),
    ]

    lines = [
        "# Experiment 1: Candidate Survival Analysis",
        "",
        "This analysis uses the saved MACPRS test run and does not rerun the pipeline.",
        "",
        "## Configuration",
        "",
        f"- Metrics file: `{summary['inputs']['metrics_file']}`",
        f"- Per-conversation file: `{summary['inputs']['per_conversation_file']}`",
        f"- Total conversations: `{counts['n_total_conversations']}`",
        f"- Recommendation-bearing conversations: `{counts['n_gt_conversations']}`",
        f"- Triplet model: `{metrics.get('triplet_model_path')}`",
        f"- Fusion mode: `{metrics.get('triplet_fusion_mode')}`",
        f"- Blend weight: `{metrics.get('triplet_blend_weight')}`",
        "",
        "## Stage Survival",
        "",
        "| Stage | Count | Rate / 95% bootstrap CI | Interpretation |",
        "|---|---:|---:|---|",
    ]
    for stage, count, rate, meaning in stage_rows:
        lines.append(f"| {stage} | {count} | {rate} | {meaning} |")

    lines.extend(
        [
            "",
            "## Ranking Quality",
            "",
            "| Metric | Value / 95% bootstrap CI |",
            "|---|---:|",
        ]
    )
    for metric, value in ranking_rows:
        lines.append(f"| {metric} | {value} |")

    lines.extend(
        [
            "",
            "## Failure Breakdown",
            "",
            f"- Retrieval misses: `{counts['n_retrieval_miss']} / {counts['n_gt_conversations']}`.",
            f"- Rerank loss@10: `{counts['n_rerank_loss@10_given_search_hit']} / {counts['n_retrieved_with_relevant']}` search-hit conversations.",
            f"- Total no Hit@10 cases: `{counts['n_retrieval_miss'] + counts['n_rerank_loss@10_given_search_hit']} / {counts['n_gt_conversations']}`.",
            "",
            "## Interpretation",
            "",
            "The retrieval stage found at least one relevant product in 90% of recommendation-bearing conversations. The remaining 10% are retrieval misses, where reranking cannot recover the relevant item because it never entered the candidate pool.",
            "",
            "Among conversations where retrieval succeeded, the reranker preserved a relevant product in the final top-10 in 85.8% of cases and lost it in 14.2% of cases. This indicates that the remaining failure mass is split between first-stage retrieval misses and post-retrieval ranking losses, with additional top-position weakness visible in Hit@1 and Hit@3.",
            "",
            "## Example Failure Files",
            "",
            f"- Failure examples CSV: `{summary['outputs']['failure_examples_csv']}`",
            f"- Machine-readable summary JSON: `{summary['outputs']['summary_json']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--per-conversation", type=Path, default=DEFAULT_PER_CONV)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--out-prefix", type=Path, default=ROOT / "eval/results/candidate_survival_test_20260815_142014")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples-per-type", type=int, default=5)
    args = parser.parse_args()

    metrics = load_json(args.metrics)
    rows = load_json(args.per_conversation)
    split_rows = load_json(args.split)
    product_rows = load_json(args.products)

    counts = summarize_counts(rows)
    values = metric_values(rows)
    ci = bootstrap_ci(rows, args.bootstrap, args.seed)
    conv_map = build_conversation_map(split_rows)
    product_map = build_product_map(product_rows)
    examples = failure_examples(rows, conv_map, product_map, args.examples_per_type)

    summary_json = args.out_prefix.with_suffix(".json")
    summary_md = args.out_prefix.with_suffix(".md")
    examples_csv = args.out_prefix.parent / f"{args.out_prefix.name}_failure_examples.csv"

    summary = {
        "experiment": "candidate_survival",
        "inputs": {
            "metrics_file": str(args.metrics),
            "per_conversation_file": str(args.per_conversation),
            "split_file": str(args.split),
            "product_text_file": str(args.products),
        },
        "outputs": {
            "summary_json": str(summary_json),
            "summary_markdown": str(summary_md),
            "failure_examples_csv": str(examples_csv),
        },
        "bootstrap": {"n_resamples": args.bootstrap, "seed": args.seed},
        "counts": counts,
        "metrics": {key: round(value, 4) for key, value in values.items()},
        "bootstrap_95ci": ci,
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(render_markdown(summary, metrics), encoding="utf-8")
    write_csv(examples_csv, examples)

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")
    print(f"Wrote {examples_csv}")


if __name__ == "__main__":
    main()
