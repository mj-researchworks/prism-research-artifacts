#!/usr/bin/env python3
"""Run reusable ablation studies for the autonomous ConServe pipeline.

Each ablation launches the full recommendation evaluator in a fresh Python
process with a different component configuration, then collects the produced
summary JSON into one comparison table. This keeps the scoring path identical
to autonomous_full_recommendation_eval and avoids notebook-only experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results" / "autonomous_ablation_study"


ABLATIONS: list[dict[str, Any]] = [
    {
        "name": "full_autonomous",
        "description": "Planner + LLM reflection + hybrid LLM critic.",
        "planner_mode": "auto",
        "env": {
            "AUTONOMOUS_PLANNER_MODE": "auto",
            "AUTONOMOUS_REFLECTION_MODE": "agent",
            "AUTONOMOUS_CRITIC_MODE": "hybrid",
        },
    },
    {
        "name": "no_reflection",
        "description": "Planner + rule reflection/state extraction + hybrid LLM critic.",
        "planner_mode": "auto",
        "env": {
            "AUTONOMOUS_PLANNER_MODE": "auto",
            "AUTONOMOUS_REFLECTION_MODE": "rule",
            "AUTONOMOUS_CRITIC_MODE": "hybrid",
        },
    },
    {
        "name": "no_critic",
        "description": "Planner + LLM reflection + hard/rule critic only.",
        "planner_mode": "auto",
        "env": {
            "AUTONOMOUS_PLANNER_MODE": "auto",
            "AUTONOMOUS_REFLECTION_MODE": "agent",
            "AUTONOMOUS_CRITIC_MODE": "rule",
        },
    },
    {
        "name": "no_reflection_no_critic",
        "description": "Planner with deterministic reflection/state extraction and hard/rule critic.",
        "planner_mode": "auto",
        "env": {
            "AUTONOMOUS_PLANNER_MODE": "auto",
            "AUTONOMOUS_REFLECTION_MODE": "rule",
            "AUTONOMOUS_CRITIC_MODE": "rule",
        },
    },
    {
        "name": "rule_planner_full_tools",
        "description": "Rule planner with LLM reflection and hybrid LLM critic.",
        "planner_mode": "rule",
        "env": {
            "AUTONOMOUS_PLANNER_MODE": "rule",
            "AUTONOMOUS_REFLECTION_MODE": "agent",
            "AUTONOMOUS_CRITIC_MODE": "hybrid",
        },
    },
    {
        "name": "deterministic_control",
        "description": "Rule planner + rule reflection/state extraction + hard/rule critic.",
        "planner_mode": "rule",
        "env": {
            "AUTONOMOUS_PLANNER_MODE": "rule",
            "AUTONOMOUS_REFLECTION_MODE": "rule",
            "AUTONOMOUS_CRITIC_MODE": "rule",
        },
    },
]


SUMMARY_KEYS = [
    "n_conversations_total",
    "n_conversations_failed",
    "n_conversations_with_gt",
    "n_turns",
    "elapsed_seconds",
    "ndcg@10",
    "hit@3",
    "hit@5",
    "hit@10",
    "mrr",
    "displayed_hit@5",
    "displayed_semantic_hit_rate",
    "internal_ranked_hit@5",
    "internal_ranked_semantic_hit_rate",
    "recall@50",
    "search_hit_rate",
    "retrieval_miss_rate",
    "rerank_loss@3_rate",
    "atr",
    "success_rate@5",
    "llm_planner_turn_rate",
    "planner_unchanged_execution_rate",
    "planner_fallback_turn_rate",
    "guard_intervention_turn_rate",
    "reflection_call_turn_rate",
    "reflection_skip_turn_rate",
    "reflection_cache_hit_rate",
    "critic_call_turn_rate",
    "critic_skip_turn_rate",
    "critic_cache_hit_rate",
    "llm_calls_per_conversation",
    "retrieval_call_turn_rate",
    "actual_retrieval_call_turn_rate",
    "retrieval_cache_hit_rate",
    "fallback_retrieval_calls",
    "full_stack_retrieval_failures",
    "search_again_actions",
    "latency_ms_avg",
    "planner_duration_ms_avg",
    "planner_tokens_per_turn",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max-convs", type=int, default=None)
    parser.add_argument("--retrieval-backend", default=os.getenv("AUTONOMOUS_RETRIEVAL_BACKEND", "conserve_stack"))
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Optional subset of ablation names. Default runs all configs.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only full_autonomous, no_reflection, no_critic, and no_reflection_no_critic.",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Allow ablation on test split. Use only after freezing design.",
    )
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test:
        print("Refusing test ablation without --allow-test. Use --split val while tuning.", file=sys.stderr)
        raise SystemExit(2)

    selected = select_configs(args)
    run_root = args.out_dir / f"{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for config in selected:
        row = run_ablation(config, args, run_root)
        rows.append(row)
        baseline = rows[0]
        add_deltas(row, baseline)
        print(
            f"{row['name']}: hit@5={row.get('hit@5')} "
            f"ndcg@10={row.get('ndcg@10')} mrr={row.get('mrr')} "
            f"latency={row.get('latency_ms_avg')}ms "
            f"llm_calls/conv={row.get('llm_calls_per_conversation')} "
            f"fallback={row.get('planner_fallback_turn_rate')} "
            f"failed={row.get('n_conversations_failed')}",
            flush=True,
        )

    baseline = rows[0] if rows else {}
    for row in rows:
        add_deltas(row, baseline)

    summary_csv = run_root / "ablation_summary.csv"
    summary_json = run_root / "ablation_summary.json"
    write_csv(summary_csv, rows)
    summary_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nAblation summary CSV  -> {summary_csv}")
    print(f"Ablation summary JSON -> {summary_json}")


def select_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.quick:
        names = {"full_autonomous", "no_reflection", "no_critic", "no_reflection_no_critic"}
    elif args.configs:
        names = set(args.configs)
    else:
        names = {config["name"] for config in ABLATIONS}

    configs = [config for config in ABLATIONS if config["name"] in names]
    missing = names - {config["name"] for config in configs}
    if missing:
        known = ", ".join(config["name"] for config in ABLATIONS)
        raise SystemExit(f"Unknown ablation config(s): {sorted(missing)}. Known: {known}")
    return configs


def run_ablation(config: dict[str, Any], args: argparse.Namespace, run_root: Path) -> dict[str, Any]:
    config_out = run_root / config["name"]
    config_out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("AUTONOMOUS_REASONING_BUDGET", "adaptive")
    env.setdefault("AUTONOMOUS_PLAN_POLICY_MODE", "validator")
    env.setdefault("AUTONOMOUS_PLANNER_PROVIDER", "deepseek")
    env.setdefault("AUTONOMOUS_PLANNER_MODEL", "deepseek-v4-flash")
    env.setdefault("AUTONOMOUS_PLANNER_TIMEOUT", "60")
    env.setdefault("AUTONOMOUS_REFLECTION_PROVIDER", "deepseek")
    env.setdefault("AUTONOMOUS_REFLECTION_MODEL", "deepseek-v4-flash")
    env.setdefault("AUTONOMOUS_REFLECTION_TIMEOUT", "60")
    env.setdefault("AUTONOMOUS_CRITIC_PROVIDER", "deepseek")
    env.setdefault("AUTONOMOUS_CRITIC_MODEL", "deepseek-v4-flash")
    env.setdefault("AUTONOMOUS_CRITIC_TIMEOUT", "60")
    env.setdefault("AUTONOMOUS_REFLECTION_WRITES_INTENT", "1")
    env.setdefault("AUTONOMOUS_INTENT_MODE", "lightweight")
    env.setdefault("MACPRS_TRIPLET_FUSION_MODE", "score")
    env.setdefault("MACPRS_TRIPLET_BLEND_WEIGHT", "0.6")
    env.setdefault("MACPRS_USE_TOP5_DIVERSITY_RERANK", "1")
    env.update(config["env"])
    env["AUTONOMOUS_RETRIEVAL_BACKEND"] = args.retrieval_backend
    env["PYTHONPATH"] = str(ROOT)

    command = [
        sys.executable,
        "-m",
        "eval.autonomous_full_recommendation_eval",
        "--split",
        args.split,
        "--planner-mode",
        str(config["planner_mode"]),
        "--retrieval-backend",
        args.retrieval_backend,
        "--out-dir",
        str(config_out),
    ]
    if args.max_convs:
        command.extend(["--max-convs", str(args.max_convs)])

    log_lines: list[str] = []
    proc = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_lines.append(line)
    returncode = proc.wait()
    (config_out / "run.log").write_text("".join(log_lines), encoding="utf-8")

    summary_path = latest_summary(config_out, args.split)
    row: dict[str, Any] = {
        "name": config["name"],
        "description": config["description"],
        "returncode": returncode,
        "planner_mode": config["planner_mode"],
        "reflection_mode": config["env"].get("AUTONOMOUS_REFLECTION_MODE"),
        "critic_mode": config["env"].get("AUTONOMOUS_CRITIC_MODE"),
        "retrieval_backend": args.retrieval_backend,
    }
    if summary_path:
        row.update(flatten_summary(summary_path))
        row["summary_path"] = str(summary_path)
    else:
        row["error"] = "summary not produced"
    return row


def latest_summary(path: Path, split: str) -> Path | None:
    summaries = sorted(path.glob(f"summary_{split}_*.json"), key=lambda item: item.stat().st_mtime)
    return summaries[-1] if summaries else None


def flatten_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {key: data.get(key) for key in SUMMARY_KEYS}
    diagnostics = data.get("triplet_reranker_diagnostics") or {}
    row["triplet_product_cache_count"] = diagnostics.get("triplet_product_cache_count")
    row["triplet_available"] = diagnostics.get("available")
    row["triplet_load_error"] = diagnostics.get("triplet_load_error")
    row["retrieval_actual_backends"] = json.dumps(data.get("retrieval_actual_backends") or {}, sort_keys=True)
    row["guard_intervention_reasons"] = json.dumps(data.get("guard_intervention_reasons") or {}, sort_keys=True)
    row["reflection_call_reasons"] = json.dumps(data.get("reflection_call_reasons") or {}, sort_keys=True)
    row["critic_call_reasons"] = json.dumps(data.get("critic_call_reasons") or {}, sort_keys=True)
    return row


def add_deltas(row: dict[str, Any], baseline: dict[str, Any]) -> None:
    if not baseline:
        return
    for key in ("hit@5", "ndcg@10", "mrr", "latency_ms_avg", "llm_calls_per_conversation"):
        row[f"{key}_delta_vs_full"] = round(as_float(row.get(key)) - as_float(baseline.get(key)), 4)


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
