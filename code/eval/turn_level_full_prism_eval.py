#!/usr/bin/env python3
"""Turn-level full-PRISM recommendation evaluation.

This is a thin evaluator for the "correct recommendation turn" experiment.
It does not change PRISM itself. It selects only user turns whose next
dataset/system action is Recommend, hides that original recommendation, runs
full PRISM on the conversation prefix up to that user turn, and scores the
generated ranking against the products attached to that same hidden
recommendation turn.

Run from the PRISM repository root, for example:

    python eval/turn_level_full_prism_eval.py --split test --max-cases 5

For a full run, omit --max-cases.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.autonomous_full_recommendation_eval import (  # noqa: E402
    hit_rate_at_k,
    load_split,
    mean_key,
    ndcg_at_k,
    percentile_values,
    product_ids,
    reciprocal_rank,
    score_turn,
    write_csv,
)


DEFAULT_OUT_DIR = ROOT / "eval" / "results" / "turn_level_full_prism"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-convs", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--planner-mode", default=os.getenv("AUTONOMOUS_PLANNER_MODE", "rule"))
    parser.add_argument(
        "--label-source",
        default="union",
        choices=["recommended", "search_results", "union"],
        help=(
            "Turn-level ground truth source. The default 'union' matches the "
            "original full-system evaluator: recommended_products ∪ search_results. "
            "Use 'recommended' only for a stricter displayed-product diagnostic."
        ),
    )
    parser.add_argument("--execution-scope", default="full")
    parser.add_argument(
        "--retrieval-backend",
        default=os.getenv("AUTONOMOUS_RETRIEVAL_BACKEND", "fallback"),
        help="Use fallback/local_safe for Mac-safe eval, or auto for full ConServe stack.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.environ["AUTONOMOUS_EXECUTION_SCOPE"] = args.execution_scope
    os.environ["AUTONOMOUS_RETRIEVAL_BACKEND"] = args.retrieval_backend
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from src.autonomous.eval_logger import EvalLogger
    from src.autonomous.runner import AutonomousRunner

    conversations = load_split(args.split)
    if args.max_convs:
        conversations = conversations[: args.max_convs]

    cases = build_recommendation_turn_cases(conversations, label_source=args.label_source)
    if args.max_cases:
        cases = cases[: args.max_cases]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Turn-level full PRISM eval: split={args.split} "
        f"conversations={len(conversations)} cases={len(cases)}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    started = time.time()
    for case_index, case in enumerate(cases, start=1):
        cid = case["conversation_id"]
        relevant_ids = set(case["target_product_ids"])
        print(
            f"[{case_index:4}/{len(cases)}] conv={cid} "
            f"eval_user_turn={case['eval_user_turn_index']} targets={len(relevant_ids)}",
            flush=True,
        )
        try:
            runner = AutonomousRunner(
                planner_mode=args.planner_mode,
                execution_scope=args.execution_scope,
                eval_logger=EvalLogger(enabled=False),
            )

            response = ""
            state: dict[str, Any] = {}
            scored_latency_ms = 0
            scored_before_trace_len = 0

            for prefix_index, turn in enumerate(case["prefix_user_turns"], start=1):
                user_message = str(turn.get("content") or "").strip()
                if not user_message:
                    continue
                before_trace_len = len(runner.blackboard.snapshot().get("planner_trace") or [])
                turn_started = time.perf_counter()
                response, state = runner.run_turn(user_message)
                latency_ms = int((time.perf_counter() - turn_started) * 1000)
                if prefix_index == len(case["prefix_user_turns"]):
                    scored_latency_ms = latency_ms
                    scored_before_trace_len = before_trace_len

            row = score_turn(
                cid=cid,
                turn_index=int(case["eval_user_turn_index"]),
                turn=case["eval_turn"],
                response=response,
                state=state,
                relevant_ids=relevant_ids,
                before_trace_len=scored_before_trace_len,
                latency_ms=scored_latency_ms,
            )
            row.update(case_metadata(case, row, relevant_ids))
        except Exception as exc:
            row = failed_case_row(case, exc)
            print(f"  failed: {row['eval_error']}", flush=True)
        rows.append(row)

    summary = summarize_cases(rows, args, elapsed=time.time() - started)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows_path = args.out_dir / f"turn_level_full_prism_{args.split}_{timestamp}.csv"
    summary_path = args.out_dir / f"turn_level_full_prism_summary_{args.split}_{timestamp}.json"
    write_csv(rows_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nTurn-level full PRISM evaluation complete")
    print(f"Rows    -> {rows_path}")
    print(f"Summary -> {summary_path}")
    print(
        "Main metrics: "
        f"n={summary['scored_turns']} "
        f"Hit@5={summary['hit@5']} "
        f"NDCG@5={summary['ndcg@5']} "
        f"MRR={summary['mrr']} "
        f"p95_latency_ms={summary['latency_ms_p95']}"
    )
    return 0


def build_recommendation_turn_cases(
    conversations: list[dict[str, Any]],
    *,
    label_source: str = "union",
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for conv in conversations:
        cid = str(conv.get("conv_id"))
        prefix_user_turns: list[dict[str, Any]] = []
        pending_user: dict[str, Any] | None = None
        user_turn_index = 0

        for item in conv.get("conversation", []):
            role = item.get("role")
            if role == "user":
                user_turn_index += 1
                pending_user = dict(item)
                pending_user["user_turn_index"] = user_turn_index
                prefix_user_turns.append(pending_user)
                continue

            if role != "system" or pending_user is None:
                continue

            pending_user["next_system_action"] = item.get("action")
            pending_user["next_system_content"] = item.get("content")
            pending_user["clarifying_attribute"] = item.get("clarifying_attribute")

            if item.get("action") == "Recommend":
                target_ids = product_ids_from_recommend_turn(item, label_source=label_source)
                if target_ids:
                    cases.append(
                        {
                            "conversation_id": cid,
                            "eval_user_turn_index": user_turn_index,
                            "eval_dataset_turn_id": pending_user.get("turn_id"),
                            "eval_turn": dict(pending_user),
                            "hidden_system_turn_id": item.get("turn_id"),
                            "hidden_system_content": item.get("content") or "",
                            "target_product_ids": target_ids,
                            "target_product_ids_text": " ".join(target_ids),
                            "prefix_user_turns": [dict(turn) for turn in prefix_user_turns],
                            "prefix_user_messages": " || ".join(
                                str(turn.get("content") or "") for turn in prefix_user_turns
                            ),
                        }
                    )
            pending_user = None
    return cases


def product_ids_from_recommend_turn(turn: dict[str, Any], *, label_source: str = "union") -> list[str]:
    ids: list[str] = []
    keys_by_source = {
        "recommended": ("recommended_products",),
        "search_results": ("search_results",),
        "union": ("recommended_products", "search_results"),
    }
    for key in keys_by_source[label_source]:
        for item in turn.get(key) or []:
            product_id = item if isinstance(item, str) else item.get("product_id") if isinstance(item, dict) else None
            if product_id:
                ids.append(str(product_id))
    return list(dict.fromkeys(ids))


def case_metadata(case: dict[str, Any], row: dict[str, Any], relevant_ids: set[str]) -> dict[str, Any]:
    recommended_ids = str(row.get("recommended_ids") or "").split()
    ranked_ids = str(row.get("verified_candidate_ids") or "").split() or recommended_ids
    return {
        "eval_failed": 0,
        "eval_error": "",
        "case_conversation_id": case["conversation_id"],
        "eval_user_turn_index": case["eval_user_turn_index"],
        "eval_dataset_turn_id": case.get("eval_dataset_turn_id") or "",
        "hidden_system_turn_id": case.get("hidden_system_turn_id") or "",
        "target_product_ids": case["target_product_ids_text"],
        "prefix_user_turn_count": len(case["prefix_user_turns"]),
        "prefix_user_messages": case["prefix_user_messages"],
        "turn_level_ndcg@5": ndcg_at_k(ranked_ids, relevant_ids, 5),
        "turn_level_hit@5": hit_rate_at_k(ranked_ids, relevant_ids, 5),
        "turn_level_mrr": reciprocal_rank(ranked_ids, relevant_ids),
    }


def failed_case_row(case: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "conversation_id": case["conversation_id"],
        "case_conversation_id": case["conversation_id"],
        "turn_index": case["eval_user_turn_index"],
        "eval_user_turn_index": case["eval_user_turn_index"],
        "eval_dataset_turn_id": case.get("eval_dataset_turn_id") or "",
        "hidden_system_turn_id": case.get("hidden_system_turn_id") or "",
        "expected_action": "Recommend",
        "eval_failed": 1,
        "eval_error": f"{type(exc).__name__}: {exc}",
        "n_relevant": len(case["target_product_ids"]),
        "target_product_ids": case["target_product_ids_text"],
        "recommended": 0,
        "recommended_ids": "",
        "verified_candidate_ids": "",
        "turn_level_ndcg@5": 0.0,
        "turn_level_hit@5": 0.0,
        "turn_level_mrr": 0.0,
        "latency_ms": "",
        "planner_total_tokens": 0,
        "prefix_user_turn_count": len(case["prefix_user_turns"]),
        "prefix_user_messages": case["prefix_user_messages"],
    }


def summarize_cases(rows: list[dict[str, Any]], args: argparse.Namespace, elapsed: float) -> dict[str, Any]:
    scored = [row for row in rows if not int(row.get("eval_failed") or 0)]
    return {
        "experiment": "turn_level_full_prism",
        "split": args.split,
        "planner_mode": args.planner_mode,
        "label_source": args.label_source,
        "execution_scope": args.execution_scope,
        "retrieval_backend": os.getenv("AUTONOMOUS_RETRIEVAL_BACKEND", ""),
        "elapsed_seconds": round(elapsed, 2),
        "cases_total": len(rows),
        "scored_turns": len(scored),
        "failed_cases": sum(int(row.get("eval_failed") or 0) for row in rows),
        "recommendation_coverage": round(
            sum(int(row.get("recommended") or 0) for row in scored) / max(len(scored), 1),
            4,
        ),
        "hit@5": mean_key(scored, "turn_level_hit@5"),
        "ndcg@5": mean_key(scored, "turn_level_ndcg@5"),
        "mrr": mean_key(scored, "turn_level_mrr"),
        "hit@10": mean_key(scored, "hit@10"),
        "ndcg@10": mean_key(scored, "ndcg@10"),
        "search_hit_rate": mean_key(scored, "search_hit_rate"),
        "retrieval_miss_rate": round(
            sum(1 for row in scored if row.get("retrieval_miss") == 1) / max(len(scored), 1),
            4,
        ),
        "latency_ms_avg": mean_key(scored, "latency_ms"),
        "latency_ms_p95": percentile_values([row.get("latency_ms") for row in scored], 95),
        "tokens_per_turn": mean_key(scored, "planner_total_tokens"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
