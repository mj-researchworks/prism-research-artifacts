#!/usr/bin/env python3
"""Evaluate the PRISM-native fixed-workflow baseline.

The evaluator keeps the frozen PRISM scoring protocol and fold assets.  Only
the orchestration changes: the executive planner is removed and the declared
component sequence is executed once per turn.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from eval.autonomous_full_recommendation_eval import (
    build_ground_truth,
    configure_fold,
    extract_user_turns,
    failed_conversation_row,
    load_split,
    score_turn,
    summarize,
    summarize_conversation,
    write_csv,
)
from fixed_workflow_runner import FixedWorkflowRunner
from src.autonomous.eval_logger import EvalLogger


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "eval" / "results" / "fixed_workflow"


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True, choices=["fold1", "fold2", "fold3", "fold4"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-convs", type=int, default=None)
    parser.add_argument("--retrieval-backend", default="conserve_stack")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    import os
    os.environ["AUTONOMOUS_RETRIEVAL_BACKEND"] = args.retrieval_backend
    os.environ["AUTONOMOUS_EXECUTION_SCOPE"] = "fixed_workflow"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    split_dir = configure_fold(args.fold)
    conversations = load_split(args.split, split_dir=split_dir)
    if args.max_convs:
        conversations = conversations[: args.max_convs]
    ground_truth = build_ground_truth(conversations)
    out_dir = args.out_dir or RESULTS_DIR / f"{args.fold}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    turn_rows: list[dict] = []
    conversation_rows: list[dict] = []
    states_path = out_dir / "states.jsonl"
    started = time.time()
    with states_path.open("w", encoding="utf-8") as states:
        for index, conv in enumerate(conversations, 1):
            cid = str(conv.get("conv_id"))
            relevant_ids = ground_truth.get(cid, set())
            logger = EvalLogger(enabled=False)
            logger.conversation_id = cid
            runner = FixedWorkflowRunner(eval_logger=logger)
            conv_rows = []
            try:
                turns = extract_user_turns(conv)
                print(f"[{index:3}/{len(conversations)}] conv {cid} turns={len(turns)}", flush=True)
                for turn_index, turn in enumerate(turns, 1):
                    user_message = str(turn.get("content") or "").strip()
                    if not user_message:
                        continue
                    before = len(runner.blackboard.snapshot().get("planner_trace") or [])
                    turn_started = time.perf_counter()
                    response, state = runner.run_turn(user_message)
                    latency_ms = int((time.perf_counter() - turn_started) * 1000)
                    states.write(json.dumps({
                        "conversation_id": cid,
                        "turn_index": turn_index,
                        "user_message": user_message,
                        "response": response,
                        "state": state,
                    }, ensure_ascii=False, default=_json_default) + "\n")
                    states.flush()
                    row = score_turn(cid=cid, turn_index=turn_index, turn=turn,
                                     response=response, state=state,
                                     relevant_ids=relevant_ids,
                                     before_trace_len=before, latency_ms=latency_ms)
                    turn_rows.append(row)
                    conv_rows.append(row)
                conversation_rows.append(summarize_conversation(cid, conv_rows, relevant_ids))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"[{index:3}/{len(conversations)}] conv {cid} failed: {error}", flush=True)
                conversation_rows.append(failed_conversation_row(cid, relevant_ids, error))

    summary_args = argparse.Namespace(split=args.split, fold=args.fold,
                                      planner_mode="fixed_workflow", execution_scope="fixed_workflow")
    summary = summarize(turn_rows, conversation_rows, summary_args, elapsed=time.time() - started)
    summary.update({
        "fixed_workflow": True,
        "workflow_sequence": ["intent_agent", "preference_reflection_agent",
                               "search_research_agent", "pre_retrieval_critic",
                               "conserve_search_tool", "post_retrieval_critic",
                               "recommendation_decision_agent", "response_tool"],
        "adaptive_planner": False,
        "adaptive_retry_or_replan": False,
        "recovery_policy": "terminal clarification/no-match; no search-again",
        "retrieval_backend": args.retrieval_backend,
        "candidate_budget": {"search_depth": 50, "top_k": 10, "display_k": 3,
                              "max_query_variants": 3},
        "notes": "PRISM-native fixed sequence; same fold assets and scoring protocol as Full PRISM.",
    })
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    write_csv(out_dir / f"turns_{args.split}_{timestamp}.csv", turn_rows)
    write_csv(out_dir / f"conversations_{args.split}_{timestamp}.csv", conversation_rows)
    (out_dir / f"summary_{args.split}_{timestamp}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

