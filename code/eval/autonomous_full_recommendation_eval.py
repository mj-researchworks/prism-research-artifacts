#!/usr/bin/env python3
"""Evaluate the full autonomous ConServe recommendation workflow.

This replays existing PSCoN/ConServe split conversations through the new
planner-blackboard-specialist loop and scores final recommendations against
the same frozen product-id ground truth used by eval.evaluate:

  - PSCoN system Recommend turns define relevant products.
  - relevant = recommended_products union search_results.

The evaluator also records autonomy-specific behavior: planner steps, planner
backend use, specialist/tool activation, reflection calls, critic calls,
search-again/replan actions, latency, and token usage when the planner/provider
reports it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.autonomous.flags import combined_flags
from eval.run_evidence import RunEvidence, EvidenceWriteError

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "outputs" / "splits"
RESULTS_DIR = ROOT / "eval" / "results" / "autonomous_full_recommendation"
CATEGORY_MAP_PATH = ROOT / "data" / "product_category_map.json"
SEMANTIC_MATCH_THRESHOLD = 0.6
_semantic_embed_cache: dict[str, Any] = {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--fold",
        choices=["fold1", "fold2", "fold3", "fold4"],
        help="Use this fold's split and matching triplet checkpoint/cache. Overrides inherited reranker paths.",
    )
    parser.add_argument("--max-convs", type=int, default=None)
    parser.add_argument("--planner-mode", default=os.getenv("AUTONOMOUS_PLANNER_MODE", "rule"))
    parser.add_argument("--execution-scope", default="full")
    parser.add_argument(
        "--retrieval-backend",
        default=os.getenv("AUTONOMOUS_RETRIEVAL_BACKEND", "fallback"),
        help="Use fallback/local_safe for Mac-safe eval, or auto for the existing full ConServe stack.",
    )
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    os.environ["AUTONOMOUS_EXECUTION_SCOPE"] = args.execution_scope
    os.environ["AUTONOMOUS_RETRIEVAL_BACKEND"] = args.retrieval_backend
    split_dir = configure_fold(args.fold) if args.fold else SPLITS_DIR
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from src.autonomous.eval_logger import EvalLogger
    from src.autonomous.runner import AutonomousRunner

    conversations = load_split(args.split, split_dir=split_dir)
    if args.max_convs:
        conversations = conversations[: args.max_convs]

    ground_truth = build_ground_truth(conversations)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    evidence = RunEvidence(root=ROOT, out_dir=args.out_dir, args=args,
                           conversations=conversations, split_dir=split_dir)
    print(f"Research evidence -> {evidence.directory}", flush=True)

    turn_rows: list[dict[str, Any]] = []
    conversation_rows: list[dict[str, Any]] = []

    started = time.time()
    for index, conv in enumerate(conversations, start=1):
        cid = str(conv.get("conv_id"))
        relevant_ids = ground_truth.get(cid, set())
        turn_index = None
        user_message = None
        runner = None
        try:
            runner = AutonomousRunner(
                planner_mode=args.planner_mode,
                execution_scope=args.execution_scope,
                eval_logger=EvalLogger(enabled=False),
            )
            turns = extract_user_turns(conv)
            conv_turn_rows = []
            print(
                f"[{index:3}/{len(conversations)}] conv {cid} "
                f"turns={len(turns)} relevant={len(relevant_ids)}",
                flush=True,
            )

            for turn_index, turn in enumerate(turns, start=1):
                user_message = str(turn.get("content") or "").strip()
                if not user_message:
                    continue
                before_trace_len = len(runner.blackboard.snapshot().get("planner_trace") or [])
                turn_started = time.perf_counter()
                response, state = runner.run_turn(user_message)
                latency_ms = int((time.perf_counter() - turn_started) * 1000)
                # Save the complete returned state BEFORE scoring: a scorer
                # error must not discard an already-paid inference result.
                evidence.turn(cid=cid, turn_index=turn_index, user_message=user_message,
                              response=response, state=state, latency_ms=latency_ms)
                row = score_turn(
                    cid=cid,
                    turn_index=turn_index,
                    turn=turn,
                    response=response,
                    state=state,
                    relevant_ids=relevant_ids,
                    before_trace_len=before_trace_len,
                    latency_ms=latency_ms,
                )
                turn_rows.append(row)
                conv_turn_rows.append(row)
                evidence.append('scored_turns.jsonl', row)

            conv_summary = summarize_conversation(cid, conv_turn_rows, relevant_ids)
            conversation_rows.append(conv_summary)
            evidence.append('conversations.jsonl', conv_summary)
        except EvidenceWriteError:
            raise  # Do not continue paid inference when evidence storage fails.
        except (KeyboardInterrupt, SystemExit) as exc:
            evidence.append('errors.jsonl', {
                'conversation_id':cid, 'turn_index':turn_index, 'user_message':user_message,
                'error':type(exc).__name__, 'status':'interrupted',
                'partial_state':runner.blackboard.snapshot() if runner is not None else None,
            })
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[{index:3}/{len(conversations)}] conv {cid} failed: {error}", flush=True)
            failure = failed_conversation_row(cid, relevant_ids, error)
            conversation_rows.append(failure)
            evidence.append('errors.jsonl', {
                'conversation_id':cid, 'turn_index':turn_index, 'user_message':user_message,
                'error':error, 'status':'failed',
                'partial_state':runner.blackboard.snapshot() if runner is not None else None,
            })
            evidence.append('conversations.jsonl', failure)

    summary = summarize(turn_rows, conversation_rows, args, elapsed=time.time() - started)
    summary['research_evidence_directory'] = str(evidence.directory)
    summary['research_run_id'] = evidence.run_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    turn_path = args.out_dir / f"turns_{args.split}_{timestamp}.csv"
    conv_path = args.out_dir / f"conversations_{args.split}_{timestamp}.csv"
    summary_path = args.out_dir / f"summary_{args.split}_{timestamp}.json"

    write_csv(turn_path, turn_rows)
    write_csv(conv_path, conversation_rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    evidence.finish(summary=summary, output_files=[turn_path, conv_path, summary_path])

    print("\nAutonomous full recommendation evaluation complete")
    print(f"Summary       -> {summary_path}")
    print(f"Turn rows     -> {turn_path}")
    print(f"Conversations -> {conv_path}")
    print(f"Full evidence -> {evidence.directory}")
    print()
    print(f"Conversations: {summary['n_conversations_total']}")
    print(f"Failed conversations: {summary['n_conversations_failed']}")
    print(f"With GT labels: {summary['n_conversations_with_gt']}")
    print(f"NDCG@3/NDCG@10: {summary['ndcg@3']} / {summary['ndcg@10']}")
    print(f"Hit@3/Hit@10: {summary['hit@3']} / {summary['hit@10']}")
    print(f"MRR: {summary['mrr']}")
    print(
        "Displayed NDCG@3/NDCG@10: "
        f"{summary['displayed_ndcg@3']} / {summary['displayed_ndcg@10']}"
    )
    print(
        "Displayed Hit@3/Hit@10: "
        f"{summary['displayed_hit@3']} / {summary['displayed_hit@10']}"
    )
    print(f"Displayed MRR: {summary['displayed_mrr']}")
    print(
        "Displayed Semantic Hit/F1: "
        f"{summary['displayed_semantic_hit_rate']} / {summary['displayed_semantic_f1']}"
    )
    print(
        "Internal ranked NDCG@3/NDCG@10: "
        f"{summary['internal_ranked_ndcg@3']} / {summary['internal_ranked_ndcg@10']}"
    )
    print(
        "Internal ranked Hit@3/Hit@10: "
        f"{summary['internal_ranked_hit@3']} / {summary['internal_ranked_hit@10']}"
    )
    print(f"Internal ranked MRR: {summary['internal_ranked_mrr']}")
    print(
        "Internal ranked Semantic Hit/F1: "
        f"{summary['internal_ranked_semantic_hit_rate']} / {summary['internal_ranked_semantic_f1']}"
    )
    print(f"Recall@50: {summary['recall@50']}")
    print(f"Search hit rate: {summary['search_hit_rate']}")
    print(f"ATR: {summary['atr']}")
    print(f"LLM planner turn rate: {summary['llm_planner_turn_rate']}")
    print(f"Planner unchanged execution rate: {summary['planner_unchanged_execution_rate']}")
    print(f"Planner fallback turn rate: {summary['planner_fallback_turn_rate']}")
    print(f"Guard intervention turn rate: {summary['guard_intervention_turn_rate']}")
    print(f"Guard intervention reasons: {summary['guard_intervention_reasons']}")
    print(f"Avg planner steps/turn: {summary['planner_steps_avg']}")
    print(f"Avg components/turn: {summary['components_per_turn_avg']}")
    print(f"Avg LLM calls/conversation: {summary['llm_calls_per_conversation']}")
    print(f"Avg latency ms/turn: {summary['latency_ms_avg']}")
    print(f"P50/P95 latency ms/turn: {summary['latency_ms_p50']} / {summary['latency_ms_p95']}")
    print(
        "Avg/P50/P95 latency ms/conversation: "
        f"{summary['conversation_latency_ms_avg']} / "
        f"{summary['conversation_latency_ms_p50']} / "
        f"{summary['conversation_latency_ms_p95']}"
    )
    print(f"Avg planner ms/turn: {summary['planner_duration_ms_avg']}")
    print(f"Component duration ms total: {summary['component_duration_ms_total']}")
    print(f"Retrieval call turn rate: {summary['retrieval_call_turn_rate']}")
    print(
        "Retrieval cache hits/misses/reuse: "
        f"{summary['retrieval_cache_hits']} / "
        f"{summary['retrieval_cache_misses']} / "
        f"{summary['retrieval_candidate_reuse_calls']}"
    )
    print(f"Retrieval cache hit rate: {summary['retrieval_cache_hit_rate']}")
    print(f"Reflection calls/failures: {summary['reasoning_reflection_calls']} / {summary['reasoning_reflection_failures']}")
    print(f"Reflection skips/cache hits: {summary['reasoning_reflection_skips']} / {summary['reasoning_reflection_cache_hits']}")
    print(f"Critic calls/failures: {summary['critic_reasoning_calls']} / {summary['critic_reasoning_failures']}")
    print(f"Critic skips/cache hits: {summary['critic_reasoning_skips']} / {summary['critic_reasoning_cache_hits']}")
    print(f"Reasoning depths: {summary['reasoning_depths']}")
    print(f"Quality risks: {summary['quality_risks']}")
    print(f"Control flags: {summary['control_flags']}")
    print(f"Preference flags: {summary['preference_flags']}")
    print(f"Safety flags: {summary['safety_flags']}")
    print(f"Unclassified flags: {summary['unclassified_flags']}")
    print(
        "Memory events total/rate/types: "
        f"{summary['memory_events_total']} / "
        f"{summary['memory_event_turn_rate']} / "
        f"{summary['memory_event_types']}"
    )
    intent_diag = summary.get("intent_diagnostics") or {}
    print(
        "Intent accuracy/macro-F1/weighted-F1: "
        f"{intent_diag.get('accuracy')} / "
        f"{intent_diag.get('macro_f1')} / "
        f"{intent_diag.get('weighted_f1')}"
    )
    print(f"Retrieval actual backends: {summary['retrieval_actual_backends']}")
    print(f"Fallback retrieval calls: {summary['fallback_retrieval_calls']}")
    print(f"Full-stack retrieval failures: {summary['full_stack_retrieval_failures']}")
    print(f"Triplet reranker diagnostics: {summary['triplet_reranker_diagnostics']}")
    if summary["planner_errors"]:
        print(f"Planner errors: {summary['planner_errors']}")
    if summary["reflection_errors"]:
        print(f"Reflection errors: {summary['reflection_errors']}")
    if summary["critic_errors"]:
        print(f"Critic errors: {summary['critic_errors']}")
    print(f"Search-again actions: {summary['search_again_actions']}")


def configure_fold(fold: str) -> Path:
    split_dir = ROOT / "outputs" / fold
    model_path = ROOT / f"e5_{fold}_best_model"
    cache_path = ROOT / f"product_embedding_cache_{fold}.pt"
    required_paths = [
        split_dir / "train_conversations.json",
        split_dir / "val_conversations.json",
        split_dir / "test_conversations.json",
        model_path,
        cache_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot run {fold}: required split/model artifacts are missing: {', '.join(missing)}"
        )
    os.environ["MACPRS_TRIPLET_MODEL_PATH"] = str(model_path)
    os.environ["MACPRS_TRIPLET_PRODUCT_CACHE_PATH"] = str(cache_path)
    return split_dir


def load_split(split: str, *, split_dir: Path = SPLITS_DIR) -> list[dict[str, Any]]:
    with (split_dir / f"{split}_conversations.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def build_ground_truth(conversations: list[dict[str, Any]]) -> dict[str, set[str]]:
    with CATEGORY_MAP_PATH.open(encoding="utf-8") as handle:
        category_map = json.load(handle)

    ground_truth: dict[str, set[str]] = {}
    for conv in conversations:
        cid = str(conv["conv_id"])
        relevant_ids: set[str] = set()
        for turn in conv.get("conversation", []):
            if turn.get("role") != "system" or turn.get("action") != "Recommend":
                continue
            for key in ("recommended_products", "search_results"):
                for rec in turn.get(key) or []:
                    pid = rec if isinstance(rec, str) else rec.get("product_id") if isinstance(rec, dict) else None
                    if pid and pid in category_map:
                        relevant_ids.add(str(pid))
        if relevant_ids:
            ground_truth[cid] = relevant_ids
    return ground_truth


def extract_user_turns(conv: dict[str, Any]) -> list[dict[str, Any]]:
    turns = []
    last_user = None
    for item in conv.get("conversation", []):
        if item.get("role") == "user":
            last_user = dict(item)
            turns.append(last_user)
        elif item.get("role") == "system" and last_user is not None:
            last_user["next_system_action"] = item.get("action")
            last_user["next_system_content"] = item.get("content")
            last_user["clarifying_attribute"] = item.get("clarifying_attribute")
            last_user = None
    return turns


def score_turn(
    *,
    cid: str,
    turn_index: int,
    turn: dict[str, Any],
    response: str,
    state: dict[str, Any],
    relevant_ids: set[str],
    before_trace_len: int,
    latency_ms: int,
) -> dict[str, Any]:
    trace = state.get("planner_trace") or []
    turn_trace = trace[before_trace_len:]
    final_request = to_plain(state.get("final_search_request")) or {}
    retrieval_plan = to_plain(state.get("retrieval_plan")) or {}
    raw_candidates = state.get("raw_candidates") or []
    verified_candidates = state.get("verified_candidates") or []
    recommendations = state.get("last_recommendations") or []
    candidate_ledger = to_plain(state.get("candidate_ledger") or {})
    retrieval_diagnostics = candidate_ledger.get("diagnostics") if isinstance(candidate_ledger, dict) else {}
    retrieval_diagnostics = retrieval_diagnostics or {}
    adapter_diagnostics = (
        candidate_ledger.get("adapter_diagnostics")
        if isinstance(candidate_ledger, dict)
        else {}
    ) or retrieval_diagnostics.get("adapter") or {}
    retrieval_backend_actual = retrieval_diagnostics.get("actual_backend") or retrieval_diagnostics.get("backend") or ""
    retrieval_cache_hit = bool(retrieval_diagnostics.get("cache_hit") or adapter_diagnostics.get("cache_hit"))
    retrieval_candidate_reuse = bool(retrieval_diagnostics.get("candidate_reuse_used"))
    retrieval_cache_miss = bool(retrieval_backend_actual and not retrieval_cache_hit and not retrieval_candidate_reuse)

    raw_ids = product_ids(raw_candidates)
    verified_ids = product_ids(verified_candidates)
    recommended_ids = product_ids(recommendations)
    internal_ranked_ids = verified_ids or recommended_ids
    text_by_id = _semantic_vector_texts()
    displayed_semantic = semantic_match_metrics(recommended_ids, relevant_ids, text_by_id, k=3)
    internal_semantic = semantic_match_metrics(internal_ranked_ids, relevant_ids, text_by_id, k=3)
    planner_usage = usage_totals(turn_trace)
    reflection_notes = current_turn_notes(state.get("reflection_notes") or [], state.get("turn_id"))
    critic_notes = current_turn_notes(state.get("critic_notes") or [], state.get("turn_id"))
    component_durations = component_duration_totals(state.get("tool_observations") or [])
    planner_duration_ms = sum(
        int(step.get("planner_duration_ms") or 0)
        for step in turn_trace
        if isinstance(step.get("planner_duration_ms"), (int, float))
    )
    planner_errors = [str(step.get("planner_error")) for step in turn_trace if step.get("planner_error")]
    reflection_errors = [str(note.get("error")) for note in reflection_notes if note.get("error")]
    critic_errors = [str(note.get("error")) for note in critic_notes if note.get("error")]
    reflection_skip_reasons = [
        str(note.get("call_reason") or note.get("reason") or "")
        for note in reflection_notes
        if note.get("mode") == "reasoning_skipped"
    ]
    reflection_call_reasons = [
        str(note.get("call_reason") or "")
        for note in reflection_notes
        if note.get("mode") in {"reasoning", "reasoning_failed"}
    ]
    critic_skip_reasons = [
        str(note.get("skip_reason") or "")
        for note in critic_notes
        if note.get("mode") == "critic_reasoning_skipped"
    ]
    critic_call_reasons = [
        str(note.get("call_reason") or "")
        for note in critic_notes
        if note.get("mode") in {"critic_reasoning", "critic_reasoning_failed"}
    ]
    turn_analysis = latest_turn_analysis(turn_trace)
    runtime_intent = state.get("intent") or {}
    dataset_intent = _normalize_intent_label(turn.get("intent"))
    predicted_pscon_intent = _predicted_pscon_intent(runtime_intent)
    memory_events = to_plain(state.get("preference_events") or [])
    memory_event_types = Counter(str(event.get("type") or "") for event in memory_events if event.get("type"))
    memory_event_scopes = Counter(str(event.get("event_scope") or "") for event in memory_events if event.get("event_scope"))
    planner_step_trace = [
        {
            "step": step.get("step"),
            "action": step.get("action"),
            "target": step.get("target"),
            "backend": step.get("planner_backend") or step.get("planner_mode"),
            "reason": step.get("reason"),
        }
        for step in turn_trace
    ]

    return {
        "conversation_id": cid,
        "turn_index": turn_index,
        "dataset_intent": turn.get("intent") or "",
        "dataset_intent_normalized": dataset_intent,
        "predicted_pscon_intent": predicted_pscon_intent,
        "runtime_intent_label": runtime_intent.get("label") or "",
        "runtime_action_label": runtime_intent.get("action_label") or "",
        "runtime_intent_source": runtime_intent.get("source") or "",
        "intent_correct": (
            int(dataset_intent == predicted_pscon_intent)
            if dataset_intent and predicted_pscon_intent
            else ""
        ),
        "expected_action": turn.get("next_system_action") or "",
        "user_message": turn.get("content") or "",
        "response": response,
        "recommended": int(bool(recommended_ids)),
        "recommended_ids": " ".join(recommended_ids[:10]),
        "raw_candidate_ids": " ".join(raw_ids[:50]),
        "verified_candidate_ids": " ".join(verified_ids[:50]),
        "n_relevant": len(relevant_ids),
        "n_raw": len(raw_ids),
        "n_verified": len(verified_ids),
        "n_recommended": len(recommended_ids),
        "ndcg@3": ndcg_at_k(recommended_ids, relevant_ids, 3),
        "ndcg@10": ndcg_at_k(internal_ranked_ids, relevant_ids, 10),
        "hit@1": hit_rate_at_k(recommended_ids, relevant_ids, 1),
        "hit@3": hit_rate_at_k(recommended_ids, relevant_ids, 3),
        "hit@5": hit_rate_at_k(internal_ranked_ids, relevant_ids, 5),
        "hit@10": hit_rate_at_k(internal_ranked_ids, relevant_ids, 10),
        "mrr": reciprocal_rank(internal_ranked_ids, relevant_ids),
        "displayed_ndcg@3": ndcg_at_k(recommended_ids, relevant_ids, 3),
        "displayed_ndcg@10": ndcg_at_k(recommended_ids, relevant_ids, 10),
        "displayed_hit@1": hit_rate_at_k(recommended_ids, relevant_ids, 1),
        "displayed_hit@3": hit_rate_at_k(recommended_ids, relevant_ids, 3),
        "displayed_hit@5": hit_rate_at_k(recommended_ids, relevant_ids, 5),
        "displayed_hit@10": hit_rate_at_k(recommended_ids, relevant_ids, 10),
        "displayed_mrr": reciprocal_rank(recommended_ids, relevant_ids),
        "internal_ranked_ndcg@3": ndcg_at_k(internal_ranked_ids, relevant_ids, 3),
        "internal_ranked_ndcg@10": ndcg_at_k(internal_ranked_ids, relevant_ids, 10),
        "internal_ranked_hit@1": hit_rate_at_k(internal_ranked_ids, relevant_ids, 1),
        "internal_ranked_hit@3": hit_rate_at_k(internal_ranked_ids, relevant_ids, 3),
        "internal_ranked_hit@5": hit_rate_at_k(internal_ranked_ids, relevant_ids, 5),
        "internal_ranked_hit@10": hit_rate_at_k(internal_ranked_ids, relevant_ids, 10),
        "internal_ranked_mrr": reciprocal_rank(internal_ranked_ids, relevant_ids),
        "recall@50": recall_at_k(raw_ids, relevant_ids, 50),
        "search_hit_rate": search_hit_rate(raw_ids, relevant_ids),
        "search_recall": item_level_recall(raw_ids, relevant_ids),
        "raw_relevant_count": relevant_count(raw_ids, relevant_ids),
        "ranked_relevant_count@3": relevant_count(recommended_ids, relevant_ids, 3),
        "ranked_relevant_count@10": relevant_count(internal_ranked_ids, relevant_ids, 10),
        "exact_precision@3": precision_recall_f1_at_k(recommended_ids, relevant_ids, 3)["precision"],
        "exact_recall@3": precision_recall_f1_at_k(recommended_ids, relevant_ids, 3)["recall"],
        "exact_f1@3": precision_recall_f1_at_k(recommended_ids, relevant_ids, 3)["f1"],
        "displayed_semantic_hit_rate": displayed_semantic["hit_rate"],
        "displayed_semantic_precision": displayed_semantic["precision"],
        "displayed_semantic_recall": displayed_semantic["recall"],
        "displayed_semantic_f1": displayed_semantic["f1"],
        "internal_ranked_semantic_hit_rate": internal_semantic["hit_rate"],
        "internal_ranked_semantic_precision": internal_semantic["precision"],
        "internal_ranked_semantic_recall": internal_semantic["recall"],
        "internal_ranked_semantic_f1": internal_semantic["f1"],
        "retrieval_miss": int(search_hit_rate(raw_ids, relevant_ids) == 0.0) if relevant_ids else "",
        "rerank_loss@3": int(search_hit_rate(raw_ids, relevant_ids) == 1.0 and hit_rate_at_k(recommended_ids, relevant_ids, 3) == 0.0) if relevant_ids else "",
        "query_text": final_request.get("query_text") or "",
        "category": final_request.get("category") or "",
        "hard_constraints": json.dumps(final_request.get("hard_constraints") or {}, ensure_ascii=False),
        "soft_preferences": json.dumps(final_request.get("soft_preferences") or {}, ensure_ascii=False),
        "preferred": json.dumps(final_request.get("preferred") or {}, ensure_ascii=False),
        "exclusions": json.dumps(final_request.get("exclusions") or {}, ensure_ascii=False),
        "retrieval_strategy": retrieval_plan.get("strategy") or "",
        "retrieval_query_variants": json.dumps(retrieval_plan.get("query_variants") or [], ensure_ascii=False),
        "retrieval_search_depth": retrieval_plan.get("search_depth") or "",
        "retrieval_backend_requested": retrieval_diagnostics.get("requested_backend") or "",
        "retrieval_backend_actual": retrieval_backend_actual,
        "retrieval_fallback_used": int(bool(retrieval_diagnostics.get("fallback_used"))),
        "retrieval_cache_hit": int(retrieval_cache_hit),
        "retrieval_cache_miss": int(retrieval_cache_miss),
        "retrieval_candidate_reuse": int(retrieval_candidate_reuse),
        "retrieval_cache_size": retrieval_diagnostics.get("cache_size") or "",
        "retrieval_error": retrieval_diagnostics.get("error") or "",
        "retrieval_fallback_reason": retrieval_diagnostics.get("fallback_reason") or "",
        "adapter_mapped_slots": json.dumps(adapter_diagnostics.get("mapped_slot_keys") or [], ensure_ascii=False),
        "adapter_post_filter_counts": json.dumps(adapter_diagnostics.get("post_filter_counts") or {}, ensure_ascii=False),
        "adapter_search_depth": adapter_diagnostics.get("search_depth") or "",
        "adapter_query_variant_count": adapter_diagnostics.get("query_variant_count") or "",
        "critic_decision": state.get("critic_decision") or "",
        "critic_next_action": state.get("critic_next_action") or "",
        "security_flags": json.dumps(combined_flags(state), ensure_ascii=False),
        "control_flags": json.dumps(state.get("control_flags") or [], ensure_ascii=False),
        "preference_flags": json.dumps(state.get("preference_flags") or [], ensure_ascii=False),
        "safety_flags": json.dumps(state.get("safety_flags") or [], ensure_ascii=False),
        "unclassified_flags": json.dumps(state.get("unclassified_flags") or [], ensure_ascii=False),
        "preference_events": json.dumps(memory_events, ensure_ascii=False),
        "memory_event_count": len(memory_events),
        "memory_event_types": json.dumps(dict(memory_event_types), ensure_ascii=False),
        "memory_event_scopes": json.dumps(dict(memory_event_scopes), ensure_ascii=False),
        "recent_memory_events": json.dumps(to_plain(state.get("recent_memory_events") or []), ensure_ascii=False),
        "planner_steps": len(turn_trace),
        "planner_duration_ms": planner_duration_ms,
        "component_duration_ms": json.dumps(component_durations, ensure_ascii=False),
        "planner_backends": json.dumps(list(dict.fromkeys(step.get("planner_backend") for step in turn_trace if step.get("planner_backend"))), ensure_ascii=False),
        "planner_step_trace": json.dumps(planner_step_trace, ensure_ascii=False),
        "planner_errors": json.dumps(planner_errors, ensure_ascii=False),
        "reasoning_depth": turn_analysis.get("reasoning_depth") or "",
        "quality_risk": turn_analysis.get("quality_risk") or "",
        "components_called": json.dumps([step.get("target") for step in turn_trace if step.get("target")], ensure_ascii=False),
        "reasoning_reflection_called": int(any(note.get("mode") in {"reasoning", "reasoning_failed"} for note in reflection_notes)),
        "reasoning_reflection_failed": int(any(note.get("mode") == "reasoning_failed" for note in reflection_notes)),
        "reasoning_reflection_skipped": int(any(note.get("mode") == "reasoning_skipped" for note in reflection_notes)),
        "reasoning_reflection_cache_hit": int(any(note.get("mode") == "reasoning" and note.get("cache_hit") for note in reflection_notes)),
        "reflection_call_reasons": json.dumps([reason for reason in reflection_call_reasons if reason], ensure_ascii=False),
        "reflection_skip_reasons": json.dumps([reason for reason in reflection_skip_reasons if reason], ensure_ascii=False),
        "reflection_errors": json.dumps(reflection_errors, ensure_ascii=False),
        "critic_reasoning_called": int(any(note.get("mode") in {"critic_reasoning", "critic_reasoning_failed"} for note in critic_notes)),
        "critic_reasoning_failed": int(any(note.get("mode") == "critic_reasoning_failed" for note in critic_notes)),
        "critic_reasoning_skipped": int(any(note.get("mode") == "critic_reasoning_skipped" for note in critic_notes)),
        "critic_reasoning_cache_hit": int(any(note.get("mode") == "critic_reasoning" and note.get("cache_hit") for note in critic_notes)),
        "critic_call_reasons": json.dumps([reason for reason in critic_call_reasons if reason], ensure_ascii=False),
        "critic_skip_reasons": json.dumps([reason for reason in critic_skip_reasons if reason], ensure_ascii=False),
        "critic_errors": json.dumps(critic_errors, ensure_ascii=False),
        "search_again_action": int(state.get("critic_next_action") == "search_again" or any(step.get("target") == "search_research_agent" and state.get("search_attempts", 0) for step in turn_trace)),
        "latency_ms": latency_ms,
        "planner_prompt_tokens": planner_usage["prompt_tokens"],
        "planner_completion_tokens": planner_usage["completion_tokens"],
        "planner_total_tokens": planner_usage["total_tokens"],
    }


def summarize_conversation(cid: str, rows: list[dict[str, Any]], relevant_ids: set[str]) -> dict[str, Any]:
    final_row = last_recommendation_row(rows) or rows[-1] if rows else {}
    first_rec_idx = next((row["turn_index"] for row in rows if row["recommended"]), None)
    backends = Counter()
    components = Counter()
    retrieval_backends = Counter(row["retrieval_backend_actual"] for row in rows if row.get("retrieval_backend_actual"))
    for row in rows:
        backends.update(json.loads(row["planner_backends"] or "[]"))
        components.update(json.loads(row["components_called"] or "[]"))
    return {
        "conversation_id": cid,
        "eval_failed": 0,
        "eval_error": "",
        "turns": len(rows),
        "n_relevant": len(relevant_ids),
        "recommended": int(any(row["recommended"] for row in rows)),
        "turns_to_rec": first_rec_idx or len(rows),
        "final_query": final_row.get("query_text", ""),
        "final_category": final_row.get("category", ""),
        "final_recommended_ids": final_row.get("recommended_ids", ""),
        "ndcg@3": final_row.get("ndcg@3", 0.0),
        "ndcg@10": final_row.get("ndcg@10", 0.0),
        "hit@1": final_row.get("hit@1", 0.0),
        "hit@3": final_row.get("hit@3", 0.0),
        "hit@5": final_row.get("hit@5", 0.0),
        "hit@10": final_row.get("hit@10", 0.0),
        "mrr": final_row.get("mrr", 0.0),
        "displayed_ndcg@3": final_row.get("displayed_ndcg@3", 0.0),
        "displayed_ndcg@10": final_row.get("displayed_ndcg@10", 0.0),
        "displayed_hit@1": final_row.get("displayed_hit@1", 0.0),
        "displayed_hit@3": final_row.get("displayed_hit@3", 0.0),
        "displayed_hit@5": final_row.get("displayed_hit@5", 0.0),
        "displayed_hit@10": final_row.get("displayed_hit@10", 0.0),
        "displayed_mrr": final_row.get("displayed_mrr", 0.0),
        "internal_ranked_ndcg@3": final_row.get("internal_ranked_ndcg@3", 0.0),
        "internal_ranked_ndcg@10": final_row.get("internal_ranked_ndcg@10", 0.0),
        "internal_ranked_hit@1": final_row.get("internal_ranked_hit@1", 0.0),
        "internal_ranked_hit@3": final_row.get("internal_ranked_hit@3", 0.0),
        "internal_ranked_hit@5": final_row.get("internal_ranked_hit@5", 0.0),
        "internal_ranked_hit@10": final_row.get("internal_ranked_hit@10", 0.0),
        "internal_ranked_mrr": final_row.get("internal_ranked_mrr", 0.0),
        "recall@50": final_row.get("recall@50"),
        "search_hit_rate": final_row.get("search_hit_rate"),
        "search_recall": final_row.get("search_recall"),
        "exact_precision@3": final_row.get("exact_precision@3"),
        "exact_recall@3": final_row.get("exact_recall@3"),
        "exact_f1@3": final_row.get("exact_f1@3"),
        "displayed_semantic_hit_rate": final_row.get("displayed_semantic_hit_rate"),
        "displayed_semantic_precision": final_row.get("displayed_semantic_precision"),
        "displayed_semantic_recall": final_row.get("displayed_semantic_recall"),
        "displayed_semantic_f1": final_row.get("displayed_semantic_f1"),
        "internal_ranked_semantic_hit_rate": final_row.get("internal_ranked_semantic_hit_rate"),
        "internal_ranked_semantic_precision": final_row.get("internal_ranked_semantic_precision"),
        "internal_ranked_semantic_recall": final_row.get("internal_ranked_semantic_recall"),
        "internal_ranked_semantic_f1": final_row.get("internal_ranked_semantic_f1"),
        "retrieval_miss": final_row.get("retrieval_miss", ""),
        "rerank_loss@3": final_row.get("rerank_loss@3", ""),
        "planner_steps_total": sum(int(row["planner_steps"]) for row in rows),
        "planner_steps_avg": average([row["planner_steps"] for row in rows]),
        "reasoning_reflection_calls": sum(int(row["reasoning_reflection_called"]) for row in rows),
        "reasoning_reflection_failures": sum(int(row["reasoning_reflection_failed"]) for row in rows),
        "reasoning_reflection_skips": sum(int(row.get("reasoning_reflection_skipped") or 0) for row in rows),
        "reasoning_reflection_cache_hits": sum(int(row.get("reasoning_reflection_cache_hit") or 0) for row in rows),
        "critic_reasoning_calls": sum(int(row["critic_reasoning_called"]) for row in rows),
        "critic_reasoning_failures": sum(int(row["critic_reasoning_failed"]) for row in rows),
        "critic_reasoning_skips": sum(int(row.get("critic_reasoning_skipped") or 0) for row in rows),
        "critic_reasoning_cache_hits": sum(int(row.get("critic_reasoning_cache_hit") or 0) for row in rows),
        "search_again_actions": sum(int(row["search_again_action"]) for row in rows),
        "fallback_retrieval_calls": sum(int(row["retrieval_fallback_used"]) for row in rows),
        "full_stack_retrieval_failures": sum(1 for row in rows if row.get("retrieval_backend_actual") == "conserve_stack_failed"),
        "latency_ms_avg": average([row["latency_ms"] for row in rows]),
        "latency_ms_max": max([int(row["latency_ms"]) for row in rows], default=0),
        "planner_total_tokens": sum(int(row["planner_total_tokens"]) for row in rows),
        "retrieval_actual_backends": json.dumps(dict(retrieval_backends), ensure_ascii=False),
        "planner_backends": json.dumps(dict(backends), ensure_ascii=False),
        "components_called": json.dumps(dict(components), ensure_ascii=False),
    }


def failed_conversation_row(cid: str, relevant_ids: set[str], error: str) -> dict[str, Any]:
    return {
        "conversation_id": cid,
        "eval_failed": 1,
        "eval_error": error,
        "turns": 0,
        "n_relevant": len(relevant_ids),
        "recommended": 0,
        "turns_to_rec": 0,
        "final_query": "",
        "final_category": "",
        "final_recommended_ids": "",
        "retrieval_miss": "",
        "rerank_loss@3": "",
        "planner_steps_total": 0,
        "planner_steps_avg": None,
        "reasoning_reflection_calls": 0,
        "reasoning_reflection_failures": 0,
        "reasoning_reflection_skips": 0,
        "reasoning_reflection_cache_hits": 0,
        "critic_reasoning_calls": 0,
        "critic_reasoning_failures": 0,
        "critic_reasoning_skips": 0,
        "critic_reasoning_cache_hits": 0,
        "search_again_actions": 0,
        "fallback_retrieval_calls": 0,
        "full_stack_retrieval_failures": 0,
        "latency_ms_avg": None,
        "latency_ms_max": 0,
        "planner_total_tokens": 0,
        "retrieval_actual_backends": "{}",
        "planner_backends": "{}",
        "components_called": "{}",
    }


def summarize(
    turn_rows: list[dict[str, Any]],
    conversation_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    elapsed: float,
) -> dict[str, Any]:
    evaluated = [
        row
        for row in conversation_rows
        if int(row["n_relevant"]) > 0 and not int(row.get("eval_failed") or 0)
    ]
    all_convs = conversation_rows
    component_counts = Counter()
    backend_counts = Counter()
    retrieval_backend_counts = Counter()
    planner_errors = Counter()
    reflection_errors = Counter()
    critic_errors = Counter()
    reflection_call_reasons = Counter()
    reflection_skip_reasons = Counter()
    critic_call_reasons = Counter()
    critic_skip_reasons = Counter()
    reasoning_depths = Counter()
    quality_risks = Counter()
    component_durations = Counter()
    control_flags = Counter()
    preference_flags = Counter()
    safety_flags = Counter()
    unclassified_flags = Counter()
    memory_event_types = Counter()
    memory_event_scopes = Counter()
    critic_actions = Counter(row.get("critic_next_action") for row in turn_rows if row.get("critic_next_action"))
    critic_decisions = Counter(row.get("critic_decision") for row in turn_rows if row.get("critic_decision"))
    guard_intervention_reasons = Counter()
    turns_with_llm_planner = 0
    turns_with_fallback = 0
    turns_with_guard = 0
    turns_with_unchanged_llm_plan = 0
    turns_with_retrieval = 0
    component_calls_per_turn = []
    for row in turn_rows:
        components = json.loads(row["components_called"] or "[]")
        backends = json.loads(row["planner_backends"] or "[]")
        planner_step_trace = json.loads(row.get("planner_step_trace") or "[]")
        component_counts.update(components)
        backend_counts.update(backends)
        component_calls_per_turn.append(len(components))
        component_durations.update(json.loads(row.get("component_duration_ms") or "{}"))
        has_llm_planner = any(backend in {"llm_plan", "llm_plan_cached"} for backend in backends)
        has_fallback = any("fallback" in str(backend) for backend in backends)
        has_guard = any("guard" in str(backend) for backend in backends)
        if has_llm_planner:
            turns_with_llm_planner += 1
        if has_fallback:
            turns_with_fallback += 1
        if has_guard:
            turns_with_guard += 1
            guard_intervention_reasons.update(_guard_reason_buckets(planner_step_trace))
        if has_llm_planner and not has_guard and not has_fallback:
            turns_with_unchanged_llm_plan += 1
        if "conserve_search_tool" in components:
            turns_with_retrieval += 1
        if row.get("retrieval_backend_actual"):
            retrieval_backend_counts.update([row["retrieval_backend_actual"]])
        planner_errors.update(json.loads(row["planner_errors"] or "[]"))
        reflection_errors.update(json.loads(row["reflection_errors"] or "[]"))
        critic_errors.update(json.loads(row["critic_errors"] or "[]"))
        reflection_call_reasons.update(json.loads(row.get("reflection_call_reasons") or "[]"))
        reflection_skip_reasons.update(json.loads(row.get("reflection_skip_reasons") or "[]"))
        critic_call_reasons.update(json.loads(row.get("critic_call_reasons") or "[]"))
        critic_skip_reasons.update(json.loads(row.get("critic_skip_reasons") or "[]"))
        if row.get("reasoning_depth"):
            reasoning_depths.update([row["reasoning_depth"]])
        if row.get("quality_risk"):
            quality_risks.update([row["quality_risk"]])
        control_flags.update(json.loads(row.get("control_flags") or "[]"))
        preference_flags.update(json.loads(row.get("preference_flags") or "[]"))
        safety_flags.update(json.loads(row.get("safety_flags") or "[]"))
        unclassified_flags.update(json.loads(row.get("unclassified_flags") or "[]"))
        memory_event_types.update(json.loads(row.get("memory_event_types") or "{}"))
        memory_event_scopes.update(json.loads(row.get("memory_event_scopes") or "{}"))

    n_search_hit = sum(1 for row in evaluated if row.get("search_hit_rate") == 1.0)
    n_retrieval_miss = sum(1 for row in evaluated if row.get("retrieval_miss") == 1)
    n_rerank_loss_3 = sum(1 for row in evaluated if row.get("rerank_loss@3") == 1)
    retrieval_cache_hits = sum(int(row.get("retrieval_cache_hit") or 0) for row in turn_rows)
    retrieval_cache_misses = sum(int(row.get("retrieval_cache_miss") or 0) for row in turn_rows)
    retrieval_candidate_reuse_calls = sum(int(row.get("retrieval_candidate_reuse") or 0) for row in turn_rows)
    intent_metrics = intent_classification_metrics(turn_rows)
    total_memory_events = sum(int(row.get("memory_event_count") or 0) for row in turn_rows)
    turns_with_memory_events = sum(1 for row in turn_rows if int(row.get("memory_event_count") or 0) > 0)
    conv_latencies = [row.get("latency_ms_avg") for row in all_convs if row.get("latency_ms_avg") not in ("", None)]

    return {
        "split": args.split,
        "fold": args.fold or "",
        "split_directory": str((ROOT / "outputs" / args.fold) if args.fold else SPLITS_DIR),
        "planner_mode": args.planner_mode,
        "execution_scope": args.execution_scope,
        "retrieval_backend": os.getenv("AUTONOMOUS_RETRIEVAL_BACKEND", ""),
        "effective_config": effective_config(),
        "n_conversations_total": len(all_convs),
        "n_conversations_failed": sum(int(row.get("eval_failed") or 0) for row in all_convs),
        "n_conversations_with_gt": len(evaluated),
        "n_turns": len(turn_rows),
        "intent_diagnostics": intent_metrics,
        "elapsed_seconds": round(elapsed, 2),
        "n_conversations_recommended": sum(int(row["recommended"]) for row in all_convs),
        "ndcg@3": mean_key(evaluated, "ndcg@3"),
        "ndcg@10": mean_key(evaluated, "ndcg@10"),
        "hit@1": mean_key(evaluated, "hit@1"),
        "hit@3": mean_key(evaluated, "hit@3"),
        "hit@5": mean_key(evaluated, "hit@5"),
        "hit@10": mean_key(evaluated, "hit@10"),
        "mrr": mean_key(evaluated, "mrr"),
        "displayed_ndcg@3": mean_key(evaluated, "displayed_ndcg@3"),
        "displayed_ndcg@10": mean_key(evaluated, "displayed_ndcg@10"),
        "displayed_hit@1": mean_key(evaluated, "displayed_hit@1"),
        "displayed_hit@3": mean_key(evaluated, "displayed_hit@3"),
        "displayed_hit@5": mean_key(evaluated, "displayed_hit@5"),
        "displayed_hit@10": mean_key(evaluated, "displayed_hit@10"),
        "displayed_mrr": mean_key(evaluated, "displayed_mrr"),
        "internal_ranked_ndcg@3": mean_key(evaluated, "internal_ranked_ndcg@3"),
        "internal_ranked_ndcg@10": mean_key(evaluated, "internal_ranked_ndcg@10"),
        "internal_ranked_hit@1": mean_key(evaluated, "internal_ranked_hit@1"),
        "internal_ranked_hit@3": mean_key(evaluated, "internal_ranked_hit@3"),
        "internal_ranked_hit@5": mean_key(evaluated, "internal_ranked_hit@5"),
        "internal_ranked_hit@10": mean_key(evaluated, "internal_ranked_hit@10"),
        "internal_ranked_mrr": mean_key(evaluated, "internal_ranked_mrr"),
        "recall@50": mean_key(evaluated, "recall@50"),
        "search_hit_rate": mean_key(evaluated, "search_hit_rate"),
        "search_recall": mean_key(evaluated, "search_recall"),
        "retrieval_miss_rate": round(n_retrieval_miss / max(len(evaluated), 1), 4),
        "rerank_loss@3_rate": round(n_rerank_loss_3 / max(len(evaluated), 1), 4),
        "rerank_loss@3_given_search_hit": round(n_rerank_loss_3 / max(n_search_hit, 1), 4),
        "exact_recommendation_hit@3": mean_key(evaluated, "hit@3"),
        "exact_recommendation_precision": mean_key(evaluated, "exact_precision@3"),
        "exact_recommendation_recall": mean_key(evaluated, "exact_recall@3"),
        "exact_recommendation_f1": mean_key(evaluated, "exact_f1@3"),
        "displayed_semantic_hit_rate": mean_key(evaluated, "displayed_semantic_hit_rate"),
        "displayed_semantic_precision": mean_key(evaluated, "displayed_semantic_precision"),
        "displayed_semantic_recall": mean_key(evaluated, "displayed_semantic_recall"),
        "displayed_semantic_f1": mean_key(evaluated, "displayed_semantic_f1"),
        "internal_ranked_semantic_hit_rate": mean_key(evaluated, "internal_ranked_semantic_hit_rate"),
        "internal_ranked_semantic_precision": mean_key(evaluated, "internal_ranked_semantic_precision"),
        "internal_ranked_semantic_recall": mean_key(evaluated, "internal_ranked_semantic_recall"),
        "internal_ranked_semantic_f1": mean_key(evaluated, "internal_ranked_semantic_f1"),
        "atr": mean_key(all_convs, "turns_to_rec"),
        "success_rate@3": round(sum(1 for row in all_convs if float(row.get("hit@3") or 0) > 0) / max(len(all_convs), 1), 4),
        "success_rate@5": round(sum(1 for row in all_convs if float(row.get("hit@5") or 0) > 0) / max(len(all_convs), 1), 4),
        "success_rate@10": round(sum(1 for row in all_convs if float(row.get("hit@10") or 0) > 0) / max(len(all_convs), 1), 4),
        "llm_planner_turn_rate": round(turns_with_llm_planner / max(len(turn_rows), 1), 4),
        "planner_unchanged_execution_rate": round(turns_with_unchanged_llm_plan / max(turns_with_llm_planner, 1), 4),
        "planner_unchanged_turn_rate": round(turns_with_unchanged_llm_plan / max(len(turn_rows), 1), 4),
        "planner_fallback_turn_rate": round(turns_with_fallback / max(len(turn_rows), 1), 4),
        "guard_intervention_turn_rate": round(turns_with_guard / max(len(turn_rows), 1), 4),
        "guard_intervention_reasons": dict(guard_intervention_reasons),
        "planner_steps_avg": mean_key(turn_rows, "planner_steps"),
        "planner_steps_max": max([int(row["planner_steps"]) for row in turn_rows], default=0),
        "components_per_turn_avg": average(component_calls_per_turn),
        "components_per_turn_max": max(component_calls_per_turn, default=0),
        "planner_duration_ms_avg": mean_key(turn_rows, "planner_duration_ms"),
        "component_duration_ms_total": dict(component_durations),
        "retrieval_call_turn_rate": round(turns_with_retrieval / max(len(turn_rows), 1), 4),
        "retrieval_cache_hits": retrieval_cache_hits,
        "retrieval_cache_misses": retrieval_cache_misses,
        "retrieval_candidate_reuse_calls": retrieval_candidate_reuse_calls,
        "retrieval_cache_hit_rate": round(retrieval_cache_hits / max(turns_with_retrieval, 1), 4),
        "retrieval_cache_miss_rate": round(retrieval_cache_misses / max(turns_with_retrieval, 1), 4),
        "actual_retrieval_call_turn_rate": round(retrieval_cache_misses / max(len(turn_rows), 1), 4),
        "reasoning_reflection_calls": sum(int(row["reasoning_reflection_called"]) for row in turn_rows),
        "reasoning_reflection_failures": sum(int(row["reasoning_reflection_failed"]) for row in turn_rows),
        "reasoning_reflection_skips": sum(int(row.get("reasoning_reflection_skipped") or 0) for row in turn_rows),
        "reasoning_reflection_cache_hits": sum(int(row.get("reasoning_reflection_cache_hit") or 0) for row in turn_rows),
        "reflection_call_turn_rate": round(sum(int(row["reasoning_reflection_called"]) for row in turn_rows) / max(len(turn_rows), 1), 4),
        "reflection_skip_turn_rate": round(sum(int(row.get("reasoning_reflection_skipped") or 0) for row in turn_rows) / max(len(turn_rows), 1), 4),
        "reflection_cache_hit_rate": round(sum(int(row.get("reasoning_reflection_cache_hit") or 0) for row in turn_rows) / max(sum(int(row["reasoning_reflection_called"]) for row in turn_rows), 1), 4),
        "reflection_failure_rate": round(sum(int(row["reasoning_reflection_failed"]) for row in turn_rows) / max(sum(int(row["reasoning_reflection_called"]) for row in turn_rows), 1), 4),
        "critic_reasoning_calls": sum(int(row["critic_reasoning_called"]) for row in turn_rows),
        "critic_reasoning_failures": sum(int(row["critic_reasoning_failed"]) for row in turn_rows),
        "critic_reasoning_skips": sum(int(row.get("critic_reasoning_skipped") or 0) for row in turn_rows),
        "critic_reasoning_cache_hits": sum(int(row.get("critic_reasoning_cache_hit") or 0) for row in turn_rows),
        "critic_call_turn_rate": round(sum(int(row["critic_reasoning_called"]) for row in turn_rows) / max(len(turn_rows), 1), 4),
        "critic_skip_turn_rate": round(sum(int(row.get("critic_reasoning_skipped") or 0) for row in turn_rows) / max(len(turn_rows), 1), 4),
        "critic_cache_hit_rate": round(sum(int(row.get("critic_reasoning_cache_hit") or 0) for row in turn_rows) / max(sum(int(row["critic_reasoning_called"]) for row in turn_rows), 1), 4),
        "critic_failure_rate": round(sum(int(row["critic_reasoning_failed"]) for row in turn_rows) / max(sum(int(row["critic_reasoning_called"]) for row in turn_rows), 1), 4),
        "llm_calls_per_conversation": round(
            (
                turns_with_llm_planner
                + sum(int(row["reasoning_reflection_called"]) for row in turn_rows)
                + sum(int(row["critic_reasoning_called"]) for row in turn_rows)
            )
            / max(len(all_convs), 1),
            2,
        ),
        "search_again_actions": sum(int(row["search_again_action"]) for row in turn_rows),
        "search_again_turn_rate": round(sum(int(row["search_again_action"]) for row in turn_rows) / max(len(turn_rows), 1), 4),
        "fallback_retrieval_calls": sum(int(row["retrieval_fallback_used"]) for row in turn_rows),
        "full_stack_retrieval_failures": sum(1 for row in turn_rows if row.get("retrieval_backend_actual") == "conserve_stack_failed"),
        "fallback_retrieval_rate": round(sum(int(row["retrieval_fallback_used"]) for row in turn_rows) / max(turns_with_retrieval, 1), 4),
        "full_stack_failure_rate": round(sum(1 for row in turn_rows if row.get("retrieval_backend_actual") == "conserve_stack_failed") / max(turns_with_retrieval, 1), 4),
        "latency_ms_avg": mean_key(turn_rows, "latency_ms"),
        "latency_ms_p50": percentile_key(turn_rows, "latency_ms", 50),
        "latency_ms_p95": percentile_key(turn_rows, "latency_ms", 95),
        "latency_ms_max": max([int(row["latency_ms"]) for row in turn_rows], default=0),
        "conversation_latency_ms_avg": average(conv_latencies),
        "conversation_latency_ms_p50": percentile_values(conv_latencies, 50),
        "conversation_latency_ms_p95": percentile_values(conv_latencies, 95),
        "planner_prompt_tokens": sum(int(row["planner_prompt_tokens"]) for row in turn_rows),
        "planner_completion_tokens": sum(int(row["planner_completion_tokens"]) for row in turn_rows),
        "planner_total_tokens": sum(int(row["planner_total_tokens"]) for row in turn_rows),
        "planner_tokens_per_turn": round(sum(int(row["planner_total_tokens"]) for row in turn_rows) / max(len(turn_rows), 1), 2),
        "planner_tokens_per_conversation": round(sum(int(row["planner_total_tokens"]) for row in turn_rows) / max(len(all_convs), 1), 2),
        "reasoning_depths": dict(reasoning_depths),
        "quality_risks": dict(quality_risks),
        "control_flags": dict(control_flags),
        "preference_flags": dict(preference_flags),
        "safety_flags": dict(safety_flags),
        "unclassified_flags": dict(unclassified_flags),
        "memory_events_total": total_memory_events,
        "memory_event_turn_rate": round(turns_with_memory_events / max(len(turn_rows), 1), 4),
        "memory_events_per_turn": round(total_memory_events / max(len(turn_rows), 1), 4),
        "memory_events_per_conversation": round(total_memory_events / max(len(all_convs), 1), 4),
        "memory_event_types": dict(memory_event_types),
        "memory_event_scopes": dict(memory_event_scopes),
        "preference_revision_events": sum(
            count for event, count in memory_event_types.items()
            if "revision" in event or "clear" in event
        ),
        "preference_rejection_events": sum(
            count for event, count in memory_event_types.items()
            if "reject" in event or "rejection" in event
        ),
        "preference_acceptance_events": sum(
            count for event, count in memory_event_types.items()
            if "accept" in event
        ),
        "planner_backends": dict(backend_counts),
        "retrieval_actual_backends": dict(retrieval_backend_counts),
        "triplet_reranker_diagnostics": triplet_reranker_diagnostics(),
        "planner_errors": dict(planner_errors),
        "reflection_errors": dict(reflection_errors),
        "critic_errors": dict(critic_errors),
        "reflection_call_reasons": dict(reflection_call_reasons),
        "reflection_skip_reasons": dict(reflection_skip_reasons),
        "critic_call_reasons": dict(critic_call_reasons),
        "critic_skip_reasons": dict(critic_skip_reasons),
        "components_called": dict(component_counts),
        "critic_decisions": dict(critic_decisions),
        "critic_next_actions": dict(critic_actions),
    }


def last_recommendation_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in reversed(rows):
        if row.get("recommended") and int(row.get("n_raw") or 0) > 0:
            return row
    for row in reversed(rows):
        if row.get("recommended"):
            return row
    return None


def component_duration_totals(observations: list[dict[str, Any]]) -> dict[str, int]:
    totals: Counter = Counter()
    for observation in observations:
        component = observation.get("component")
        duration = observation.get("duration_ms")
        if component and isinstance(duration, (int, float)):
            totals[str(component)] += int(duration)
    return dict(totals)


def _guard_reason_buckets(planner_step_trace: list[dict[str, Any]]) -> list[str]:
    buckets: list[str] = []
    for step in planner_step_trace:
        backend = str(step.get("backend") or "")
        if "guard" not in backend:
            continue
        reason = str(step.get("reason") or "").lower()
        if "product/category is known" in reason or "build the structured search request" in reason:
            bucket = "workflow_sequence_build_request"
        elif "pre-retrieval critic" in reason or "critic readiness" in reason or "requires pre-retrieval" in reason:
            bucket = "workflow_sequence_pre_retrieval_critic"
        elif "ask the user before retrieval" in reason or "missing" in reason or "clarification" in reason:
            bucket = "missing_state_or_clarification"
        elif "accepted current recommendations" in reason or "skip retrieval" in reason or "no need to search" in reason:
            bucket = "acceptance_or_skip_retrieval"
        elif "search again" in reason or "search-again" in reason:
            bucket = "search_again_recovery"
        elif "before responding" in reason or "before recommendation" in reason or "before retrieval" in reason:
            bucket = "workflow_contract_enforcement"
        elif "invalid" in reason or "unsupported" in reason:
            bucket = "invalid_planner_action"
        elif "safety" in reason or "unsafe" in reason or "security" in reason:
            bucket = "safety_or_constraint"
        else:
            bucket = "other_guard_intervention"
        buckets.append(bucket)
    return buckets


def effective_config() -> dict[str, str]:
    keys = [
        "AUTONOMOUS_RETRIEVAL_BACKEND",
        "AUTONOMOUS_EXECUTION_SCOPE",
        "AUTONOMOUS_PLANNER_MODE",
        "AUTONOMOUS_PLANNER_PROVIDER",
        "AUTONOMOUS_PLANNER_MODEL",
        "AUTONOMOUS_PLANNER_TIMEOUT",
        "AUTONOMOUS_PLAN_POLICY_MODE",
        "AUTONOMOUS_REFLECTION_MODE",
        "AUTONOMOUS_REFLECTION_PROVIDER",
        "AUTONOMOUS_REFLECTION_MODEL",
        "AUTONOMOUS_REFLECTION_TIMEOUT",
        "AUTONOMOUS_CRITIC_MODE",
        "AUTONOMOUS_CRITIC_PROVIDER",
        "AUTONOMOUS_CRITIC_MODEL",
        "AUTONOMOUS_CRITIC_TIMEOUT",
        "AUTONOMOUS_RESPONSE_MODE",
        "AUTONOMOUS_RESPONSE_PROVIDER",
        "AUTONOMOUS_RESPONSE_MODEL",
        "AUTONOMOUS_RESPONSE_TIMEOUT",
        "AUTONOMOUS_RESPONSE_LLM_RECOMMENDATIONS",
        "AUTONOMOUS_REASONING_BUDGET",
        "AUTONOMOUS_DECISION_DIVERSITY",
        "AUTONOMOUS_AGENT_ADJUSTMENT_WEIGHT",
        "AUTONOMOUS_TRIPLET_CONTEXT",
        "MACPRS_USE_TRIPLET_RERANKER",
        "MACPRS_TRIPLET_MODEL_PATH",
        "MACPRS_TRIPLET_FUSION_MODE",
        "MACPRS_TRIPLET_BLEND_WEIGHT",
    ]
    return {key: os.getenv(key, "") for key in keys}


def triplet_reranker_diagnostics() -> dict[str, Any]:
    try:
        from src.agents.reranker import get_triplet_reranker_diagnostics
        diagnostics = dict(get_triplet_reranker_diagnostics())
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    diagnostics["available"] = True
    return diagnostics


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    relevances = [1 if pid in relevant_ids else 0 for pid in ranked_ids[:k]]
    ideal = [1] * min(len(relevant_ids), k) + [0] * max(0, k - len(relevant_ids))
    ideal_dcg = dcg(ideal)
    return round(dcg(relevances) / ideal_dcg, 4) if ideal_dcg else 0.0


def hit_rate_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return 1.0 if any(pid in relevant_ids for pid in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    for index, pid in enumerate(ranked_ids, start=1):
        if pid in relevant_ids:
            return round(1.0 / index, 4)
    return 0.0


def recall_at_k(candidate_ids: list[str], relevant_ids: set[str], k: int) -> float | None:
    if not relevant_ids:
        return None
    return round(sum(1 for pid in relevant_ids if pid in candidate_ids[:k]) / len(relevant_ids), 4)


def search_hit_rate(candidate_ids: list[str], relevant_ids: set[str]) -> float | None:
    if not relevant_ids:
        return None
    return 1.0 if any(pid in relevant_ids for pid in candidate_ids) else 0.0


def item_level_recall(candidate_ids: list[str], relevant_ids: set[str]) -> float | None:
    if not relevant_ids:
        return None
    return round(sum(1 for pid in relevant_ids if pid in candidate_ids) / len(relevant_ids), 4)


def relevant_count(candidate_ids: list[str], relevant_ids: set[str], k: int | None = None) -> int:
    ids = candidate_ids[:k] if k is not None else candidate_ids
    return sum(1 for pid in ids if pid in relevant_ids)


def precision_recall_f1_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> dict[str, float | None]:
    if not relevant_ids:
        return {"precision": None, "recall": None, "f1": None}
    top_k = ranked_ids[:k]
    hits = sum(1 for pid in top_k if pid in relevant_ids)
    precision = hits / len(top_k) if top_k else 0.0
    recall = hits / len(relevant_ids)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def semantic_match_metrics(
    shown_ids: list[str],
    relevant_ids: set[str],
    text_by_id: dict[str, str],
    k: int = 3,
) -> dict[str, float | None]:
    if not relevant_ids:
        return {"hit_rate": None, "precision": None, "recall": None, "f1": None}
    if not text_by_id:
        return {"hit_rate": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    top_k = shown_ids[:k]
    rel_vecs = [
        vector
        for vector in (_semantic_embedding(product_id, text_by_id) for product_id in relevant_ids)
        if vector is not None
    ]
    shown_vecs = [
        (product_id, vector)
        for product_id in top_k
        for vector in [_semantic_embedding(product_id, text_by_id)]
        if vector is not None
    ]

    if not rel_vecs or not shown_vecs:
        return {"hit_rate": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    import numpy as np

    rel_matrix = np.stack(rel_vecs)
    shown_hits = []
    for _product_id, vector in shown_vecs:
        similarities = rel_matrix @ vector
        shown_hits.append(bool(similarities.max() >= SEMANTIC_MATCH_THRESHOLD))

    shown_matrix = np.stack([vector for _product_id, vector in shown_vecs])
    rel_hits = []
    for rel_vector in rel_vecs:
        similarities = shown_matrix @ rel_vector
        rel_hits.append(bool(similarities.max() >= SEMANTIC_MATCH_THRESHOLD))

    hit_rate = 1.0 if any(shown_hits) else 0.0
    precision = sum(shown_hits) / len(shown_hits) if shown_hits else 0.0
    recall = sum(rel_hits) / len(rel_hits) if rel_hits else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "hit_rate": hit_rate,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _semantic_embedding(product_id: str, text_by_id: dict[str, str]):
    if product_id in _semantic_embed_cache:
        return _semantic_embed_cache[product_id]

    from src.agents.search_agent import _get_embedder

    text = text_by_id.get(product_id, "")
    if not text:
        _semantic_embed_cache[product_id] = None
        return None
    vector = _get_embedder().encode([text], normalize_embeddings=True)[0]
    _semantic_embed_cache[product_id] = vector
    return vector


def _semantic_vector_texts() -> dict[str, str]:
    if _local_numpy_stack_is_incompatible():
        return {}
    try:
        from src.agents.search_agent import _get_vector_texts

        return _get_vector_texts()
    except Exception:
        return {}


def _local_numpy_stack_is_incompatible() -> bool:
    try:
        import numpy as np

        major, minor, *_ = [int(part) for part in np.__version__.split(".")[:2]]
    except Exception:
        return False
    return (major, minor) >= (2, 3)


def product_ids(products: list[dict[str, Any]], limit: int | None = None) -> list[str]:
    ids = [str(product.get("product_id")) for product in products if product.get("product_id")]
    return ids[:limit] if limit else ids


def current_turn_notes(notes: list[dict[str, Any]], turn_id: int | None) -> list[dict[str, Any]]:
    return [note for note in to_plain(notes) if note.get("turn_id") == turn_id]


def usage_totals(turn_trace: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    seen = set()
    for step in turn_trace:
        usage = step.get("planner_usage") or {}
        key = json.dumps(usage, sort_keys=True)
        if not usage or key in seen:
            continue
        seen.add(key)
        for field in totals:
            try:
                totals[field] += int(usage.get(field) or 0)
            except (TypeError, ValueError):
                pass
    return totals


def latest_turn_analysis(turn_trace: list[dict[str, Any]]) -> dict[str, Any]:
    for step in turn_trace:
        analysis = step.get("turn_analysis")
        if isinstance(analysis, dict) and analysis:
            return analysis
    return {}


def _normalize_intent_label(label: Any) -> str:
    normalized = str(label or "").strip().lower()
    aliases = {
        "reveal": "reveal",
        "interpret": "interpret",
        "revise": "revise",
        "inquire": "inquire",
        "chitchat": "chitchat",
        "chat": "chitchat",
    }
    return aliases.get(normalized, normalized)


def _predicted_pscon_intent(intent: dict[str, Any]) -> str:
    explicit = _normalize_intent_label(
        intent.get("pscon_label")
        or intent.get("reflection_pscon_intent_label")
        or intent.get("encoder_pscon_label")
    )
    if explicit in {"reveal", "interpret", "revise", "inquire", "chitchat"}:
        return explicit

    action_label = str(intent.get("action_label") or "").strip().lower()
    workflow_label = str(intent.get("label") or "").strip().lower()
    if action_label == "chitchat" or workflow_label == "chitchat":
        return "chitchat"
    if action_label == "inquiry" or workflow_label == "inquire":
        return "inquire"
    if action_label in {"rejection", "revision"} or workflow_label == "reject_or_revise":
        return "revise"
    if action_label == "new_request":
        return "reveal"
    if action_label in {"new_request_or_refinement", "clarification"} or workflow_label == "recommend":
        return "interpret"
    if action_label == "acceptance" or workflow_label == "accept":
        return "interpret"
    return ""


def intent_classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ["reveal", "interpret", "revise", "inquire", "chitchat"]
    pairs = [
        (row.get("dataset_intent_normalized"), row.get("predicted_pscon_intent"))
        for row in rows
        if row.get("dataset_intent_normalized") in labels and row.get("predicted_pscon_intent") in labels
    ]
    support = Counter(gold for gold, _pred in pairs)
    predictions = Counter(pred for _gold, pred in pairs)
    confusion: dict[str, dict[str, int]] = {
        gold: {pred: 0 for pred in labels}
        for gold in labels
    }
    for gold, pred in pairs:
        confusion[gold][pred] += 1

    per_class = {}
    f1_values = []
    weighted_f1_sum = 0.0
    for label in labels:
        tp = confusion[label][label]
        fp = predictions[label] - tp
        fn = support[label] - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall / max(precision + recall, 1e-12)) if (precision or recall) else 0.0
        f1_values.append(f1)
        weighted_f1_sum += f1 * support[label]
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support[label],
        }

    total = len(pairs)
    correct = sum(1 for gold, pred in pairs if gold == pred)
    return {
        "n_turns_with_gold_and_prediction": total,
        "accuracy": round(correct / max(total, 1), 4) if total else None,
        "macro_f1": round(sum(f1_values) / len(labels), 4) if total else None,
        "weighted_f1": round(weighted_f1_sum / max(total, 1), 4) if total else None,
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def mean_key(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if value in ("", None):
            continue
        values.append(float(value))
    return round(sum(values) / len(values), 4) if values else None


def average(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value not in ("", None)]
    return round(sum(numeric) / len(numeric), 4) if numeric else None


def percentile_key(rows: list[dict[str, Any]], key: str, percentile: float) -> float | None:
    return percentile_values([row.get(key) for row in rows], percentile)


def percentile_values(values: list[Any], percentile: float) -> float | None:
    numeric = sorted(float(value) for value in values if value not in ("", None))
    if not numeric:
        return None
    if len(numeric) == 1:
        return round(numeric[0], 4)
    rank = (len(numeric) - 1) * (percentile / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return round(numeric[lower], 4)
    weight = rank - lower
    return round(numeric[lower] * (1 - weight) + numeric[upper] * weight, 4)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    return value


if __name__ == "__main__":
    main()
