#!/usr/bin/env python3
"""Single-agent LLM baseline using the same PRISM retrieval/reranking stack.

This evaluator is intentionally separate from the autonomous PRISM runner. It
uses one prompt-based controller to infer the current shopping state and search
request, then calls the existing ConServe search tool. There is no multi-agent
planner, reflection agent, critic agent, or search-again loop.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.autonomous_full_recommendation_eval import (
    RESULTS_DIR,
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
from src.autonomous.tools.conserve_search_tool import ConServeSearchTool


BASELINE_RESULTS_DIR = RESULTS_DIR.parent / "single_agent_baseline"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--fold", choices=["fold1", "fold2", "fold3", "fold4"], default=None)
    parser.add_argument("--max-convs", type=int, default=None)
    parser.add_argument(
        "--retrieval-backend",
        default=os.getenv("AUTONOMOUS_RETRIEVAL_BACKEND", "conserve_stack"),
    )
    parser.add_argument(
        "--controller-provider",
        default=os.getenv("SINGLE_AGENT_PROVIDER", "deepseek"),
        choices=["deepseek", "rule"],
    )
    parser.add_argument(
        "--controller-model",
        default=os.getenv("SINGLE_AGENT_MODEL")
        or os.getenv("AUTONOMOUS_PLANNER_MODEL", "deepseek-v4-flash"),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--display-k", type=int, default=3)
    parser.add_argument(
        "--search-depth",
        type=int,
        default=50,
        help="One-pass candidate budget; match PRISM's initial retrieval budget.",
    )
    parser.add_argument(
        "--max-query-variants",
        type=int,
        default=3,
        help="Maximum variants per retrieval request; resource-match PRISM's typical query expansion.",
    )
    parser.add_argument("--out-dir", type=Path, default=BASELINE_RESULTS_DIR)
    args = parser.parse_args()

    os.environ["AUTONOMOUS_RETRIEVAL_BACKEND"] = args.retrieval_backend
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    split_dir = configure_fold(args.fold) if args.fold else None
    conversations = load_split(args.split, split_dir=split_dir) if split_dir else load_split(args.split)
    if args.max_convs:
        conversations = conversations[: args.max_convs]

    ground_truth = build_ground_truth(conversations)
    controller = SingleAgentController(
        provider=args.controller_provider,
        model=args.controller_model,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    turn_rows: list[dict[str, Any]] = []
    conversation_rows: list[dict[str, Any]] = []

    started = time.time()
    for index, conv in enumerate(conversations, start=1):
        cid = str(conv.get("conv_id"))
        relevant_ids = ground_truth.get(cid, set())
        state = _initial_state()
        search_tool = ConServeSearchTool(top_k=args.top_k)
        history: list[dict[str, Any]] = []
        conv_turn_rows: list[dict[str, Any]] = []

        try:
            turns = extract_user_turns(conv)
            print(
                f"[{index:3}/{len(conversations)}] conv {cid} "
                f"turns={len(turns)} relevant={len(relevant_ids)}",
                flush=True,
            )
            for turn_index, turn in enumerate(turns, start=1):
                user_message = str(turn.get("content") or "").strip()
                if not user_message:
                    continue

                state["turn_id"] = turn_index
                state["latest_user_message"] = user_message
                history.append({"role": "user", "content": user_message})
                state["history"] = list(history)

                before_trace_len = len(state.get("planner_trace") or [])
                turn_started = time.perf_counter()
                response = run_single_agent_turn(
                    state=state,
                    controller=controller,
                    search_tool=search_tool,
                    display_k=args.display_k,
                    search_depth=args.search_depth,
                    max_query_variants=args.max_query_variants,
                )
                latency_ms = int((time.perf_counter() - turn_started) * 1000)
                history.append({"role": "assistant", "content": response})
                state["history"] = list(history)

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

            conversation_rows.append(summarize_conversation(cid, conv_turn_rows, relevant_ids))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[{index:3}/{len(conversations)}] conv {cid} failed: {error}", flush=True)
            conversation_rows.append(failed_conversation_row(cid, relevant_ids, error))

    summary_args = argparse.Namespace(
        split=args.split,
        fold=args.fold or "",
        planner_mode="single_agent_baseline",
        execution_scope="single_agent",
    )
    summary = summarize(
        turn_rows,
        conversation_rows,
        summary_args,
        elapsed=time.time() - started,
    )
    summary["single_agent_controller_provider"] = args.controller_provider
    summary["single_agent_controller_model"] = args.controller_model
    summary["retrieval_candidate_budget"] = args.search_depth
    summary["max_query_variants"] = args.max_query_variants
    summary["single_agent_notes"] = (
        "One prompt-based controller owns state revision and direct retrieval "
        "request construction; the same ConServe retrieval/reranking tool is used. "
        "Planner, reflection, critic, search-request agent, decision agent, and "
        "search-again recovery are disabled. Query expansion is capped to the "
        "configured resource-matched budget."
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    turn_path = args.out_dir / f"turns_{args.split}_{timestamp}.csv"
    conv_path = args.out_dir / f"conversations_{args.split}_{timestamp}.csv"
    summary_path = args.out_dir / f"summary_{args.split}_{timestamp}.json"
    write_csv(turn_path, turn_rows)
    write_csv(conv_path, conversation_rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSingle-agent baseline evaluation complete")
    print(f"Summary       -> {summary_path}")
    print(f"Turn rows     -> {turn_path}")
    print(f"Conversations -> {conv_path}")
    print()
    print(f"Conversations: {summary['n_conversations_total']}")
    print(f"Failed conversations: {summary['n_conversations_failed']}")
    print(f"Hit@5/Hit@10: {summary['hit@5']} / {summary['hit@10']}")
    print(f"NDCG@10/MRR: {summary['ndcg@10']} / {summary['mrr']}")
    print(f"Search hit/retrieval miss: {summary['search_hit_rate']} / {summary['retrieval_miss_rate']}")
    print(f"P95 latency ms/turn: {summary['latency_ms_p95']}")
    print(f"Planner tokens/turn: {summary['planner_tokens_per_turn']}")


def run_single_agent_turn(
    *,
    state: dict[str, Any],
    controller: "SingleAgentController",
    search_tool: ConServeSearchTool,
    display_k: int,
    search_depth: int,
    max_query_variants: int,
) -> str:
    turn_id = int(state.get("turn_id") or 0)
    decision_started = time.perf_counter()
    decision, usage, backend = controller.decide(
        history=state.get("history") or [],
        latest_user_message=str(state.get("latest_user_message") or ""),
        memory=state.get("single_agent_memory") or {},
    )
    decision_ms = int((time.perf_counter() - decision_started) * 1000)
    request = _normalize_search_request(decision, state)
    state["single_agent_memory"] = _update_single_agent_memory(
        state.get("single_agent_memory") or {},
        request,
        decision,
    )
    _project_memory_to_prism_state(state)

    state["intent"] = _intent_from_decision(decision)
    state["control_flags"] = []
    state["preference_flags"] = list(decision.get("preference_flags") or [])
    state["safety_flags"] = []
    state["unclassified_flags"] = []
    state["preference_events"] = _preference_events(decision, turn_id)

    _append_trace(
        state,
        {
            "step": len(state.get("planner_trace") or []) + 1,
            "action": "CALL_TOOL" if decision.get("should_recommend", True) else "RESPOND",
            "target": "single_agent_controller",
            "reason": str(decision.get("reason") or "Single-agent controller built the search request."),
            "planner_backend": "llm_plan" if backend.startswith("deepseek") else backend,
            "planner_duration_ms": decision_ms,
            "planner_usage": usage,
            "turn_analysis": {
                "reasoning_depth": "single_agent",
                "quality_risk": "no_reflection_or_critic",
            },
        },
    )

    if not bool(decision.get("should_recommend", True)):
        state["raw_candidates"] = []
        state["verified_candidates"] = []
        state["last_recommendations"] = []
        _append_trace(
            state,
            {
                "step": len(state.get("planner_trace") or []) + 1,
                "action": "CALL_TOOL",
                "target": "response_tool",
                "reason": "Single-agent controller chose not to retrieve products for this turn.",
                "planner_backend": "single_agent_rule",
            },
        )
        return str(decision.get("response") or "Could you clarify what you would like to change or compare?")

    # Pass the controller's request directly to retrieval. Calling PRISM's
    # SearchResearchAgent would both add a specialist and discard this LLM's
    # query variants, so it would not be a single-agent baseline.
    state["final_search_request"] = request
    state["last_final_search_request"] = request
    state["retrieval_plan"] = _retrieval_plan(
        request,
        decision=decision,
        state=state,
        search_depth=search_depth,
        max_query_variants=max_query_variants,
    )
    state["last_retrieval_plan"] = state["retrieval_plan"]
    state["retrieval_readiness"] = "ready"
    state["search_attempts"] = 0
    state["raw_candidates"] = []
    state["verified_candidates"] = []
    state["candidate_ledger"] = {}

    search_started = time.perf_counter()
    result = search_tool.run(state)
    search_ms = int((time.perf_counter() - search_started) * 1000)
    _apply_patch_to_state(state, result.patch)
    observation = result.patch.get("tool_observation") or {}
    observation["duration_ms"] = search_ms
    state.setdefault("tool_observations", []).append(observation)

    verified = list(state.get("verified_candidates") or [])
    recommendations = verified[:display_k]
    state["last_recommendations"] = recommendations
    state["last_raw_candidates"] = list(state.get("raw_candidates") or [])
    state["last_verified_candidates"] = verified
    state["last_candidate_ledger"] = state.get("candidate_ledger") or {}

    _append_trace(
        state,
        {
            "step": len(state.get("planner_trace") or []) + 1,
            "action": "CALL_TOOL",
            "target": "conserve_search_tool",
            "reason": "Single-agent controller handed one search request to the shared retrieval stack.",
            "planner_backend": "single_agent_rule",
        },
    )
    return _response_from_recommendations(recommendations, decision)


class SingleAgentController:
    def __init__(self, *, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.timeout = float(os.getenv("SINGLE_AGENT_TIMEOUT", os.getenv("AUTONOMOUS_PLANNER_TIMEOUT", "60")))
        self.endpoint = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
        self.thinking = os.getenv("DEEPSEEK_THINKING", "disabled").lower()
        self.reasoning_effort = os.getenv("DEEPSEEK_REASONING_EFFORT", "low").lower()

    def decide(
        self,
        *,
        history: list[dict[str, Any]],
        latest_user_message: str,
        memory: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, int], str]:
        if self.provider == "rule":
            return _rule_decision(latest_user_message), {}, "single_agent_rule"
        decision, usage = self._call_deepseek(history, memory)
        return decision, usage, f"deepseek:{self.model}"

    def _call_deepseek(
        self,
        history: list[dict[str, Any]],
        memory: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, int]]:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a single-agent conversational shopping controller. "
                        "Infer the active user preferences and produce one structured "
                        "search request only when the user is asking for one. "
                        "Do not create a new recommendation after an acknowledgement, "
                        "acceptance, thanks, order confirmation, or closing message. "
                        "Return only valid JSON."
                    ),
                },
                {"role": "user", "content": _controller_prompt(history, memory)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 650,
            "stream": False,
        }
        if self.thinking in {"enabled", "disabled"}:
            thinking: dict[str, Any] = {"type": self.thinking}
            if self.thinking == "enabled":
                thinking["reasoning_effort"] = self.reasoning_effort
            payload["thinking"] = thinking

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek single-agent HTTP {exc.code}: {body[:300]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("DeepSeek single-agent endpoint is unavailable") from exc

        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("DeepSeek response did not include message content") from exc
        return _validate_decision(_parse_json_object(content)), _usage(raw.get("usage") or {})


def _controller_prompt(history: list[dict[str, Any]], memory: dict[str, Any]) -> str:
    compact_history = []
    for turn in history[-10:]:
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "").strip()
        if content:
            compact_history.append({"role": role, "content": content[:1200]})
    return f"""Task: act as the ONLY controller for a product recommender baseline.

Use the conversation history to infer the active shopping request. Respect changed
or rejected preferences. Set should_recommend=true only when the user explicitly
asks for a product, more options, a replacement, a comparison, or a preference
change that requires a new recommendation.

Set should_recommend=false for acknowledgements, acceptance, thanks, order
confirmation, or closing turns (for example, "yes, I like this one", "that looks
good", "place the order", or "thanks") unless that same turn also explicitly asks
for another product, a replacement, comparison, or changed preference. In those
cases, return a short response and leave query_text/category empty. Never invent a
generic new request merely because the conversation has an active product category.

Return only JSON with this exact shape:
{{
  "should_recommend": true,
  "query_text": "short product search query with key active preferences",
  "category": "broad product noun only, e.g. shoes, controller, humidifier, t-shirt",
  "hard_constraints": {{"budget_min": null, "budget_max": null}},
  "soft_preferences": {{"use_case": "", "occasion": "", "recipient": ""}},
  "preferred": {{"brands": [], "features": []}},
  "exclusions": {{"brands": [], "products": [], "product_types": [], "features": []}},
  "query_variants": [],
  "clear_memory_fields": [],
  "active_preferences": [],
  "rejected_preferences": [],
  "reason": "brief reason",
  "response": ""
}}

Rules:
- Keep unchanged requirements active after a revision.
- If a user says a brand no longer matters, do not exclude that brand.
- If a user explicitly says do not recommend/show a brand, product type, item,
  or feature, add it to exclusions.
- Use budget_max only for explicit upper-price requests.
- Do not fabricate product IDs.
- Put only the broad product type in category. Put color, style, brand, recipient,
  use case, compatibility, and subjective requirements in query_text/features.
- query_variants should include exactly 3 useful searches: full active request,
  broad category-only, and category plus the most important feature or constraint.
- Use clear_memory_fields only when the user explicitly says a previous
  preference no longer matters. Valid values are: preferred.brands,
  preferred.features, exclusions.brands, exclusions.products,
  exclusions.product_types, exclusions.features, hard_constraints.budget_max,
  soft_preferences.use_case, soft_preferences.recipient.

Current structured memory from this same single agent:
{json.dumps(memory, ensure_ascii=False, indent=2)}

Conversation history:
{json.dumps(compact_history, ensure_ascii=False, indent=2)}
"""


def _normalize_search_request(decision: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    memory = state.get("single_agent_memory") or {}
    query = str(decision.get("query_text") or state.get("latest_user_message") or "").strip()
    prior_category = _clean_text(memory.get("category"))
    category = _normalize_category(str(decision.get("category") or prior_category).strip(), query)
    hard = _as_dict(decision.get("hard_constraints"))
    soft = _as_dict(decision.get("soft_preferences"))
    preferred = _as_dict(decision.get("preferred"))
    exclusions = _as_dict(decision.get("exclusions"))
    memory_hard = _as_dict(memory.get("hard_constraints"))
    memory_soft = _as_dict(memory.get("soft_preferences"))
    memory_preferred = _as_dict(memory.get("preferred"))
    memory_exclusions = _as_dict(memory.get("exclusions"))
    clear_fields = set(_clean_list(decision.get("clear_memory_fields")))

    hard = {
        "budget_min": _number_or_none(hard.get("budget_min"))
        if "hard_constraints.budget_min" in clear_fields or hard.get("budget_min") is not None
        else _number_or_none(memory_hard.get("budget_min")),
        "budget_max": _number_or_none(hard.get("budget_max"))
        if "hard_constraints.budget_max" in clear_fields or hard.get("budget_max") is not None
        else _number_or_none(memory_hard.get("budget_max")),
    }
    soft = {
        "use_case": _merged_text(memory_soft.get("use_case"), soft.get("use_case"), "soft_preferences.use_case", clear_fields),
        "occasion": _merged_text(memory_soft.get("occasion"), soft.get("occasion"), "soft_preferences.occasion", clear_fields),
        "recipient": _merged_text(memory_soft.get("recipient"), soft.get("recipient"), "soft_preferences.recipient", clear_fields),
    }
    preferred = {
        "brands": _merged_list(memory_preferred.get("brands"), preferred.get("brands"), "preferred.brands", clear_fields),
        "features": _merged_list(memory_preferred.get("features"), preferred.get("features"), "preferred.features", clear_fields),
    }
    exclusions = {
        "brands": _merged_list(memory_exclusions.get("brands"), exclusions.get("brands"), "exclusions.brands", clear_fields),
        "products": _merged_list(memory_exclusions.get("products"), exclusions.get("products"), "exclusions.products", clear_fields),
        "product_types": _merged_list(memory_exclusions.get("product_types"), exclusions.get("product_types"), "exclusions.product_types", clear_fields),
        "features": _merged_list(memory_exclusions.get("features"), exclusions.get("features"), "exclusions.features", clear_fields),
    }
    return {
        "query_text": query,
        "category": category,
        "hard_constraints": hard,
        "soft_preferences": soft,
        "preferred": preferred,
        "exclusions": exclusions,
        "control_flags": [],
        "preference_flags": list(decision.get("preference_flags") or []),
        "safety_flags": [],
        "unclassified_flags": [],
    }


def _retrieval_plan(
    request: dict[str, Any],
    *,
    decision: dict[str, Any],
    state: dict[str, Any],
    search_depth: int,
    max_query_variants: int,
) -> dict[str, Any]:
    query = str(request.get("query_text") or "").strip()
    category = str(request.get("category") or "").strip()
    preferred = _as_dict(request.get("preferred"))
    soft = _as_dict(request.get("soft_preferences"))
    hard = _as_dict(request.get("hard_constraints"))
    variants = []
    variants.extend(_clean_list(decision.get("query_variants")))
    variants.append(query)
    if category and category.lower() not in query.lower():
        variants.append(f"{category} {query}".strip())
    if category:
        variants.append(category)
    for feature in _clean_list(preferred.get("features"))[:4]:
        if category:
            variants.append(f"{category} {feature}")
        variants.append(feature)
    for brand in _clean_list(preferred.get("brands"))[:2]:
        if category:
            variants.append(f"{brand} {category}")
    for value in (soft.get("use_case"), soft.get("recipient"), soft.get("occasion")):
        cleaned = _clean_text(value)
        if cleaned and category:
            variants.append(f"{category} {cleaned}")
    if hard.get("budget_max") is not None and category:
        variants.append(f"budget {category}")
    variants.extend(state.get("single_agent_query_memory") or [])
    variants = _broaden_common_category_variants(variants, category)
    return {
        "strategy": "single_agent_wide_one_pass",
        "query_variants": list(dict.fromkeys(v for v in variants if v))[
            :max(1, int(max_query_variants))
        ],
        "search_depth": search_depth,
        "top_k": 10,
        "reason": "Single-agent baseline uses one resource-capped retrieval request per turn.",
    }


def _intent_from_decision(decision: dict[str, Any]) -> dict[str, Any]:
    if decision.get("should_recommend", True):
        return {
            "label": "recommend",
            "action_label": "new_request_or_refinement",
            "source": "single_agent_controller",
        }
    return {
        "label": "inquire",
        "action_label": "inquiry",
        "source": "single_agent_controller",
    }


def _preference_events(decision: dict[str, Any], turn_id: int) -> list[dict[str, Any]]:
    events = []
    for value in _clean_list(decision.get("active_preferences")):
        events.append({
            "type": "single_agent_active_preference",
            "slot": "preference",
            "value": value,
            "event_scope": "turn",
            "source_turn": turn_id,
        })
    for value in _clean_list(decision.get("rejected_preferences")):
        events.append({
            "type": "single_agent_rejected_preference",
            "slot": "preference",
            "value": value,
            "event_scope": "turn",
            "source_turn": turn_id,
        })
    return events


def _response_from_recommendations(recommendations: list[dict[str, Any]], decision: dict[str, Any]) -> str:
    if not recommendations:
        return "I could not find a strong matching product from the current catalogue."
    lines = ["Here are the best matches I found:"]
    for idx, product in enumerate(recommendations, start=1):
        title = str(product.get("title") or product.get("name") or product.get("product_id") or "Product")
        pid = str(product.get("product_id") or "")
        brand = str(product.get("brand") or "").strip()
        prefix = f"{idx}. {title}"
        if brand:
            prefix += f" ({brand})"
        if pid:
            prefix += f" [{pid}]"
        lines.append(prefix)
    reason = str(decision.get("reason") or "").strip()
    if reason:
        lines.append(f"Reason: {reason}")
    return "\n".join(lines)


def _initial_state() -> dict[str, Any]:
    return {
        "turn_id": 0,
        "history": [],
        "planner_trace": [],
        "tool_observations": [],
        "reflection_notes": [],
        "critic_notes": [],
        "raw_candidates": [],
        "verified_candidates": [],
        "last_recommendations": [],
        "single_agent_memory": {
            "category": "",
            "hard_constraints": {"budget_min": None, "budget_max": None},
            "soft_preferences": {"use_case": "", "occasion": "", "recipient": ""},
            "preferred": {"brands": [], "features": []},
            "exclusions": {"brands": [], "products": [], "product_types": [], "features": []},
            "active_preferences": [],
            "rejected_preferences": [],
        },
    }


def _append_trace(state: dict[str, Any], step: dict[str, Any]) -> None:
    state.setdefault("planner_trace", []).append(step)


def _apply_patch_to_state(state: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in (patch or {}).items():
        if key == "tool_observation":
            continue
        state[key] = value
        if key == "final_search_request":
            state["last_final_search_request"] = value
        elif key == "retrieval_plan":
            state["last_retrieval_plan"] = value
        elif key == "raw_candidates":
            state["last_raw_candidates"] = value
        elif key == "verified_candidates":
            state["last_verified_candidates"] = value


def _rule_decision(latest_user_message: str) -> dict[str, Any]:
    text = latest_user_message.strip()
    category = _guess_category(text)
    return {
        "should_recommend": True,
        "query_text": text,
        "category": category,
        "hard_constraints": {"budget_min": None, "budget_max": _extract_budget_max(text)},
        "soft_preferences": {"use_case": "", "occasion": "", "recipient": ""},
        "preferred": {"brands": _extract_preferred_brands(text), "features": []},
        "exclusions": {
            "brands": _extract_excluded_brands(text),
            "products": [],
            "product_types": [],
            "features": [],
        },
        "active_preferences": [text],
        "rejected_preferences": [],
        "reason": "Rule controller smoke-test request.",
        "response": "",
    }


def _guess_category(text: str) -> str:
    lowered = text.lower()
    patterns = [
        "game controller",
        "controller",
        "laptop",
        "headphones",
        "headphone",
        "humidifier",
        "camera",
        "chair",
        "shoes",
        "shoe",
        "socks",
        "sock",
        "t-shirt",
        "shirt",
        "jacket",
        "piano",
        "tablet",
        "iphone",
        "phone",
        "toy gun",
        "toy",
        "alarm clock",
        "clock",
        "water glass",
        "glass",
        "cup",
        "bike",
        "bicycle",
        "phone case",
        "mouse pad",
        "book",
        "mirror",
        "blanket",
    ]
    for pattern in patterns:
        if pattern in lowered:
            return pattern
    return ""


def _normalize_category(category: str, query: str) -> str:
    text = " ".join(str(category or "").lower().strip(" .?!,;:").split())
    query_text = " ".join(str(query or "").lower().strip(" .?!,;:").split())
    combined = f"{text} {query_text}".strip()
    if not combined:
        return ""

    noun_map = [
        ("phone case", ("phone case", "protective case", "phone shell")),
        ("game controller", ("game controller", "controller", "pc controller")),
        ("alarm clock", ("alarm clock",)),
        ("water glass", ("water glass", "glass cup", "drinking glass")),
        ("mouse pad", ("mouse pad", "mousepad")),
        ("t-shirt", ("t-shirt", "tee shirt", "tee", "shirt")),
        ("headphones", ("headphones", "headphone", "headset", "earbuds")),
        ("shoes", ("shoes", "shoe", "sneakers", "cleats")),
        ("socks", ("socks", "sock")),
        ("laptop", ("laptop", "notebook computer")),
        ("tablet", ("tablet",)),
        ("phone", ("iphone", "smartphone", "phone")),
        ("camera", ("camera",)),
        ("chair", ("chair",)),
        ("bike", ("bike", "bicycle")),
        ("piano", ("piano",)),
        ("humidifier", ("humidifier",)),
        ("jacket", ("jacket", "coat")),
        ("toy", ("toy",)),
        ("book", ("book", "literature")),
        ("mirror", ("mirror",)),
        ("blanket", ("blanket",)),
        ("cup", ("cup", "mug")),
    ]
    for normalized, aliases in noun_map:
        if any(re.search(rf"\b{re.escape(alias)}s?\b", combined) for alias in aliases):
            return normalized

    stop = {
        "good", "best", "cheap", "cheaper", "black", "white", "red", "blue",
        "new", "nice", "fashion", "traditional", "acoustic", "discount",
        "simple", "cute", "fancy", "casual", "dry", "indoor", "winter",
        "outside", "inside", "with", "for", "under", "more", "less",
    }
    tokens = [token for token in re.findall(r"[a-z0-9-]+", text) if token not in stop]
    if not tokens:
        return ""
    return " ".join(tokens[-2:])


def _broaden_common_category_variants(variants: list[str], category: str) -> list[str]:
    result = list(variants)
    expansions = {
        "shoes": ["sport shoes", "athletic shoes", "sneakers"],
        "game controller": ["controller", "wireless controller", "pc controller"],
        "headphones": ["headphones", "headset", "earbuds"],
        "t-shirt": ["shirt", "tee shirt", "casual shirt"],
        "socks": ["socks", "athletic socks"],
        "humidifier": ["humidifier", "cool mist humidifier"],
        "alarm clock": ["clock", "digital alarm clock"],
        "water glass": ["drinking glass", "glass cup"],
        "toy": ["toy", "kids toy"],
        "piano": ["piano", "keyboard piano"],
        "jacket": ["jacket", "casual jacket"],
        "phone": ["iphone", "smartphone", "mobile phone"],
    }
    for value in expansions.get(str(category or "").lower(), []):
        if value not in result:
            result.append(value)
    return result


def _update_single_agent_memory(
    memory: dict[str, Any],
    request: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    updated = {
        "category": _clean_text(request.get("category")),
        "hard_constraints": _as_dict(request.get("hard_constraints")),
        "soft_preferences": _as_dict(request.get("soft_preferences")),
        "preferred": _as_dict(request.get("preferred")),
        "exclusions": _as_dict(request.get("exclusions")),
        "active_preferences": _merged_list(
            memory.get("active_preferences"),
            decision.get("active_preferences"),
            "active_preferences",
            set(),
        )[-12:],
        "rejected_preferences": _merged_list(
            memory.get("rejected_preferences"),
            decision.get("rejected_preferences"),
            "rejected_preferences",
            set(),
        )[-12:],
    }
    return updated


def _project_memory_to_prism_state(state: dict[str, Any]) -> None:
    memory = state.get("single_agent_memory") or {}
    hard = _as_dict(memory.get("hard_constraints"))
    soft = _as_dict(memory.get("soft_preferences"))
    preferred = _as_dict(memory.get("preferred"))
    exclusions = _as_dict(memory.get("exclusions"))
    preferences = {}
    if memory.get("category"):
        preferences["product_type"] = {"value": memory["category"], "status": "single_agent"}
    if hard.get("budget_max") is not None:
        preferences["budget_max"] = {"value": hard.get("budget_max"), "status": "single_agent"}
    for slot in ("use_case", "occasion", "recipient"):
        if soft.get(slot):
            preferences[slot] = {"value": soft.get(slot), "status": "single_agent"}
    state["preferences"] = preferences
    state["preferred_brands"] = list(preferred.get("brands") or [])
    state["preferred_features"] = list(preferred.get("features") or [])
    state["rejected_brands"] = list(exclusions.get("brands") or [])
    state["rejected_products"] = list(exclusions.get("products") or [])
    state["rejected_product_types"] = list(exclusions.get("product_types") or [])
    state["rejected_features"] = list(exclusions.get("features") or [])


def _merged_text(previous: Any, current: Any, clear_key: str, clear_fields: set[str]) -> str:
    if clear_key in clear_fields:
        return ""
    cleaned = _clean_text(current)
    return cleaned if cleaned else _clean_text(previous)


def _merged_list(previous: Any, current: Any, clear_key: str, clear_fields: set[str]) -> list[str]:
    if clear_key in clear_fields:
        base: list[str] = []
    else:
        base = _clean_list(previous)
    for item in _clean_list(current):
        if item not in base:
            base.append(item)
    return base


def _extract_budget_max(text: str) -> float | None:
    lowered = text.lower()
    match = re.search(r"(?:under|below|less than|at most|max(?:imum)?)\s*\$?\s*(\d+(?:\.\d+)?)", lowered)
    if match:
        return float(match.group(1))
    return None


def _extract_preferred_brands(text: str) -> list[str]:
    brands = []
    lowered = text.lower()
    for brand in ("hp", "dell", "sony", "apple", "samsung", "playstation", "nike", "adidas"):
        if re.search(rf"\b(prefer|like|want)\b[^.?!]*\b{re.escape(brand)}\b", lowered):
            brands.append(brand)
    return brands


def _extract_excluded_brands(text: str) -> list[str]:
    brands = []
    lowered = text.lower()
    for brand in ("hp", "dell", "sony", "apple", "samsung", "playstation", "nike", "adidas"):
        if re.search(rf"\b(no|not|don't|do not|exclude|without)\b[^.?!]*\b{re.escape(brand)}\b", lowered):
            brands.append(brand)
    return brands


def _validate_decision(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("single-agent controller did not return a JSON object")
    # A missing decision must not cause an unsolicited recommendation. The normal
    # prompt always supplies this field for genuine recommendation requests.
    value.setdefault("should_recommend", False)
    value.setdefault("query_text", "")
    value.setdefault("category", "")
    value.setdefault("hard_constraints", {})
    value.setdefault("soft_preferences", {})
    value.setdefault("preferred", {})
    value.setdefault("exclusions", {})
    value.setdefault("query_variants", [])
    value.setdefault("clear_memory_fields", [])
    value.setdefault("active_preferences", [])
    value.setdefault("rejected_preferences", [])
    value.setdefault("reason", "")
    value.setdefault("response", "")
    return value


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _usage(raw: dict[str, Any]) -> dict[str, int]:
    return {
        "prompt_tokens": int(raw.get("prompt_tokens") or 0),
        "completion_tokens": int(raw.get("completion_tokens") or 0),
        "total_tokens": int(raw.get("total_tokens") or 0),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        cleaned = _clean_text(item)
        if cleaned:
            result.append(cleaned)
    return result


def _number_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
