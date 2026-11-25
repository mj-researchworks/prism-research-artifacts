"""Adaptive reasoning budget helpers for bounded autonomous agents."""

from __future__ import annotations

import os
from typing import Any

from src.autonomous.flags import combined_flags


DEPTHS = {"FAST", "REFLECT", "VERIFY", "DEEP"}


def adaptive_budget_enabled() -> bool:
    return os.getenv("AUTONOMOUS_REASONING_BUDGET", "off").lower() in {"adaptive", "on", "1", "true", "yes"}


def latest_turn_analysis(state: dict[str, Any]) -> dict[str, Any]:
    turn_id = state.get("turn_id")
    for step in reversed(state.get("planner_trace") or []):
        if step.get("turn_id") != turn_id:
            continue
        analysis = step.get("turn_analysis")
        if isinstance(analysis, dict):
            return analysis
    return {}


def reasoning_depth(state: dict[str, Any]) -> str:
    analysis = latest_turn_analysis(state)
    raw = str(analysis.get("reasoning_depth") or analysis.get("reasoning_budget") or "").strip().upper()
    if raw in DEPTHS:
        return raw
    return infer_reasoning_depth(state, analysis)


def infer_reasoning_depth(state: dict[str, Any], analysis: dict[str, Any] | None = None) -> str:
    analysis = analysis or latest_turn_analysis(state)
    turn_type = str(analysis.get("turn_type") or "").lower()
    preference_change = str(analysis.get("preference_change") or "").lower()
    confidence = str(analysis.get("planner_confidence") or "").lower()
    text = f" {str(state.get('latest_user_message') or '').lower()} "

    if confidence == "low":
        return "DEEP"
    if combined_flags(state):
        return "DEEP"
    if turn_type in {"rejection", "acceptance"} or preference_change in {"rejection", "acceptance", "revision", "uncertain"}:
        return "DEEP"
    if any(phrase in text for phrase in (" actually ", " instead ", " rather ", " not ", " no ", " avoid ", " without ")):
        return "DEEP"
    if turn_type in {"action_followup", "ask_detail"}:
        return "VERIFY"
    if turn_type in {"new_request", "refinement"} or preference_change == "new_preference":
        return "REFLECT"
    return "FAST"


def reflection_llm_needed(state: dict[str, Any], preferences: dict, events: list, flags: list, rule_reason: str | None) -> bool:
    if not adaptive_budget_enabled():
        return True
    if rule_reason:
        return True
    depth = reasoning_depth(state)
    if depth in {"REFLECT", "DEEP"}:
        return True
    if flags:
        return True
    if any(event.get("type") in _COMPLEX_EVENT_TYPES for event in events):
        return True
    text = f" {str(state.get('latest_user_message') or '').lower()} "
    if any(phrase in text for phrase in (" actually ", " instead ", " rather ", " not ", " no ", " avoid ", " without ")):
        return True
    return False


def critic_llm_needed(state: dict[str, Any], hard_patch: dict) -> bool:
    if not adaptive_budget_enabled():
        return True
    depth = reasoning_depth(state)
    hard_decision = hard_patch.get("critic_decision")
    hard_next_action = hard_patch.get("critic_next_action")
    if depth in {"VERIFY", "DEEP"}:
        return True
    if hard_decision in {"SEARCH_AGAIN", "ASK_USER", "RERANK", "UPDATE_STATE"}:
        return True
    if hard_next_action in {"search_again", "rerank", "update_state", "ask_user"}:
        return True
    if combined_flags(state) or state.get("preference_events"):
        return True
    request = state.get("final_search_request")
    if _request_has_constraints(request):
        return True
    if state.get("raw_candidates") and len(state.get("verified_candidates") or []) < 3:
        return True
    return False


def budget_note(state: dict[str, Any], component: str, used_llm: bool) -> dict[str, Any]:
    return {
        "component": component,
        "reasoning_budget": reasoning_depth(state),
        "adaptive_budget": adaptive_budget_enabled(),
        "llm_used": used_llm,
    }


def _request_has_constraints(request) -> bool:
    if not request:
        return False
    hard_constraints = _request_value(request, "hard_constraints") or {}
    exclusions = _request_value(request, "exclusions") or {}
    preferred = _request_value(request, "preferred") or {}
    return bool(
        hard_constraints
        or exclusions.get("brands")
        or exclusions.get("products")
        or exclusions.get("features")
        or preferred.get("brands")
        or preferred.get("features")
    )


def _request_value(request, key: str, default=None):
    if isinstance(request, dict):
        return request.get(key, default)
    return getattr(request, key, default)


_COMPLEX_EVENT_TYPES = {
    "brand_rejection",
    "product_rejection",
    "feature_rejection",
    "product_acceptance",
    "preference_revision",
    "preference_clear",
}
