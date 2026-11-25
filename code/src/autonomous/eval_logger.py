"""JSONL evaluation logging for autonomous ConServe sessions."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from src.autonomous.flags import combined_flags


class EvalLogger:
    """Writes one compact research record per user turn."""

    def __init__(self, root: str | Path | None = None, enabled: bool | None = None):
        if enabled is None:
            enabled = os.getenv("AUTONOMOUS_EVAL_LOG", "1").lower() not in {"0", "false", "no"}
        self.enabled = enabled
        self.conversation_id = uuid.uuid4().hex[:12]
        root_path = Path(root or os.getenv("AUTONOMOUS_EVAL_LOG_DIR", "outputs/autonomous_logs"))
        if not root_path.is_absolute():
            root_path = Path(__file__).resolve().parents[2] / root_path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = root_path / f"conversation_{timestamp}_{self.conversation_id}.jsonl"
        self.effective_config = _effective_config()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_turn(self, *, user_message: str, response: str, state: dict, started_at: float, ended_at: float) -> None:
        if not self.enabled:
            return
        turn_id = state.get("turn_id")
        trace = state.get("planner_trace") or []
        turn_trace = [step for step in trace if step.get("turn_id") == turn_id]
        record = {
            "conversation_id": self.conversation_id,
            "turn_id": turn_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "latency_ms": int((ended_at - started_at) * 1000),
            "effective_config": self.effective_config,
            "user_message": user_message,
            "response": response,
            "planner_backends": list(dict.fromkeys(step.get("planner_backend") for step in turn_trace if step.get("planner_backend"))),
            "planner_trace": _jsonish(turn_trace),
            "turn_analysis": _jsonish(_latest_turn_analysis(turn_trace)),
            "components_called": [step.get("target") for step in turn_trace if step.get("target")],
            "tool_observations": _jsonish(state.get("tool_observations", [])),
            "preferences": _jsonish(state.get("preferences", {})),
            "preference_events": _jsonish(state.get("preference_events", [])),
            "response_notes": _jsonish(state.get("response_notes", [])),
            "control_flags": list(state.get("control_flags", [])),
            "preference_flags": list(state.get("preference_flags", [])),
            "safety_flags": list(state.get("safety_flags", [])),
            "unclassified_flags": list(state.get("unclassified_flags", [])),
            "security_flags": list(state.get("security_flags", [])),
            "combined_flags": combined_flags(state),
            "final_search_request": _jsonish(state.get("final_search_request")),
            "last_final_search_request": _jsonish(state.get("last_final_search_request")),
            "retrieval_plan": _jsonish(state.get("retrieval_plan")),
            "last_retrieval_plan": _jsonish(state.get("last_retrieval_plan")),
            "candidate_counts": {
                "raw": len(state.get("raw_candidates") or []),
                "verified": len(state.get("verified_candidates") or []),
                "recommended": len(state.get("last_recommendations") or []),
            },
            "critic_decision": state.get("critic_decision"),
            "critic_next_action": state.get("critic_next_action"),
            "critic_reasons": list(state.get("critic_reasons", [])),
            "critic_notes": _jsonish(state.get("critic_notes", [])),
            "retrieval_readiness": state.get("retrieval_readiness"),
            "recommendations": _summarize_recommendations(state.get("last_recommendations") or []),
            "rejections": {
                "brands": list(state.get("rejected_brands", [])),
                "products": list(state.get("rejected_products", [])),
                "product_types": list(state.get("rejected_product_types", [])),
                "features": list(state.get("rejected_features", [])),
            },
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _latest_turn_analysis(turn_trace: list[dict]) -> dict:
    for step in turn_trace:
        analysis = step.get("turn_analysis")
        if analysis:
            return analysis
    return {}


def _summarize_recommendations(products: list[dict]) -> list[dict]:
    summary = []
    for product in products[:5]:
        summary.append({
            "product_id": product.get("product_id"),
            "title": product.get("title"),
            "brand": product.get("brand"),
            "price": product.get("price"),
            "rating": product.get("rating"),
            "decision_score": product.get("decision_score"),
            "reasons": product.get("decision_reasons"),
        })
    return summary


def _effective_config() -> dict:
    keys = [
        "AUTONOMOUS_EXECUTION_SCOPE",
        "AUTONOMOUS_PLANNER_MODE",
        "AUTONOMOUS_PLANNER_PROVIDER",
        "AUTONOMOUS_PLANNER_MODEL",
        "AUTONOMOUS_PLANNER_TIMEOUT",
        "AUTONOMOUS_RETRIEVAL_BACKEND",
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
    ]
    return {key: os.getenv(key, "") for key in keys}


def _jsonish(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _jsonish(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonish(item) for item in value]
    return value
