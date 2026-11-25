"""Shared shopping memory for the autonomous control loop."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .flags import classify_flags
from .schemas import PreferenceItem, SearchRequest

HISTORY_WINDOW = 12
EVENT_WINDOW = 40
NOTE_WINDOW = 40
TRACE_WINDOW = 120


class Blackboard:
    """Central state store read by the planner and written via patches."""

    def __init__(self, conversation_id: str = "interactive"):
        self.state: dict[str, Any] = {
            "conversation_id": conversation_id,
            "turn_id": 0,
            "history": [],
            "latest_user_message": "",
            "current_goal": None,
            "intent": {
                "label": None,
                "confidence": None,
                "previous_label": None,
                "shift_detected": False,
            },
            "preferences": {},
            "preference_events": [],
            "memory_event_ledger": [],
            "recent_memory_events": [],
            "reflection_notes": [],
            "response_notes": [],
            "last_recommendations": [],
            "recommendation_decisions": [],
            "previous_recommendations": [],
            "rejected_products": [],
            "rejected_product_types": [],
            "rejected_brands": [],
            "rejected_features": [],
            "preferred_brands": [],
            "preferred_features": [],
            "accepted_products": [],
            "candidate_ledger": {},
            "last_candidate_ledger": {},
            "control_flags": [],
            "preference_flags": [],
            "safety_flags": [],
            "unclassified_flags": [],
            "security_flags": [],
            "planner_trace": [],
            "tool_observations": [],
            "final_search_request": None,
            "last_final_search_request": None,
            "retrieval_plan": None,
            "last_retrieval_plan": None,
            "retrieval_readiness": None,
            "search_attempts": 0,
            "raw_candidates": [],
            "verified_candidates": [],
            "last_raw_candidates": [],
            "last_verified_candidates": [],
            "critic_decision": None,
            "critic_next_action": None,
            "critic_reasons": [],
            "critic_notes": [],
            "response": None,
            "done": False,
        }

    def observe_user_turn(self, message: str) -> None:
        previous_intent = self.state["intent"].get("label")
        previous_recommendations = list(
            self.state.get("last_recommendations")
            or self.state.get("previous_recommendations")
            or []
        )
        self.state["turn_id"] += 1
        self.state["latest_user_message"] = message
        self.state["history"].append(
            {"turn_id": self.state["turn_id"], "role": "user", "content": message}
        )
        self._cap_long_lived_lists()

        # Preserve long-lived shopping memory, but force the planner to rebuild
        # all per-turn evidence before retrieval.
        self.state["intent"] = {
            "label": None,
            "confidence": None,
            "previous_label": previous_intent,
            "shift_detected": False,
        }
        self.state["security_flags"] = []
        self.state["control_flags"] = []
        self.state["preference_flags"] = []
        self.state["safety_flags"] = []
        self.state["unclassified_flags"] = []
        self.state["tool_observations"] = []
        self.state["final_search_request"] = None
        self.state["retrieval_plan"] = None
        self.state["retrieval_readiness"] = None
        self.state["search_attempts"] = 0
        self.state["raw_candidates"] = []
        self.state["verified_candidates"] = []
        self.state["previous_recommendations"] = previous_recommendations
        self.state["last_recommendations"] = []
        self.state["recommendation_decisions"] = []
        self.state["candidate_ledger"] = {}
        self.state["critic_decision"] = None
        self.state["critic_next_action"] = None
        self.state["critic_reasons"] = []
        self.state["critic_notes"] = []
        self.state["response"] = None
        self.state["response_notes"] = []
        self.state["done"] = False

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self.state)

    def apply_patch(self, patch: dict[str, Any]) -> None:
        if not patch:
            return

        if "intent" in patch:
            current = self.state["intent"].get("label")
            incoming = dict(patch["intent"])
            incoming.setdefault("previous_label", current)
            incoming.setdefault(
                "shift_detected",
                bool(current and incoming.get("label") and incoming.get("label") != current),
            )
            self.state["intent"].update(incoming)

        if "preferences" in patch:
            for slot, raw_item in patch["preferences"].items():
                existing = self.state["preferences"].get(slot)
                item = (
                    raw_item
                    if isinstance(raw_item, PreferenceItem)
                    else PreferenceItem(slot=slot, **raw_item)
                )
                if existing and item.previous_value is None:
                    item.previous_value = existing.value
                self.state["preferences"][slot] = item

        for key in (
            "current_goal",
            "last_recommendations",
            "recommendation_decisions",
            "rejected_products",
            "rejected_product_types",
            "rejected_brands",
            "rejected_features",
            "preferred_brands",
            "preferred_features",
            "accepted_products",
            "candidate_ledger",
            "raw_candidates",
            "verified_candidates",
            "retrieval_readiness",
            "retrieval_plan",
            "critic_decision",
            "critic_next_action",
            "critic_reasons",
            "search_attempts",
            "response",
            "done",
        ):
            if key in patch:
                self.state[key] = patch[key]

        if patch.get("candidate_ledger"):
            self.state["last_candidate_ledger"] = deepcopy(patch["candidate_ledger"])
        if patch.get("raw_candidates"):
            self.state["last_raw_candidates"] = deepcopy(patch["raw_candidates"])
        if patch.get("verified_candidates"):
            self.state["last_verified_candidates"] = deepcopy(patch["verified_candidates"])

        if "reflection_notes" in patch:
            self.state["reflection_notes"].extend(patch["reflection_notes"])
            self.state["reflection_notes"] = self.state["reflection_notes"][-NOTE_WINDOW:]

        if "critic_notes" in patch:
            self.state["critic_notes"].extend(patch["critic_notes"])
            self.state["critic_notes"] = self.state["critic_notes"][-NOTE_WINDOW:]

        if "response_notes" in patch:
            self.state["response_notes"].extend(patch["response_notes"])
            self.state["response_notes"] = self.state["response_notes"][-NOTE_WINDOW:]

        incoming_flags = []
        for key in ("security_flags", "control_flags", "preference_flags", "safety_flags", "unclassified_flags"):
            incoming_flags.extend(patch.get(key) or [])
        if incoming_flags:
            buckets = classify_flags(incoming_flags)
            for key in ("control_flags", "preference_flags", "safety_flags", "unclassified_flags", "security_flags"):
                for flag in buckets[key]:
                    if flag not in self.state[key]:
                        self.state[key].append(flag)

        if "preference_events" in patch:
            events = [
                self._normalize_memory_event(event)
                for event in patch["preference_events"]
            ]
            self.state["preference_events"].extend(events)
            self.state["preference_events"] = self.state["preference_events"][-EVENT_WINDOW:]
            self.state["memory_event_ledger"].extend(events)
            self.state["memory_event_ledger"] = self.state["memory_event_ledger"][-EVENT_WINDOW:]
            self.state["recent_memory_events"] = self._compact_memory_events(
                self.state["memory_event_ledger"][-8:]
            )

        if "tool_observation" in patch:
            self.state["tool_observations"].append(patch["tool_observation"])

        if "final_search_request" in patch:
            request = patch["final_search_request"]
            normalized = request if isinstance(request, SearchRequest) else SearchRequest(**request)
            self.state["final_search_request"] = normalized
            self.state["last_final_search_request"] = normalized

        if "retrieval_plan" in patch:
            self.state["retrieval_plan"] = patch["retrieval_plan"]
            self.state["last_retrieval_plan"] = patch["retrieval_plan"]

    def log_planner_decision(self, decision: dict[str, Any]) -> None:
        self.state["planner_trace"].append(
            {"turn_id": self.state["turn_id"], "step": len(self.state["planner_trace"]) + 1, **decision}
        )
        self.state["planner_trace"] = self.state["planner_trace"][-TRACE_WINDOW:]

    def _cap_long_lived_lists(self) -> None:
        self.state["history"] = self.state["history"][-HISTORY_WINDOW:]
        self.state["memory_event_ledger"] = self.state.get("memory_event_ledger", [])[-EVENT_WINDOW:]
        self.state["recent_memory_events"] = self._compact_memory_events(
            self.state["memory_event_ledger"][-8:]
        )
        self.state["reflection_notes"] = self.state["reflection_notes"][-NOTE_WINDOW:]
        self.state["critic_notes"] = self.state["critic_notes"][-NOTE_WINDOW:]

    def _normalize_memory_event(self, event: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(event or {})
        normalized.setdefault("turn_id", self.state.get("turn_id"))
        normalized.setdefault("evidence_text", self.state.get("latest_user_message", ""))
        normalized.setdefault("source_component", "preference_reflection_agent")
        normalized.setdefault("confidence", None)
        normalized.setdefault("downstream_action", self._current_downstream_action())
        normalized.setdefault("event_scope", self._event_scope(normalized))
        return normalized

    def _current_downstream_action(self) -> str:
        intent = self.state.get("intent") or {}
        action = intent.get("action_label") or intent.get("next_action_hint")
        return str(action or "")

    @staticmethod
    def _event_scope(event: dict[str, Any]) -> str:
        event_type = str(event.get("type") or "").lower()
        slot = str(event.get("slot") or "").lower()
        if "reject" in event_type or "clear" in event_type or "revision" in event_type:
            return "state_change"
        if "accept" in event_type:
            return "acceptance"
        if slot or event.get("new_value") is not None:
            return "preference_update"
        return "memory_note"

    @staticmethod
    def _compact_memory_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact = []
        for event in events:
            compact.append({
                "turn_id": event.get("turn_id"),
                "type": event.get("type"),
                "slot": event.get("slot"),
                "old_value": event.get("old_value"),
                "new_value": event.get("new_value"),
                "event_scope": event.get("event_scope"),
                "source_component": event.get("source_component"),
            })
        return compact
