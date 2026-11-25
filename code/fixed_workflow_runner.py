"""PRISM-native fixed-workflow runner.

This is a controlled baseline, not a second PRISM implementation.  It reuses
the frozen Blackboard and registered PRISM components, but removes the
executive planner and executes one declared sequence.  No adaptive retry or
planner-selected component invocation is allowed.
"""

from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from typing import Any

from src.autonomous.blackboard import Blackboard
from src.autonomous.eval_logger import EvalLogger
from src.autonomous.registry import build_registry


class FixedWorkflowRunner:
    """Run the same PRISM components in a fixed, auditable order."""

    mode = "fixed_workflow"

    def __init__(self, *, eval_logger: EvalLogger | None = None):
        self.eval_logger = eval_logger or EvalLogger()
        self.blackboard = Blackboard(conversation_id=self.eval_logger.conversation_id)
        self.registry = build_registry()

    def run_turn(self, user_message: str) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        self.blackboard.observe_user_turn(user_message)

        # Every turn begins with the same state interpretation and reflection.
        self._run_component("intent_agent")
        self._run_component("preference_reflection_agent")
        state = self.blackboard.snapshot()

        # Acknowledgements/closing turns do not create a new search in either
        # system.  This is a declared route, not planner-selected adaptation.
        label = str((state.get("intent") or {}).get("label") or "").lower()
        if label in {"accept", "chitchat"}:
            self._run_component("response_tool")
            return self._finish(user_message, started)

        # Fixed recommendation path. The first critic call is the same
        # pre-retrieval readiness gate used by PRISM's planner path.
        self._run_component("search_research_agent")
        self._fix_retrieval_budget()
        self._run_component("critic_verification_agent")
        if self.blackboard.snapshot().get("critic_decision") != "ACCEPT":
            self._run_component("response_tool")
            return self._finish(user_message, started)

        # Only an accepted search request may retrieve. After retrieval, run
        # the post-retrieval critic; decision selection is never run on reject.
        self._run_component("conserve_search_tool")
        self._run_component("critic_verification_agent")
        self._disable_adaptive_recovery()
        if self.blackboard.snapshot().get("critic_decision") != "ACCEPT":
            self._run_component("response_tool")
            return self._finish(user_message, started)
        self._run_component("recommendation_decision_agent")
        self._run_component("response_tool")
        return self._finish(user_message, started)

    def _run_component(self, name: str) -> None:
        component = self.registry[name]
        started = time.perf_counter()
        result = component.run(self.blackboard.snapshot())
        duration_ms = int((time.perf_counter() - started) * 1000)
        result.duration_ms = duration_ms
        observation = result.patch.get("tool_observation") if isinstance(result.patch, dict) else None
        if isinstance(observation, dict):
            observation["duration_ms"] = duration_ms
        self.blackboard.apply_patch(result.patch)
        self.blackboard.log_planner_decision({
            "planner_mode": self.mode,
            "planner_backend": self.mode,
            "action": "FIXED_COMPONENT",
            "target": name,
            "reason": "Declared fixed-workflow sequence; no planner selection.",
            "planner_duration_ms": 0,
            "component_duration_ms": duration_ms,
        })
        if result.message:
            self.blackboard.apply_patch({"response": result.message, "done": True})

    def _fix_retrieval_budget(self) -> None:
        """Use the same declared retrieval cap as the PRISM baseline runs."""
        state = self.blackboard.snapshot()
        plan = dict(state.get("retrieval_plan") or {})
        variants = list(plan.get("query_variants") or [])[:3]
        plan.update({"search_depth": 50, "top_k": 10, "query_variants": variants})
        self.blackboard.apply_patch({
            "retrieval_plan": plan,
            "last_retrieval_plan": plan,
        })
        self.blackboard.log_planner_decision({
            "planner_mode": self.mode,
            "planner_backend": self.mode,
            "action": "FIXED_BUDGET",
            "target": "retrieval_budget",
            "reason": "Fixed evaluation cap: search_depth=50, top_k=10, max_query_variants=3.",
            "planner_duration_ms": 0,
            "component_duration_ms": 0,
        })

    def _disable_adaptive_recovery(self) -> None:
        """Make critic rejection terminal; fixed workflow never searches again."""
        state = self.blackboard.snapshot()
        if state.get("critic_next_action") == "search_again" or state.get("critic_decision") == "SEARCH_AGAIN":
            reasons = list(state.get("critic_reasons") or [])
            reasons.append("Fixed workflow: adaptive search-again recovery disabled.")
            self.blackboard.apply_patch({
                "critic_decision": "ASK_USER",
                "critic_next_action": "clarify",
                "critic_reasons": reasons,
                "retrieval_readiness": "needs_clarification",
                "control_flags": list(state.get("control_flags") or []) + ["fixed_recovery_disabled"],
            })
            self.blackboard.log_planner_decision({
                "planner_mode": self.mode,
                "planner_backend": self.mode,
                "action": "FIXED_TERMINAL_RECOVERY",
                "target": "response_tool",
                "reason": "Critic rejected candidates; fixed workflow returns clarification without retry.",
                "planner_duration_ms": 0,
                "component_duration_ms": 0,
            })

    def _finish(self, user_message: str, started: float) -> tuple[str, dict[str, Any]]:
        state = self.blackboard.snapshot()
        response = str(state.get("response") or "")
        self.eval_logger.write_turn(
            user_message=user_message,
            response=response,
            state=state,
            started_at=started,
            ended_at=time.perf_counter(),
        )
        return response, state


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value

