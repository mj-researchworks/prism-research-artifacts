"""Runner for the first autonomous planner-blackboard loop."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any

from .blackboard import Blackboard
from .eval_logger import EvalLogger
from .flags import combined_flags
from .llm_planner import LLMExecutivePlanner
from .planner import RuleBasedPlanner
from .registry import build_registry


class AutonomousRunner:
    def __init__(
        self,
        max_steps: int = 20,
        planner_mode: str | None = None,
        eval_logger: EvalLogger | None = None,
        execution_scope: str | None = None,
    ):
        self.planner_mode = (planner_mode or os.getenv("AUTONOMOUS_PLANNER_MODE", "rule")).lower()
        self.execution_scope = (execution_scope or os.getenv("AUTONOMOUS_EXECUTION_SCOPE", "full")).lower()
        self.planner = self._build_planner(self.planner_mode)
        self.registry = build_registry()
        self.max_steps = max_steps
        self.eval_logger = eval_logger or EvalLogger()
        self.blackboard = Blackboard(conversation_id=self.eval_logger.conversation_id)

    def _build_planner(self, mode: str):
        if mode in {"llm", "auto"}:
            return LLMExecutivePlanner(fallback=RuleBasedPlanner())
        return RuleBasedPlanner()

    def run_turn(self, user_message: str) -> tuple[str, dict[str, Any]]:
        started_at = time.perf_counter()
        self.blackboard.observe_user_turn(user_message)

        for _ in range(self.max_steps):
            state = self.blackboard.snapshot()
            planner_started_at = time.perf_counter()
            decision = self.planner.decide(state)
            planner_duration_ms = int((time.perf_counter() - planner_started_at) * 1000)
            trace_decision = _serialize(decision)
            trace_decision["planner_mode"] = self.planner_mode
            trace_decision["planner_duration_ms"] = planner_duration_ms
            if hasattr(self.planner, "last_backend"):
                trace_decision["planner_backend"] = self.planner.last_backend
                trace_decision["planner_error"] = self.planner.last_error
                trace_decision["planner_consecutive_errors"] = getattr(self.planner, "consecutive_errors", 0)
            if hasattr(self.planner, "provider"):
                trace_decision["planner_provider"] = self.planner.provider
                trace_decision["planner_model"] = getattr(self.planner, "model", None)
            if hasattr(self.planner, "last_usage") and self.planner.last_usage:
                trace_decision["planner_usage"] = self.planner.last_usage
            if hasattr(self.planner, "last_turn_analysis") and self.planner.last_turn_analysis:
                trace_decision["turn_analysis"] = self.planner.last_turn_analysis
            self.blackboard.log_planner_decision(trace_decision)

            if decision.action in {"CALL_AGENT", "CALL_TOOL"}:
                if (
                    self.execution_scope == "pre_retrieval"
                    and decision.target == "conserve_search_tool"
                    and self._pre_retrieval_ready(self.blackboard.snapshot())
                ):
                    return self._finish_turn(user_message, self._format_pre_retrieval_response(self.blackboard.snapshot()), started_at)
                if not decision.target or decision.target not in self.registry:
                    return self._finish_turn(user_message, "I cannot safely continue because the planner selected an unknown component.", started_at)
                result = self._run_component(decision.target)
                self.blackboard.apply_patch(result.patch)
                if result.message:
                    return self._finish_turn(user_message, result.message, started_at)
                if self.execution_scope == "pre_retrieval" and self._pre_retrieval_ready(self.blackboard.snapshot()):
                    return self._finish_turn(user_message, self._format_pre_retrieval_response(self.blackboard.snapshot()), started_at)
                continue

            if decision.action == "RESPOND":
                result = self._run_component("response_tool")
                self.blackboard.apply_patch(result.patch)
                return self._finish_turn(user_message, result.message or self.blackboard.snapshot().get("response") or "", started_at)

            if decision.action == "STOP":
                return self._finish_turn(user_message, "Stopped before retrieval because the planner decided no safe action remains.", started_at)

        self.blackboard.apply_patch({
            "control_flags": ["step_budget_exhausted"],
            "tool_observation": {
                "component": "runner",
                "summary": "step_budget_exhausted",
                "status": "degraded",
            },
        })
        result = self._run_component("response_tool")
        self.blackboard.apply_patch(result.patch)
        fallback_response = result.message or "I need one more detail before I can continue safely."
        return self._finish_turn(user_message, fallback_response, started_at)

    def _run_component(self, target: str):
        component = self.registry[target]
        component_started_at = time.perf_counter()
        result = component.run(self.blackboard.snapshot())
        duration_ms = int((time.perf_counter() - component_started_at) * 1000)
        result.duration_ms = duration_ms
        observation = result.patch.get("tool_observation") if isinstance(result.patch, dict) else None
        if isinstance(observation, dict):
            observation["duration_ms"] = duration_ms
        return result

    def _pre_retrieval_ready(self, state: dict[str, Any]) -> bool:
        if not state.get("final_search_request"):
            return False
        if state.get("raw_candidates"):
            return False
        blocking_flags = {"underspecified_product_type", "missing_search_request", "ask_user_before_retrieval"}
        if blocking_flags.intersection(set(combined_flags(state))):
            return False
        if state.get("critic_decision") != "ACCEPT":
            return False
        if state.get("retrieval_readiness") != "ready":
            return False
        return True

    def _format_pre_retrieval_response(self, state: dict[str, Any]) -> str:
        request = state.get("final_search_request")
        if not request:
            return "Pre-retrieval planner stopped before a structured search request was ready."
        data = _serialize(request)
        return (
            "Pre-retrieval search request finalized:\n"
            f"  query: {data.get('query_text')}\n"
            f"  category: {data.get('category')}\n"
            f"  hard_constraints: {data.get('hard_constraints')}\n"
            f"  soft_preferences: {data.get('soft_preferences')}\n"
            f"  exclusions: {data.get('exclusions')}\n"
            f"  retrieval_plan: {_serialize(state.get('retrieval_plan'))}"
        )

    def _finish_turn(self, user_message: str, response: str, started_at: float) -> tuple[str, dict[str, Any]]:
        state = self.blackboard.snapshot()
        self.eval_logger.write_turn(
            user_message=user_message,
            response=response,
            state=state,
            started_at=started_at,
            ended_at=time.perf_counter(),
        )
        return response, state


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def demo() -> None:
    import sys

    runner = AutonomousRunner()
    raw = " ".join(sys.argv[1:])
    turns = [part.strip() for part in raw.split("||") if part.strip()]
    if not turns:
        turns = ["I need a gift for my son"]

    state = {}
    for index, user_message in enumerate(turns, start=1):
        response, state = runner.run_turn(user_message)
        print(f"\nUSER {index}: {user_message}")
        print(f"ASSISTANT {index}: {response}")

    print("\nFinal search request:", state.get("final_search_request"))
    print("Planner trace:", state.get("planner_trace"))


if __name__ == "__main__":
    demo()
