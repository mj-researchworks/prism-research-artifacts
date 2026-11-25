"""Interactive CLI for the autonomous ConServe recommender."""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import Any

from .runner import AutonomousRunner


def main() -> None:
    planner_mode = os.getenv("AUTONOMOUS_PLANNER_MODE", "rule")
    execution_scope = os.getenv("AUTONOMOUS_EXECUTION_SCOPE", "full")
    runner = AutonomousRunner(planner_mode=planner_mode, execution_scope=execution_scope)

    print("Autonomous ConServe chat")
    print(f"Planner mode: {planner_mode}")
    print(f"Execution scope: {execution_scope}")
    print("Commands: :trace, :state, :planner, :log, :reset, :quit")
    print()

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nConServe: Goodbye.")
            return

        if not user_message:
            continue

        command = user_message.lower()
        if command in {":quit", ":q", "exit", "quit"}:
            print("ConServe: Goodbye.")
            return
        if command == ":reset":
            runner = AutonomousRunner(planner_mode=planner_mode, execution_scope=execution_scope)
            print("ConServe: Conversation memory reset.")
            continue
        if command == ":trace":
            _print_trace(runner.blackboard.snapshot())
            continue
        if command == ":state":
            _print_state(runner.blackboard.snapshot())
            continue
        if command == ":planner":
            _print_planner(runner)
            continue
        if command == ":log":
            print(f"Eval log: {runner.eval_logger.path}")
            continue

        response, _state = runner.run_turn(user_message)
        print(f"ConServe: {response}")
        print()


def _print_trace(state: dict) -> None:
    trace = state.get("planner_trace") or []
    if not trace:
        print("Trace: no planner steps yet.")
        return
    print("Trace:")
    for step in trace[-12:]:
        target = step.get("target") or step.get("action")
        reason = step.get("reason", "")
        mode = step.get("planner_mode", "")
        backend = step.get("planner_backend")
        backend_text = f"/{backend}" if backend else ""
        error = step.get("planner_error")
        error_text = f" | error={error}" if error else ""
        print(f"  t{step.get('turn_id')} s{step.get('step')} [{mode}{backend_text}] {target}: {reason}{error_text}")
    print()


def _print_planner(runner: AutonomousRunner) -> None:
    planner = runner.planner
    print("Planner:")
    print(f"  mode: {runner.planner_mode}")
    print(f"  execution_scope: {runner.execution_scope}")
    print(f"  class: {planner.__class__.__name__}")
    print(f"  provider: {getattr(planner, 'provider', None)}")
    print(f"  backend: {getattr(planner, 'last_backend', 'rule')}")
    print(f"  model: {getattr(planner, 'model', None)}")
    print(f"  endpoint: {getattr(planner, 'endpoint', None)}")
    print(f"  deepseek_endpoint: {getattr(planner, 'deepseek_endpoint', None)}")
    print(f"  last_error: {getattr(planner, 'last_error', '')}")
    print(f"  consecutive_errors: {getattr(planner, 'consecutive_errors', 0)}")
    print(f"  last_usage: {getattr(planner, 'last_usage', {})}")
    print(f"  disabled_after_error: {getattr(planner, 'disabled_after_error', False)}")
    if hasattr(planner, "last_plan"):
        print(f"  last_plan: {planner.last_plan}")
        print(f"  queued_plan: {planner.queued_plan}")
        print(f"  last_plan_reason: {planner.last_plan_reason}")
        print(f"  last_turn_analysis: {getattr(planner, 'last_turn_analysis', {})}")
    print()


def _print_state(state: dict) -> None:
    request = state.get("final_search_request")
    last_request = state.get("last_final_search_request")
    retrieval_plan = state.get("retrieval_plan")
    print("State:")
    print(f"  turn_id: {state.get('turn_id')}")
    print(f"  preferences: {_jsonish(state.get('preferences', {}))}")
    print(f"  preference_events: {_jsonish(state.get('preference_events', []))}")
    print(f"  reflection_notes: {_jsonish(state.get('reflection_notes', []))}")
    print(f"  control_flags: {state.get('control_flags', [])}")
    print(f"  preference_flags: {state.get('preference_flags', [])}")
    print(f"  safety_flags: {state.get('safety_flags', [])}")
    print(f"  unclassified_flags: {state.get('unclassified_flags', [])}")
    print(f"  rejected_brands: {state.get('rejected_brands', [])}")
    print(f"  rejected_products: {state.get('rejected_products', [])}")
    print(f"  rejected_product_types: {state.get('rejected_product_types', [])}")
    print(f"  critic_decision: {state.get('critic_decision')}")
    print(f"  critic_next_action: {state.get('critic_next_action')}")
    print(f"  critic_reasons: {state.get('critic_reasons', [])}")
    print(f"  critic_notes: {_jsonish(state.get('critic_notes', []))}")
    print(f"  response_notes: {_jsonish(state.get('response_notes', []))}")
    print(f"  retrieval_readiness: {state.get('retrieval_readiness')}")
    print(f"  final_search_request: {_jsonish(request)}")
    print(f"  last_final_search_request: {_jsonish(last_request)}")
    print(f"  retrieval_plan: {_jsonish(retrieval_plan)}")
    print(f"  raw_candidates: {len(state.get('raw_candidates') or [])}")
    print(f"  verified_candidates: {len(state.get('verified_candidates') or [])}")
    print(f"  last_recommendations: {len(state.get('last_recommendations') or [])}")
    print()


def _jsonish(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _jsonish(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonish(item) for item in value]
    return value


if __name__ == "__main__":
    main()
