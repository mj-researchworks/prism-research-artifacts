"""Summarize autonomous ConServe JSONL evaluation logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable


DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "outputs" / "autonomous_logs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize autonomous ConServe evaluation logs.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_LOG_DIR),
        help="Path to a JSONL log file or a directory of JSONL logs.",
    )
    parser.add_argument("--latest", action="store_true", help="Summarize only the newest JSONL file in a directory.")
    args = parser.parse_args()

    files = _resolve_files(Path(args.path), latest=args.latest)
    if not files:
        raise SystemExit(f"No .jsonl logs found at {args.path}")

    records = list(_read_records(files))
    if not records:
        raise SystemExit("No records found in selected logs")

    summary = summarize(records, files)
    print_summary(summary)


def summarize(records: list[dict], files: list[Path]) -> dict:
    planner_backend_counts = Counter()
    component_counts = Counter()
    critic_counts = Counter()
    event_counts = Counter()
    security_flag_counts = Counter()
    control_flag_counts = Counter()
    preference_flag_counts = Counter()
    safety_flag_counts = Counter()
    unclassified_flag_counts = Counter()
    component_durations = defaultdict(list)
    planner_durations = []
    turn_type_counts = Counter()
    preference_change_counts = Counter()
    retrieval_readiness_counts = Counter()
    search_request_status_counts = Counter()
    planner_confidence_counts = Counter()
    state_change_counts = Counter()
    required_specialist_counts = Counter()
    planner_need_counts = Counter()
    conversation_ids = set()
    latencies = []
    recommendation_counts = []
    search_turns = 0
    turns_with_acceptance = 0
    turns_with_rejection = 0
    turns_with_preference_events = 0
    total_acceptance_events = 0
    total_rejection_events = 0
    total_preference_events = 0
    fallback_turns = 0
    guard_turns = 0
    llm_turns = 0
    turns_by_conversation = defaultdict(int)

    for record in records:
        conversation_id = record.get("conversation_id")
        if conversation_id:
            conversation_ids.add(conversation_id)
            turns_by_conversation[conversation_id] += 1

        latency = record.get("latency_ms")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))

        rec_count = (record.get("candidate_counts") or {}).get("recommended")
        if isinstance(rec_count, int):
            recommendation_counts.append(rec_count)

        backends = record.get("planner_backends") or []
        planner_backend_counts.update(backends)
        if any("fallback" in str(item) for item in backends):
            fallback_turns += 1
        if any("guard" in str(item) for item in backends):
            guard_turns += 1
        if any(str(item).startswith("llm") for item in backends):
            llm_turns += 1

        components = record.get("components_called") or []
        component_counts.update(components)
        if "conserve_search_tool" in components:
            search_turns += 1

        critic = record.get("critic_decision")
        if critic:
            critic_counts[critic] += 1

        security_flag_counts.update(record.get("security_flags") or [])
        control_flag_counts.update(record.get("control_flags") or [])
        preference_flag_counts.update(record.get("preference_flags") or [])
        safety_flag_counts.update(record.get("safety_flags") or [])
        unclassified_flag_counts.update(record.get("unclassified_flags") or [])

        for step in record.get("planner_trace") or []:
            duration = step.get("planner_duration_ms")
            if isinstance(duration, (int, float)):
                planner_durations.append(float(duration))
        for observation in record.get("tool_observations") or []:
            component = observation.get("component")
            duration = observation.get("duration_ms")
            if component and isinstance(duration, (int, float)):
                component_durations[str(component)].append(float(duration))

        analysis = record.get("turn_analysis") or {}
        turn_type = analysis.get("turn_type")
        if turn_type:
            turn_type_counts[str(turn_type)] += 1
        preference_change = analysis.get("preference_change")
        if preference_change:
            preference_change_counts[str(preference_change)] += 1
        retrieval_readiness = analysis.get("retrieval_readiness")
        if retrieval_readiness:
            retrieval_readiness_counts[str(retrieval_readiness)] += 1
        search_request_status = analysis.get("search_request_status")
        if search_request_status:
            search_request_status_counts[str(search_request_status)] += 1
        planner_confidence = analysis.get("planner_confidence")
        if planner_confidence:
            planner_confidence_counts[str(planner_confidence)] += 1
        state_change = analysis.get("state_change_expected")
        if state_change:
            state_change_counts[str(state_change)] += 1
        required_specialist_counts.update(analysis.get("required_specialists") or [])
        if analysis.get("needs_retrieval") is True:
            planner_need_counts["needs_retrieval"] += 1
        if analysis.get("needs_critic") is True:
            planner_need_counts["needs_critic"] += 1

        turn_has_acceptance = False
        turn_has_rejection = False
        turn_events = record.get("preference_events") or []
        if turn_events:
            turns_with_preference_events += 1
        for event in turn_events:
            event_type = event.get("type")
            if event_type:
                total_preference_events += 1
                event_counts[event_type] += 1
                if "accept" in event_type:
                    total_acceptance_events += 1
                    turn_has_acceptance = True
                if "reject" in event_type or "rejection" in event_type:
                    total_rejection_events += 1
                    turn_has_rejection = True
        if turn_has_acceptance:
            turns_with_acceptance += 1
        if turn_has_rejection:
            turns_with_rejection += 1

    return {
        "files": files,
        "records": records,
        "conversation_count": len(conversation_ids),
        "turn_count": len(records),
        "turns_by_conversation": dict(turns_by_conversation),
        "latencies": latencies,
        "recommendation_counts": recommendation_counts,
        "planner_backend_counts": planner_backend_counts,
        "component_counts": component_counts,
        "critic_counts": critic_counts,
        "event_counts": event_counts,
        "security_flag_counts": security_flag_counts,
        "control_flag_counts": control_flag_counts,
        "preference_flag_counts": preference_flag_counts,
        "safety_flag_counts": safety_flag_counts,
        "unclassified_flag_counts": unclassified_flag_counts,
        "component_durations": dict(component_durations),
        "planner_durations": planner_durations,
        "turn_type_counts": turn_type_counts,
        "preference_change_counts": preference_change_counts,
        "retrieval_readiness_counts": retrieval_readiness_counts,
        "search_request_status_counts": search_request_status_counts,
        "planner_confidence_counts": planner_confidence_counts,
        "state_change_counts": state_change_counts,
        "required_specialist_counts": required_specialist_counts,
        "planner_need_counts": planner_need_counts,
        "search_turns": search_turns,
        "turns_with_acceptance": turns_with_acceptance,
        "turns_with_rejection": turns_with_rejection,
        "turns_with_preference_events": turns_with_preference_events,
        "total_acceptance_events": total_acceptance_events,
        "total_rejection_events": total_rejection_events,
        "total_preference_events": total_preference_events,
        "fallback_turns": fallback_turns,
        "guard_turns": guard_turns,
        "llm_turns": llm_turns,
    }


def print_summary(summary: dict) -> None:
    latencies = summary["latencies"]
    rec_counts = summary["recommendation_counts"]
    print("Autonomous ConServe Log Summary")
    print("=" * 34)
    print(f"Files: {len(summary['files'])}")
    for file in summary["files"]:
        print(f"  - {file}")
    print(f"Conversations: {summary['conversation_count']}")
    print(f"Turns: {summary['turn_count']}")
    print(f"Turns with LLM planner: {summary['llm_turns']}")
    print(f"Turns with policy guard: {summary['guard_turns']}")
    print(f"Turns with fallback: {summary['fallback_turns']}")
    print(f"Turns with retrieval: {summary['search_turns']}")
    print(f"Turns with preference events: {summary['turns_with_preference_events']}")
    print(f"Turns with acceptance: {summary['turns_with_acceptance']}")
    print(f"Turns with rejection: {summary['turns_with_rejection']}")
    print(f"Total preference events: {summary['total_preference_events']}")
    print(f"Total acceptance events: {summary['total_acceptance_events']}")
    print(f"Total rejection events: {summary['total_rejection_events']}")

    if latencies:
        print(f"Latency ms avg/median/max: {mean(latencies):.0f} / {median(latencies):.0f} / {max(latencies):.0f}")
    if rec_counts:
        print(f"Recommended count avg/max: {mean(rec_counts):.2f} / {max(rec_counts)}")
    if summary["planner_durations"]:
        durations = summary["planner_durations"]
        print(f"Planner decision ms avg/median/max: {mean(durations):.0f} / {median(durations):.0f} / {max(durations):.0f}")

    _print_counter("Planner backends", summary["planner_backend_counts"])
    _print_counter("Components called", summary["component_counts"])
    _print_component_durations(summary["component_durations"])
    _print_counter("Critic decisions", summary["critic_counts"])
    _print_counter("Planner turn types", summary["turn_type_counts"])
    _print_counter("Planner preference changes", summary["preference_change_counts"])
    _print_counter("Planner retrieval readiness", summary["retrieval_readiness_counts"])
    _print_counter("Planner search request status", summary["search_request_status_counts"])
    _print_counter("Planner confidence", summary["planner_confidence_counts"])
    _print_counter("Planner state changes", summary["state_change_counts"])
    _print_counter("Planner required specialists", summary["required_specialist_counts"])
    _print_counter("Planner declared needs", summary["planner_need_counts"])
    _print_counter("Preference events", summary["event_counts"])
    _print_counter("Security/state flags", summary["security_flag_counts"])
    _print_counter("Control flags", summary["control_flag_counts"])
    _print_counter("Preference flags", summary["preference_flag_counts"])
    _print_counter("Safety flags", summary["safety_flag_counts"])
    _print_counter("Unclassified flags", summary["unclassified_flag_counts"])


def _print_counter(title: str, counter: Counter) -> None:
    print()
    print(f"{title}:")
    if not counter:
        print("  none")
        return
    for key, count in counter.most_common():
        print(f"  {key}: {count}")


def _print_component_durations(durations_by_component: dict[str, list[float]]) -> None:
    print()
    print("Component duration ms:")
    if not durations_by_component:
        print("  none")
        return
    for component, durations in sorted(durations_by_component.items()):
        print(f"  {component}: avg={mean(durations):.0f} median={median(durations):.0f} max={max(durations):.0f} calls={len(durations)}")

def _resolve_files(path: Path, latest: bool) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
    if latest and files:
        return [files[0]]
    return list(reversed(files))


def _read_records(files: Iterable[Path]) -> Iterable[dict]:
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON in {file}:{line_number}: {exc}") from exc


if __name__ == "__main__":
    main()
