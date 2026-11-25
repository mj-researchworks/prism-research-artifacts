"""Flag taxonomy for autonomous recommender state."""

from __future__ import annotations


CONTROL_FLAGS = {
    "underspecified_product_type",
    "missing_search_request",
    "ask_user_before_retrieval",
    "weak_candidate_set",
    "retrieval_tool_failed",
    "step_budget_exhausted",
}

PREFERENCE_FLAGS = {
    "preference_instability_signal",
    "intent_shift_signal",
    "contradiction_signal",
}

SAFETY_FLAGS = {
    "prompt_injection_signal",
    "unsafe_instruction_signal",
    "corrupted_state_signal",
    "untrusted_tool_output_signal",
}


def classify_flags(flags: list[str] | tuple[str, ...] | set[str] | None) -> dict[str, list[str]]:
    buckets = {
        "control_flags": [],
        "preference_flags": [],
        "safety_flags": [],
        "unclassified_flags": [],
        "security_flags": [],
    }
    for raw in flags or []:
        flag = str(raw or "").strip()
        if not flag:
            continue
        if flag in CONTROL_FLAGS:
            buckets["control_flags"].append(flag)
        elif flag in PREFERENCE_FLAGS:
            buckets["preference_flags"].append(flag)
        elif flag in SAFETY_FLAGS:
            buckets["safety_flags"].append(flag)
            buckets["security_flags"].append(flag)
        else:
            buckets["unclassified_flags"].append(flag)
    return {key: list(dict.fromkeys(value)) for key, value in buckets.items()}


def combined_flags(state: dict) -> list[str]:
    flags = []
    for key in ("control_flags", "preference_flags", "safety_flags"):
        flags.extend(state.get(key) or [])
    return list(dict.fromkeys(str(flag) for flag in flags if str(flag or "").strip()))
