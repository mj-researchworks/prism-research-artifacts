"""Intent evidence agent for the autonomous workflow.

The autonomous planner owns workflow control, but it should not have to infer
intent from scratch on every step. This agent provides local intent evidence:
the trained ConServe BERT intent encoder is the primary signal for normal
shopping turns, while lightweight gates handle dialogue acts the encoder was
not trained to model directly, such as acceptance and product questions.
"""

from __future__ import annotations

import os
import re

from src.autonomous.schemas import ComponentResult


class IntentEvidenceAgent:
    name = "intent_agent"

    def run(self, state: dict) -> ComponentResult:
        mode = os.getenv("AUTONOMOUS_INTENT_MODE", "hybrid").strip().lower()
        lightweight = _lightweight_intent(state)
        encoder = None
        error = None

        disabled_reason = _encoder_disabled_reason()
        if mode in {"encoder", "bert", "hybrid"} and not _should_skip_encoder(lightweight) and not disabled_reason:
            try:
                encoder = _encoder_intent(state)
            except Exception as exc:  # pragma: no cover - depends on local model files
                error = f"{type(exc).__name__}: {exc}"
        elif disabled_reason:
            error = disabled_reason

        selected, selection_source = _select_intent(mode, lightweight, encoder)
        selected = _enrich_intent(selected, lightweight, encoder)

        if mode in {"encoder", "bert"} and encoder:
            selected["source"] = encoder.get("source", "bert:raw_bert")
        elif mode in {"encoder", "bert"} and not encoder:
            selected["source"] = "lightweight_after_encoder_error"
        elif selection_source == "action_gate":
            selected["source"] = "action_gate"
        elif selection_source == "bert":
            selected["source"] = encoder.get("source", "bert:raw_bert") if encoder else "bert"
        else:
            selected["source"] = "lightweight"

        observation = {
            "component": self.name,
            "summary": (
                f"intent={selected['label']} "
                f"source={selected.get('source', 'unknown')}"
            ),
            "status": "success" if error is None else "degraded",
            "details": {
                "mode": mode,
                "lightweight": lightweight,
                "encoder": encoder,
                "selection_source": selection_source,
            },
        }
        if error:
            observation["details"]["encoder_error"] = error

        return ComponentResult(
            name=self.name,
            patch={
                "intent": selected,
                "tool_observation": observation,
            },
        )


def _lightweight_intent(state: dict) -> dict:
    text = state.get("latest_user_message", "").lower()
    stripped = text.strip()
    if any(term in text for term in ("thank", "i like", "take it", "buy it", "this works", "looks good", "sounds good")):
        label = "accept"
        action_label = "acceptance"
        confidence = 0.95
    elif stripped in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}:
        label = "chitchat"
        action_label = "chitchat"
        confidence = 0.95
    elif stripped in {"no", "nope"} or re.match(r"^(not|no)\s+\w+", stripped):
        label = "reject_or_revise"
        action_label = "rejection"
        confidence = 0.9
    elif any(term in text for term in ("don't like", "do not like", "change", "another", "cheaper", "different option", "show different")):
        label = "reject_or_revise"
        action_label = "rejection"
        confidence = 0.86
    elif "?" in text or text.startswith(("what", "how", "does", "can you")):
        label = "inquire"
        action_label = "inquiry"
        confidence = 0.82
    else:
        label = "recommend"
        action_label = "new_request_or_refinement"
        confidence = 0.68
    return {
        "label": label,
        "action_label": action_label,
        "confidence": confidence,
        "source": "lightweight",
    }


def _should_skip_encoder(lightweight: dict) -> bool:
    """Keep only out-of-vocabulary dialogue acts away from the encoder.

    The old BERT model was trained on Reveal/Interpret/Revise/Chitchat. It did
    not directly learn the autonomous labels "accept" and "inquire", so those
    high-confidence gates remain local. Rejection/revision and normal shopping
    requests go to the encoder first in hybrid mode.
    """

    return lightweight.get("label") in {"accept", "inquire", "chitchat"} and float(
        lightweight.get("confidence") or 0.0
    ) >= 0.9


def _encoder_is_confident(encoder: dict) -> bool:
    return float(encoder.get("confidence") or 0.0) >= 0.62


def _encoder_intent(state: dict) -> dict:
    from src.agents.intent_agent import IntentAgent

    user_message = state.get("latest_user_message", "")
    history = [
        {"role": item.get("role", ""), "content": item.get("content", "")}
        for item in state.get("history", [])[:-1]
    ]
    result = IntentAgent().classify_raw_encoder(user_message=user_message, history=history)
    mapped = _map_encoder_label(result.get("intent"))
    confidence = float(result.get("confidence") or 0.0)
    return {
        "label": mapped,
        "action_label": _action_label_from_pscon(result.get("intent")),
        "confidence": round(confidence, 4),
        "source": f"bert:{result.get('source', 'unknown')}",
        "pscon_label": _pscon_label(result.get("intent")),
        "encoder_label": result.get("intent"),
        "encoder_confidence": round(confidence, 4),
        "encoder_margin": result.get("margin"),
        "encoder_run": bool(result.get("run_bert")),
    }


def _encoder_disabled_reason() -> str | None:
    """Avoid importing the BERT stack in known-broken local environments."""

    try:
        import numpy as np

        major, minor, *_ = [int(part) for part in np.__version__.split(".")[:2]]
    except Exception:
        return None
    if (major, minor) >= (2, 3):
        return (
            "encoder_skipped: local numpy version is incompatible with the "
            "installed scipy/sklearn stack"
        )
    return None


def _session_stage(state: dict) -> str:
    if state.get("previous_recommendations"):
        return "recommending"
    if state.get("preferences", {}).get("product_type"):
        return "searching"
    if state.get("intent", {}).get("previous_label"):
        return "clarifying"
    return "new"


def _map_encoder_label(label: str | None) -> str:
    normalized = (label or "").strip().lower()
    if normalized == "acceptance":
        return "accept"
    if normalized == "revise":
        return "reject_or_revise"
    if normalized == "inquire":
        return "inquire"
    if normalized == "chitchat":
        return "chitchat"
    return "recommend"


def _select_intent(mode: str, lightweight: dict, encoder: dict | None) -> tuple[dict, str]:
    if mode in {"lightweight", "rules", "rule"}:
        return dict(lightweight), "lightweight"

    if _is_action_gate(lightweight):
        return dict(lightweight), "action_gate"

    if encoder and _encoder_is_confident(encoder):
        return dict(encoder), "bert"

    return dict(lightweight), "lightweight"


def _is_action_gate(lightweight: dict) -> bool:
    return lightweight.get("action_label") in {
        "acceptance",
        "inquiry",
        "rejection",
        "chitchat",
    } and float(lightweight.get("confidence") or 0.0) >= 0.8


def _enrich_intent(selected: dict, lightweight: dict, encoder: dict | None) -> dict:
    enriched = dict(selected)
    encoder_pscon_label = None
    if encoder:
        encoder_pscon_label = encoder.get("pscon_label") or _pscon_label(encoder.get("encoder_label"))
        enriched["encoder_pscon_label"] = encoder_pscon_label
        enriched["encoder_label"] = encoder.get("encoder_label")
        enriched["encoder_confidence"] = encoder.get("encoder_confidence")
        enriched["encoder_margin"] = encoder.get("encoder_margin")
        enriched["encoder_run"] = encoder.get("encoder_run")
    action_pscon_label = _pscon_from_action(enriched.get("action_label"))
    enriched["action_pscon_label"] = action_pscon_label
    enriched["pscon_label"] = enriched.get("pscon_label") or encoder_pscon_label or action_pscon_label
    enriched["intent_disagreement"] = bool(
        encoder_pscon_label
        and action_pscon_label
        and encoder_pscon_label != action_pscon_label
        and _is_action_gate(enriched)
    )
    enriched["action_label"] = enriched.get("action_label") or _action_label_from_pscon(
        enriched.get("pscon_label")
    )
    enriched["planner_intent"] = enriched.get("label")
    enriched["next_action_hint"] = _next_action_hint(enriched.get("action_label"))
    enriched["requires_state_update"] = enriched.get("action_label") in {
        "new_request",
        "new_request_or_refinement",
        "refinement",
        "revision",
        "rejection",
    }
    enriched["requires_retrieval_decision"] = enriched.get("action_label") in {
        "new_request",
        "new_request_or_refinement",
        "refinement",
        "revision",
        "rejection",
    }
    return enriched


def _pscon_label(label: str | None) -> str | None:
    normalized = (label or "").strip().lower()
    mapping = {
        "reveal": "Reveal",
        "interpret": "Interpret",
        "revise": "Revise",
        "chitchat": "Chitchat",
    }
    return mapping.get(normalized)


def _action_label_from_pscon(label: str | None) -> str:
    normalized = (label or "").strip().lower()
    if normalized == "revise":
        return "revision"
    if normalized == "interpret":
        return "refinement"
    if normalized == "chitchat":
        return "chitchat"
    return "new_request"


def _pscon_from_action(action_label: str | None) -> str:
    if action_label == "chitchat":
        return "Chitchat"
    if action_label in {"rejection", "revision"}:
        return "Revise"
    if action_label in {"refinement", "inquiry", "acceptance"}:
        return "Interpret"
    return "Reveal"


def _next_action_hint(action_label: str | None) -> str:
    mapping = {
        "acceptance": "stop_or_confirm_current_recommendations",
        "inquiry": "answer_question_or_ask_context",
        "rejection": "update_memory_and_retrieve",
        "revision": "update_memory_and_retrieve",
        "refinement": "update_memory_and_decide_retrieval",
        "new_request": "update_memory_and_decide_clarify_or_retrieve",
        "new_request_or_refinement": "update_memory_and_decide_clarify_or_retrieve",
        "chitchat": "respond_without_retrieval",
    }
    return mapping.get(action_label or "", "planner_decides")
