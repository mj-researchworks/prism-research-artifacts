"""Optional LLM reasoning helper for preference reflection."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, is_dataclass
from typing import Any


class ReasoningReflectionClient:
    """Calls a small reasoning model for complex preference-state updates."""

    def __init__(self) -> None:
        self.provider = os.getenv("AUTONOMOUS_REFLECTION_PROVIDER", "ollama").lower()
        self.model = os.getenv("AUTONOMOUS_REFLECTION_MODEL", "llama3.2:3b")
        self.timeout_seconds = float(os.getenv("AUTONOMOUS_REFLECTION_TIMEOUT", "45"))
        self.ollama_endpoint = os.getenv("AUTONOMOUS_REFLECTION_OLLAMA_URL", "http://localhost:11434/api/generate")
        self.deepseek_endpoint = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
        self.deepseek_thinking = os.getenv("AUTONOMOUS_REFLECTION_DEEPSEEK_THINKING", "disabled").lower()
        self.last_error = ""
        self.last_usage: dict[str, Any] = {}

    def reflect(self, state: dict[str, Any], rule_draft: dict[str, Any] | None = None) -> dict[str, Any]:
        compact_state = _compact_reflection_state(state)
        prompt = _reflection_prompt(compact_state, rule_draft or {})
        if self.provider == "deepseek":
            return self._reflect_deepseek(prompt)
        return self._reflect_ollama(prompt)

    def _reflect_ollama(self, prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 360},
        }
        request = urllib.request.Request(
            self.ollama_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError("Reflection Ollama endpoint is unavailable") from exc
        self.last_usage = {}
        self.last_error = ""
        return _parse_json_object(raw.get("response", ""))

    def _reflect_deepseek(self, prompt: str) -> dict[str, Any]:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a preference reflection specialist. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 520,
            "stream": False,
        }
        if self.deepseek_thinking in {"enabled", "disabled"}:
            payload["thinking"] = {"type": self.deepseek_thinking}
        request = urllib.request.Request(
            self.deepseek_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            self.last_error = f"HTTP {exc.code}: {body[:240]}"
            raise RuntimeError(f"DeepSeek reflection HTTP {exc.code}: {body[:240]}") from exc
        except urllib.error.URLError as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError("DeepSeek reflection endpoint is unavailable") from exc

        self.last_usage = raw.get("usage") or {}
        self.last_error = ""
        content = raw["choices"][0]["message"].get("content") or ""
        return _parse_json_object(content)


def _reflection_prompt(state: dict[str, Any], rule_draft: dict[str, Any]) -> str:
    intent_detection_text = ""
    intent_shape_text = ""
    intent_guidance_text = ""
    if _reflection_intent_fields_enabled():
        intent_detection_text = "- PSCon annotation intent: reveal, interpret, revise, inquire, or chitchat.\n"
        intent_shape_text = (
            '  "pscon_intent_label": "reveal|interpret|revise|inquire|chitchat",\n'
        )
        intent_guidance_text = (
            "\nAssign pscon_intent_label using PSCon-style annotation rules, separately from workflow strategy:\n"
            "- reveal: the user introduces a new product need or begins a product-search request.\n"
            "- interpret: the user gives details, answers clarification, refines constraints, rejects options\n"
            "  within the same product need, accepts/selects a shown option, or asks to see/compare more options.\n"
            "- revise: the user clearly switches to a different product/category or replaces the main shopping goal.\n"
            "- inquire: the user asks an explicit factual question about shown products or product attributes.\n"
            "- chitchat: greeting, thanks, social, or non-shopping conversation.\n"
            "Do not label ordinary rejection, cheaper/more/another-option feedback, or acceptance as revise unless\n"
            "the main product/category changes. Capture those behaviors in preference_change and strategy instead.\n"
        )
    return f"""You are the Preference & Reflection Agent inside an autonomous product recommender.

Your job is not to recommend products. Diagnose how the latest user turn changes shopping memory.
Use conversation context and current blackboard state. Return ONLY one valid JSON object.
Do not include markdown, prose, comments, or chain-of-thought.
The first character of your response must be {{ and the last character must be }}.

You receive a rule_draft from a lightweight extractor. Treat it as a draft, not truth.
If rule_draft is empty, reason directly from State as the primary reflection agent.
In autonomous agent mode, use rule_draft only as evidence. You must decide whether to keep, reject,
or correct each draft item in your JSON output.
Return the corrected latest-turn delta only. Do not repeat unchanged memory.

Preserve existing product_type unless the latest user turn clearly names a replacement product family.
Do not convert acceptance, status, or vague modifier phrases into product_type.
A modifier such as cheaper, more portable, or less expensive updates preferences for the existing
product_type; it is not a new category by itself.

Detect:
- stable preference: user adds normal details without conflict.
- revision: user changes an earlier product/category/feature/budget/use case.
- rejection: user rejects a brand, product, feature, or previous recommendation.
- acceptance: user accepts current recommendation/options.
- uncertainty: the turn is vague or needs clarification.
{intent_detection_text}

JSON shape:
{{
{intent_shape_text}  "preference_change": "none|new_preference|revision|rejection|acceptance|uncertain",
  "strategy": "ask_clarification|update_memory|update_query|search_again|accept_current|no_action",
  "confidence": 0.0,
  "preferences": {{
    "product_type": null,
    "budget_max": null,
    "occasion": null,
    "recipient": null,
    "use_case": null
  }},
  "preferred_brands": [],
  "preferred_features": [],
  "rejected_brands": [],
  "rejected_features": [],
  "reject_previous_recommendations": false,
  "clear_preferences": [],
  "remove_preferred_brands": [],
  "remove_preferred_features": [],
  "remove_rejected_brands": [],
  "remove_rejected_features": [],
  "accepted": false,
  "needs_clarification": false,
  "reason": "one short sentence"
}}

{intent_guidance_text}

Use clear_preferences only when the latest turn invalidates a stored slot and does not provide a replacement.
Allowed clear slots: product_type, budget_max, use_case, occasion, recipient.
Example: if current product_type is "headset" and the latest turn is "I don't want a headset",
return "clear_preferences": ["product_type"] and preference_change "rejection".

For brand/feature replacements, return the old stored values in remove_preferred_brands
or remove_preferred_features and the new values in preferred_brands or preferred_features.
Retiring a preference does not mean rejecting it: use rejected_* only for explicit exclusions.
To withdraw an exclusion, use remove_rejected_brands or remove_rejected_features without
inventing a positive preference. Return exact stored values for removal; omitted fields preserve memory.
If a turn expresses acceptance but also asks for a change, interpret the outstanding request
in context and use the appropriate update_query/search_again strategy rather than accept_current.
Use accept_current when no further requested change remains. Do not infer acceptance from a phrase alone.

State:
{json.dumps(state, separators=(",", ":"))}

Rule draft:
{json.dumps(rule_draft, separators=(",", ":"))}
"""


def _reflection_intent_fields_enabled() -> bool:
    return os.getenv("AUTONOMOUS_REFLECTION_WRITES_INTENT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _compact_reflection_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": state.get("turn_id"),
        "latest_user_message": state.get("latest_user_message"),
        "history_tail": state.get("history", [])[-6:],
        "intent": state.get("intent"),
        "preferences": _serialize(state.get("preferences", {})),
        "preferred_brands": state.get("preferred_brands", []),
        "preferred_features": state.get("preferred_features", []),
        "rejected_brands": state.get("rejected_brands", []),
        "rejected_features": state.get("rejected_features", []),
        "has_previous_recommendations": bool(state.get("previous_recommendations")),
        "previous_recommendation_titles": [
            str(item.get("title") or "")[:100]
            for item in (state.get("previous_recommendations") or [])[:3]
        ],
        "last_final_search_request": _serialize(state.get("last_final_search_request")),
    }


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def _parse_json_object(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        snippet = text[:160].replace("\n", " ")
        raise ValueError(f"Reflection model did not return a JSON object: {snippet!r}")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Reflection model JSON was not an object")
    return parsed
