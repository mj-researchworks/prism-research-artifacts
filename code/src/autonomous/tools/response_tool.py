"""Response tool for the autonomous planner loop."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, is_dataclass
from typing import Any

from src.autonomous.flags import combined_flags
from src.autonomous.schemas import ComponentResult


class ResponseTool:
    name = "response_tool"

    def __init__(self) -> None:
        self.mode = os.getenv("AUTONOMOUS_RESPONSE_MODE", "template").lower()
        self.client = ResponseReasoningClient() if self.mode in {"agent", "hybrid"} else None

    def run(self, state: dict) -> ComponentResult:
        text, response_type = self._draft_response(state)
        text, response_note = self._maybe_reasoned_response(state, text, response_type)

        return ComponentResult(
            name=self.name,
            message=text,
            patch={
                "response": text,
                "done": True,
                "last_recommendations": list(state.get("previous_recommendations") or [])
                if state.get("intent", {}).get("label") == "accept" and state.get("previous_recommendations")
                else state.get("last_recommendations") or [],
                "response_notes": [response_note] if response_note else [],
                "tool_observation": {
                    "component": self.name,
                    "summary": f"response generated; type={response_type}; mode={response_note.get('mode') if response_note else 'template'}",
                    "status": "success",
                },
            },
        )

    def _draft_response(self, state: dict) -> tuple[str, str]:
        if state.get("intent", {}).get("label") == "accept" and state.get("previous_recommendations"):
            text = "Great, I will keep those options as the current recommendation set."
            return text, "acceptance"

        if state.get("intent", {}).get("label") == "accept":
            request = state.get("last_final_search_request") or state.get("final_search_request")
            if request:
                text = (
                    "I have a finalized search request, but I do not have recommendations in memory to accept yet: "
                    f"query='{_request_value(request, 'query_text')}', category='{_request_value(request, 'category')}'."
                )
            else:
                text = "I do not have recommendations in memory to accept yet."
            return text, "acceptance_without_recommendations"

        recommendations = state.get("last_recommendations") or []
        critic_decision = state.get("critic_decision")
        critic_reasons = state.get("critic_reasons") or []
        if recommendations and critic_decision == "ACCEPT":
            text = _format_recommendations(
                recommendations,
                slots=(state.get("candidate_ledger") or {}).get("slots") or {},
            )
            return text, "recommendation"

        if critic_decision == "ASK_USER":
            if state.get("raw_candidates") and not state.get("verified_candidates"):
                if state.get("rejected_products") or state.get("rejected_brands") or state.get("rejected_features"):
                    text = "I searched again, but the remaining results were weak after excluding the previous choices. Could you add one more preference, like brand, budget, or use case?"
                else:
                    text = "I found products, but none passed the current hard constraints. Should I relax one constraint and search again?"
            elif state.get("retrieval_readiness") == "needs_clarification":
                reason = critic_reasons[0] if critic_reasons else "I need one more preference before retrieval."
                if "availability only" in reason.lower():
                    text = "Yes, I can help with that. Do you have a price range, brand, or feature preference?"
                elif "vague" in reason.lower():
                    text = "Could you add one more preference, like price range, brand, style, or use case?"
                else:
                    text = "Could you add one more preference before I search, like price range, brand, style, or use case?"
            else:
                text = "Could you add one more preference before I search?"
        elif critic_decision == "SEARCH_AGAIN":
            reason = critic_reasons[0] if critic_reasons else "The candidate set is weak."
            if int(state.get('search_attempts') or 0) >= 2:
                text = ("I searched again, but could not verify enough suitable options. "
                        "Would you like to change a preference or relax a constraint?")
            else:
                text = (f"I could not finalize recommendations on this turn: {reason} "
                        "Would you like me to try again or change a preference?")
        elif state.get("raw_candidates") and not state.get("verified_candidates"):
            text = "I found products, but they conflict with the current constraints. Should I relax one constraint and search again?"
        else:
            request = state.get("final_search_request")
            flags = combined_flags(state)
            if request is None and "underspecified_product_type" in flags:
                text = "What type of product are you considering for this gift?"
            elif request is None:
                text = "I need a little more information before I can build a search request."
            elif "underspecified_product_type" in combined_flags(request if isinstance(request, dict) else vars(request)):
                text = "What type of product are you considering for this gift?"
            else:
                text = (
                    "I built a structured search request: "
                    f"query='{_request_value(request, 'query_text')}', category='{_request_value(request, 'category')}', "
                    f"hard_constraints={_request_value(request, 'hard_constraints')}, "
                    f"soft_preferences={_request_value(request, 'soft_preferences')}."
                )

        return text, "clarification_or_status"

    def _maybe_reasoned_response(self, state: dict, draft: str, response_type: str) -> tuple[str, dict[str, Any] | None]:
        if not self.client:
            return draft, None
        if response_type == "recommendation" and os.getenv("AUTONOMOUS_RESPONSE_LLM_RECOMMENDATIONS", "0").lower() not in {"1", "true", "yes", "on"}:
            return draft, {
                "turn_id": state.get("turn_id"),
                "mode": "template",
                "reason": "Recommendation responses stay templated unless explicitly enabled to avoid hallucinated product facts.",
            }
        try:
            result = self.client.respond(state=state, draft=draft, response_type=response_type)
            message = _bounded_message(result.get("message"), fallback=draft)
            if not _response_preserves_grounding(message, draft, state, response_type):
                return draft, {
                    "turn_id": state.get("turn_id"),
                    "mode": "reasoning_response_rejected",
                    "provider": self.client.provider,
                    "model": self.client.model,
                    "response_type": response_type,
                    "reason": "LLM rewrite did not preserve required structured facts.",
                    "usage": self.client.last_usage,
                }
            return message, {
                "turn_id": state.get("turn_id"),
                "mode": "reasoning_response",
                "provider": self.client.provider,
                "model": self.client.model,
                "response_type": response_type,
                "confidence": result.get("confidence"),
                "reason": result.get("reason"),
                "usage": self.client.last_usage,
            }
        except Exception as exc:
            if self.mode == "agent":
                # Even in agent mode, response generation must fail safely.
                pass
            return draft, {
                "turn_id": state.get("turn_id"),
                "mode": "reasoning_response_failed",
                "provider": self.client.provider,
                "model": self.client.model,
                "response_type": response_type,
                "error": f"{type(exc).__name__}: {exc}",
            }


class ResponseReasoningClient:
    """Optional bounded LLM writer for user-facing clarification/status text."""

    def __init__(self) -> None:
        self.provider = os.getenv("AUTONOMOUS_RESPONSE_PROVIDER", os.getenv("AUTONOMOUS_PLANNER_PROVIDER", "ollama")).lower()
        self.model = os.getenv("AUTONOMOUS_RESPONSE_MODEL", os.getenv("AUTONOMOUS_PLANNER_MODEL", "llama3.2:3b"))
        self.timeout_seconds = float(os.getenv("AUTONOMOUS_RESPONSE_TIMEOUT", "30"))
        self.ollama_endpoint = os.getenv("AUTONOMOUS_RESPONSE_OLLAMA_URL", "http://localhost:11434/api/generate")
        self.deepseek_endpoint = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
        self.last_usage: dict[str, Any] = {}

    def respond(self, *, state: dict[str, Any], draft: str, response_type: str) -> dict[str, Any]:
        prompt = _response_prompt(_compact_response_state(state), draft, response_type)
        if self.provider == "deepseek":
            return self._respond_deepseek(prompt)
        return self._respond_ollama(prompt)

    def _respond_ollama(self, prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 260},
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
            raise RuntimeError("Response Ollama endpoint is unavailable") from exc
        self.last_usage = {}
        return _parse_json_object(raw.get("response", ""))

    def _respond_deepseek(self, prompt: str) -> dict[str, Any]:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a bounded response agent. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 320,
            "stream": False,
        }
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
            raise RuntimeError(f"DeepSeek response HTTP {exc.code}: {body[:240]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("DeepSeek response endpoint is unavailable") from exc
        self.last_usage = raw.get("usage") or {}
        content = raw["choices"][0]["message"].get("content") or ""
        return _parse_json_object(content)


def _format_recommendations(candidates: list, slots: dict) -> str:
    lines = ["Here are some grounded options I found for you:\n"]
    for index, product in enumerate(candidates[:3], start=1):
        title = product.get("title", "Unknown product")
        brand = product.get("brand") or ""
        price = product.get("price")
        rating = product.get("rating")
        price_text = _format_price(price)
        rating_text = f"  rating {float(rating):.1f}" if _is_number(rating) else ""
        brand_text = f" - {brand}" if brand else ""
        reason = _first_reason(product.get("decision_reasons") or [])
        reason_text = f"\n   Why: {reason}" if reason else ""
        lines.append(f"{index}. {title}{brand_text}\n   {price_text}{rating_text}{reason_text}")

    filters = []
    if slots.get("price_max") is not None:
        filters.append(f"under ${slots['price_max']}")
    if slots.get("use_case"):
        filters.append(f"for {slots['use_case']}")
    if filters:
        lines.append("\nApplied constraints: " + ", ".join(filters))
    lines.append("\nWould you like details on any option, or should I refine the search?")
    return "\n".join(lines)


def _format_price(value) -> str:
    return f"${float(value):.2f}" if _is_number(value) else "Price not listed"


def _is_number(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _first_reason(reasons: list) -> str:
    for reason in reasons:
        reason = str(reason or "").strip()
        if reason and not reason.startswith("blocked"):
            return reason
    return ""


def _request_value(request: Any, key: str):
    if isinstance(request, dict):
        return request.get(key)
    return getattr(request, key, None)


def _response_prompt(state: dict[str, Any], draft: str, response_type: str) -> str:
    return f"""You are the Response Agent in a bounded autonomous product recommender.

Your job is to write the user-facing message for the current turn.
Do not choose tools. Do not recommend products unless verified recommendation titles are provided.
Do not invent product facts, prices, brands, ratings, or policies.
Use the draft as the safe baseline. You may rewrite it to be clearer and more conversational.
For clarification, ask exactly one concise question.
For status/search-request messages, preserve the exact structured facts.
Return ONLY valid JSON.

JSON shape:
{{
  "message": "short user-facing response",
  "confidence": 0.0,
  "reason": "one short sentence"
}}

Response type:
{response_type}

Compact state:
{json.dumps(state, separators=(",", ":"))}

Safe draft:
{json.dumps(draft)}
"""


def _compact_response_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": state.get("turn_id"),
        "latest_user_message": state.get("latest_user_message"),
        "intent": state.get("intent"),
        "preferences": _jsonish(state.get("preferences", {})),
        "control_flags": state.get("control_flags", []),
        "preference_flags": state.get("preference_flags", []),
        "safety_flags": state.get("safety_flags", []),
        "critic_decision": state.get("critic_decision"),
        "critic_next_action": state.get("critic_next_action"),
        "critic_reasons": state.get("critic_reasons", []),
        "retrieval_readiness": state.get("retrieval_readiness"),
        "final_search_request": _jsonish(state.get("final_search_request")),
        "recommendations": [
            {
                "title": item.get("title"),
                "brand": item.get("brand"),
                "price": item.get("price"),
                "rating": item.get("rating"),
                "reasons": item.get("decision_reasons"),
            }
            for item in (state.get("last_recommendations") or [])[:3]
        ],
    }


def _bounded_message(message: Any, fallback: str) -> str:
    text = str(message or "").strip()
    if not text:
        return fallback
    text = " ".join(text.split())
    if len(text) > 700:
        return fallback
    return text


def _response_preserves_grounding(message: str, draft: str, state: dict[str, Any], response_type: str) -> bool:
    if response_type == "recommendation":
        return True
    required = _required_response_facts(draft, state)
    normalized_message = _normalize_for_grounding(message)
    return all(_normalize_for_grounding(value) in normalized_message for value in required)


def _required_response_facts(draft: str, state: dict[str, Any]) -> list[str]:
    request = state.get("final_search_request") or state.get("last_final_search_request")
    if not request:
        return []
    required = []
    for key in ("query_text", "category"):
        value = _request_value(request, key)
        if value and str(value) in draft:
            required.append(str(value))
    hard_constraints = _request_value(request, "hard_constraints") or {}
    if isinstance(hard_constraints, dict):
        for value in hard_constraints.values():
            if value is not None and str(value) in draft:
                required.append(str(value))
    return list(dict.fromkeys(required))


def _normalize_for_grounding(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _jsonish(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _jsonish(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonish(item) for item in value]
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
        raise ValueError("Response model did not return a JSON object")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Response model JSON was not an object")
    return parsed
