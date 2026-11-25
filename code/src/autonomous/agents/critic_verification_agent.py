"""Critic and verification specialist for autonomous recommendation decisions."""

from __future__ import annotations

import json
import os
import hashlib
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any

from src.autonomous.flags import combined_flags
from src.autonomous.reasoning_budget import adaptive_budget_enabled, critic_llm_needed, reasoning_depth
from src.autonomous.schemas import ComponentResult


ALLOWED_DECISIONS = {"ACCEPT", "RERANK", "SEARCH_AGAIN", "UPDATE_STATE", "ASK_USER"}
ALLOWED_NEXT_ACTIONS = {"retrieve", "ask_user", "search_again", "rerank", "update_state", "respond"}


class CriticVerificationAgent:
    name = "critic_verification_agent"

    def __init__(self, critic_client: "CriticReasoningClient | None" = None, critic_mode: str | None = None):
        self.critic_mode = (critic_mode or os.getenv("AUTONOMOUS_CRITIC_MODE", "rule")).lower()
        self.critic_client = critic_client
        self._critique_cache: dict[str, dict[str, Any]] = {}

    def run(self, state: dict) -> ComponentResult:
        hard_result = self._hard_verify(state)
        if not self._should_call_critic_llm(state, hard_result.patch):
            patch = dict(hard_result.patch)
            notes = list(patch.get("critic_notes") or [])
            notes.append({
                "turn_id": state.get("turn_id"),
                "mode": "critic_reasoning_skipped",
                "reasoning_budget": reasoning_depth(state),
                "adaptive_budget": adaptive_budget_enabled(),
                "hard_decision": hard_result.patch.get("critic_decision"),
                "hard_next_action": hard_result.patch.get("critic_next_action"),
                "skip_reason": self._critic_skip_reason(state, hard_result.patch),
            })
            patch["critic_notes"] = notes
            observation = dict(patch.get("tool_observation") or {})
            if observation:
                observation["summary"] = f"{observation.get('summary', '')}; llm=False"
                patch["tool_observation"] = observation
            return ComponentResult(name=self.name, patch=patch)

        try:
            client = self.critic_client or CriticReasoningClient()
            cache_key = self._critique_cache_key(state, hard_result.patch, client)
            cached_critique = self._critique_cache.get(cache_key)
            if cached_critique is not None:
                critique = deepcopy(cached_critique)
            else:
                critique = client.critique(state, hard_result.patch)
                self._critique_cache[cache_key] = deepcopy(critique)
            return self._merge_critic_result(
                hard_patch=hard_result.patch,
                critique=critique,
                client=client,
                state=state,
                cache_hit=cached_critique is not None,
            )
        except Exception as exc:
            patch = dict(hard_result.patch)
            notes = list(patch.get("critic_notes") or [])
            notes.append({
                "turn_id": state.get("turn_id"),
                "mode": "critic_reasoning_failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            patch["critic_notes"] = notes
            return ComponentResult(name=self.name, patch=patch)

    def _hard_verify(self, state: dict) -> ComponentResult:
        request = state.get("final_search_request")
        raw_candidates = state.get("raw_candidates") or []
        verified_candidates = state.get("verified_candidates") or []
        flags = []
        reasons = []

        request_flags = _request_value(request, "security_flags") or []
        if request and "underspecified_product_type" in request_flags:
            flags.append("ask_user_before_retrieval")
            reasons.append("Product type is missing, so retrieval should not run yet.")
            return self._result("ASK_USER", reasons, flags, readiness="needs_clarification", next_action="ask_user")

        if request and not raw_candidates:
            if self._retrieval_was_attempted(state):
                reasons.append("Retrieval completed but returned zero raw candidates.")
                if int(state.get("search_attempts") or 0) < 2:
                    return self._result("SEARCH_AGAIN", reasons, flags, readiness="needs_more_evidence", next_action="search_again")
                reasons.append("Search-again limit reached after empty retrieval; ask the user for one more preference.")
                return self._result("ASK_USER", reasons, flags, readiness="needs_clarification", next_action="ask_user")

            readiness, readiness_reasons, next_action = self._pre_retrieval_readiness(state, request)
            reasons.extend(readiness_reasons)
            if readiness == "ready":
                return self._result("ACCEPT", reasons, flags, readiness=readiness, next_action=next_action)
            flags.append("ask_user_before_retrieval")
            return self._result("ASK_USER", reasons, flags, readiness=readiness, next_action=next_action)

        if raw_candidates and not verified_candidates:
            if int(state.get("search_attempts") or 0) < 2:
                if state.get("rejected_products") or state.get("rejected_brands") or state.get("rejected_features"):
                    reasons.append("Retrieved products exist, but none passed after applying the current exclusions.")
                else:
                    reasons.append("Retrieved products exist, but none passed verification; try a broader retrieval plan before asking the user.")
                return self._result("SEARCH_AGAIN", reasons, flags, readiness="needs_more_evidence", next_action="search_again")
            if state.get("rejected_products") or state.get("rejected_brands") or state.get("rejected_features"):
                reasons.append("Retrieved products exist, but none passed after applying the current exclusions.")
            else:
                reasons.append("Retrieved products exist, but none passed hard verification constraints.")
            return self._result("ASK_USER", reasons, flags, readiness="needs_clarification", next_action="ask_user")

        if len(verified_candidates) < 3:
            reasons.append("Fewer than three verified candidates are available.")
            if int(state.get("search_attempts") or 0) >= 2:
                reasons.append("Search-again limit reached; ask the user for one more preference.")
                return self._result("ASK_USER", reasons, flags, readiness="needs_clarification", next_action="ask_user")
            return self._result("SEARCH_AGAIN", reasons, flags, readiness="needs_more_evidence", next_action="search_again")

        reasons.append("Verified candidates satisfy the current request and constraints.")
        if state.get("preference_events"):
            reasons.append("Preference instability events were incorporated into the search state.")
        return self._result("ACCEPT", reasons, flags, readiness="ready", next_action="respond")


    def _retrieval_was_attempted(self, state: dict) -> bool:
        # A re-search creates a new request and resets candidate state.  Only
        # a retrieval after the latest request-build observation belongs to
        # that request; older retrieval observations must not block recovery.
        observations = state.get("tool_observations", [])
        latest_request_index = max(
            (
                index
                for index, obs in enumerate(observations)
                if obs.get("component") == "search_research_agent"
            ),
            default=-1,
        )
        return any(
            obs.get("component") == "conserve_search_tool"
            for obs in observations[latest_request_index + 1 :]
        )

    def _should_call_critic_llm(self, state: dict, hard_patch: dict) -> bool:
        if self.critic_mode in {"0", "false", "off", "rule", "rules"}:
            return False
        if self.critic_mode in {"llm", "agent", "autonomous", "always"}:
            if adaptive_budget_enabled():
                return critic_llm_needed(state, hard_patch)
            return bool(state.get("final_search_request") or state.get("raw_candidates"))
        if self.critic_mode != "hybrid":
            return False

        hard_decision = hard_patch.get("critic_decision")
        hard_next_action = hard_patch.get("critic_next_action")
        if adaptive_budget_enabled():
            return critic_llm_needed(state, hard_patch)
        if _env_enabled("AUTONOMOUS_CRITIC_SELECTIVE") and not self._needs_llm_critique(state, hard_patch):
            return False
        if hard_next_action in {"update_state"}:
            return False
        if hard_decision in {"SEARCH_AGAIN", "ASK_USER"} and state.get("raw_candidates"):
            return True
        if hard_decision == "ACCEPT" and state.get("final_search_request"):
            return True
        if combined_flags(state) or state.get("preference_events"):
            return True
        return False

    def _needs_llm_critique(self, state: dict, hard_patch: dict) -> bool:
        hard_decision = hard_patch.get("critic_decision")
        hard_next_action = hard_patch.get("critic_next_action")
        if hard_decision in {"SEARCH_AGAIN", "ASK_USER", "RERANK", "UPDATE_STATE"}:
            return True
        if hard_next_action in {"search_again", "rerank", "update_state", "ask_user"}:
            return True
        if combined_flags(state) or state.get("preference_events"):
            return True
        request = state.get("final_search_request")
        hard_constraints = _request_value(request, "hard_constraints") or {}
        exclusions = _request_value(request, "exclusions") or {}
        preferred = _request_value(request, "preferred") or {}
        if hard_constraints or exclusions.get("brands") or exclusions.get("products") or exclusions.get("features"):
            return True
        if preferred.get("brands") or preferred.get("features"):
            return True
        if state.get("raw_candidates") and len(state.get("verified_candidates") or []) < 3:
            return True
        return False

    def _merge_critic_result(
        self,
        *,
        hard_patch: dict,
        critique: dict[str, Any],
        client: "CriticReasoningClient",
        state: dict,
        cache_hit: bool = False,
    ) -> ComponentResult:
        hard_decision = str(hard_patch.get("critic_decision") or "")
        hard_next_action = str(hard_patch.get("critic_next_action") or "")
        proposed_decision = str(critique.get("decision") or "").strip().upper()
        proposed_next_action = str(critique.get("next_action") or "").strip().lower()
        proposed_reasons = [str(reason) for reason in _as_list(critique.get("reasons")) if str(reason).strip()]

        bounded = False
        patch = dict(hard_patch)
        if proposed_decision not in ALLOWED_DECISIONS:
            proposed_decision = hard_decision
            bounded = True
        if proposed_next_action not in ALLOWED_NEXT_ACTIONS:
            proposed_next_action = hard_next_action
            bounded = True

        if hard_decision != "ACCEPT":
            proposed_decision = hard_decision
            proposed_next_action = hard_next_action
            bounded = True
        elif (
            state.get("raw_candidates")
            and proposed_decision == "SEARCH_AGAIN"
            and int(state.get("search_attempts") or 0) >= 1
        ):
            # One broadened search is enough when deterministic verification
            # already accepts the current candidate set.  Do not let a second
            # ungrounded LLM search request suppress valid recommendations.
            proposed_decision = "ACCEPT"
            proposed_next_action = "respond"
            bounded = True
        elif not state.get("raw_candidates") and proposed_decision in {"RERANK", "SEARCH_AGAIN"}:
            proposed_decision = "UPDATE_STATE"
            proposed_next_action = "update_state"
            patch["retrieval_readiness"] = "needs_revision"
            bounded = True

        # The retrieval stack already performs its deterministic reranking
        # before candidates reach this critic.  There is no separate rerank
        # component to invoke here, so treating RERANK as an unresolved state
        # would strand the turn without a recommendation.  Accept the existing
        # verified ordering and retain the LLM's reason in the audit note.
        if state.get("raw_candidates") and proposed_decision == "RERANK":
            proposed_decision = "ACCEPT"
            proposed_next_action = "respond"
            bounded = True

        # An LLM critic can request an update, but its schema contains no safe
        # state patch.  With retrieved candidates, the executable recovery is
        # a bounded broader search; after the search budget it must ask the
        # user instead of silently ending the turn.
        if state.get("raw_candidates") and proposed_decision == "UPDATE_STATE":
            if int(state.get("search_attempts") or 0) < 2:
                proposed_decision = "SEARCH_AGAIN"
                proposed_next_action = "search_again"
            else:
                proposed_decision = "ASK_USER"
                proposed_next_action = "ask_user"
                patch["retrieval_readiness"] = "needs_clarification"
            bounded = True

        if proposed_reasons:
            patch["critic_reasons"] = proposed_reasons
        patch["critic_decision"] = proposed_decision
        patch["critic_next_action"] = proposed_next_action
        patch["critic_notes"] = [{
            "turn_id": state.get("turn_id"),
            "mode": "critic_reasoning",
            "provider": getattr(client, "provider", None),
            "model": getattr(client, "model", None),
            "usage": getattr(client, "last_usage", {}),
            "hard_decision": hard_decision,
            "hard_next_action": hard_next_action,
            "proposed_decision": critique.get("decision"),
            "proposed_next_action": critique.get("next_action"),
            "final_decision": proposed_decision,
            "final_next_action": proposed_next_action,
            "bounded_by_hard_verifier": bounded,
            "recovery_resolution": _recovery_resolution(
                proposed_decision=str(critique.get("decision") or "").strip().upper(),
                final_decision=proposed_decision,
            ),
            "confidence": critique.get("confidence"),
            "reasoning_budget": reasoning_depth(state),
            "adaptive_budget": adaptive_budget_enabled(),
            "reason": critique.get("reason"),
            "call_reason": self._critic_call_reason(state, hard_patch),
            "cache_hit": cache_hit,
        }]
        patch["tool_observation"] = {
            "component": self.name,
            "summary": f"critic_decision={proposed_decision}; next_action={proposed_next_action}; llm=True",
            "status": "success",
        }
        return ComponentResult(name=self.name, patch=patch)

    def _critic_call_reason(self, state: dict, hard_patch: dict) -> str:
        decision = hard_patch.get("critic_decision")
        next_action = hard_patch.get("critic_next_action")
        if decision in {"SEARCH_AGAIN", "ASK_USER", "RERANK", "UPDATE_STATE"}:
            return f"hard verifier proposed {decision}"
        if next_action in {"search_again", "rerank", "update_state", "ask_user"}:
            return f"next action requires {next_action}"
        if combined_flags(state):
            return "state contains control/preference/safety flags"
        if state.get("preference_events"):
            return "turn contains preference events"
        if state.get("raw_candidates") and len(state.get("verified_candidates") or []) < 3:
            return "candidate set needs recovery"
        return "critic mode selected LLM verification"

    def _critic_skip_reason(self, state: dict, hard_patch: dict) -> str:
        if self.critic_mode in {"0", "false", "off", "rule", "rules"}:
            return "critic mode is rule-only"
        if not adaptive_budget_enabled():
            return "critic LLM gate disabled by mode"
        if hard_patch.get("critic_decision") == "ACCEPT":
            return "hard verifier accepted a low-risk state"
        return "adaptive critic budget judged LLM verification unnecessary"

    def _critique_cache_key(self, state: dict[str, Any], hard_patch: dict[str, Any], client: "CriticReasoningClient") -> str:
        payload = {
            "conversation_id": state.get("conversation_id"),
            "provider": getattr(client, "provider", None),
            "model": getattr(client, "model", None),
            "critic_mode": self.critic_mode,
            "reasoning_depth": reasoning_depth(state),
            "latest_user_message": state.get("latest_user_message"),
            "intent": _jsonish(state.get("intent")),
            "preferences": _jsonish(state.get("preferences", {})),
            "preference_events": _jsonish(state.get("preference_events", [])[-6:]),
            "flags": combined_flags(state),
            "final_search_request": _jsonish(state.get("final_search_request")),
            "retrieval_plan": _jsonish(state.get("retrieval_plan")),
            "raw_candidates": _candidate_fingerprint(state.get("raw_candidates") or []),
            "verified_candidates": _candidate_fingerprint(state.get("verified_candidates") or []),
            "hard_patch": _jsonish(hard_patch),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _pre_retrieval_readiness(self, state: dict, request) -> tuple[str, list[str], str]:
        reasons = []
        latest = str(state.get("latest_user_message") or "").lower().strip()
        prefs = state.get("preferences") or {}
        retrieval_plan = state.get("retrieval_plan") or {}
        soft_preferences = _request_value(request, "soft_preferences") or {}
        hard_constraints = _request_value(request, "hard_constraints") or {}
        preferred = _request_value(request, "preferred") or {}
        exclusions = _request_value(request, "exclusions") or {}
        has_preference_evidence = bool(
            soft_preferences
            or hard_constraints
            or preferred.get("brands")
            or preferred.get("features")
            or exclusions.get("brands")
            or exclusions.get("features")
            or exclusions.get("products")
        )
        has_context = len(prefs) > 1 or bool(state.get("last_final_search_request"))
        asks_recommendation = any(
            phrase in latest
            for phrase in (
                "recommend",
                "show me",
                "pick",
                "find",
                "specific product",
                "gift",
                "need",
                "looking for",
                "want to buy",
            )
        )
        availability_only = latest.startswith(("do you sell", "do you have"))
        vague_continuation = (
            latest in {"anything", "any price will do", "neither", "ok", "okay", "fine"}
            or latest.startswith("any price")
            or latest.startswith("anything")
        )

        if not _request_value(request, "category"):
            return "needs_clarification", ["Product/category is missing."], "ask_user"
        if not retrieval_plan:
            return "needs_clarification", ["Retrieval plan is missing for the structured request."], "update_state"
        if not retrieval_plan.get("strategy"):
            return "needs_clarification", ["Retrieval strategy is missing."], "update_state"
        if not retrieval_plan.get("query_variants"):
            return "needs_clarification", ["Retrieval plan has no query variants."], "update_state"
        if retrieval_plan.get("strategy") not in {"lexical", "dense", "hybrid"}:
            return "needs_clarification", [f"Unsupported retrieval strategy: {retrieval_plan.get('strategy')}."], "update_state"
        if exclusions.get("brands") and not any(str(brand) in exclusions.get("brands", []) for brand in state.get("rejected_brands") or []):
            return "needs_clarification", ["Rejected brand state is inconsistent with request exclusions."], "update_state"
        agreement_issues = _request_state_agreement_issues(state, request)
        if any(issue.get("severity") == "hard" for issue in agreement_issues):
            return "needs_revision", [issue["message"] for issue in agreement_issues], "update_state"
        if vague_continuation and not has_context:
            return "needs_clarification", ["Latest turn is vague and adds no retrieval evidence."], "ask_user"
        if availability_only and not has_preference_evidence and not has_context and _is_broad_category(_request_value(request, "category")):
            return "needs_clarification", ["User asked availability for a broad category; ask one clarification before retrieval."], "ask_user"
        if not has_context and not has_preference_evidence and not asks_recommendation and _is_broad_category(_request_value(request, "category")):
            return "needs_clarification", ["Only a product/category is known; one preference or purpose is needed."], "ask_user"

        reasons.append("Structured request has enough product and preference evidence for retrieval.")
        reasons.append(f"Retrieval plan strategy is {retrieval_plan.get('strategy')} with {len(retrieval_plan.get('query_variants') or [])} query variant(s).")
        for issue in agreement_issues:
            reasons.append(f"Soft agreement warning: {issue['message']}")
        if state.get("preference_events"):
            reasons.append("Preference changes or rejections are represented in the request.")
        return "ready", reasons, "retrieve"

    def _result(self, decision: str, reasons: list[str], flags: list[str], readiness: str | None = None, next_action: str | None = None) -> ComponentResult:
        return ComponentResult(
            name=self.name,
            patch={
                "critic_decision": decision,
                "critic_reasons": reasons,
                "retrieval_readiness": readiness,
                "critic_next_action": next_action,
                "security_flags": flags,
                "tool_observation": {
                    "component": self.name,
                    "summary": f"critic_decision={decision}; next_action={next_action}; reasons={len(reasons)}",
                    "status": "success",
                },
            },
        )


def _request_value(request, key: str):
    if isinstance(request, dict):
        return request.get(key)
    return getattr(request, key, None)


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0").lower() in {"1", "true", "yes", "on"}


def _request_state_agreement_issues(state: dict, request) -> list[dict[str, str]]:
    issues = []
    preferences = state.get("preferences") or {}
    query_text = str(_request_value(request, "query_text") or "").lower()
    category = str(_request_value(request, "category") or "").strip().lower()
    hard_constraints = _request_value(request, "hard_constraints") or {}
    preferred = _request_value(request, "preferred") or {}
    exclusions = _request_value(request, "exclusions") or {}

    product_type = _pref_value(preferences, "product_type")
    if product_type and category and not _category_compatible(product_type, category):
        issues.append({
            "severity": "hard",
            "message": "Search request category does not match current product_type memory.",
        })

    if category and not _phrase_in_text(category, query_text):
        issues.append({
            "severity": "soft",
            "message": "Search request query does not include the selected product/category.",
        })

    budget = _pref_value(preferences, "budget_max")
    request_budget = hard_constraints.get("budget_max")
    if budget is not None and request_budget != budget:
        issues.append({
            "severity": "hard",
            "message": "Search request budget constraint does not match current memory.",
        })

    state_features = {_norm(feature) for feature in state.get("preferred_features") or [] if _norm(feature)}
    request_features = {_norm(feature) for feature in preferred.get("features") or [] if _norm(feature)}
    if state_features - request_features:
        issues.append({
            "severity": "soft",
            "message": "Search request is missing one or more preferred features from memory.",
        })

    state_rejected_brands = {_norm(brand) for brand in state.get("rejected_brands") or [] if _norm(brand)}
    request_rejected_brands = {_norm(brand) for brand in exclusions.get("brands") or [] if _norm(brand)}
    if state_rejected_brands - request_rejected_brands:
        issues.append({
            "severity": "hard",
            "message": "Search request is missing one or more rejected brands from memory.",
        })

    return issues


def _pref_value(preferences: dict, slot: str):
    item = preferences.get(slot)
    if isinstance(item, dict):
        return item.get("value")
    return getattr(item, "value", None)


def _norm(value) -> str:
    return " ".join(str(value or "").lower().strip().split())


def _phrase_in_text(phrase: str, text: str) -> bool:
    phrase = _norm(phrase)
    text = _norm(text)
    if not phrase:
        return True
    return f" {phrase} " in f" {text} "


def _category_compatible(product_type, category: str) -> bool:
    product = _norm(product_type)
    category = _norm(category)
    if not product or not category:
        return True
    if product == category:
        return True
    product_tokens = set(product.split())
    category_tokens = set(category.split())
    if category_tokens and category_tokens.issubset(product_tokens):
        return True
    if product_tokens and product_tokens.issubset(category_tokens):
        return True
    singular_product = {token[:-1] if token.endswith("s") else token for token in product_tokens}
    singular_category = {token[:-1] if token.endswith("s") else token for token in category_tokens}
    return bool(singular_product & singular_category)


def _is_broad_category(category) -> bool:
    text = _norm(category)
    if not text:
        return True
    broad = {
        "book",
        "books",
        "coffee",
        "gift",
        "gifts",
        "item",
        "items",
        "lamp",
        "product",
        "products",
        "shoes",
        "shoe",
        "toy",
        "toys",
        "watch",
        "watches",
    }
    return text in broad


class CriticReasoningClient:
    """Optional LLM critic that proposes bounded verification decisions."""

    def __init__(self) -> None:
        self.provider = os.getenv("AUTONOMOUS_CRITIC_PROVIDER", os.getenv("AUTONOMOUS_PLANNER_PROVIDER", "ollama")).lower()
        self.model = os.getenv("AUTONOMOUS_CRITIC_MODEL", os.getenv("AUTONOMOUS_PLANNER_MODEL", "llama3.2:3b"))
        self.timeout_seconds = float(os.getenv("AUTONOMOUS_CRITIC_TIMEOUT", "45"))
        self.ollama_endpoint = os.getenv("AUTONOMOUS_CRITIC_OLLAMA_URL", "http://localhost:11434/api/generate")
        self.deepseek_endpoint = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
        self.deepseek_thinking = os.getenv("AUTONOMOUS_CRITIC_DEEPSEEK_THINKING", "disabled").lower()
        self.last_usage: dict[str, Any] = {}

    def critique(self, state: dict[str, Any], hard_patch: dict[str, Any]) -> dict[str, Any]:
        prompt = _critic_prompt(_compact_critic_state(state), _jsonish(hard_patch))
        if self.provider == "deepseek":
            return self._critique_deepseek(prompt)
        return self._critique_ollama(prompt)

    def _critique_ollama(self, prompt: str) -> dict[str, Any]:
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
            raise RuntimeError("Critic Ollama endpoint is unavailable") from exc
        self.last_usage = {}
        return _parse_json_object(raw.get("response", ""))

    def _critique_deepseek(self, prompt: str) -> dict[str, Any]:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a bounded critic agent. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 500,
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
            raise RuntimeError(f"DeepSeek critic HTTP {exc.code}: {body[:240]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("DeepSeek critic endpoint is unavailable") from exc
        self.last_usage = raw.get("usage") or {}
        content = raw["choices"][0]["message"].get("content") or ""
        return _parse_json_object(content)


def _critic_prompt(state: dict[str, Any], hard_verifier: dict[str, Any]) -> str:
    return f"""You are the Critic & Verification Agent in an autonomous product recommender.

Your job is to critique the current workflow state. Do not recommend products.
Return ONLY one valid JSON object. No markdown, prose, or chain-of-thought.

You may propose only these decisions: ACCEPT, RERANK, SEARCH_AGAIN, UPDATE_STATE, ASK_USER.
You may propose only these next_action values: retrieve, ask_user, search_again, rerank, update_state, respond.

Use the hard_verifier as a safety boundary. If hard_verifier reports missing category, missing retrieval plan,
invalid strategy, or failed hard constraints, do not override it to ACCEPT.

JSON shape:
{{
  "decision": "ACCEPT|RERANK|SEARCH_AGAIN|UPDATE_STATE|ASK_USER",
  "next_action": "retrieve|ask_user|search_again|rerank|update_state|respond",
  "confidence": 0.0,
  "reasons": ["short reason"],
  "reason": "one short summary"
}}

State:
{json.dumps(state, separators=(",", ":"))}

Hard verifier:
{json.dumps(hard_verifier, separators=(",", ":"))}
"""


def _compact_critic_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": state.get("turn_id"),
        "latest_user_message": state.get("latest_user_message"),
        "intent": state.get("intent"),
        "preferences": _jsonish(state.get("preferences", {})),
        "preference_events": _jsonish(state.get("preference_events", [])[-6:]),
        "control_flags": state.get("control_flags", []),
        "preference_flags": state.get("preference_flags", []),
        "safety_flags": state.get("safety_flags", []),
        "unclassified_flags": state.get("unclassified_flags", []),
        "security_flags": combined_flags(state),
        "final_search_request": _jsonish(state.get("final_search_request")),
        "retrieval_plan": _jsonish(state.get("retrieval_plan")),
        "raw_candidate_count": len(state.get("raw_candidates") or []),
        "verified_candidate_count": len(state.get("verified_candidates") or []),
    }


def _recovery_resolution(*, proposed_decision: str, final_decision: str) -> str | None:
    if proposed_decision == final_decision:
        return None
    if proposed_decision == "RERANK" and final_decision == "ACCEPT":
        return "rerank_resolved_by_existing_deterministic_reranker"
    if proposed_decision == "SEARCH_AGAIN" and final_decision == "ACCEPT":
        return "repeated_search_again_resolved_by_accepted_candidates"
    if proposed_decision == "UPDATE_STATE" and final_decision == "SEARCH_AGAIN":
        return "update_state_resolved_as_bounded_search_again"
    if proposed_decision == "UPDATE_STATE" and final_decision == "ASK_USER":
        return "update_state_budget_exhausted_asked_user"
    return "critic_decision_bounded_by_execution_contract"


def _parse_json_object(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Critic model did not return a JSON object")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Critic model JSON was not an object")
    return parsed


def _jsonish(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _jsonish(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonish(item) for item in value]
    return value


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _candidate_fingerprint(candidates: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    fingerprint = []
    for candidate in candidates[:limit]:
        fingerprint.append({
            "id": candidate.get("product_id") or candidate.get("id") or candidate.get("asin") or candidate.get("title"),
            "score": candidate.get("score"),
            "rerank_score": candidate.get("rerank_score"),
            "evidence_score": candidate.get("evidence_score"),
            "verified": candidate.get("verified"),
        })
    return fingerprint
