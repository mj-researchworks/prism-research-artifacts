"""Preference and reflection specialist for autonomous ConServe."""

from __future__ import annotations

import os
import re
import json
import hashlib
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.autonomous.agents.reasoning_reflection import ReasoningReflectionClient
from src.autonomous.flags import classify_flags
from src.autonomous.reasoning_budget import adaptive_budget_enabled, latest_turn_analysis, reasoning_depth, reflection_llm_needed
from src.autonomous.schemas import ComponentResult


FALLBACK_BRANDS = (
    "hp",
    "dell",
    "lenovo",
    "acer",
    "asus",
    "apple",
    "msi",
    "samsung",
    "sony",
    "logitech",
    "nike",
    "adidas",
    "new balance",
)
NON_BRAND_REJECTION_TERMS = {
    "leather",
    "style",
    "color",
    "material",
    "heavy",
    "expensive",
    "large",
    "small",
    "freezer",
    "freezers",
    "touchscreen",
}

CATALOG_BRANDS_PATH = Path(__file__).resolve().parents[2] / "taxonomy" / "catalog_brands.json"


def _without_value(values: list, value: str) -> list:
    target = str(value).casefold()
    return [item for item in values if str(item).casefold() != target]


class PreferenceReflectionAgent:
    name = "preference_reflection_agent"

    def __init__(self, reasoning_client: ReasoningReflectionClient | None = None, reflection_mode: str | None = None):
        self.reflection_mode = (reflection_mode or os.getenv("AUTONOMOUS_REFLECTION_MODE", "rule")).lower()
        self.reasoning_client = reasoning_client
        self._reasoning_cache: dict[str, dict[str, Any]] = {}

    def run(self, state: dict) -> ComponentResult:
        text = state.get("latest_user_message", "")
        turn_id = state.get("turn_id", 0)
        lowered = text.lower().strip()
        existing_preferences = state.get("preferences", {})
        previous_recommendations = state.get("previous_recommendations") or state.get("last_recommendations") or []

        preferences = {}
        events = []
        flags = []
        rejected_brands = list(state.get("rejected_brands") or [])
        rejected_products = list(state.get("rejected_products") or [])
        rejected_product_types = list(state.get("rejected_product_types") or [])
        rejected_features = list(state.get("rejected_features") or [])
        preferred_brands = list(state.get("preferred_brands") or [])
        preferred_features = list(state.get("preferred_features") or [])
        accepted_products = list(state.get("accepted_products") or [])
        base_rejected_brands = list(rejected_brands)
        base_rejected_product_types = list(rejected_product_types)
        base_rejected_features = list(rejected_features)
        base_preferred_brands = list(preferred_brands)
        base_preferred_features = list(preferred_features)
        base_accepted_products = list(accepted_products)

        if self._uses_agent_primary_reflection() and not _env_enabled("AUTONOMOUS_REFLECTION_AGENT_RULE_SUPPORT"):
            return self._run_agent_primary_reflection(
                state=state,
                turn_id=turn_id,
                existing_preferences=existing_preferences,
                rejected_brands=rejected_brands,
                rejected_products=rejected_products,
                rejected_product_types=rejected_product_types,
                rejected_features=rejected_features,
                preferred_brands=preferred_brands,
                preferred_features=preferred_features,
                accepted_products=accepted_products,
            )

        accepted_product = self._extract_accepted_product(lowered, previous_recommendations)
        if accepted_product:
            product_id = accepted_product.get("product_id")
            if product_id and product_id not in accepted_products:
                accepted_products.append(product_id)
            events.append({
                "turn_id": turn_id,
                "type": "product_acceptance",
                "slot": "product",
                "old_value": None,
                "new_value": product_id or accepted_product.get("title"),
                "reason": "User accepted a specific recommended product.",
            })

        product = self._guess_product_type(lowered)
        if not product:
            product = self._extract_generic_product_type(lowered)
        if product:
            preferences["product_type"] = self._preference_patch(
                slot="product_type", value=product, turn_id=turn_id, confidence=0.78,
                existing_preferences=existing_preferences, reason="Detected product/category phrase in latest user turn."
            )

        budget = self._extract_budget(lowered)
        if budget is not None:
            preferences["budget_max"] = self._preference_patch(
                slot="budget_max", value=budget, turn_id=turn_id, confidence=0.88,
                existing_preferences=existing_preferences, reason="Detected maximum budget in latest user turn."
            )

        occasion = self._extract_occasion(lowered)
        if occasion:
            preferences["occasion"] = self._preference_patch(
                slot="occasion", value=occasion, turn_id=turn_id, confidence=0.85,
                existing_preferences=existing_preferences, reason="Detected shopping occasion."
            )

        recipient = self._extract_recipient(lowered)
        if recipient:
            preferences["recipient"] = self._preference_patch(
                slot="recipient", value=recipient, turn_id=turn_id, confidence=0.86,
                existing_preferences=existing_preferences, reason="Detected recipient."
            )

        use_case = self._extract_use_case(lowered)
        if use_case:
            preferences["use_case"] = self._preference_patch(
                slot="use_case", value=use_case, turn_id=turn_id, confidence=0.84,
                existing_preferences=existing_preferences, reason="Detected use-case signal in latest user turn."
            )

        rejected_brand = self._extract_rejected_brand(lowered, previous_recommendations)
        wanted_brands = [
            brand for brand in self._extract_preferred_brands(lowered)
            if not rejected_brand or brand.casefold() != rejected_brand.casefold()
        ]
        for brand in wanted_brands:
            rejected_brands = _without_value(rejected_brands, brand)
            if brand not in preferred_brands:
                preferred_brands.append(brand)
                events.append({
                    "turn_id": turn_id,
                    "type": "brand_preference",
                    "slot": "brand",
                    "old_value": None,
                    "new_value": brand,
                    "reason": "User expressed a brand preference.",
                })

        wanted_features = self._extract_preferred_features(lowered)
        for feature in wanted_features:
            rejected_features = _without_value(rejected_features, feature)
            if feature not in preferred_features:
                preferred_features.append(feature)
                events.append({
                    "turn_id": turn_id,
                    "type": "feature_preference",
                    "slot": "feature",
                    "old_value": None,
                    "new_value": feature,
                    "reason": "User expressed a feature preference.",
                })

        if rejected_brand:
            preferred_brands = _without_value(preferred_brands, rejected_brand)
        if rejected_brand and rejected_brand not in rejected_brands:
            rejected_brands.append(rejected_brand)
            flags.append("preference_instability_signal")
            events.append({
                "turn_id": turn_id,
                "type": "brand_rejection",
                "slot": "brand",
                "old_value": rejected_brand,
                "new_value": f"exclude:{rejected_brand}",
                "reason": "User rejected a brand or recommendation associated with that brand.",
            })

        if self._is_direct_rejection(lowered) and previous_recommendations:
            rejected_ids = []
            for rejected in previous_recommendations:
                product_id = rejected.get("product_id")
                if product_id:
                    rejected_ids.append(product_id)
                    if product_id not in rejected_products:
                        rejected_products.append(product_id)
            flags.append("preference_instability_signal")
            events.append({
                "turn_id": turn_id,
                "type": "product_rejection",
                "slot": "product",
                "old_value": rejected_ids,
                "new_value": "exclude_previous_recommendations",
                "reason": "User asked for different options, so previous recommendations were excluded.",
            })

        rejected_feature = self._extract_rejected_feature(lowered)
        if rejected_feature:
            preferred_features = _without_value(preferred_features, rejected_feature)
        if rejected_feature and rejected_feature not in rejected_features:
            rejected_features.append(rejected_feature)
            flags.append("preference_instability_signal")
            events.append({
                "turn_id": turn_id,
                "type": "feature_rejection",
                "slot": "feature",
                "old_value": rejected_feature,
                "new_value": f"exclude:{rejected_feature}",
                "reason": "User rejected a feature in the latest turn.",
            })

        for slot, item in preferences.items():
            if item["status"] == "revised":
                flags.append("preference_instability_signal")
                events.append({
                    "turn_id": turn_id,
                    "type": "preference_revision",
                    "slot": slot,
                    "old_value": item.get("previous_value"),
                    "new_value": item.get("value"),
                    "reason": item.get("reason", "User revised a stored preference."),
                })

        if not product and "gift" in lowered and not _pref_value(existing_preferences, "product_type"):
            flags.append("underspecified_product_type")

        reflection_notes = []
        reflection_failed = False
        reflected_intent = None
        reasoning_call_reason = self._reasoning_call_reason(state, preferences, events, flags)
        agent_primary = self._uses_agent_primary_reflection()
        if self.reflection_mode in {"llm", "reasoning", "always"} or agent_primary:
            reasoning_call_reason = "reflection mode requires the specialized reflection agent"
        if self._should_call_reasoning(state, preferences, events, flags):
            try:
                client = self.reasoning_client or ReasoningReflectionClient()
                if agent_primary:
                    rule_draft = _reflection_agent_evidence(
                        flags=flags,
                        accepted_products=accepted_products,
                    )
                else:
                    rule_draft = _reflection_rule_draft(
                        preferences=preferences,
                        events=events,
                        flags=flags,
                        rejected_brands=rejected_brands,
                        rejected_product_types=rejected_product_types,
                        rejected_features=rejected_features,
                        preferred_brands=preferred_brands,
                        preferred_features=preferred_features,
                    )
                draft_flags = list(flags)
                reflection = client.reflect(state, rule_draft)
                if agent_primary:
                    reflection = _sanitize_agent_reflection(reflection, state)
                reflected_intent = _intent_from_reflection(reflection, state)
                if agent_primary:
                    preferences = {}
                    events = []
                    flags = []
                    rejected_brands = list(base_rejected_brands)
                    rejected_products = list(rejected_products)
                    rejected_product_types = list(base_rejected_product_types)
                    rejected_features = list(base_rejected_features)
                    preferred_brands = list(base_preferred_brands)
                    preferred_features = list(base_preferred_features)
                    accepted_products = list(base_accepted_products)
                self._merge_reasoning_reflection(
                    reflection=reflection,
                    state=state,
                    turn_id=turn_id,
                    existing_preferences=existing_preferences,
                    preferences=preferences,
                    events=events,
                    flags=flags,
                    rejected_brands=rejected_brands,
                    rejected_products=rejected_products,
                    rejected_product_types=rejected_product_types,
                    rejected_features=rejected_features,
                    preferred_brands=preferred_brands,
                    preferred_features=preferred_features,
                    accepted_products=accepted_products,
                )
                self._preserve_control_flags_from_draft(
                    draft_flags=draft_flags,
                    reflection=reflection,
                    existing_preferences=existing_preferences,
                    preferences=preferences,
                    flags=flags,
                )
                reflection_notes.append({
                    "turn_id": turn_id,
                    "mode": "reasoning",
                    "provider": getattr(client, "provider", None),
                    "model": getattr(client, "model", None),
                    "usage": getattr(client, "last_usage", {}),
                    "preference_change": reflection.get("preference_change"),
                    "strategy": reflection.get("strategy"),
                    "confidence": reflection.get("confidence"),
                    "reason": reflection.get("reason"),
                    "call_reason": reasoning_call_reason,
                    "reasoning_budget": reasoning_depth(state),
                    "adaptive_budget": adaptive_budget_enabled(),
                    "agent_primary": agent_primary,
                    "rule_draft": rule_draft,
                    "validation_notes": reflection.get("_validation_notes", []),
                })
            except Exception as exc:
                reflection_failed = True
                reflection_notes.append({
                    "turn_id": turn_id,
                    "mode": "reasoning_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })

        flag_buckets = classify_flags(flags)
        return ComponentResult(
            name=self.name,
            patch={
                **({"intent": reflected_intent} if reflected_intent else {}),
                "preferences": preferences,
                "security_flags": list(dict.fromkeys(flags)),
                "control_flags": flag_buckets["control_flags"],
                "preference_flags": flag_buckets["preference_flags"],
                "safety_flags": flag_buckets["safety_flags"],
                "unclassified_flags": flag_buckets["unclassified_flags"],
                "preference_events": events,
                "rejected_brands": rejected_brands,
                "rejected_products": rejected_products,
                "rejected_product_types": rejected_product_types,
                "rejected_features": rejected_features,
                "preferred_brands": preferred_brands,
                "preferred_features": preferred_features,
                "accepted_products": accepted_products,
                "reflection_notes": reflection_notes,
                "tool_observation": {
                    "component": self.name,
                    "summary": f"preferences={list(preferences)} events={len(events)} flags={list(dict.fromkeys(flags))} reflection={reflection_notes[-1]['mode'] if reflection_notes else 'rule'}",
                    "status": "degraded" if reflection_failed else "success",
                },
            },
        )

    def _run_agent_primary_reflection(
        self,
        *,
        state: dict,
        turn_id: int,
        existing_preferences: dict,
        rejected_brands: list,
        rejected_products: list,
        rejected_product_types: list,
        rejected_features: list,
        preferred_brands: list,
        preferred_features: list,
        accepted_products: list,
    ) -> ComponentResult:
        if self._can_skip_agent_primary_llm(state, existing_preferences):
            return self._agent_primary_skip_result(
                state=state,
                rejected_brands=rejected_brands,
                rejected_products=rejected_products,
                rejected_product_types=rejected_product_types,
                rejected_features=rejected_features,
                preferred_brands=preferred_brands,
                preferred_features=preferred_features,
                accepted_products=accepted_products,
            )

        preferences = {}
        events = []
        flags = []
        reflection_notes = []
        reflection_failed = False
        reflected_intent = None
        rule_draft = _reflection_agent_evidence(flags=[], accepted_products=accepted_products)
        reasoning_call_reason = "reflection mode requires the specialized reflection agent"
        try:
            client = self.reasoning_client or ReasoningReflectionClient()
            cache_key = self._reasoning_cache_key(state, rule_draft, client)
            cached_reflection = self._reasoning_cache.get(cache_key)
            if cached_reflection is not None:
                reflection = deepcopy(cached_reflection)
            else:
                raw_reflection = client.reflect(state, rule_draft)
                reflection = _sanitize_agent_reflection(raw_reflection, state)
                reflection['_raw_llm_output'] = deepcopy(raw_reflection)
                self._reasoning_cache[cache_key] = deepcopy(reflection)
            reflected_intent = _intent_from_reflection(reflection, state)
            self._merge_reasoning_reflection(
                reflection=reflection,
                state=state,
                turn_id=turn_id,
                existing_preferences=existing_preferences,
                preferences=preferences,
                events=events,
                flags=flags,
                rejected_brands=rejected_brands,
                rejected_products=rejected_products,
                rejected_product_types=rejected_product_types,
                rejected_features=rejected_features,
                preferred_brands=preferred_brands,
                preferred_features=preferred_features,
                accepted_products=accepted_products,
            )
            self._preserve_agent_control_flags(
                reflection=reflection,
                existing_preferences=existing_preferences,
                preferences=preferences,
                flags=flags,
            )
            self._recover_product_type_from_turn(
                state=state,
                turn_id=turn_id,
                existing_preferences=existing_preferences,
                preferences=preferences,
                events=events,
                flags=flags,
            )
            reflection_notes.append({
                "turn_id": turn_id,
                "mode": "reasoning",
                "provider": getattr(client, "provider", None),
                "model": getattr(client, "model", None),
                "usage": getattr(client, "last_usage", {}),
                "preference_change": reflection.get("preference_change"),
                "strategy": reflection.get("strategy"),
                "confidence": reflection.get("confidence"),
                "reason": reflection.get("reason"),
                "call_reason": reasoning_call_reason,
                "reasoning_budget": reasoning_depth(state),
                "adaptive_budget": adaptive_budget_enabled(),
                "agent_primary": True,
                "rule_draft": rule_draft,
                "validation_notes": reflection.get("_validation_notes", []),
                "raw_llm_output": reflection.get("_raw_llm_output"),
                "validated_output": {k: deepcopy(v) for k, v in reflection.items() if not k.startswith('_')},
                "cache_hit": cached_reflection is not None,
            })
        except Exception as exc:
            reflection_failed = True
            self._recover_product_type_from_turn(
                state=state,
                turn_id=turn_id,
                existing_preferences=existing_preferences,
                preferences=preferences,
                events=events,
                flags=flags,
            )
            reflection_notes.append({
                "turn_id": turn_id,
                "mode": "reasoning_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "agent_primary": True,
                "rule_draft": rule_draft,
            })

        flag_buckets = classify_flags(flags)
        return ComponentResult(
            name=self.name,
            patch={
                **({"intent": reflected_intent} if reflected_intent else {}),
                "preferences": preferences,
                "security_flags": list(dict.fromkeys(flags)),
                "control_flags": flag_buckets["control_flags"],
                "preference_flags": flag_buckets["preference_flags"],
                "safety_flags": flag_buckets["safety_flags"],
                "unclassified_flags": flag_buckets["unclassified_flags"],
                "preference_events": events,
                "rejected_brands": rejected_brands,
                "rejected_products": rejected_products,
                "rejected_product_types": rejected_product_types,
                "rejected_features": rejected_features,
                "preferred_brands": preferred_brands,
                "preferred_features": preferred_features,
                "accepted_products": accepted_products,
                "reflection_notes": reflection_notes,
                "tool_observation": {
                    "component": self.name,
                    "summary": f"preferences={list(preferences)} events={len(events)} flags={list(dict.fromkeys(flags))} reflection={reflection_notes[-1]['mode'] if reflection_notes else 'none'}",
                    "status": "degraded" if reflection_failed else "success",
                },
            },
        )

    def _reasoning_cache_key(self, state: dict[str, Any], rule_draft: dict[str, Any], client: ReasoningReflectionClient) -> str:
        payload = {
            "conversation_id": state.get("conversation_id"),
            "provider": getattr(client, "provider", None),
            "model": getattr(client, "model", None),
            "reflection_mode": self.reflection_mode,
            "reasoning_depth": reasoning_depth(state),
            "latest_user_message": state.get("latest_user_message"),
            "intent": _jsonish(state.get("intent")),
            "preferences": _jsonish(state.get("preferences", {})),
            "previous_recommendation_ids": _product_ids(state.get("previous_recommendations") or state.get("last_recommendations") or []),
            "rejected_brands": state.get("rejected_brands") or [],
            "rejected_products": state.get("rejected_products") or [],
            "rejected_product_types": state.get("rejected_product_types") or [],
            "rejected_features": state.get("rejected_features") or [],
            "preferred_brands": state.get("preferred_brands") or [],
            "preferred_features": state.get("preferred_features") or [],
            "accepted_products": state.get("accepted_products") or [],
            "last_final_search_request": _jsonish(state.get("last_final_search_request")),
            "rule_draft": _jsonish(rule_draft),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _can_skip_agent_primary_llm(self, state: dict, existing_preferences: dict) -> bool:
        if not (adaptive_budget_enabled() or _env_enabled("AUTONOMOUS_REFLECTION_SELECTIVE")):
            return False
        if not _pref_value(existing_preferences, "product_type") and not state.get("last_final_search_request"):
            return False
        text = f" {str(state.get('latest_user_message') or '').lower()} "
        if _has_memory_change_signal(text):
            return False
        intent = state.get("intent") or {}
        if (
            intent.get("requires_state_update") is False
            and intent.get("requires_retrieval_decision") is False
            and intent.get("label") in {"inquire", "chitchat"}
        ):
            return True
        depth = reasoning_depth(state)
        if depth in {"REFLECT", "DEEP"}:
            return False
        analysis = latest_turn_analysis(state)
        turn_type = str(analysis.get("turn_type") or "").lower()
        preference_change = str(analysis.get("preference_change") or "").lower()
        if turn_type in {"new_request", "clarification", "refinement", "rejection", "acceptance"}:
            return False
        if preference_change in {"new_preference", "revision", "rejection", "acceptance", "uncertain"}:
            return False
        stable_followup = turn_type in {"ask_detail", "action_followup", "other"} or any(
            phrase in text
            for phrase in (
                " show me ",
                " picture ",
                " image ",
                " details ",
                " compare ",
                " why ",
                " more like ",
            )
        )
        return stable_followup

    def _agent_primary_skip_result(
        self,
        *,
        state: dict,
        rejected_brands: list,
        rejected_products: list,
        rejected_product_types: list,
        rejected_features: list,
        preferred_brands: list,
        preferred_features: list,
        accepted_products: list,
    ) -> ComponentResult:
        note = {
            "turn_id": state.get("turn_id"),
            "mode": "reasoning_skipped",
            "provider": None,
            "model": None,
            "usage": {},
            "preference_change": "none",
            "strategy": "reuse_memory",
            "confidence": 0.8,
            "reason": "Planner analysis indicates a stable follow-up; existing shopping memory can be reused.",
            "call_reason": "adaptive reflection budget skipped specialized LLM call",
            "reasoning_budget": reasoning_depth(state),
            "adaptive_budget": adaptive_budget_enabled(),
            "agent_primary": True,
        }
        return ComponentResult(
            name=self.name,
            patch={
                "preferences": {},
                "security_flags": [],
                "control_flags": [],
                "preference_flags": [],
                "safety_flags": [],
                "unclassified_flags": [],
                "preference_events": [],
                "rejected_brands": rejected_brands,
                "rejected_products": rejected_products,
                "rejected_product_types": rejected_product_types,
                "rejected_features": rejected_features,
                "preferred_brands": preferred_brands,
                "preferred_features": preferred_features,
                "accepted_products": accepted_products,
                "reflection_notes": [note],
                "tool_observation": {
                    "component": self.name,
                    "summary": "preferences=[] events=0 flags=[] reflection=reasoning_skipped",
                    "status": "success",
                },
            },
        )

    def _preserve_agent_control_flags(
        self,
        *,
        reflection: dict[str, Any],
        existing_preferences: dict,
        preferences: dict,
        flags: list[str],
    ) -> None:
        has_existing_product = bool(_pref_value(existing_preferences, "product_type"))
        product_update = preferences.get("product_type")
        has_new_product = bool(product_update and product_update.get("value"))
        needs_clarification = bool(reflection.get("needs_clarification")) or reflection.get("strategy") == "ask_clarification"
        if not has_existing_product and not has_new_product and needs_clarification:
            flags.append("underspecified_product_type")

    def _preference_patch(self, slot, value, turn_id, confidence, existing_preferences, reason):
        previous = _pref_value(existing_preferences, slot)
        status = "revised" if previous is not None and previous != value else "new"
        return {
            "value": value,
            "status": status,
            "confidence": confidence,
            "source_turn": turn_id,
            "previous_value": previous,
            "reason": reason,
        }

    def _guess_product_type(self, text: str) -> str | None:
        aliases = (
            ("headphones", ("headphones", "headphone", "earbuds", "ear buds")),
            ("headset", ("headset", "headsets")),
            ("laptop", ("laptop", "laptops", "notebook computer")),
            ("phone", ("phone", "phones", "smartphone", "cell phone")),
            ("monitor", ("monitor", "monitors")),
            ("mouse pad", ("mouse pad", "mousepad")),
            ("mouse", ("mouse", "mice")),
            ("fridge", ("fridge", "refrigerator")),
            ("shoes", ("shoes", "shoe", "sneakers")),
            ("watch", ("watch", "watches")),
            ("toy", ("toy", "toys")),
            ("book", ("book", "books")),
        )
        for canonical, terms in aliases:
            for term in terms:
                if _has_phrase(text, term):
                    return canonical
        return None

    def _extract_generic_product_type(self, text: str) -> str | None:
        text = text.strip(" .?!,;:")
        if self._is_non_product_followup(text):
            return None

        patterns = (
            r"(?:i would like to buy|i'd like to buy|would like to buy|i want to buy|i wanna buy|want to buy|i want buy|want buy|i would like|i'd like|i need|need|i want|i wanna|want)\s+(?:some\s+|a\s+|an\s+|the\s+)?([a-z0-9][a-z0-9& .'-]{2,80})",
            r"(?:do you sell|do you have|show me|recommend|looking for|look for)\s+(?:some\s+|a\s+|an\s+|the\s+)?([a-z0-9][a-z0-9& .'-]{2,60})",
            r"^(?:some\s+|a\s+|an\s+|the\s+)?([a-z0-9][a-z0-9& .'-]{2,40})$",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            candidate = self._clean_product_phrase(match.group(1))
            if candidate:
                return candidate
        return None

    def _clean_product_phrase(self, phrase: str) -> str | None:
        phrase = phrase.strip(" .?!,;:")
        phrase = re.sub(r"\s+", " ", phrase)
        phrase = re.sub(r"\b(?:please|pls|thanks|thank you|ok|okay)\b.*$", "", phrase).strip()
        phrase = re.sub(r"\b(?:for me|for my choice|for recommendation|recommendations?)\b.*$", "", phrase).strip()
        phrase = re.sub(r"\b(?:with|under|below|less than|around|up to|to give|to make|to clean|that|which|what)\b.*$", "", phrase).strip()
        phrase = phrase.strip(" .?!,;:")
        if not phrase:
            return None
        if _brand_question_without_new_product(phrase):
            return None
        if _is_modifier_or_followup_phrase(phrase) or _looks_like_brand_only(phrase):
            return None
        if phrase in {"any", "one", "ones", "anything", "something", "stuff", "goods", "product", "products", "recommendation", "recommendations"}:
            return None
        if phrase.endswith(" one") or phrase.endswith(" ones"):
            return None
        if len(phrase.split()) > 6:
            return None
        return phrase

    def _is_non_product_followup(self, text: str) -> bool:
        stripped = text.strip(" .?!,;:")
        if _is_modifier_or_followup_phrase(stripped) or _looks_like_brand_only(stripped):
            return True
        if re.match(r"^(?:not|no|avoid|don't|do not)\b", stripped):
            return True
        if re.match(r"^(?:under|below|less than|up to|around|cheaper than)\s+\$?\d", stripped):
            return True
        if stripped in {
            "ok",
            "okay",
            "fine",
            "yes",
            "no",
            "thanks",
            "thank you",
            "anything",
            "any price will do",
            "neither",
        }:
            return True
        if "gift for" in stripped or stripped.startswith("gift"):
            return True
        if stripped.startswith("something ") and any(
            _has_phrase(stripped, feature)
            for feature in ("lightweight", "cheap", "cheape", "cheaper", "affordable", "portable", "wireless", "electric", "fast")
        ):
            return True
        if any(
            phrase in stripped
            for phrase in (
                "this one",
                "that one",
                "these",
                "those",
                "my choice",
                "this works",
                "that works",
                "works for me",
                "looks good",
                "sounds good",
                "i like it",
                "i like this",
            )
        ):
            return True
        return False

    def _extract_budget(self, text: str) -> int | None:
        match = re.search(r"(?:under|below|less than|up to|around|cheaper than)?\s*\$?\s*(\d{2,5})", text)
        if match and any(word in text for word in ("under", "below", "less than", "up to", "cheaper than", "$")):
            return int(match.group(1))
        return None

    def _extract_occasion(self, text: str) -> str | None:
        if _has_phrase(text, "gift") or _has_phrase(text, "birthday"):
            return "gift"
        return None

    def _extract_recipient(self, text: str) -> str | None:
        for recipient in ("son", "daughter", "kid", "kids", "student", "mother", "father", "friend"):
            if _has_phrase(text, recipient):
                return "child" if recipient in {"kid", "kids"} else recipient
        return None

    def _extract_use_case(self, text: str) -> str | None:
        if any(phrase in text for phrase in ("this works", "that works", "works for me")):
            return None
        for term in ("gaming", "school", "travel", "video editing", "programming", "college"):
            if _has_phrase(text, term):
                return "school" if term == "college" else term
        if _has_phrase(text, "work") or _has_phrase(text, "office"):
            return "work"
        return None

    def _extract_preferred_brands(self, text: str) -> list[str]:
        matches = []
        brand_list_turn = "," in text or " or " in f" {text} " or text.strip(" .?!,;:").endswith(" brand")
        if brand_list_turn or any(signal in text for signal in ("prefer", "preferably", "like", "want", "same brand")):
            for brand in _catalog_brands():
                match = _phrase_match(text, brand)
                if match:
                    matches.append((match.start(), _canonical_brand(brand)))
        ordered = []
        for _index, brand in sorted(matches, key=lambda item: item[0]):
            if brand not in ordered:
                ordered.append(brand)
        return ordered

    def _extract_preferred_features(self, text: str) -> list[str]:
        features = []
        feature_aliases = {
            "lightweight": ("lightweight", "light weight", "portable"),
            "touchscreen": ("touchscreen", "touch screen"),
            "long battery": ("long battery", "battery life"),
            "fast": ("fast", "powerful"),
            "cheap": ("cheap", "cheape", "cheaper", "affordable", "budget", "low cost", "less expensive"),
            "electric": ("electric",),
            "wireless": ("wireless", "bluetooth"),
            "large": ("large", "big", "bigger"),
            "women": ("women's", "womens", "for women", "women"),
            "men": ("men's", "mens", "for men", "men"),
        }
        for canonical, aliases in feature_aliases.items():
            if any(_has_phrase(text, alias) for alias in aliases):
                features.append(canonical)
        return features

    def _extract_rejected_brand(self, text: str, previous_recommendations: list) -> str | None:
        explicit = re.search(r"(?:don't like|do not like|not|no|avoid)\s+([a-z0-9][a-z0-9& .-]{1,40})", text)
        if explicit:
            candidate = explicit.group(1).strip(" .")
            candidate = re.split(r"(?:,|;|\.|\b(?:but|instead|show|recommend|give|with|or)\b)", candidate, maxsplit=1)[0].strip(" .,")
            candidate = re.sub(r"\b(?:brand|one|this|that|option|please)\b.*$", "", candidate).strip()
            candidate = re.sub(r"^(?:not|no|avoid|dont like|don't like|do not like)\s+", "", candidate).strip()
            if candidate.startswith(("have ", "need ", "want ", "looking ", "official ", "prescription ", "any ")):
                return None
            for brand in _catalog_brands():
                if _has_phrase(candidate, brand):
                    return _canonical_brand(brand)
            if candidate and candidate not in {"this", "that", "one", "it"} and candidate not in NON_BRAND_REJECTION_TERMS:
                return _canonical_brand(candidate)
        for product in previous_recommendations:
            brand = str(product.get("brand") or "").strip()
            if brand and brand.lower() in text and any(signal in text for signal in ("not", "avoid", "don't like", "do not like", "no ")):
                return _canonical_brand(brand)
        return None

    def _is_direct_rejection(self, text: str) -> bool:
        if text.strip() in {"no", "nope"}:
            return True
        return any(phrase in text for phrase in ("not this", "not that", "don't like this", "do not like this", "different option", "show different"))

    def _extract_rejected_feature(self, text: str) -> str | None:
        match = re.search(r"(?:not|no|avoid|don't want|do not want|without)\s+(?:too\s+)?(heavy|expensive|large|small|gaming|touchscreen|leather|freezer)", text)
        return match.group(1) if match else None

    def _extract_accepted_product(self, text: str, previous_recommendations: list) -> dict | None:
        if not previous_recommendations:
            return None
        ordinal_map = {
            "second": 1, "2": 1, "two": 1,
            "third": 2, "3": 2, "three": 2,
            "first": 0, "1": 0, "one": 0,
        }
        if any(phrase in text for phrase in ("this works", "looks good", "sounds good", "i like it", "take it")):
            return previous_recommendations[0]
        for word, index in ordinal_map.items():
            if _has_phrase(text, word) and any(signal in text for signal in ("like", "choose", "take", "select", "go with", "works")):
                if index < len(previous_recommendations):
                    return previous_recommendations[index]
        return None

    def _should_call_reasoning(self, state: dict, preferences: dict, events: list, flags: list) -> bool:
        if self.reflection_mode in {"0", "false", "off", "rule", "rules"}:
            return False
        if adaptive_budget_enabled():
            return reflection_llm_needed(
                state=state,
                preferences=preferences,
                events=events,
                flags=flags,
                rule_reason=self._reasoning_call_reason(state, preferences, events, flags),
            )
        if self.reflection_mode in {"llm", "reasoning", "always"} or self._uses_agent_primary_reflection():
            if self._uses_agent_primary_reflection() and _env_enabled("AUTONOMOUS_REFLECTION_SELECTIVE"):
                return self._agent_reasoning_needed(state, preferences, events, flags)
            return True

        return self._reasoning_call_reason(state, preferences, events, flags) is not None

    def _uses_agent_primary_reflection(self) -> bool:
        return self.reflection_mode in {"agent", "autonomous", "full_llm", "llm_agent"}

    def _reasoning_call_reason(self, state: dict, preferences: dict, events: list, flags: list) -> str | None:
        text = f" {str(state.get('latest_user_message') or '').lower()} "
        existing_preferences = state.get("preferences") or {}
        existing_product = _pref_value(existing_preferences, "product_type")
        drafted_product = preferences.get("product_type", {}).get("value")
        has_previous_recommendations = bool(state.get("previous_recommendations") or state.get("last_final_search_request"))

        if existing_product and drafted_product and str(existing_product).lower() != str(drafted_product).lower():
            return "product/category revision needs semantic validation"

        known_brand_rejection = any(event.get("type") == "brand_rejection" for event in events)
        known_feature_rejection = any(event.get("type") == "feature_rejection" for event in events)
        direct_product_rejection = any(
            phrase in text
            for phrase in (" don't want ", " do not want ", " no longer want ", " not looking for ", " not interested in ")
        )
        if direct_product_rejection and existing_product:
            return "latest turn may invalidate stored product/category"

        complex_revision = any(
            phrase in text
            for phrase in (" actually ", " instead ", " rather ", " aren’t ", " aren't ", " isnt ", " isn't ", " not for ")
        )
        if complex_revision and (existing_product or has_previous_recommendations):
            return "complex revision needs reflection"

        referential_turn = any(phrase in text for phrase in (" these ", " those ", " this one ", " that one "))
        if referential_turn and has_previous_recommendations and not any(event.get("type") == "product_acceptance" for event in events):
            return "referential feedback needs recommendation context"

        has_unknown_negation = any(phrase in text for phrase in (" not ", " no ", " avoid ", " without "))
        if has_unknown_negation and not known_brand_rejection and not known_feature_rejection:
            return "negation was not fully resolved by the rule draft"

        return None

    def _agent_reasoning_needed(self, state: dict, preferences: dict, events: list, flags: list) -> bool:
        if self._reasoning_call_reason(state, preferences, events, flags):
            return True
        meaningful_flags = set(flags) - {"underspecified_product_type"}
        if meaningful_flags:
            return True
        complex_event_types = {
            "brand_rejection",
            "product_rejection",
            "feature_rejection",
            "product_acceptance",
            "preference_revision",
            "preference_clear",
        }
        if any(event.get("type") in complex_event_types for event in events):
            return True
        text = f" {str(state.get('latest_user_message') or '').lower()} "
        if any(phrase in text for phrase in (" actually ", " instead ", " rather ", " not ", " no ", " avoid ", " without ")):
            return True
        if state.get("previous_recommendations") and any(phrase in text for phrase in (" first ", " second ", " third ", " this ", " that ", " choice ")):
            return True
        return False

    def _merge_reasoning_reflection(
        self,
        *,
        reflection: dict[str, Any],
        state: dict[str, Any],
        turn_id: int,
        existing_preferences: dict,
        preferences: dict,
        events: list,
        flags: list,
        rejected_brands: list,
        rejected_products: list,
        rejected_product_types: list,
        rejected_features: list,
        preferred_brands: list,
        preferred_features: list,
        accepted_products: list,
    ) -> None:
        confidence = _safe_confidence(reflection.get("confidence"), default=0.72)
        reason = str(reflection.get("reason") or "Reasoning reflection updated preference memory.")
        reflected_preferences = reflection.get("preferences") if isinstance(reflection.get("preferences"), dict) else {}

        # Explicit LLM deltas retire old values without inventing a rejection.
        # Omitted removal fields preserve memory, including older client outputs.
        for field, values, slot in (
            ('remove_preferred_brands', preferred_brands, 'brand'),
            ('remove_preferred_features', preferred_features, 'feature'),
            ('remove_rejected_brands', rejected_brands, 'brand_exclusion'),
            ('remove_rejected_features', rejected_features, 'feature_exclusion'),
        ):
            for value in _as_list(reflection.get(field)):
                canonical = _canonical_brand(value) if 'brands' in field else _clean_feature(value)
                if not canonical:
                    continue
                matched = any(str(item).casefold() == canonical.casefold() for item in values)
                values[:] = _without_value(values, canonical)
                if matched:
                    events.append(_event(turn_id, 'preference_withdrawal', slot, canonical, None, reason))

        for slot in _as_list(reflection.get("clear_preferences")):
            slot = str(slot or "").strip().lower()
            if slot not in {"product_type", "budget_max", "occasion", "recipient", "use_case"}:
                continue
            previous = _pref_value(existing_preferences, slot)
            if previous is None:
                continue
            preferences[slot] = {
                "value": None,
                "status": "superseded",
                "confidence": confidence,
                "source_turn": turn_id,
                "previous_value": previous,
                "reason": reason,
            }
            flags.append("preference_instability_signal")
            events.append(_event(turn_id, "preference_clear", slot, previous, None, reason))
            if slot == "product_type" and str(previous) not in rejected_product_types:
                rejected_product_types.append(str(previous))

        for slot in ("product_type", "budget_max", "occasion", "recipient", "use_case"):
            if slot in preferences and preferences[slot].get("status") == "superseded":
                continue
            value = reflected_preferences.get(slot)
            if _is_empty(value):
                continue
            value = self._normalize_reflected_preference(slot, value)
            if _is_empty(value):
                continue
            if slot == "budget_max":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
            previous = _pref_value(existing_preferences, slot)
            if previous is not None and str(previous).lower() == str(value).lower() and slot not in preferences:
                continue
            preferences[slot] = self._preference_patch(
                slot=slot,
                value=value,
                turn_id=turn_id,
                confidence=confidence,
                existing_preferences=existing_preferences,
                reason=reason,
            )

        for brand in _as_list(reflection.get("preferred_brands")):
            canonical = _canonical_brand(str(brand))
            if canonical:
                rejected_brands[:] = _without_value(rejected_brands, canonical)
            if canonical and canonical not in preferred_brands:
                preferred_brands.append(canonical)
                events.append(_event(turn_id, "brand_preference", "brand", None, canonical, reason))

        for feature in _as_list(reflection.get("preferred_features")):
            feature = _clean_feature(feature)
            if feature:
                rejected_features[:] = _without_value(rejected_features, feature)
            if feature and feature not in preferred_features:
                preferred_features.append(feature)
                events.append(_event(turn_id, "feature_preference", "feature", None, feature, reason))

        for brand in _as_list(reflection.get("rejected_brands")):
            canonical = _canonical_brand(str(brand))
            if canonical:
                preferred_brands[:] = _without_value(preferred_brands, canonical)
            if canonical and canonical not in rejected_brands:
                rejected_brands.append(canonical)
                flags.append("preference_instability_signal")
                events.append(_event(turn_id, "brand_rejection", "brand", canonical, f"exclude:{canonical}", reason))

        for feature in _as_list(reflection.get("rejected_features")):
            feature = _clean_feature(feature)
            if feature:
                preferred_features[:] = _without_value(preferred_features, feature)
            if feature and feature not in rejected_features:
                rejected_features.append(feature)
                flags.append("preference_instability_signal")
                events.append(_event(turn_id, "feature_rejection", "feature", feature, f"exclude:{feature}", reason))

        if _rejects_previous_recommendations(reflection):
            rejected_ids = _product_ids(state.get("previous_recommendations") or state.get("last_recommendations") or [])
            new_ids = [product_id for product_id in rejected_ids if product_id not in rejected_products]
            if new_ids:
                rejected_products.extend(new_ids)
                flags.append("preference_instability_signal")
                events.append(_event(
                    turn_id,
                    "product_rejection",
                    "product",
                    new_ids,
                    "exclude_previous_recommendations",
                    reason,
                ))

        if reflection.get("accepted") and "reasoning_acceptance" not in accepted_products:
            accepted_products.append("reasoning_acceptance")

        preference_change = str(reflection.get("preference_change") or "").lower()
        if preference_change in {"revision", "rejection", "acceptance", "uncertain"}:
            flags.append("preference_instability_signal")

        for slot, item in list(preferences.items()):
            if item["status"] == "revised" and not any(
                event.get("type") == "preference_revision" and event.get("slot") == slot and event.get("turn_id") == turn_id
                for event in events
            ):
                flags.append("preference_instability_signal")
                events.append(_event(turn_id, "preference_revision", slot, item.get("previous_value"), item.get("value"), reason))

    def _preserve_control_flags_from_draft(
        self,
        *,
        draft_flags: list[str],
        reflection: dict[str, Any],
        existing_preferences: dict,
        preferences: dict,
        flags: list[str],
    ) -> None:
        if "underspecified_product_type" not in draft_flags:
            return
        has_existing_product = bool(_pref_value(existing_preferences, "product_type"))
        product_update = preferences.get("product_type")
        has_new_product = bool(product_update and product_update.get("value"))
        needs_clarification = bool(reflection.get("needs_clarification")) or reflection.get("strategy") == "ask_clarification"
        if not has_existing_product and not has_new_product and needs_clarification:
            flags.append("underspecified_product_type")

    def _recover_product_type_from_turn(
        self,
        *,
        state: dict,
        turn_id: int,
        existing_preferences: dict,
        preferences: dict,
        events: list,
        flags: list[str],
    ) -> None:
        if _pref_value(existing_preferences, "product_type"):
            return
        current = preferences.get("product_type")
        if current and current.get("value"):
            return
        latest = str(state.get("latest_user_message") or "").lower().strip()
        product = self._guess_product_type(latest) or self._extract_generic_product_type(latest)
        if not product:
            return
        preferences["product_type"] = self._preference_patch(
            slot="product_type",
            value=product,
            turn_id=turn_id,
            confidence=0.72,
            existing_preferences=existing_preferences,
            reason="Recovered a product/category phrase from the latest user turn after reflection validation.",
        )
        while "underspecified_product_type" in flags:
            flags.remove("underspecified_product_type")
        events.append(_event(
            turn_id,
            "product_type_recovery",
            "product_type",
            None,
            product,
            "Recovered product/category evidence from the user turn.",
        ))

    def _normalize_reflected_preference(self, slot: str, value):
        if slot == "budget_max":
            return value
        text = re.sub(r"\s+", " ", str(value or "").strip().lower())
        if not text:
            return None
        if slot == "occasion":
            return "gift" if text in {"gift", "birthday", "present"} else text
        if slot == "use_case" and text in {"gift", "present", "birthday", "recommendation", "recommendations", "product"}:
            return None
        if slot == "product_type" and _is_invalid_reflected_product_type(text):
            return None
        return text


def _pref_value(preferences: dict, slot: str):
    item = preferences.get(slot)
    if isinstance(item, dict):
        return item.get("value")
    return getattr(item, "value", None)


def _has_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text))


def _phrase_match(text: str, phrase: str):
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text)


def _mentions_skin_need(text: str) -> bool:
    return any(
        _has_phrase(text, term)
        for term in (
            "skin",
            "oily",
            "sensitive",
            "dry",
            "acne",
            "pimple",
            "prescription",
            "treatment",
        )
    )


def _is_modifier_or_followup_phrase(text: str) -> bool:
    stripped = " ".join(str(text or "").lower().strip(" .?!,;:").split())
    if not stripped:
        return False
    stripped = re.sub(
        r"^(?:i would like to buy|i'd like to buy|would like to buy|i want to buy|i wanna buy|want to buy|i want buy|want buy|i would like|i'd like|i need|need|i want|i wanna|want)\s+",
        "",
        stripped,
    ).strip()
    stripped = stripped.removeprefix("a pair of ").removeprefix("pair of ").strip()
    stripped = stripped.removeprefix("a ").removeprefix("an ").removeprefix("the ").strip()
    if _is_gender_or_style_only(stripped):
        return True
    if stripped in {
        "different option",
        "different options",
        "other option",
        "other options",
        "another option",
        "another options",
        "any other option",
        "any other options",
        "other style",
        "other styles",
        "any other style",
        "any other styles",
        "show different options",
        "show me different options",
        "either one is ok",
        "either one is okay",
        "either is ok",
        "either is okay",
    }:
        return True
    if any(phrase in stripped for phrase in ("different option", "other option", "another option", "other style", "other styles")):
        return True
    if re.match(r"^(?:are there|do you have|can you show|show me)\b.*\b(?:other|different)\b", stripped):
        return True
    return False


def _brand_question_without_new_product(text: str) -> bool:
    stripped = " ".join(str(text or "").lower().strip(" .?!,;:").split())
    if not stripped:
        return False
    if not re.match(r"^(?:do you have|do you sell|have you got|can you show)\b", stripped):
        return False
    return any(_has_phrase(stripped, brand) for brand in _catalog_brands())


def _is_gender_or_style_only(text: str) -> bool:
    normalized = str(text or "").lower().replace("'", "").replace("’", "")
    normalized = " ".join(normalized.strip(" .?!,;:").split())
    normalized = normalized.removeprefix("a pair of ").removeprefix("pair of ").strip()
    normalized = normalized.removeprefix("a ").removeprefix("an ").removeprefix("the ").strip()
    return normalized in {"womens", "women", "for women", "mens", "men", "for men", "female", "male", "unisex"}


def _looks_like_brand_only(text: str) -> bool:
    stripped = " ".join(str(text or "").lower().strip(" .?!,;:").split())
    if not stripped:
        return False
    if stripped.endswith(" brand"):
        stripped = stripped.removesuffix(" brand").strip()
    parts = [
        part.strip()
        for part in re.split(r"\s*(?:,|/|\bor\b|\band\b)\s*", stripped)
        if part.strip()
    ]
    if not parts:
        return False
    known = set(_catalog_brands())
    return all(part in known for part in parts)


def _should_preserve_existing_product_as_context(existing_product, proposed_product, text: str) -> bool:
    if not existing_product or not proposed_product:
        return False
    existing = str(existing_product).lower()
    proposed = str(proposed_product).lower()
    if existing == proposed:
        return False
    contextual_modifiers = {
        "desk",
        "office",
        "home",
        "sports",
        "sport",
        "running",
        "quality",
        "cheap",
        "lightweight",
        "portable",
        "electric",
        "wireless",
    }
    if proposed in contextual_modifiers and len(text.split()) <= 4:
        return True
    if existing == "lamp" and proposed in {"desk", "table"}:
        return True
    return False


def _canonical_brand(value: str) -> str:
    special = {"hp": "HP", "msi": "MSI", "asus": "ASUS"}
    cleaned = value.strip().lower()
    return special.get(cleaned, cleaned.title())


@lru_cache(maxsize=1)
def _catalog_brands() -> tuple[str, ...]:
    brands = set(FALLBACK_BRANDS)
    if CATALOG_BRANDS_PATH.exists():
        try:
            raw = json.loads(CATALOG_BRANDS_PATH.read_text())
        except Exception:
            raw = []
        if isinstance(raw, list):
            for item in raw:
                cleaned = _clean_brand_candidate(item)
                if cleaned:
                    brands.add(cleaned)
    return tuple(sorted(brands, key=lambda value: (-len(value), value)))


def _clean_brand_candidate(value) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if not text or text.startswith("#"):
        return None
    if not 2 <= len(text) <= 30:
        return None
    if text in NON_BRAND_REJECTION_TERMS:
        return None
    return text


def _safe_confidence(value, default: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, confidence))


def _is_empty(value) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _clean_feature(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0").lower() in {"1", "true", "yes", "on"}


def _has_memory_change_signal(text: str) -> bool:
    change_phrases = (
        " actually ",
        " instead ",
        " rather ",
        " not ",
        " no ",
        " avoid ",
        " without ",
        " cheaper ",
        " cheap ",
        " under ",
        " less ",
        " more ",
        " different ",
        " another ",
        " other ",
        " lighter ",
        " smaller ",
        " larger ",
        " bigger ",
        " color ",
        " black ",
        " white ",
        " blue ",
        " red ",
        " brand ",
    )
    if any(phrase in text for phrase in change_phrases):
        return True
    signal_tokens = {
        "black",
        "white",
        "blue",
        "red",
        "green",
        "pink",
        "small",
        "smaller",
        "large",
        "larger",
        "big",
        "bigger",
        "cheap",
        "cheaper",
        "light",
        "lighter",
        "heavy",
        "wireless",
        "portable",
    }
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return bool(tokens & signal_tokens)


def _reflection_rule_draft(
    *,
    preferences: dict,
    events: list,
    flags: list,
    rejected_brands: list,
    rejected_product_types: list,
    rejected_features: list,
    preferred_brands: list,
    preferred_features: list,
) -> dict:
    return {
        "preferences": {
            slot: {
                "value": item.get("value"),
                "status": item.get("status"),
                "previous_value": item.get("previous_value"),
            }
            for slot, item in preferences.items()
        },
        "preference_events": events,
        "security_flags": list(dict.fromkeys(flags)),
        "rejected_brands": list(rejected_brands),
        "rejected_product_types": list(rejected_product_types),
        "rejected_features": list(rejected_features),
        "preferred_brands": list(preferred_brands),
        "preferred_features": list(preferred_features),
    }


def _reflection_agent_evidence(*, flags: list, accepted_products: list) -> dict:
    evidence = {
        "security_flags": list(dict.fromkeys(flags)),
    }
    if accepted_products:
        evidence["accepted_products"] = list(accepted_products)
    return evidence


def _sanitize_agent_reflection(reflection: dict[str, Any], state: dict) -> dict[str, Any]:
    if not isinstance(reflection, dict):
        raise ValueError('Reflection must be a JSON object')
    cleaned = dict(reflection or {})
    if cleaned.get('preferences') is not None and not isinstance(cleaned['preferences'], dict):
        raise ValueError('Reflection preferences must be an object')
    for field in ('preferred_brands', 'preferred_features', 'rejected_brands',
                  'rejected_features', 'clear_preferences', 'remove_preferred_brands',
                  'remove_preferred_features', 'remove_rejected_brands', 'remove_rejected_features'):
        value = cleaned.get(field)
        if value is not None and (not isinstance(value, list) or
                                  any(not isinstance(item, str) for item in value)):
            raise ValueError(f'Reflection {field} must be a list of strings')
    for field in ('accepted', 'needs_clarification', 'reject_previous_recommendations'):
        if field in cleaned and not isinstance(cleaned[field], bool):
            raise ValueError(f'Reflection {field} must be a Boolean')
    for field, allowed in (
        ('preference_change', {'none', 'new_preference', 'revision', 'rejection', 'acceptance', 'uncertain'}),
        ('strategy', {'ask_clarification', 'update_memory', 'update_query', 'search_again', 'accept_current', 'no_action'}),
    ):
        if field in cleaned and cleaned[field] not in allowed:
            raise ValueError(f'Invalid reflection {field}')
    reflected_preferences = dict(cleaned.get("preferences") or {})
    notes = []
    # User semantics belong to the LLM. Acceptance/price phrase detectors must
    # not rewrite an otherwise valid preference_change or strategy.

    product_type = reflected_preferences.get("product_type")
    if product_type and _is_invalid_reflected_product_type(str(product_type)):
        reflected_preferences["product_type"] = None
        notes.append("removed_non_product_product_type")

    if notes:
        cleaned["_validation_notes"] = list(dict.fromkeys(notes))
    cleaned["preferences"] = reflected_preferences
    return cleaned


def _rejects_previous_recommendations(reflection: dict[str, Any]) -> bool:
    # In agent-primary mode, the LLM is the semantic authority for whether a
    # user rejected the current recommendation set.  Do not replace that
    # decision with a brittle phrase list.
    return bool(reflection.get("reject_previous_recommendations"))


def _intent_from_reflection(reflection: dict[str, Any], state: dict) -> dict[str, Any] | None:
    if not _env_enabled("AUTONOMOUS_REFLECTION_WRITES_INTENT"):
        return None
    pscon_label = str(
        reflection.get("pscon_intent_label")
        or reflection.get("intent_label")
        or ""
    ).strip().lower()
    preference_change = str(reflection.get("preference_change") or "").strip().lower()
    strategy = str(reflection.get("strategy") or "").strip().lower()

    if pscon_label not in {"reveal", "interpret", "revise", "inquire", "chitchat"}:
        pscon_label = _pscon_intent_from_reflection_change(reflection, preference_change, strategy)
    label = _workflow_label_from_pscon(pscon_label, preference_change, strategy)
    action_label = _action_label_from_pscon_reflection(pscon_label, preference_change, strategy)
    if not label and not action_label:
        return None

    confidence = _safe_confidence(reflection.get("confidence"), default=0.72)
    return {
        "label": label or "recommend",
        "action_label": action_label or "new_request_or_refinement",
        "confidence": round(confidence, 4),
        "source": "reflection",
        "pscon_label": pscon_label,
        "reflection_pscon_intent_label": pscon_label,
        "planner_intent": label or "recommend",
        "next_action_hint": _next_action_hint_from_action(action_label),
        "requires_state_update": action_label
        in {"new_request_or_refinement", "rejection", "clarification"},
        "requires_retrieval_decision": label in {"recommend", "reject_or_revise", "clarify"},
        "reflection_strategy": strategy,
        "reflection_preference_change": preference_change,
        "previous_label": (state.get("intent") or {}).get("label"),
}


def _pscon_intent_from_reflection_change(reflection: dict[str, Any], preference_change: str, strategy: str) -> str:
    if preference_change == "acceptance" or strategy == "accept_current":
        return "interpret"
    reflected_preferences = reflection.get("preferences") if isinstance(reflection.get("preferences"), dict) else {}
    if preference_change == "revision" and not _is_empty(reflected_preferences.get("product_type")):
        return "revise"
    if preference_change in {"revision", "rejection"} or strategy == "search_again":
        return "interpret"
    if strategy == "ask_clarification":
        return "interpret"
    if preference_change == "new_preference":
        return "reveal"
    if preference_change == "uncertain" or strategy in {"update_memory", "update_query"}:
        return "interpret"
    return ""


def _workflow_label_from_pscon(pscon_label: str, preference_change: str, strategy: str) -> str:
    # Acceptance describes feedback; strategy describes the outstanding action.
    # A valid LLM update/clarification must not be discarded because feedback
    # is also positive. No user-text phrase detection is involved here.
    if strategy == "ask_clarification":
        return "clarify"
    if strategy == "search_again":
        return "reject_or_revise"
    if strategy == "update_query":
        return "reject_or_revise" if pscon_label == "revise" else "recommend"
    if preference_change == "acceptance" or strategy == "accept_current":
        return "accept"
    if pscon_label == "revise":
        return "reject_or_revise"
    if pscon_label == "inquire":
        return "inquire"
    if pscon_label == "chitchat":
        return "chitchat"
    if pscon_label in {"reveal", "interpret"}:
        return "recommend"
    return ""


def _action_label_from_pscon_reflection(pscon_label: str, preference_change: str, strategy: str) -> str:
    if strategy == "ask_clarification":
        return "clarification"
    if strategy == "search_again":
        return "rejection"
    if strategy == "update_query":
        return "rejection" if pscon_label == "revise" or preference_change in {"revision", "rejection"} else "new_request_or_refinement"
    if preference_change == "acceptance" or strategy == "accept_current":
        return "acceptance"
    if pscon_label == "inquire":
        return "inquiry"
    if pscon_label == "chitchat":
        return "chitchat"
    if strategy == "ask_clarification":
        return "clarification"
    if pscon_label == "revise" or preference_change in {"revision", "rejection"} or strategy == "search_again":
        return "rejection"
    if pscon_label in {"reveal", "interpret"} or preference_change in {"new_preference", "uncertain"} or strategy in {"update_memory", "update_query"}:
        return "new_request_or_refinement"
    return ""


def _next_action_hint_from_action(action_label: str) -> str:
    return {
        "acceptance": "respond",
        "inquiry": "respond_or_answer_from_context",
        "chitchat": "respond",
        "rejection": "update_memory_then_retrieve",
        "clarification": "update_memory_or_ask_clarification",
        "new_request_or_refinement": "update_memory_then_retrieve",
    }.get(action_label, "update_memory_then_retrieve")


def _is_acceptance_turn(text: str) -> bool:
    padded = f" {text} "
    return any(
        phrase in padded
        for phrase in (
            " this works ",
            " that works ",
            " works for me ",
            " looks good ",
            " sounds good ",
            " i like it ",
            " i like this ",
            " good option ",
            " take it ",
            " i'll take it ",
        )
    )


def _is_price_preference_turn(text: str) -> bool:
    return any(
        _has_phrase(text, phrase)
        for phrase in ("cheap", "cheape", "cheaper", "affordable", "budget", "low cost", "less expensive")
    )


def _is_invalid_reflected_product_type(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if not text:
        return True
    blocked = {
        "gift",
        "present",
        "recommendation",
        "recommendations",
        "product",
        "products",
        "item",
        "items",
        "this works",
        "that works",
        "works for me",
        "looks good",
        "sounds good",
        "i like it",
        "i like this",
        "cheap",
        "cheape",
        "cheaper",
        "affordable",
        "budget",
        "low cost",
        "less expensive",
        "different options",
        "show different options",
        "show more",
    }
    if text in blocked:
        return True
    if text.startswith("something ") and any(
        _has_phrase(text, phrase)
        for phrase in ("cheap", "cheape", "cheaper", "affordable", "budget", "lightweight", "portable")
    ):
        return True
    return False


def _event(turn_id: int, event_type: str, slot: str, old_value, new_value, reason: str) -> dict:
    return {
        "turn_id": turn_id,
        "type": event_type,
        "slot": slot,
        "old_value": old_value,
        "new_value": new_value,
        "reason": reason,
    }


def _product_ids(products: list[dict[str, Any]]) -> list[str]:
    ids = []
    for product in products:
        product_id = product.get("product_id") or product.get("id") or product.get("asin") or product.get("title")
        if product_id:
            ids.append(str(product_id))
    return ids


def _jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonish(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonish(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _jsonish(item) for key, item in vars(value).items()}
    return value
