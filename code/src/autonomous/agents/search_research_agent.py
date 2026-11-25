"""Search and research specialist that builds the structured search request."""

from __future__ import annotations

import re

from src.autonomous.flags import classify_flags, combined_flags
from src.autonomous.schemas import ComponentResult


class SearchResearchAgent:
    name = "search_research_agent"

    def run(self, state: dict) -> ComponentResult:
        prefs = state.get("preferences", {})
        raw_product_type = _pref_value(prefs, "product_type")
        product_type = raw_product_type
        last_request = state.get("last_final_search_request")
        previous_category, _previous_use_case = _normalize_product_type(_request_value(last_request, "category"))
        recovered_from_memory = False
        if not product_type and _can_reuse_last_request(state):
            product_type = _request_value(last_request, "category")
            recovered_from_memory = bool(product_type)
        product_type, inferred_use_case = _normalize_product_type(product_type)
        product_type, phrase_features = _normalize_category_phrase(
            product_type,
            latest_message=state.get("latest_user_message", ""),
            previous_category=previous_category,
        )
        if not product_type and _can_reuse_last_request(state):
            product_type = _request_value(last_request, "category")
            product_type, _prior_inferred_use_case = _normalize_product_type(product_type)
            product_type, phrase_features = _normalize_category_phrase(
                product_type,
                latest_message=state.get("latest_user_message", ""),
                previous_category=previous_category,
            )
            recovered_from_memory = bool(product_type)
        budget = _pref_value(prefs, "budget_max") or _request_value(last_request, "hard_constraints", {}).get("budget_max")
        query_parts = []
        preferred_features = _dedupe([*(state.get("preferred_features") or []), *phrase_features])
        preferred_brands = list(state.get("preferred_brands") or [])
        search_again = state.get("critic_next_action") == "search_again"
        search_attempts = int(state.get("search_attempts") or 0)

        if not product_type:
            return ComponentResult(
                name=self.name,
                patch={
                    "security_flags": ["missing_search_request"],
                    "tool_observation": {
                        "component": self.name,
                        "summary": "search request not built because product/category is missing",
                        "status": "success",
                    },
                },
            )

        _add_query_part(query_parts, str(product_type))
        if _pref_value(prefs, "occasion") == "gift":
            _add_query_part(query_parts, "gift")
        if _pref_value(prefs, "recipient"):
            _add_query_part(query_parts, f"for {_pref_value(prefs, 'recipient')}")
        use_case = _pref_value(prefs, "use_case") or inferred_use_case
        if use_case:
            use_case_part = f"for {use_case}" if inferred_use_case and use_case == inferred_use_case else str(use_case)
            _add_query_part(query_parts, use_case_part)
        for brand in preferred_brands:
            _add_query_part(query_parts, str(brand))
        for feature in preferred_features:
            _add_query_part(query_parts, str(feature))
        if budget:
            _add_query_part(query_parts, f"under {budget}")

        query_text = " ".join(query_parts).strip()
        if not query_text and _can_reuse_last_request(state):
            query_text = _request_value(state.get("last_final_search_request"), "query_text") or ""
        if not query_text:
            query_text = state.get("latest_user_message", "")
        retrieval_plan = _build_retrieval_plan(
            query_text=query_text,
            product_type=product_type,
            budget=budget,
            preferred_brands=preferred_brands,
            preferred_features=preferred_features,
            rejected_brands=list(state.get("rejected_brands") or []),
            rejected_features=list(state.get("rejected_features") or []),
            security_flags=combined_flags(state),
            search_again=search_again,
            prior_plan=state.get("retrieval_plan") or state.get("last_retrieval_plan") or {},
        )

        hard_constraints = {}
        if budget:
            hard_constraints["budget_max"] = budget

        flag_buckets = classify_flags(combined_flags(state))
        request = {
            "query_text": query_text,
            "category": product_type,
            "hard_constraints": hard_constraints,
            "soft_preferences": {
                k: use_case if k == "use_case" else _pref_value(prefs, k)
                for k in ("occasion", "recipient", "use_case")
                if (use_case if k == "use_case" else _pref_value(prefs, k))
            },
            "preferred": {
                "brands": list(state.get("preferred_brands") or []),
                "features": preferred_features,
            },
            "exclusions": {
                "brands": list(state.get("rejected_brands") or []),
                "products": list(state.get("rejected_products") or []),
                "product_types": list(state.get("rejected_product_types") or []),
                "features": list(state.get("rejected_features") or []),
            },
            "control_flags": flag_buckets["control_flags"],
            "preference_flags": flag_buckets["preference_flags"],
            "safety_flags": flag_buckets["safety_flags"],
            "unclassified_flags": flag_buckets["unclassified_flags"],
            "security_flags": flag_buckets["security_flags"],
        }

        return ComponentResult(
            name=self.name,
            patch={
                "final_search_request": request,
                "retrieval_plan": retrieval_plan,
                "preferred_features": preferred_features,
                "search_attempts": search_attempts + 1 if search_again else search_attempts,
                "raw_candidates": [],
                "verified_candidates": [],
                "candidate_ledger": {},
                "critic_decision": None,
                "critic_next_action": None,
                "critic_reasons": [],
                "tool_observation": {
                    "component": self.name,
                    "summary": f"structured search request built; strategy={retrieval_plan['strategy']}; recovered={recovered_from_memory}",
                    "status": "success",
                },
            },
        )


def _pref_value(preferences: dict, slot: str):
    item = preferences.get(slot)
    if isinstance(item, dict):
        return item.get("value")
    return getattr(item, "value", None)


def _request_value(request, key: str, default=None):
    if isinstance(request, dict):
        return request.get(key, default)
    return getattr(request, key, default)


def _can_reuse_last_request(state: dict) -> bool:
    last = state.get("last_final_search_request")
    if not last:
        return False
    latest_raw = str(state.get("latest_user_message") or "").lower()
    if _is_non_category_phrase(latest_raw):
        return True
    latest = f" {latest_raw} "
    if any(
        phrase in latest
        for phrase in (
            " this one ",
            " that one ",
            " these ",
            " those ",
            " my choice ",
            " first one ",
            " second one ",
            " third one ",
            " another ",
            " different ",
            " cheaper ",
            " more ",
            " less ",
            " show ",
            " picture ",
            " yes ",
            " thanks ",
            " thank you ",
        )
    ):
        return True
    return False


def _normalize_product_type(product_type) -> tuple[str | None, str | None]:
    if not product_type:
        return None, None
    text = " ".join(str(product_type).lower().strip(" .?!,;:").split())
    if not text:
        return None, None
    text = text.removeprefix("a ").removeprefix("an ").removeprefix("the ")
    text = text.removeprefix("some ")
    text = text.removeprefix("pair of ")
    text = text.removeprefix("a pair of ")
    if _is_non_category_phrase(text):
        return None, None
    if text in {"gift", "gifts", "present", "recommendation", "recommendations", "product", "products"}:
        return None, None
    if text.startswith("gift for ") or text.startswith("gifts for "):
        return None, text
    for marker in (" for ", " to "):
        if marker in text:
            head, tail = text.split(marker, 1)
            head = head.strip()
            tail = tail.strip()
            if head and head not in {"gift", "gifts", "something", "anything"}:
                return head, tail or None
    return text, None


def _normalize_category_phrase(product_type, *, latest_message: str, previous_category: str | None) -> tuple[str | None, list[str]]:
    if not product_type:
        return None, []
    text = " ".join(str(product_type or "").lower().strip(" .?!,;:").split())
    latest = " ".join(str(latest_message or "").lower().strip(" .?!,;:").split())
    previous = " ".join(str(previous_category or "").lower().strip(" .?!,;:").split())
    features: list[str] = []

    product_nouns = (
        "backpack",
        "lamp",
        "laptop",
        "headphones",
        "headphone",
        "headset",
        "shoes",
        "shoe",
        "sneakers",
        "lipstick",
        "pen",
        "bottle opener",
        "mouse pad",
        "seasoning packs",
        "seasoning pack",
    )
    feature_words = {
        "stylish",
        "durable",
        "washable",
        "waterproof",
        "water-resistant",
        "water resistant",
        "electric",
        "cheap",
        "cheaper",
        "quality",
        "lightweight",
        "wireless",
        "portable",
    }

    if previous == "lamp" and text in {"desk", "table", "floor", "bedside", "office"}:
        return f"{text} lamp", [text]
    if previous and text in {"style", "styles", "different", "other", "another", "more"}:
        return previous, []

    split_category, split_features = _split_category_head_and_modifiers(text)
    if split_category:
        return split_category, _dedupe([*features, *split_features])

    for feature in feature_words:
        if re.search(rf"\b{re.escape(feature)}\b", f"{text} {latest}"):
            normalized_feature = feature.replace("-", " ")
            if normalized_feature != text:
                features.append(normalized_feature)

    for noun in sorted(product_nouns, key=len, reverse=True):
        if re.search(rf"\b{re.escape(noun)}s?\b", text):
            canonical = _canonical_category(noun)
            return canonical, _dedupe(features)

    return text, _dedupe(features)


def _split_category_head_and_modifiers(text: str) -> tuple[str | None, list[str]]:
    text = " ".join(str(text or "").lower().split())
    if not text:
        return None, []
    protected_compounds = {
        "body lotion",
        "bottle opener",
        "comic book",
        "desk lamp",
        "head shaping device",
        "head-shaping device",
        "mouse pad",
        "smart watch",
    }
    if text in protected_compounds:
        return _canonical_category(text), []

    tokens = text.split()
    if len(tokens) < 2 or len(tokens) > 4:
        return None, []

    category_heads = {
        "backpack",
        "book",
        "headphones",
        "headphone",
        "headset",
        "jacket",
        "lamp",
        "laptop",
        "lipstick",
        "pen",
        "shoes",
        "shoe",
        "toy",
        "watch",
    }
    first = tokens[0]
    last = tokens[-1]
    if first in category_heads:
        return _canonical_category(first), tokens[1:]
    if last in category_heads:
        return _canonical_category(last), tokens[:-1]
    return None, []


def _canonical_category(value: str) -> str:
    value = value.lower().strip()
    if value == "headphone":
        return "headphones"
    if value in {"shoe", "sneakers"}:
        return "shoes"
    if value == "seasoning pack":
        return "seasoning packs"
    if value == "head-shaping device":
        return "head shaping device"
    return value


def _is_non_category_phrase(text: str) -> bool:
    text = " ".join(str(text or "").lower().strip(" .?!,;:").split())
    if not text:
        return False
    text = re.sub(
        r"^(?:i would like to buy|i'd like to buy|would like to buy|i want to buy|i wanna buy|want to buy|i want buy|want buy|i would like|i'd like|i need|need|i want|i wanna|want)\s+",
        "",
        text,
    ).strip()
    normalized = text.replace("'", "").replace("’", "")
    normalized = normalized.removeprefix("a pair of ").removeprefix("pair of ").strip()
    normalized = normalized.removeprefix("a ").removeprefix("an ").removeprefix("the ").strip()
    if normalized in {"womens", "women", "for women", "mens", "men", "for men", "female", "male", "unisex"}:
        return True
    if text in {
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
    if any(phrase in text for phrase in ("different option", "other option", "another option", "other style", "other styles")):
        return True
    if re.match(r"^(?:are there|do you have|can you show|show me)\b.*\b(?:other|different)\b", text):
        return True
    return False


def _build_retrieval_plan(
    *,
    query_text: str,
    product_type,
    budget,
    preferred_brands: list,
    preferred_features: list,
    rejected_brands: list,
    rejected_features: list,
    security_flags: list,
    search_again: bool = False,
    prior_plan: dict | None = None,
) -> dict:
    variants = _query_variants(
        query_text,
        product_type,
        preferred_features,
        budget,
        preferred_brands=preferred_brands,
        search_again=search_again,
    )
    constraint_count = sum(
        [
            bool(budget),
            bool(preferred_brands),
            bool(preferred_features),
            bool(rejected_brands),
            bool(rejected_features),
        ]
    )
    if constraint_count >= 2 or preferred_features:
        strategy = "hybrid"
        tool_sequence = ["bm25", "dense_e5", "reranker", "evidence_scorer", "verifier"]
        search_depth = 75
        reason = "Use hybrid retrieval because structured preferences need both lexical matching and semantic coverage."
    elif product_type and len(str(product_type).split()) > 2:
        strategy = "lexical"
        tool_sequence = ["bm25", "category_router", "reranker", "evidence_scorer", "verifier"]
        search_depth = 60
        reason = "Use lexical-first retrieval because the product/category phrase is specific."
    else:
        strategy = "hybrid"
        tool_sequence = ["bm25", "dense_e5", "reranker", "evidence_scorer", "verifier"]
        search_depth = 50
        reason = "Use balanced hybrid retrieval for a known product category."

    if security_flags:
        search_depth = max(search_depth, 75)
        reason += " Preference instability or safety flags require a wider candidate pool before verification."

    if search_again:
        prior_plan = prior_plan or {}
        prior_variants = list(prior_plan.get("query_variants") or [])
        variants = _broaden_variants(variants, prior_variants, product_type, preferred_features)
        strategy = "hybrid"
        tool_sequence = ["bm25", "dense_e5", "reranker", "evidence_scorer", "verifier"]
        search_depth = max(search_depth, int(prior_plan.get("search_depth") or 0), 100)
        reason = "Search again requested by critic; broaden query variants and increase candidate depth before verification."

    return {
        "strategy": strategy,
        "tool_sequence": tool_sequence,
        "query_variants": variants,
        "search_depth": search_depth,
        "needs_search_again_check": True,
        "relaxation_plan": _relaxation_plan(product_type, preferred_features, preferred_brands, budget),
        "reason": reason,
    }


def _query_variants(
    query_text: str,
    product_type,
    preferred_features: list,
    budget,
    *,
    preferred_brands: list | None = None,
    search_again: bool = False,
) -> list[str]:
    variants = []
    category = str(product_type or "").strip()
    feature_query = " ".join([category, *[str(feature) for feature in preferred_features]]).strip()
    brandless_query = _remove_terms(query_text, preferred_brands or [])
    expansions = _category_query_expansions(category) if search_again else []
    for value in (
        query_text,
        brandless_query,
        category,
        feature_query,
        f"budget {product_type}" if budget and product_type else "",
        *expansions,
    ):
        value = " ".join(str(value or "").split())
        if value and value not in variants:
            variants.append(value)
    limit = 8 if search_again else 6
    return variants[:limit]


def _broaden_variants(current: list[str], prior: list[str], product_type, preferred_features: list) -> list[str]:
    variants = []
    for value in [*current, *prior, str(product_type or "")]:
        value = " ".join(str(value or "").split())
        if value and value not in variants:
            variants.append(value)
    for feature in preferred_features:
        value = " ".join([str(product_type or ""), str(feature or "")]).strip()
        value = " ".join(value.split())
        if value and value not in variants:
            variants.append(value)
    for value in _category_query_expansions(str(product_type or "")):
        if value and value not in variants:
            variants.append(value)
    return variants[:8]


def _category_query_expansions(product_type: str) -> list[str]:
    normalized = " ".join(str(product_type or "").lower().split())
    expansions = {
        "seasoning packs": ["seasoning mix", "spice blend", "marinade seasoning", "cooking seasoning"],
        "backpack": ["backpack", "durable backpack", "water resistant backpack"],
        "desk lamp": ["desk lamp", "lamp"],
        "lamp": ["lamp", "desk lamp", "table lamp"],
        "shoes": ["sport shoes", "athletic shoes", "sneakers"],
        "lipstick": ["lipstick", "lip color"],
        "headphones": ["headphones", "headset", "earbuds"],
        "headset": ["headset", "headphones", "wireless headset"],
        "head-shaping devices": ["head shaping device", "baby head pillow", "infant head support"],
        "bottle opener": ["bottle opener", "electric bottle opener"],
    }
    return expansions.get(normalized, [])


def _relaxation_plan(product_type, preferred_features: list, preferred_brands: list, budget) -> list[dict]:
    plan = []
    if preferred_features:
        plan.append({"step": "drop_soft_preferences", "keep": "product_type"})
    if preferred_brands:
        plan.append({"step": "drop_brand_preference", "keep": "product_type"})
    if budget:
        plan.append({"step": "widen_budget", "keep": "product_type"})
    if product_type:
        plan.append({"step": "product_type_only", "query": str(product_type)})
        for value in _category_query_expansions(str(product_type)):
            plan.append({"step": "category_expansion", "query": value})
    return plan[:6]


def _remove_terms(text: str, terms: list) -> str:
    cleaned = str(text or "")
    for term in terms:
        if not term:
            continue
        cleaned = re.sub(rf"\b{re.escape(str(term))}\b", " ", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def _dedupe(values: list) -> list:
    result = []
    for value in values:
        value = " ".join(str(value or "").split()).strip()
        if value and value not in result:
            result.append(value)
    return result


def _add_query_part(parts: list[str], value: str) -> None:
    value = " ".join(str(value or "").split()).strip()
    if not value:
        return
    value_key = _part_key(value)
    for existing in parts:
        existing_key = _part_key(existing)
        if value_key == existing_key:
            return
        if _is_contained_phrase(value_key, existing_key):
            return
        if _is_contained_phrase(existing_key, value_key):
            parts.remove(existing)
            break
    parts.append(value)


def _part_key(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def _is_contained_phrase(needle: str, haystack: str) -> bool:
    if not needle or not haystack:
        return False
    if needle == haystack:
        return True
    return f" {needle} " in f" {haystack} "
