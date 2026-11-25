"""Recommendation decision specialist for ranking and selecting final options."""

from __future__ import annotations

import re
import os
from typing import Any

from src.autonomous.schemas import ComponentResult


class RecommendationDecisionAgent:
    name = "recommendation_decision_agent"

    def run(self, state: dict) -> ComponentResult:
        candidates = list(state.get("verified_candidates") or [])
        request = state.get("final_search_request")
        ledger = state.get("candidate_ledger") or {}
        slots = ledger.get("slots") or {}
        rejected_products = set(state.get("rejected_products") or [])
        rejected_brands = {str(b).lower() for b in state.get("rejected_brands") or []}
        rejected_features = {str(f).lower() for f in state.get("rejected_features") or []}

        scored = [
            self._score_candidate(
                candidate,
                request,
                slots,
                rejected_products,
                rejected_brands,
                rejected_features,
                candidate_index=index,
            )
            for index, candidate in enumerate(candidates)
        ]
        scored = [item for item in scored if not item["blocked"]]
        scored.sort(key=lambda item: (-item["decision_score"], item["source_rank"]))

        selected_items = self._select_candidates(scored, limit=3)
        selected = []
        decisions = []
        for rank, item in enumerate(selected_items, start=1):
            product = dict(item["product"])
            product["decision_score"] = item["decision_score"]
            product["source_rerank_score"] = item["source_score"]
            product["agent_adjustment"] = item["agent_adjustment"]
            product["raw_agent_adjustment"] = item["raw_agent_adjustment"]
            product["decision_reasons"] = item["reasons"]
            product["tradeoff_summary"] = item["tradeoff_summary"]
            product["matched_preferences"] = item["matched_preferences"]
            product["missing_evidence"] = item["missing_evidence"]
            selected.append(product)
            decisions.append({
                "rank": rank,
                "product_id": product.get("product_id"),
                "title": product.get("title"),
                "brand": product.get("brand"),
                "decision_score": item["decision_score"],
                "source_rerank_score": item["source_score"],
                "agent_adjustment": item["agent_adjustment"],
                "raw_agent_adjustment": item["raw_agent_adjustment"],
                "ranking_policy": _selection_policy_name(),
                "reasons": item["reasons"],
                "matched_preferences": item["matched_preferences"],
                "missing_evidence": item["missing_evidence"],
                "tradeoff_summary": item["tradeoff_summary"],
            })

        return ComponentResult(
            name=self.name,
            patch={
                "last_recommendations": selected,
                "recommendation_decisions": decisions,
                "tool_observation": {
                    "component": self.name,
                    "summary": f"selected={len(selected)} recommendations from {len(candidates)} verified candidates; blocked={len(candidates) - len(scored)}; policy={_selection_policy_name()}",
                    "status": "success" if selected else "empty",
                },
            },
        )

    def _score_candidate(
        self,
        product: dict,
        request: Any,
        slots: dict,
        rejected_products: set,
        rejected_brands: set,
        rejected_features: set,
        *,
        candidate_index: int = 0,
    ) -> dict:
        product_id = product.get("product_id")
        brand = str(product.get("brand") or "").strip()
        text = _product_text(product)
        reasons = []
        matched_preferences = []
        missing_evidence = []
        blocked = False

        source_score = _source_score(product)
        raw_adjustment = 0.0
        if source_score:
            reasons.append("high retrieval/rerank evidence")

        if product_id in rejected_products:
            blocked = True
            reasons.append("blocked: previously rejected product")
        if brand.lower() in rejected_brands:
            blocked = True
            reasons.append("blocked: rejected brand")
        for feature in rejected_features:
            if feature and feature in text:
                blocked = True
                reasons.append(f"blocked: rejected feature {feature}")

        query_text = str(_request_value(request, "query_text") or "").lower()
        query_tokens = _tokens(query_text)
        if query_tokens:
            matched_tokens = [token for token in query_tokens if token in text]
            query_match = len(matched_tokens) / max(len(query_tokens), 1)
            if query_match >= 0.5:
                raw_adjustment += min(0.16, query_match * 0.16)
                reasons.append("matches query terms")
                matched_preferences.append("query_terms")
            elif query_match <= 0.2:
                raw_adjustment -= 0.06
                missing_evidence.append("query_terms")

        product_type = slots.get("product_type") or _request_value(request, "category")
        product_type_text = str(product_type or "").lower()
        title_text = str(product.get("title") or "").lower()
        if product_type and product_type_text in title_text:
            raw_adjustment += 0.16
            reasons.append(f"matches product type {product_type}")
            matched_preferences.append(f"product_type:{product_type}")
        elif product_type and product_type_text in text:
            raw_adjustment += 0.08
            reasons.append(f"mentions product type {product_type}")
            matched_preferences.append(f"product_type:{product_type}")
        elif product_type:
            raw_adjustment -= 0.18
            missing_evidence.append(f"product_type:{product_type}")

        use_case = slots.get("use_case") or (_request_value(request, "soft_preferences") or {}).get("use_case")
        if use_case and str(use_case).lower() in text:
            raw_adjustment += 0.1
            reasons.append(f"matches use case {use_case}")
            matched_preferences.append(f"use_case:{use_case}")
        elif use_case:
            raw_adjustment -= 0.03
            missing_evidence.append(f"use_case:{use_case}")

        preferred = _request_value(request, "preferred") or {}
        for feature in preferred.get("features") or []:
            feature_text = str(feature).lower()
            if feature_text and feature_text in text:
                raw_adjustment += 0.09
                reasons.append(f"matches preferred feature {feature}")
                matched_preferences.append(f"feature:{feature}")
            else:
                raw_adjustment -= 0.03
                missing_evidence.append(f"feature:{feature}")
        preferred_brands = {str(item).lower() for item in preferred.get("brands") or []}
        if preferred_brands:
            if brand.lower() in preferred_brands:
                raw_adjustment += 0.12
                reasons.append(f"matches preferred brand {brand}")
                matched_preferences.append(f"brand:{brand}")
            else:
                raw_adjustment -= 0.03
                missing_evidence.append("preferred_brand")

        price_max = slots.get("price_max") or (_request_value(request, "hard_constraints") or {}).get("budget_max")
        price = _number(product.get("price"), None)
        if price_max is not None:
            if price is not None and price <= float(price_max):
                budget_ratio = max(0.0, min(1.0, 1.0 - (price / max(float(price_max), 1.0))))
                raw_adjustment += 0.08 + (budget_ratio * 0.05)
                reasons.append("fits budget")
                matched_preferences.append(f"budget_max:{price_max}")
            elif price is None:
                raw_adjustment -= 0.02
                reasons.append("price missing")
                missing_evidence.append("price")
            else:
                blocked = True
                reasons.append("blocked: exceeds hard budget")

        evidence = product.get("evidence") or {}
        if evidence.get("violates_constraints"):
            blocked = True
            reasons.append("blocked: violates evidence constraints")
        if _number(product.get("evidence_score"), 0.0) > 0:
            raw_adjustment += min(_number(product.get("evidence_score"), 0.0), 1.0) * 0.08
            reasons.append("has supporting evidence")
        evidence_alignment = _evidence_alignment_bonus(evidence)
        if evidence_alignment > 0:
            raw_adjustment += evidence_alignment
            reasons.append("strong evidence alignment")
            matched_preferences.append("evidence_alignment")
        elif evidence and evidence_alignment < 0:
            raw_adjustment += evidence_alignment
            missing_evidence.append("evidence_alignment")

        rating = _number(product.get("rating"), None)
        if rating is not None and rating >= 4.0:
            raw_adjustment += min((rating - 4.0) * 0.03, 0.04)
            reasons.append("strong user rating")

        if not reasons:
            reasons.append("available verified candidate")

        tradeoff_summary = _tradeoff_summary(
            matched_preferences=matched_preferences,
            missing_evidence=missing_evidence,
            price=price,
            price_max=price_max,
            brand=brand,
        )

        adjustment = raw_adjustment * _agent_adjustment_weight()
        return {
            "product": product,
            "source_score": round(source_score, 4),
            "agent_adjustment": round(adjustment, 4),
            "raw_agent_adjustment": round(raw_adjustment, 4),
            "decision_score": round(source_score + adjustment, 4),
            "reasons": reasons[:5],
            "matched_preferences": list(dict.fromkeys(matched_preferences)),
            "missing_evidence": list(dict.fromkeys(missing_evidence)),
            "tradeoff_summary": tradeoff_summary,
            "blocked": blocked,
            "source_rank": int(product.get("scope_rank") or product.get("rank") or candidate_index + 1),
            "brand": brand.lower(),
        }

    def _select_candidates(self, scored: list, limit: int) -> list:
        if not _diversity_enabled():
            return scored[:limit]
        return self._select_diverse(scored, limit=limit)

    def _select_diverse(self, scored: list, limit: int) -> list:
        if not scored:
            return []
        selected = []
        used_brands = set()
        for item in scored:
            brand = item.get("brand") or "unknown"
            if (
                brand in used_brands
                and len(scored) - len(selected) > limit - len(selected)
                and _has_close_alternative_brand(item, scored, selected, limit)
            ):
                continue
            selected.append(item)
            used_brands.add(brand)
            if len(selected) == limit:
                return selected
        for item in scored:
            if item not in selected:
                selected.append(item)
            if len(selected) == limit:
                break
        return selected


def _request_value(request: Any, key: str):
    if isinstance(request, dict):
        return request.get(key)
    return getattr(request, key, None)


def _product_text(product: dict) -> str:
    fields = [
        product.get("title"), product.get("brand"), product.get("description"),
        product.get("category"), product.get("l1"), product.get("l2"),
    ]
    attributes = product.get("attributes")
    if isinstance(attributes, dict):
        fields.extend(attributes.values())
    return " ".join(str(value or "").lower() for value in fields)


def _source_score(product: dict) -> float:
    return _number(
        product.get("final_score"),
        _number(product.get("rerank_score"), _number(product.get("score"), 0.0)),
    )


def _agent_adjustment_weight() -> float:
    return max(0.0, min(2.0, _number(os.getenv("AUTONOMOUS_AGENT_ADJUSTMENT_WEIGHT"), 0.75)))


def _diversity_enabled() -> bool:
    return os.getenv("AUTONOMOUS_DECISION_DIVERSITY", "0").lower() in {"1", "true", "yes", "on"}


def _selection_policy_name() -> str:
    if _diversity_enabled():
        return "reranker_plus_preference_evidence_diversity_v2"
    return "reranker_plus_preference_evidence_v2"


def _number(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _tokens(value: str) -> set[str]:
    stopwords = {
        "a", "an", "the", "for", "with", "and", "or", "to", "of", "my", "me",
        "i", "need", "want", "gift", "show", "recommend", "something",
        "quality", "normal", "good", "best", "nice",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) > 2 and token not in stopwords
    }


def _evidence_alignment_bonus(evidence: dict) -> float:
    if not isinstance(evidence, dict) or not evidence:
        return 0.0
    query_match = _number(evidence.get("query_match"), 0.0)
    product_type_match = _number(evidence.get("product_type_match"), 0.0)
    feature_match = _number(evidence.get("feature_match"), 0.0)
    category_match = _number(evidence.get("category_match"), 0.0)
    alignment = 0.0
    if query_match >= 0.6:
        alignment += 0.08
    elif query_match <= 0.2:
        alignment -= 0.04
    if product_type_match >= 0.75:
        alignment += 0.1
    elif product_type_match <= 0.2:
        alignment -= 0.06
    if feature_match >= 0.5:
        alignment += 0.06
    if category_match >= 0.5:
        alignment += 0.05
    return max(-0.12, min(0.22, alignment))


def _tradeoff_summary(*, matched_preferences: list[str], missing_evidence: list[str], price, price_max, brand: str) -> str:
    if price_max is not None and price is not None:
        budget_text = "within budget" if price <= float(price_max) else "over budget"
    elif price_max is not None:
        budget_text = "price unknown"
    else:
        budget_text = "no hard budget"

    match_count = len(set(matched_preferences))
    if missing_evidence:
        return f"{brand or 'candidate'} matches {match_count} preference(s), with {budget_text}, but lacks evidence for {', '.join(missing_evidence[:2])}."
    return f"{brand or 'candidate'} matches {match_count} preference(s) with {budget_text}."


def _has_close_alternative_brand(item: dict, scored: list, selected: list, limit: int) -> bool:
    used = {candidate.get("brand") or "unknown" for candidate in selected}
    remaining_slots = limit - len(selected)
    threshold = float(os.getenv("AUTONOMOUS_DIVERSITY_SCORE_WINDOW", "0.25"))
    current_score = item.get("decision_score") or 0.0
    alternatives = [
        candidate
        for candidate in scored
        if candidate not in selected
        and candidate is not item
        and (candidate.get("brand") or "unknown") not in used
        and current_score - (candidate.get("decision_score") or 0.0) <= threshold
    ]
    return len(alternatives) >= remaining_slots
