"""
Evidence and verification agents
--------------------------------
Lightweight, deterministic guards adapted for the taxonomy-aware pipeline.

EvidenceScorer adds transparent match features to candidates. VerifierAgent
removes products that violate hard user constraints before recommendations
are shown.
"""

import os
import re

from src.agents.normalized_metadata import (
    NORMALIZED_METADATA_ENABLED,
    get_normalized_product,
    compute_normalized_metadata_features,
)


EVIDENCE_SCORE_WEIGHT = float(os.getenv("EVIDENCE_SCORE_WEIGHT", "0.12"))
STRICT_PRICE_FILTER = os.getenv("STRICT_PRICE_FILTER", "0").strip().lower() in {"1", "true", "yes", "on"}
STRICT_COLOR_FILTER = os.getenv("STRICT_COLOR_FILTER", "0").strip().lower() in {"1", "true", "yes", "on"}

# Same suggested small weights as the reranker (src/agents/reranker.py) —
# kept identical so the two layers agree on how much normalized metadata
# should move a score. Additive here (EvidenceScorer's score is already an
# additive sum of independently-weighted 0-1 subscores, not a weighted
# average), so these add directly without renormalization.
NORMALIZED_METADATA_EVIDENCE_WEIGHTS = {
    "norm_brand_match":       0.05,
    "norm_category_overlap":  0.04,
    "norm_rating_score":      0.03,
    "norm_review_score":      0.02,
    "norm_metadata_quality":  0.02,
    "norm_attribute_overlap": 0.05,
}


class EvidenceScorer:
    """Add evidence fields and a mild final-score adjustment to candidates."""

    def score_products(self, products: list, slots: dict, query: str = "") -> list:
        slots = slots or {}
        scored = []
        for product in products or []:
            evidence = self._evidence(product, slots, query)
            item = dict(product)
            item["evidence"] = evidence
            item["evidence_score"] = evidence["score"]
            base = float(item.get("rerank_score") or item.get("score") or 0.0)
            item["final_score"] = round(base + evidence["score"] * EVIDENCE_SCORE_WEIGHT, 4)
            scored.append(item)
        scored.sort(key=lambda p: (-p.get("final_score", 0.0), p.get("scope_rank", 9)))
        return scored

    def _evidence(self, product: dict, slots: dict, query: str) -> dict:
        text = self._product_text(product)
        query_match = self._coverage(self._tokens(query), text)
        product_type_match = self._coverage(self._tokens(slots.get("product_type")), text)
        feature_match = self._coverage(self._slot_list_tokens(slots, "features"), text)
        brand_match = self._coverage(self._tokens(slots.get("brand")), text)
        color_match = self._color_match(product, slots, text)
        price_match = self._price_match(product, slots)
        excluded = self._violates_exclusions(product, slots, text)
        category_match = self._category_match(product, slots)

        score = 0.0
        score += query_match * 0.30
        score += product_type_match * 0.25
        score += feature_match * 0.18
        score += brand_match * 0.14
        score += color_match * 0.08
        score += price_match * 0.08
        score += category_match * 0.10
        if excluded:
            score -= 1.0

        result = {
            "score": round(score, 4),
            "query_match": round(query_match, 3),
            "product_type_match": round(product_type_match, 3),
            "feature_match": round(feature_match, 3),
            "brand_match": round(brand_match, 3),
            "color_match": bool(color_match),
            "price_match": bool(price_match),
            "category_match": round(category_match, 3),
            "violates_constraints": bool(excluded),
        }

        if NORMALIZED_METADATA_ENABLED:
            norm_features = compute_normalized_metadata_features(
                slots, product.get("product_id"), get_normalized_product(product.get("product_id"))
            )
            norm_score = sum(
                norm_features.get(key, 0.5 if key != "norm_brand_match" else 0) * weight
                for key, weight in NORMALIZED_METADATA_EVIDENCE_WEIGHTS.items()
            )
            norm_score += norm_features.get("norm_soft_penalty", 0.0)
            result["normalized_metadata"] = norm_features
            result["score"] = round(score + norm_score, 4)

        return result

    def _product_text(self, product: dict) -> str:
        fields = [
            product.get("title", ""),
            product.get("brand", ""),
            product.get("color", ""),
            product.get("description", ""),
            product.get("category", ""),
            product.get("l1", ""),
            product.get("l2", ""),
        ]
        return " ".join(str(f).lower() for f in fields if f)

    def _tokens(self, value) -> list:
        return [
            t for t in re.findall(r"[a-z0-9]+", str(value or "").lower())
            if len(t) > 2
        ]

    def _slot_list_tokens(self, slots: dict, key: str) -> list:
        value = slots.get(key) or []
        if not isinstance(value, list):
            value = [value]
        tokens = []
        for item in value:
            tokens.extend(self._tokens(item))
        return list(dict.fromkeys(tokens))

    def _coverage(self, terms: list, text: str) -> float:
        unique = list(dict.fromkeys(t for t in terms if t))
        if not unique:
            return 0.0
        return sum(1 for term in unique if term in text) / len(unique)

    def _color_match(self, product: dict, slots: dict, text: str) -> float:
        color = str(slots.get("color") or "").lower().strip()
        if not color:
            return 0.0
        product_color = str(product.get("color") or "").lower()
        return 1.0 if color in product_color or color in text else 0.0

    def _price_match(self, product: dict, slots: dict) -> float:
        price = product.get("price")
        if price is None:
            return 0.0
        price_min = slots.get("price_min")
        price_max = slots.get("price_max")
        if price_min is None and price_max is None:
            return 0.0
        if price_min is not None and price < price_min:
            return 0.0
        if price_max is not None and price > price_max:
            return 0.0
        return 1.0

    def _category_match(self, product: dict, slots: dict) -> float:
        product_type = self._tokens(slots.get("product_type"))
        if not product_type:
            return 0.0
        text = " ".join(str(product.get(k) or "").lower() for k in ("title", "category", "l1", "l2"))
        return self._coverage(product_type, text)

    def _violates_exclusions(self, product: dict, slots: dict, text: str) -> bool:
        brand_text = str(product.get("brand") or "").lower()
        for brand in slots.get("exclude_brands") or []:
            brand = str(brand).lower().strip()
            if brand and (brand in brand_text or brand in text):
                return True
        for feature in slots.get("exclude_features") or []:
            feature = str(feature).lower().strip()
            if feature and feature in text:
                return True
        return False


class VerifierAgent:
    """Filter candidates that violate hard constraints."""

    def verify(self, products: list, slots: dict, top_k: int = 10) -> dict:
        slots = slots or {}
        accepted = []
        rejected = []
        warnings = []

        for product in products or []:
            reasons = self._rejection_reasons(product, slots)
            if reasons:
                rejected.append({"product_id": product.get("product_id"), "reasons": reasons})
                continue
            soft = self._soft_warning_reasons(product, slots)
            item = dict(product)
            if soft:
                item["verification_warnings"] = soft
                warnings.append({"product_id": product.get("product_id"), "reasons": soft})
            accepted.append(item)

        return {
            "products": accepted[:top_k],
            "rejected": rejected,
            "warnings": warnings,
            "needs_relaxation": len(accepted) < min(top_k, len(products or [])),
        }

    def _rejection_reasons(self, product: dict, slots: dict) -> list:
        reasons = []
        text = self._product_text(product)
        brand_text = str(product.get("brand") or "").lower()

        for brand in slots.get("exclude_brands") or []:
            brand = str(brand).lower().strip()
            if brand and (brand in brand_text or brand in text):
                reasons.append(f"excluded_brand:{brand}")

        for feature in slots.get("exclude_features") or []:
            feature = str(feature).lower().strip()
            if feature and feature in text:
                reasons.append(f"excluded_feature:{feature}")

        if STRICT_PRICE_FILTER:
            price_reason = self._price_rejection_reason(product, slots)
            if price_reason:
                reasons.append(price_reason)

        if STRICT_COLOR_FILTER:
            color_reason = self._color_rejection_reason(product, slots, text)
            if color_reason:
                reasons.append(color_reason)

        return reasons

    def _soft_warning_reasons(self, product: dict, slots: dict) -> list:
        reasons = []
        price_reason = self._price_rejection_reason(product, slots)
        if price_reason:
            reasons.append(f"{price_reason}_soft")
        color_reason = self._color_rejection_reason(product, slots, self._product_text(product))
        if color_reason:
            reasons.append(f"{color_reason}_soft")
        return reasons

    def _price_rejection_reason(self, product: dict, slots: dict) -> str:
        price = product.get("price")
        price_min = slots.get("price_min")
        price_max = slots.get("price_max")
        if price is None:
            if price_min is not None or price_max is not None:
                return "missing_price"
            return ""
        if price_min is not None and price < price_min:
            return "below_min_price"
        if price_max is not None and price > price_max:
            return "over_budget"
        return ""

    def _color_rejection_reason(self, product: dict, slots: dict, text: str) -> str:
        color = str(slots.get("color") or "").lower().strip()
        if not color:
            return ""
        product_color = str(product.get("color") or "").lower()
        if product_color and color not in product_color and color not in text:
            return "color_mismatch"
        return ""

    def _product_text(self, product: dict) -> str:
        fields = [
            product.get("title", ""),
            product.get("brand", ""),
            product.get("color", ""),
            product.get("description", ""),
            product.get("category", ""),
        ]
        return " ".join(str(f).lower() for f in fields if f)
