"""
Normalized Metadata Features
-----------------------------
Optional soft reranking/evidence features sourced from
outputs/normalized_product_index.json (built by
scripts/build_normalized_product_index.py).

Gated behind MACPRS_USE_NORMALIZED_METADATA (default "0" — off, legacy
behavior unchanged). When enabled, these features add small, capped-weight
signals to the reranker and evidence scorer. They are NEVER used to hard-
filter or hard-reject a candidate, and never touch retrieval/FAISS — only
post-retrieval reranking/evidence scoring reads this module.

Missing/unmatched normalized metadata always falls back to neutral values,
so no candidate is penalized, boosted, or dropped just because normalized
metadata wasn't found for it — the candidate scores exactly as it would if
this module didn't exist.
"""

import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_NORMALIZED_METADATA_PATH = ROOT / "outputs" / "normalized_product_index.json"

NORMALIZED_METADATA_ENABLED = os.getenv(
    "MACPRS_USE_NORMALIZED_METADATA", "0"
).strip().lower() in {"1", "true", "yes", "on"}

# Absolute cap on the soft penalty contribution — never larger in magnitude,
# regardless of how many weak-conflict signals stack up.
MAX_SOFT_PENALTY = 0.05

_normalized_metadata_cache: dict | None = None
_normalized_metadata_path_loaded: str | None = None

_diagnostics = {
    "candidates_seen": 0,
    "candidates_with_metadata": 0,
    "sum_category_overlap": 0.0,
    "sum_attribute_overlap": 0.0,
}


# ── Loading ─────────────────────────────────────────────────────────────────

def load_normalized_metadata(path=None) -> dict:
    """
    Load outputs/normalized_product_index.json once and cache it.

    Returns {} (not an error) if the file is missing or malformed — the
    feature layer simply stays inert rather than crashing the pipeline.
    """
    global _normalized_metadata_cache, _normalized_metadata_path_loaded
    path = str(path or DEFAULT_NORMALIZED_METADATA_PATH)
    if _normalized_metadata_cache is not None and _normalized_metadata_path_loaded == path:
        return _normalized_metadata_cache
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    _normalized_metadata_cache = data
    _normalized_metadata_path_loaded = path
    return data


def get_normalized_product(product_id: str, path=None) -> dict | None:
    """Safe lookup by product_id — returns None if not loaded/not found, never raises."""
    if not product_id:
        return None
    metadata = load_normalized_metadata(path)
    return metadata.get(str(product_id))


# ── Diagnostics ─────────────────────────────────────────────────────────────

def reset_diagnostics():
    """Call at the start of an eval run so per-run averages aren't cross-contaminated."""
    global _diagnostics
    _diagnostics = {
        "candidates_seen": 0,
        "candidates_with_metadata": 0,
        "sum_category_overlap": 0.0,
        "sum_attribute_overlap": 0.0,
    }


def _record_diagnostics(matched: bool, category_overlap: float, attribute_overlap: float):
    _diagnostics["candidates_seen"] += 1
    if matched:
        _diagnostics["candidates_with_metadata"] += 1
    _diagnostics["sum_category_overlap"] += category_overlap
    _diagnostics["sum_attribute_overlap"] += attribute_overlap


def get_normalized_metadata_diagnostics(path=None) -> dict:
    """Summary block for eval output — safe to call even if disabled (returns zeros)."""
    seen = _diagnostics["candidates_seen"] or 1
    products_loaded = len(load_normalized_metadata(path)) if NORMALIZED_METADATA_ENABLED else 0
    return {
        "normalized_metadata_enabled": NORMALIZED_METADATA_ENABLED,
        "normalized_metadata_products_loaded": products_loaded,
        "candidates_with_normalized_metadata": _diagnostics["candidates_with_metadata"],
        "avg_norm_category_overlap": round(_diagnostics["sum_category_overlap"] / seen, 4),
        "avg_norm_attribute_overlap": round(_diagnostics["sum_attribute_overlap"] / seen, 4),
    }


# ── Feature computation ─────────────────────────────────────────────────────

def _norm_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _empty_features() -> dict:
    return {
        "norm_brand_match":       0,
        "norm_category_overlap":  0.5,
        "norm_rating_score":      0.5,
        "norm_review_score":      0.5,
        "norm_metadata_quality":  0.5,
        "norm_attribute_overlap": 0.5,
        "norm_soft_penalty":      0.0,
    }


def _brand_feature(query_context: dict, record: dict) -> tuple:
    """Returns (norm_brand_match: 0/1, penalty_contribution: float)."""
    brand_norm = _norm_text(record.get("brand_norm"))
    if not brand_norm:
        return 0, 0.0

    desired_brand = _norm_text(query_context.get("brand"))
    match = 1 if desired_brand and (desired_brand in brand_norm or brand_norm in desired_brand) else 0

    penalty = 0.0
    for excluded in (query_context.get("exclude_brands") or []):
        excluded_norm = _norm_text(excluded)
        if excluded_norm and (excluded_norm in brand_norm or brand_norm in excluded_norm):
            penalty = -MAX_SOFT_PENALTY
            break

    return match, penalty


def _category_overlap(query_context: dict, record: dict) -> float:
    """
    Fraction of category hint terms (product_type + routed l1/l2, if given)
    found among the record's category_candidates. Neutral 0.5 when there's
    nothing to compare (no hint, or record has no category_candidates) —
    matches "if uncertain, leave score unchanged."
    """
    candidates = record.get("category_candidates") or []
    if not candidates:
        return 0.5

    hints = []
    if query_context.get("product_type"):
        hints.append(str(query_context["product_type"]))
    category = query_context.get("category") or {}
    if isinstance(category, dict):
        if category.get("l1"):
            hints.append(str(category["l1"]))
        if category.get("l2"):
            hints.append(str(category["l2"]))
    if not hints:
        return 0.5

    candidate_text = " ".join(_norm_text(c) for c in candidates)
    hits = sum(1 for h in hints if _norm_text(h) and _norm_text(h) in candidate_text)
    return round(hits / len(hints), 4)


def _rating_score(record: dict) -> float:
    rating = record.get("rating_norm")
    if rating is None:
        return 0.5
    try:
        return round(min(max(float(rating), 0.0), 5.0) / 5.0, 4)
    except (TypeError, ValueError):
        return 0.5


def _review_score(record: dict) -> float:
    """Log-scaled review count — log10(1)=0 .. log10(10000)=4, capped at 1.0."""
    count = record.get("review_count_norm")
    if count is None:
        return 0.5
    try:
        count = float(count)
    except (TypeError, ValueError):
        return 0.5
    if count <= 0:
        return 0.5
    return round(min(math.log10(count + 1) / 4.0, 1.0), 4)


def _metadata_quality_score(record: dict) -> float:
    quality = (record.get("metadata_quality") or {}).get("quality_score")
    if quality is None:
        return 0.5
    try:
        return round(min(max(float(quality), 0.0), 1.0), 4)
    except (TypeError, ValueError):
        return 0.5


def _attribute_text(record: dict) -> str:
    parts = []
    key_specs = record.get("key_specs") or {}
    if isinstance(key_specs, dict):
        parts.extend(str(v) for v in key_specs.values() if v)
    for field in ("compatibility", "materials", "features"):
        values = record.get(field) or []
        if isinstance(values, list):
            parts.extend(str(v) for v in values if v)
        elif values:
            parts.append(str(values))
    return _norm_text(" ".join(parts))


# Query slot keys that plausibly overlap with key_specs/compatibility/
# materials/features text (RAM, storage, Bluetooth, USB-C, material, water
# resistant, dishwasher safe, compatible devices, etc.)
_ATTRIBUTE_QUERY_SLOT_KEYS = (
    "product_type", "brand", "color", "use_case", "compatibility", "screen_size",
    "ram", "storage", "processor", "gpu", "os", "battery_life",
    "diet_type", "wet_or_dry", "life_stage", "protein_source",
    "size", "style", "material", "occasion", "skin_type", "benefit", "spf",
)


def _query_terms(query_context: dict) -> list:
    terms = []
    for key in _ATTRIBUTE_QUERY_SLOT_KEYS:
        val = query_context.get(key)
        if val:
            terms.append(str(val))
    for feat in (query_context.get("features") or []):
        terms.append(str(feat))
    return terms


def _attribute_overlap(query_context: dict, record: dict) -> float:
    terms = _query_terms(query_context)
    if not terms:
        return 0.5
    attr_text = _attribute_text(record)
    if not attr_text:
        return 0.5
    hits = sum(1 for t in terms if _norm_text(t) and _norm_text(t) in attr_text)
    return round(hits / len(terms), 4)


def _weak_signal_penalty(query_context: dict, record: dict) -> float:
    """
    Additional soft penalty from color/exclude_features conflicts against
    normalized fields — deliberately weak (a fraction of MAX_SOFT_PENALTY on
    its own) and only applied on a clear conflict, never on missing data.
    color_norm/product_type_norm/gender/negative_constraints are read-only
    evidence here, never hard filters (VerifierAgent already owns hard
    rejection and is untouched by this module).
    """
    penalty = 0.0

    desired_color = _norm_text(query_context.get("color"))
    product_color = _norm_text(record.get("color_norm"))
    if desired_color and product_color and desired_color != product_color \
            and desired_color not in product_color and product_color not in desired_color:
        penalty -= MAX_SOFT_PENALTY * 0.3

    exclude_features = [_norm_text(f) for f in (query_context.get("exclude_features") or [])]
    if exclude_features:
        negatives = " ".join(_norm_text(n) for n in (record.get("negative_constraints") or []))
        attr_text = _attribute_text(record)
        for ef in exclude_features:
            if ef and (ef in negatives or ef in attr_text):
                penalty -= MAX_SOFT_PENALTY * 0.5
                break

    return round(max(penalty, -MAX_SOFT_PENALTY), 4)


def compute_normalized_metadata_features(query_context: dict, product_id: str, normalized_record: dict | None) -> dict:
    """
    Compute soft evidence features from normalized metadata for one product.

    Never raises, never returns a value that could hard-reject a candidate.
    `normalized_record=None` (no metadata found for this product_id) returns
    fully-neutral features — the candidate is scored exactly as if the
    normalized-metadata layer didn't exist for it.

    query_context: the session slots dict, optionally with a "category"
    key ({"l1": ..., "l2": ...}) for category-overlap scoring.
    """
    query_context = query_context or {}

    if not normalized_record:
        features = _empty_features()
        _record_diagnostics(False, features["norm_category_overlap"], features["norm_attribute_overlap"])
        return features

    brand_match, brand_penalty = _brand_feature(query_context, normalized_record)
    weak_penalty = _weak_signal_penalty(query_context, normalized_record)
    total_penalty = round(max(brand_penalty + weak_penalty, -MAX_SOFT_PENALTY), 4)

    category_overlap = _category_overlap(query_context, normalized_record)
    attribute_overlap = _attribute_overlap(query_context, normalized_record)

    _record_diagnostics(True, category_overlap, attribute_overlap)

    return {
        "norm_brand_match":       brand_match,
        "norm_category_overlap":  category_overlap,
        "norm_rating_score":      _rating_score(normalized_record),
        "norm_review_score":      _review_score(normalized_record),
        "norm_metadata_quality":  _metadata_quality_score(normalized_record),
        "norm_attribute_overlap": attribute_overlap,
        "norm_soft_penalty":      total_penalty,
    }
