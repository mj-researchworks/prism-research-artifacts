"""
Reranker
---------
Category-aware scoring of search results. No LLM in the ranking path.

Design: rather than a small set of hand-picked static weight templates keyed
off node_config's `ranking_features` (most nodes' declared features didn't
match any template, e.g. "diet_fit"/"style_fit"/"color_fit" had no scorer at
all and silently fell back to a generic default), weights are now built
per-query: a component only participates if the slot(s) it depends on are
actually filled. A laptop query with `ram`/`gpu` filled gets spec scoring; a
pet-food query with `diet_type`/`wet_or_dry` filled gets diet scoring; an
unfilled slot simply doesn't dilute the score with a meaningless neutral
value. This uses whatever CategorySlotAgent/NERAgent actually extracted,
which generalises across categories without per-node config maintenance.

Scoring components:
  - semantic_score      : cosine similarity from FAISS (always on)
  - price_fit           : price vs price_min/price_max (always on)
  - rating_score        : normalised product rating (always on)
  - keyword_match_score : fraction of ALL filled slot terms found in
                          title + generated description (always on) —
                          broader than feature_fit, covers every slot the
                          user gave us, not just the "features" list
  - brand_fit, feature_fit, use_case_fit, gender_fit, compat_fit,
    screen_size_fit, age_fit, color_fit                (slot-gated)
  - ram_fit, storage_fit, gpu_fit, processor_fit         (electronics)
  - diet_fit, life_stage_fit, wet_or_dry_fit, protein_fit (pet food)
  - skin_type_fit, spf_fit, fragrance_free_fit, benefit_fit (beauty)
  - style_fit, material_fit, size_fit                    (clothing/shoes)
  - norm_brand_match, norm_category_overlap, norm_rating_score,
    norm_review_score, norm_metadata_quality, norm_attribute_overlap
    (optional, small-weight — see src/agents/normalized_metadata.py, gated
    behind MACPRS_USE_NORMALIZED_METADATA, default off)

exclusion_fit is NOT part of the weighted sum — it's a hard multiplicative
gate (0.0 or 1.0) applied to the final score, since an excluded brand or
feature should tank a candidate regardless of how well everything else
matches, not just nudge a weighted average.

Final score = weighted sum of active components, times exclusion_fit.
No LLM call, no external API — pure arithmetic.
"""

import json
import os
import re
from pathlib import Path

from src.taxonomy.text_norm import tokenize as _tokenize
from src.agents.normalized_metadata import (
    NORMALIZED_METADATA_ENABLED,
    get_normalized_product,
    compute_normalized_metadata_features,
)


# ── Always-on baseline weights ─────────────────────────────────────────────────
BASE_WEIGHTS = {
    "semantic_score":      0.35,
    "price_fit":           0.15,
    "rating_score":        0.10,
    "keyword_match_score": 0.20,
}

# ── Slot-gated component weights — included only if the relevant slot(s)
# are filled. (slot_keys, weight) — slot_keys is what must be non-empty in
# `slots` for this component to participate.
SLOT_GATED_WEIGHTS = {
    "brand_fit":            (["brand"],            0.12),
    "feature_fit":          (["features"],          0.12),
    "use_case_fit":         (["use_case"],          0.12),
    "gender_fit":           (["gender"],            0.10),
    "compat_fit":           (["compatibility"],     0.12),
    "screen_size_fit":      (["screen_size"],       0.10),
    "age_fit":              (["age_range"],         0.10),
    "color_fit":            (["color"],             0.08),
    # Electronics
    "ram_fit":              (["ram"],               0.12),
    "storage_fit":          (["storage"],           0.12),
    "gpu_fit":              (["gpu"],               0.12),
    "processor_fit":        (["processor"],         0.12),
    # Pet food
    "diet_fit":             (["diet_type"],         0.12),
    "life_stage_fit":       (["life_stage"],        0.10),
    "wet_or_dry_fit":       (["wet_or_dry"],        0.10),
    "protein_fit":          (["protein_source"],    0.10),
    # Beauty
    "skin_type_fit":        (["skin_type"],         0.12),
    "spf_fit":              (["spf"],               0.12),
    "fragrance_free_fit":   (["fragrance_free"],    0.10),
    "benefit_fit":          (["benefit"],           0.12),
    # Clothing / shoes
    "style_fit":            (["style"],             0.12),
    "material_fit":         (["material"],          0.10),
    "size_fit":             (["size", "dog_size"],  0.10),
}

SCOPE_SCORE_MULTIPLIER = {
    0: 1.00,  # routed node, usually L2
    1: 0.96,  # L1 backup: slight penalty, still allowed to win
    2: 0.90,  # global fallback: stronger penalty
}

# ── Normalized-metadata soft features (MACPRS_USE_NORMALIZED_METADATA) ────────
# Gated behind src.agents.normalized_metadata.NORMALIZED_METADATA_ENABLED
# (default off). Weights are deliberately small and, like SLOT_GATED_WEIGHTS,
# only enter the weighted average when there's a real query signal to compare
# against — a brand-match weight is meaningless noise if the user never
# named a brand. Kept separate from SLOT_GATED_WEIGHTS so this whole block
# can be disabled/removed without touching the existing scoring logic.
NORMALIZED_METADATA_WEIGHTS = {
    "norm_brand_match":       0.05,
    "norm_category_overlap":  0.04,
    "norm_rating_score":      0.03,
    "norm_review_score":      0.02,
    "norm_metadata_quality":  0.02,
    "norm_attribute_overlap": 0.05,
}

REVEAL_RERANK_RESCUE_ENABLED = os.getenv(
    "MACPRS_USE_REVEAL_RERANK_RESCUE_V1", "0"
).strip().lower() in {"1", "true", "yes", "on"}

# Controlled reranker experiment. Default off so the stable baseline is unchanged.
# Enables product identity/title fit, category agreement, accessory mismatch
# penalty, and a weak raw-retrieval-rank prior.
CONTROLLED_RERANKER_V1_ENABLED = os.getenv(
    "MACPRS_USE_CONTROLLED_RERANKER_V1", "0"
).strip().lower() in {"1", "true", "yes", "on"}

CONTROLLED_RERANKER_WEIGHTS = {
    "product_identity_fit": 0.12,
    "category_fit": 0.08,
    "accessory_fit": 0.08,
    "retrieval_rank_score": 0.04,
}

PRODUCT_TYPE_GUARD_ENABLED = os.getenv(
    "MACPRS_USE_PRODUCT_TYPE_GUARD", "1"
).strip().lower() in {"1", "true", "yes", "on"}

PRODUCT_TYPE_GUARD_WEIGHTS = {
    "product_identity_fit": 0.16,
    "category_fit": 0.08,
    "accessory_fit": 0.10,
}

REVEAL_RESCUE_MAX_BOOST = 0.04

# Learned triplet reranker. Enabled by default for the integrated final system.
# It fuses the current rule reranker with a locally trained triplet model over
# the same retrieved candidate pool.
TRIPLET_RERANKER_ENABLED = os.getenv(
    "MACPRS_USE_TRIPLET_RERANKER", "1"
).strip().lower() in {"1", "true", "yes", "on"}

def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default

TRIPLET_BLEND_WEIGHT = max(0.0, min(1.0, _float_env("MACPRS_TRIPLET_BLEND_WEIGHT", 0.6)))
TRIPLET_FUSION_MODE = os.getenv("MACPRS_TRIPLET_FUSION_MODE", "score").strip().lower()
if TRIPLET_FUSION_MODE not in {"rank", "score"}:
    TRIPLET_FUSION_MODE = "rank"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRIPLET_MODEL_PATH = Path(os.getenv(
    "MACPRS_TRIPLET_MODEL_PATH",
    str(_PROJECT_ROOT / "e5_780_best_model"),
))
TRIPLET_PRODUCT_TEXT_PATH = Path(os.getenv(
    "MACPRS_TRIPLET_PRODUCT_TEXT_PATH",
    str(_PROJECT_ROOT / "data" / "product_base_retrieval" / "product_base_retrieval_text.json"),
))
TRIPLET_PRODUCT_CACHE_PATH = Path(os.getenv(
    "MACPRS_TRIPLET_PRODUCT_CACHE_PATH",
    str(TRIPLET_MODEL_PATH.parent / "product_embedding_cache.pt"),
))

# Top-5 diversity pass. Enabled by default for the integrated final system to
# improve Hit@5 when top results contain near-duplicate variants from the same
# brand/product family.
TOP5_DIVERSITY_RERANK_ENABLED = os.getenv(
    "MACPRS_USE_TOP5_DIVERSITY_RERANK", "1"
).strip().lower() in {"1", "true", "yes", "on"}
TOP5_DIVERSITY_THRESHOLD = max(0.0, min(1.0, _float_env("MACPRS_TOP5_DIVERSITY_THRESHOLD", 0.25)))
# When the user explicitly asks for a brand, same-brand products are expected.
# Only delay same-brand candidates then if they are very near-identical variants.
TOP5_DIVERSITY_BRAND_REQUESTED_THRESHOLD = max(
    0.0, min(1.0, _float_env("MACPRS_TOP5_DIVERSITY_BRAND_REQUESTED_THRESHOLD", 0.75))
)
try:
    TOP5_DIVERSITY_MAX_SIMILAR = max(1, int(os.getenv("MACPRS_TOP5_DIVERSITY_MAX_SIMILAR", "1")))
except ValueError:
    TOP5_DIVERSITY_MAX_SIMILAR = 1

_TRIPLET_DEVICE = os.getenv("MACPRS_TRIPLET_DEVICE", "cpu")
_TRIPLET_STATE = None
_TRIPLET_PRODUCT_EMBEDDINGS = {}
_TRIPLET_LOAD_ERROR = None


def _load_triplet_family_config() -> dict:
    marker_path = TRIPLET_MODEL_PATH / "macprs_triplet_family.json"
    if not marker_path.exists():
        marker_path = TRIPLET_MODEL_PATH.parent / "inference_config.json"
    if not marker_path.exists():
        return {}
    try:
        with marker_path.open() as f:
            return json.load(f)
    except Exception:
        return {}


def _is_qwen_decoder_triplet(family_config: dict) -> bool:
    family = str(family_config.get("triplet_model_family") or "").lower()
    pooling = str(family_config.get("pooling") or "").lower()
    return "qwen" in family or pooling == "last_non_padding_token"


def _strip_legacy_prefix(text: str, prefix: str) -> str:
    text = str(text or "").strip()
    marker = prefix.lower() + ":"
    while text.lower().startswith(marker):
        text = text[len(marker):].strip()
    return text


def _format_triplet_texts(texts: list[str], state: dict, is_query: bool = False) -> list[str]:
    if not state.get("is_qwen_decoder"):
        return texts
    if is_query:
        instruction = state.get("query_instruction") or (
            "Given a shopping request, retrieve products that satisfy "
            "the user's current requirements and preferences"
        )
        return [
            f"Instruct: {instruction}\nQuery: {_strip_legacy_prefix(text, 'query')}"
            for text in texts
        ]
    return [_strip_legacy_prefix(text, "passage") for text in texts]


def _last_token_pool(last_hidden_states, attention_mask):
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        last_hidden_states.new_tensor(range(batch_size), dtype=sequence_lengths.dtype),
        sequence_lengths,
    ]


def _load_triplet_state():
    """Lazy-load the local triplet model only when the experiment flag is on."""
    global _TRIPLET_STATE, _TRIPLET_LOAD_ERROR, _TRIPLET_PRODUCT_EMBEDDINGS
    if _TRIPLET_STATE is not None or _TRIPLET_LOAD_ERROR is not None:
        return _TRIPLET_STATE
    try:
        os.environ.setdefault("HF_MODULES_CACHE", "/private/tmp/macprs_hf_modules_cache")
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer

        family_config = _load_triplet_family_config()
        is_qwen_decoder = _is_qwen_decoder_triplet(family_config)

        tokenizer = AutoTokenizer.from_pretrained(
            str(TRIPLET_MODEL_PATH),
            local_files_only=True,
            trust_remote_code=True,
        )
        if is_qwen_decoder:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(
            str(TRIPLET_MODEL_PATH),
            local_files_only=True,
            trust_remote_code=True,
        )
        model.config.use_cache = False
        model.to(_TRIPLET_DEVICE)
        model.eval()

        product_texts = {}
        with TRIPLET_PRODUCT_TEXT_PATH.open() as f:
            rows = json.load(f)
        if isinstance(rows, dict):
            iterator = rows.items()
            for pid, row in iterator:
                if isinstance(row, dict):
                    product_texts[str(row.get("product_id") or pid)] = str(row.get("text") or "")
                else:
                    product_texts[str(pid)] = str(row or "")
        else:
            for row in rows:
                if isinstance(row, dict) and row.get("product_id"):
                    product_texts[str(row["product_id"])] = str(row.get("text") or "")

        if TRIPLET_PRODUCT_CACHE_PATH.exists():
            cache = torch.load(TRIPLET_PRODUCT_CACHE_PATH, map_location="cpu")
            embeddings = cache.get("embeddings") if isinstance(cache, dict) else None
            if isinstance(embeddings, dict):
                _TRIPLET_PRODUCT_EMBEDDINGS = {
                    str(pid): emb.detach().cpu() if hasattr(emb, "detach") else torch.as_tensor(emb)
                    for pid, emb in embeddings.items()
                }

        _TRIPLET_STATE = {
            "torch": torch,
            "F": F,
            "tokenizer": tokenizer,
            "model": model,
            "product_texts": product_texts,
            "family_config": family_config,
            "is_qwen_decoder": is_qwen_decoder,
            "query_instruction": family_config.get("query_instruction"),
            "product_cache_path": TRIPLET_PRODUCT_CACHE_PATH,
            "product_cache_count": len(_TRIPLET_PRODUCT_EMBEDDINGS),
        }
    except Exception as exc:  # pragma: no cover - defensive experiment fallback
        _TRIPLET_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        _TRIPLET_STATE = None
    return _TRIPLET_STATE


def _triplet_encode(texts: list[str], state: dict, is_query: bool = False):
    torch = state["torch"]
    F = state["F"]
    tokenizer = state["tokenizer"]
    model = state["model"]
    formatted_texts = _format_triplet_texts(texts, state, is_query=is_query)
    with torch.no_grad():
        encoded = tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {k: v.to(_TRIPLET_DEVICE) for k, v in encoded.items()}
        output = model(**encoded, use_cache=False)
        token_embeddings = output.last_hidden_state
        if state.get("is_qwen_decoder"):
            pooled = _last_token_pool(token_embeddings, encoded["attention_mask"])
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=1).cpu()


def get_triplet_reranker_diagnostics() -> dict:
    return {
        "triplet_reranker_enabled": TRIPLET_RERANKER_ENABLED,
        "triplet_fusion_mode": TRIPLET_FUSION_MODE,
        "triplet_blend_weight": TRIPLET_BLEND_WEIGHT,
        "triplet_model_path": str(TRIPLET_MODEL_PATH),
        "triplet_product_text_path": str(TRIPLET_PRODUCT_TEXT_PATH),
        "triplet_product_cache_path": str(TRIPLET_PRODUCT_CACHE_PATH),
        "triplet_product_cache_count": (
            _TRIPLET_STATE.get("product_cache_count") if _TRIPLET_STATE else len(_TRIPLET_PRODUCT_EMBEDDINGS)
        ),
        "triplet_load_error": _TRIPLET_LOAD_ERROR,
    }

REVEAL_RESCUE_GENERIC_TERMS = {
    "need", "want", "show", "buy", "looking", "look", "good", "best",
    "product", "item", "thing", "something", "ones", "one", "option",
    "cheap", "cheaper", "expensive", "more", "less", "another",
    "different", "similar", "same",
}

KNOWN_COLORS = {
    "black", "white", "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "brown", "gray", "grey", "silver", "gold", "beige", "cream",
    "navy", "teal", "clear",
}

_reveal_rescue_diagnostics = {
    "candidates_scored": 0,
    "candidates_boosted": 0,
    "sum_boost": 0.0,
    "max_boost": 0.0,
    "top10_changed_count": 0,
}


def reset_reveal_rerank_rescue_diagnostics():
    global _reveal_rescue_diagnostics
    _reveal_rescue_diagnostics = {
        "candidates_scored": 0,
        "candidates_boosted": 0,
        "sum_boost": 0.0,
        "max_boost": 0.0,
        "top10_changed_count": 0,
    }


def get_reveal_rerank_rescue_diagnostics() -> dict:
    scored = _reveal_rescue_diagnostics["candidates_scored"] or 1
    return {
        "reveal_rerank_rescue_enabled": REVEAL_RERANK_RESCUE_ENABLED,
        "reveal_rerank_rescue_candidates_scored": _reveal_rescue_diagnostics["candidates_scored"],
        "reveal_rerank_rescue_candidates_boosted": _reveal_rescue_diagnostics["candidates_boosted"],
        "reveal_rerank_rescue_avg_boost": round(_reveal_rescue_diagnostics["sum_boost"] / scored, 4),
        "reveal_rerank_rescue_max_boost": round(_reveal_rescue_diagnostics["max_boost"], 4),
        "reveal_rerank_rescue_top10_changed_count": _reveal_rescue_diagnostics["top10_changed_count"],
    }


def _slot_filled(slots: dict, key: str) -> bool:
    val = slots.get(key)
    if val is None:
        return False
    if isinstance(val, list):
        return len(val) > 0
    return bool(str(val).strip())


def _product_text(product: dict) -> str:
    return ((product.get("title") or "") + " " + (product.get("description") or "")).lower()


def _product_title(product: dict) -> str:
    return (product.get("title") or "").lower()


def _product_brand(product: dict) -> str:
    return str(product.get("brand") or "").lower().strip()


def _diversity_token_set(product: dict) -> set[str]:
    text = " ".join(_flatten_values({
        "title": product.get("title"),
        "brand": product.get("brand"),
        "product_type": product.get("product_type"),
        "l1": product.get("l1"),
        "l2": product.get("l2"),
    }))
    generic = REVEAL_RESCUE_GENERIC_TERMS | {
        "product", "title", "with", "from", "pack", "black", "white",
        "large", "small", "medium", "men", "women", "kids", "new",
        "set", "and", "for", "the", "of", "inch", "oz",
    }
    return {
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 2 and t not in generic
    }


def _token_jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _meaningful_tokens(text: str) -> list[str]:
    stop = REVEAL_RESCUE_GENERIC_TERMS | {
        "for", "with", "without", "and", "or", "the", "a", "an",
        "from", "under", "over", "less", "more", "new", "used",
    }
    return [
        t for t in re.findall(r"[a-z0-9+#&/\-]+", str(text or "").lower())
        if len(t) > 2 and t not in stop
    ]


def _compact_alnum(text: str) -> str:
    """Case/space/punctuation-insensitive form for model/spec matching."""
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _normalised_spec_text(text: str) -> str:
    """
    Normalize common catalog/query spec variants:
    - "RTX 4070 Ti" == "rtx4070ti"
    - "16 GB" == "16gb"
    - "1 TB" == "1tb"
    """
    text = str(text or "").lower()
    text = re.sub(r"\b(\d+)\s*(gb|tb|mb|hz|mah|mp)\b", r"\1\2", text)
    text = re.sub(r"\brtx\s*(\d{3,4})\s*ti\b", r"rtx\1ti", text)
    text = re.sub(r"\bgtx\s*(\d{3,4})\b", r"gtx\1", text)
    text = re.sub(r"\bryzen\s*([3579])\b", r"ryzen\1", text)
    text = re.sub(r"\bcore\s*i\s*([3579])\b", r"i\1", text)
    return text


def _contains(text: str, term) -> bool:
    """
    Raw substring match first (cheap, precise for phrases); falls back to a
    stemmed head-noun check for plural/singular and filler-word mismatches
    the substring check misses — e.g. query "knives" against a title saying
    "Knife Set", or "computers" against "Computer Desktop". Found via
    reranker failure audit (scripts/11_reranker_failure_audit.py): these
    accounted for the single largest per-component score gaps between the
    target and the wrongly top-ranked product.

    The fallback checks only the LAST stemmed token, not all of them. Query
    product_types often come out of NER as noisy noun phrases with a leading
    filler ("pair of sunglasses", "box of tea bags") where English puts the
    head noun last. Requiring every token was actively wrong: for "pair of
    sunglasses", the true match (titled plainly "Sunglasses") lost to an
    unrelated pair of clip-ons whose title coincidentally said "3 Pair" —
    matching the filler word "pair" is not evidence of relevance, matching
    the head noun "sunglasses" is.
    """
    if not term:
        return False
    term = str(term).lower()
    normalized_text = _normalised_spec_text(text)
    normalized_term = _normalised_spec_text(term)
    if term in text or normalized_term in normalized_text:
        return True
    compact_term = _compact_alnum(normalized_term)
    compact_text = _compact_alnum(normalized_text)
    if len(compact_term) >= 3 and compact_term in compact_text:
        return True
    term_tokens = _tokenize(normalized_term, bigrams=False)
    if not term_tokens:
        return False
    text_tokens = set(_tokenize(normalized_text, bigrams=False))
    singular_text_tokens = {tok[:-1] if tok.endswith("s") and len(tok) > 3 else tok for tok in text_tokens}
    head = term_tokens[-1]
    singular_head = head[:-1] if head.endswith("s") and len(head) > 3 else head
    return head in text_tokens or singular_head in singular_text_tokens


def _flatten_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        out = []
        for v in value.values():
            out.extend(_flatten_values(v))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for v in value:
            out.extend(_flatten_values(v))
        return out
    return [str(value)]


def _rescue_text(product: dict) -> str:
    values = []
    for key in (
        "title", "description", "brand", "color", "category", "l1", "l2",
        "product_type", "product_type_norm", "text",
    ):
        values.extend(_flatten_values(product.get(key)))
    values.extend(_flatten_values(product.get("ranking_metadata")))
    values.extend(_flatten_values(product.get("all_original_attributes")))
    record = get_normalized_product(product.get("product_id"))
    if record:
        values.extend(_flatten_values(record))
    return " ".join(values).lower()


def _clean_rescue_term(term: str) -> str:
    text = re.sub(r"[^a-z0-9.+#&/\-\s]", " ", str(term or "").lower())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    words = [w for w in text.split() if w and w not in REVEAL_RESCUE_GENERIC_TERMS]
    if not words:
        return ""
    return " ".join(words[:6])


def _rescue_terms(value) -> list[str]:
    out = []
    for item in _flatten_values(value):
        term = _clean_rescue_term(item)
        if term and term not in out:
            out.append(term)
    return out


def _keyword_overlap_boost(text: str, query: str, max_boost: float) -> float:
    terms = [
        t for t in re.findall(r"[a-z0-9+#&/\-]+", str(query or "").lower())
        if len(t) > 2 and t not in REVEAL_RESCUE_GENERIC_TERMS
    ][:8]
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if _contains(text, term))
    if not hits:
        return 0.0
    return max_boost * min(1.0, hits / max(len(terms), 1))


def _has_clear_reveal_contradiction(product: dict, text: str, reveal_context: dict) -> bool:
    constraints = reveal_context.get("explicit_constraints") or {}
    product_brand = str(product.get("brand") or "").lower()
    for brand in constraints.get("exclude_brands") or []:
        brand = _clean_rescue_term(brand)
        if brand and (brand in product_brand or _contains(text, brand)):
            return True

    desired_color = _clean_rescue_term(constraints.get("color"))
    product_color = _clean_rescue_term(product.get("color"))
    if desired_color and product_color and desired_color in KNOWN_COLORS and product_color in KNOWN_COLORS:
        if desired_color != product_color:
            return True

    price_max = constraints.get("price_max")
    price = product.get("price")
    if price_max is not None and price is not None:
        try:
            if float(price) > float(price_max):
                return True
        except (TypeError, ValueError):
            pass
    return False


def compute_reveal_rerank_rescue_score(product: dict, reveal_context: dict) -> float:
    """Small reveal-only additive boost. Returns 0.0 when disabled/irrelevant."""
    if not REVEAL_RERANK_RESCUE_ENABLED or not reveal_context:
        return 0.0

    text = _rescue_text(product)
    if _has_clear_reveal_contradiction(product, text, reveal_context):
        return 0.0

    constraints = reveal_context.get("explicit_constraints") or {}
    query_views = reveal_context.get("query_views") or {}
    boost = 0.0

    product_type = _clean_rescue_term(reveal_context.get("explicit_product_type"))
    if product_type and _contains(text, product_type):
        boost += 0.015

    category_hits = 0
    for hint in _rescue_terms(reveal_context.get("explicit_category_hints"))[:3]:
        if _contains(text, hint):
            category_hits += 1
    if category_hits:
        boost += 0.005

    brand = _clean_rescue_term(constraints.get("brand"))
    if brand and (_contains(text, brand) or brand in str(product.get("brand") or "").lower()):
        boost += 0.015

    color = _clean_rescue_term(constraints.get("color"))
    if color and (_contains(text, color) or color in _clean_rescue_term(product.get("color"))):
        boost += 0.01

    feature_hits = 0
    for term in _rescue_terms(constraints.get("features"))[:4]:
        if _contains(text, term):
            feature_hits += 1
    if feature_hits:
        boost += min(0.01, 0.005 * feature_hits)

    compat_hits = 0
    for term in _rescue_terms(constraints.get("compatibility"))[:3]:
        if _contains(text, term):
            compat_hits += 1
    if compat_hits:
        boost += min(0.01, 0.005 * compat_hits)

    product_tokens = set(product_type.split()) if product_type else set()
    for key in ("attribute_query", "fallback_keyword_query"):
        query = _clean_rescue_term(query_views.get(key))
        query_tokens = set(query.split())
        if product_tokens and query_tokens and query_tokens <= product_tokens:
            continue
        boost += _keyword_overlap_boost(text, query, 0.0025)

    return round(min(REVEAL_RESCUE_MAX_BOOST, max(0.0, boost)), 4)


def _has_slot_price_contradiction(product: dict, slots: dict) -> bool:
    price = product.get("price")
    if price is None:
        return False
    try:
        price = float(price)
    except (TypeError, ValueError):
        return False

    price_max = slots.get("price_max")
    if price_max is not None:
        try:
            if price > float(price_max):
                return True
        except (TypeError, ValueError):
            pass

    price_min = slots.get("price_min")
    if price_min is not None:
        try:
            if price < float(price_min):
                return True
        except (TypeError, ValueError):
            pass
    return False


# ── Always-on components ───────────────────────────────────────────────────────

def _price_fit(product: dict, slots: dict) -> float:
    """
    1.0 if within budget, scaled penalty if over. Neutral 0.5 if the user
    gave no price constraint at all — previously this returned 1.0 for any
    product with a known price and 0.5 for unknown-price products even with
    no stated budget, which rewarded "has price metadata" over actual fit.
    Found via reranker failure audit: affected 14/36 sampled failures, the
    single most common cause of the wrong product outscoring the target.
    """
    price_max = slots.get("price_max")
    price_min = slots.get("price_min")
    if price_max is None and price_min is None:
        return 0.5  # no stated budget — can't judge fit either way

    price = product.get("price")
    if price is None:
        return 0.5  # unknown price — neutral

    score = 1.0
    if price_max is not None and price > price_max:
        overage = (price - price_max) / max(price_max, 1)
        score *= max(0.0, 1.0 - overage)
    if price_min is not None and price < price_min:
        score *= 0.7  # soft penalty for being below min (may be too cheap)

    return round(score, 4)


def _rating_score(product: dict) -> float:
    """Normalise rating 0-5 -> 0-1. Missing rating -> 0.5."""
    rating = product.get("rating")
    if rating is None:
        return 0.5
    return round(min(max(float(rating), 0), 5) / 5.0, 4)


_KEYWORD_SLOT_KEYS = [
    "product_type", "brand", "color", "use_case", "compatibility", "screen_size",
    "ram", "storage", "processor", "gpu", "os", "battery_life",
    "diet_type", "wet_or_dry", "life_stage", "protein_source",
    "size", "style", "material", "occasion",
    "skin_type", "benefit", "spf",
]


def _collect_keyword_terms(slots: dict) -> list:
    terms = []
    for key in _KEYWORD_SLOT_KEYS:
        val = slots.get(key)
        if not val:
            continue
        terms.append(str(val))
    for feat in (slots.get("features") or []):
        terms.append(str(feat))
    if slots.get("fragrance_free"):
        terms.append("fragrance free")
    return terms


def _keyword_match_score(product: dict, slots: dict) -> float:
    """Fraction of every filled-slot term found (exact/substring) in title+description."""
    terms = _collect_keyword_terms(slots)
    if not terms:
        return 0.5
    text = _product_text(product)
    hits = sum(1 for t in terms if _contains(text, t))
    return round(hits / len(terms), 4)


def _product_identity_fit(product: dict, slots: dict) -> float:
    """Prefer products whose title clearly names the requested product type."""
    product_type = (slots.get("product_type") or "").strip().lower()
    if not product_type:
        return 0.5
    title = _product_title(product)
    text = _product_text(product)
    if _contains(title, product_type):
        return 1.0
    tokens = _meaningful_tokens(product_type)
    if not tokens:
        return 0.5
    title_hits = sum(1 for tok in tokens if _contains(title, tok))
    if title_hits == len(tokens):
        return 0.9
    if title_hits >= max(1, len(tokens) - 1):
        return 0.75
    head = tokens[-1]
    if _contains(title, head):
        return 0.65
    if _contains(text, product_type):
        return 0.55
    return 0.2


def _category_fit(product: dict, slots: dict) -> float:
    """Soft category agreement; never a hard filter."""
    target_l1 = (slots.get("_rerank_category_l1") or "").strip().lower()
    target_l2 = (slots.get("_rerank_category_l2") or "").strip().lower()
    product_l1 = (product.get("l1") or "").strip().lower()
    product_l2 = (product.get("l2") or "").strip().lower()
    if not target_l1 and not target_l2:
        return 0.5
    if target_l2 and product_l2 and target_l2 == product_l2:
        return 1.0
    if target_l1 and product_l1 and target_l1 == product_l1:
        return 0.78
    return 0.25


def _accessory_fit(product: dict, slots: dict) -> float:
    """Penalize accessory products when the user asked for the core product."""
    product_type = (slots.get("product_type") or "").lower()
    if not product_type:
        return 0.5
    title = _product_title(product)
    accessory_by_core = {
        "phone": ("phone case", "case", "charger", "charging cable", "screen protector", "mount", "tripod"),
        "smartphone": ("phone case", "case", "charger", "charging cable", "screen protector", "mount", "tripod"),
        "laptop": ("sleeve", "case", "charger", "adapter", "stand", "laptop bag", "cooling pad"),
        "computer": ("desk", "table", "stand", "adapter", "cable", "computer desk"),
        "mouse": ("mouse pad", "mousepad", "pad", "mat", "toy mouse", "pet mouse"),
        "keyboard": ("keycap", "keycaps", "cover", "wrist rest", "switches"),
        "camera": ("camera bag", "tripod", "lens cap", "mount", "case"),
        "watch": ("band", "strap", "charger", "case"),
        "headphone": ("case", "ear pads", "earpads", "cable", "stand"),
        "headphones": ("case", "ear pads", "earpads", "cable", "stand"),
    }
    for core, accessories in accessory_by_core.items():
        if core in product_type:
            if any(acc in product_type for acc in accessories):
                return 1.0
            if any(acc in title for acc in accessories):
                return 0.15
    return 1.0


def _retrieval_rank_score(product: dict, slots: dict) -> float:
    rank = product.get("_retrieval_rank")
    if rank is None:
        return 0.5
    try:
        rank = int(rank)
    except (TypeError, ValueError):
        return 0.5
    if rank <= 1:
        return 1.0
    # Smooth prior: rank 10 ~= .87, rank 50 ~= .43, rank 100 ~= .19.
    return round(max(0.1, 1.0 - ((rank - 1) / 110.0)), 4)


# ── Hard exclusion gate (multiplicative, not part of the weighted sum) ────────

def _exclusion_fit(product: dict, slots: dict) -> float:
    """0.0 if the product hits an excluded brand/feature, else 1.0."""
    text = _product_text(product)
    product_brand = (product.get("brand") or "").lower()

    for eb in (slots.get("exclude_brands") or []):
        if eb.lower() in product_brand:
            return 0.0
    for ef in (slots.get("exclude_features") or []):
        if _contains(text, ef):
            return 0.0
    return 1.0


# ── Slot-gated components ──────────────────────────────────────────────────────

def _brand_fit(product: dict, slots: dict) -> float:
    desired = (slots.get("brand") or "").lower().strip()
    product_brand = (product.get("brand") or "").lower()
    if desired in product_brand or product_brand in desired:
        return 1.0
    return 0.0


def _feature_fit(product: dict, slots: dict) -> float:
    features = slots.get("features") or []
    text = _product_text(product)
    hits = sum(1 for f in features if _contains(text, f))
    return round(hits / len(features), 4) if features else 0.5


def _use_case_fit(product: dict, slots: dict) -> float:
    use_case = (slots.get("use_case") or "").lower()
    text = (product.get("title") or "").lower()
    return 1.0 if use_case in text else 0.2


def _gender_fit(product: dict, slots: dict) -> float:
    gender = (slots.get("gender") or "").lower()
    product_gender = (product.get("gender") or "").lower()
    if not product_gender:
        return 0.5
    if product_gender == "unisex":
        return 0.8
    if gender in product_gender:
        return 1.0
    return 0.1


def _compat_fit(product: dict, slots: dict) -> float:
    compat = (slots.get("compatibility") or "").lower()
    text = _product_text(product)
    return 1.0 if compat in text else 0.2


def _screen_size_fit(product: dict, slots: dict) -> float:
    desired_str = slots.get("screen_size") or ""
    m = re.search(r"(\d+(?:\.\d+)?)", desired_str)
    if not m:
        return 0.5
    desired = float(m.group(1))

    text = (product.get("title") or "").lower()
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*(?:inch|\"|in\b)", text)
    if not m2:
        return 0.3
    actual = float(m2.group(1))

    diff = abs(actual - desired)
    if diff <= 1:
        return 1.0
    if diff <= 2:
        return 0.7
    return 0.2


def _age_fit(product: dict, slots: dict) -> float:
    age_range = slots.get("age_range") or ""
    text = (product.get("title") or "").lower()
    nums = re.findall(r"\d+", age_range)
    if any(n in text for n in nums):
        return 1.0
    return 0.3


def _color_fit(product: dict, slots: dict) -> float:
    desired = (slots.get("color") or "").lower()
    product_color = (product.get("color") or "").lower()
    if not product_color:
        return 0.5  # unknown — neutral
    if desired in product_color or product_color in desired:
        return 1.0
    return 0.15


# ── Electronics ─────────────────────────────────────────────────────────────────

def _spec_slot_fit(product: dict, slots: dict, slot_key: str) -> float:
    """Generic exact-substring fit for a normalised spec slot (ram/storage/gpu/processor)."""
    desired = (slots.get(slot_key) or "").lower()
    text = _product_text(product)
    if _contains(text, desired):
        return 1.0
    # Loose family match, e.g. desired "rtx 3060" vs text mentioning "rtx" only
    family = _normalised_spec_text(desired).split()[0] if desired else ""
    if family and _contains(text, family):
        return 0.5
    return 0.15


def _ram_fit(product, slots):       return _spec_slot_fit(product, slots, "ram")
def _storage_fit(product, slots):   return _spec_slot_fit(product, slots, "storage")
def _gpu_fit(product, slots):       return _spec_slot_fit(product, slots, "gpu")
def _processor_fit(product, slots): return _spec_slot_fit(product, slots, "processor")


# ── Pet food ────────────────────────────────────────────────────────────────────

def _diet_fit(product, slots):
    return 1.0 if _contains(_product_text(product), slots.get("diet_type")) else 0.2


def _life_stage_fit(product, slots):
    return 1.0 if _contains(_product_text(product), slots.get("life_stage")) else 0.3


def _wet_or_dry_fit(product, slots):
    desired = (slots.get("wet_or_dry") or "").lower()
    text = _product_text(product)
    if desired == "wet":
        return 1.0 if any(t in text for t in ("wet food", "canned", "pate")) else 0.2
    if desired == "dry":
        return 1.0 if any(t in text for t in ("dry food", "kibble")) else 0.2
    return 0.5


def _protein_fit(product, slots):
    return 1.0 if _contains(_product_text(product), slots.get("protein_source")) else 0.3


# ── Beauty ──────────────────────────────────────────────────────────────────────

def _skin_type_fit(product, slots):
    skin_type = slots.get("skin_type")
    return 1.0 if skin_type and f"{skin_type.lower()} skin" in _product_text(product) else 0.4


def _spf_fit(product, slots):
    text = _product_text(product)
    m = re.search(r"spf\s*(\d{2,3})", text)
    desired = slots.get("spf")
    if not desired:
        return 0.5
    try:
        desired_val = int(re.search(r"\d+", str(desired)).group())
    except (AttributeError, ValueError):
        return 0.5
    if not m:
        return 0.2  # no SPF info at all — likely doesn't have it
    actual_val = int(m.group(1))
    return 1.0 if actual_val >= desired_val else 0.3


def _fragrance_free_fit(product, slots):
    if not slots.get("fragrance_free"):
        return 0.5
    text = _product_text(product)
    return 1.0 if ("fragrance free" in text or "fragrance-free" in text or "unscented" in text) else 0.2


def _benefit_fit(product, slots):
    return 1.0 if _contains(_product_text(product), slots.get("benefit")) else 0.3


# ── Clothing / shoes ────────────────────────────────────────────────────────────

def _style_fit(product, slots):
    return 1.0 if _contains(_product_text(product), slots.get("style")) else 0.3


def _material_fit(product, slots):
    return 1.0 if _contains(_product_text(product), slots.get("material")) else 0.4


def _size_fit(product, slots):
    desired = slots.get("size") or slots.get("dog_size")
    return 1.0 if desired and _contains(_product_text(product), desired) else 0.4


_COMPONENT_SCORERS = {
    "product_identity_fit": _product_identity_fit,
    "category_fit": _category_fit,
    "accessory_fit": _accessory_fit,
    "retrieval_rank_score": _retrieval_rank_score,
    "brand_fit": _brand_fit, "feature_fit": _feature_fit, "use_case_fit": _use_case_fit,
    "gender_fit": _gender_fit, "compat_fit": _compat_fit, "screen_size_fit": _screen_size_fit,
    "age_fit": _age_fit, "color_fit": _color_fit,
    "ram_fit": _ram_fit, "storage_fit": _storage_fit, "gpu_fit": _gpu_fit, "processor_fit": _processor_fit,
    "diet_fit": _diet_fit, "life_stage_fit": _life_stage_fit,
    "wet_or_dry_fit": _wet_or_dry_fit, "protein_fit": _protein_fit,
    "skin_type_fit": _skin_type_fit, "spf_fit": _spf_fit,
    "fragrance_free_fit": _fragrance_free_fit, "benefit_fit": _benefit_fit,
    "style_fit": _style_fit, "material_fit": _material_fit, "size_fit": _size_fit,
}


# ── Public class ───────────────────────────────────────────────────────────────

class Reranker:
    """
    Re-ranks FAISS results using dynamic, slot-driven scoring.

    Usage:
        reranker = Reranker()
        ranked = reranker.rank(
            candidates=[...],       # from SearchAgent.search()
            slots={...},            # current session slots
            node_config={...},      # unused for weighting now, kept for API compat
            top_k=10,
        )
    """

    def rank(
        self,
        candidates: list,
        slots: dict,
        node_config: dict,
        top_k: int = 10,
        reveal_context: dict = None,
        triplet_context: str = None,
    ) -> list:
        query_context = self._build_query_context(slots, node_config)
        weights = self._build_weights(query_context)

        scored = []
        for idx, product in enumerate(candidates, start=1):
            product_for_score = dict(product)
            if CONTROLLED_RERANKER_V1_ENABLED:
                product_for_score["_retrieval_rank"] = idx
            score = self._score(product_for_score, query_context, weights)
            rescue_score = 0.0
            if REVEAL_RERANK_RESCUE_ENABLED and reveal_context:
                if self._passes_reveal_rescue_slot_gate(product_for_score, query_context, weights):
                    rescue_score = compute_reveal_rerank_rescue_score(product_for_score, reveal_context)
            final_score = score + rescue_score
            item = {**product_for_score, "rerank_score": round(final_score, 4)}
            if REVEAL_RERANK_RESCUE_ENABLED and reveal_context:
                item["base_rerank_score"] = round(score, 4)
                item["reveal_rescue_score"] = rescue_score
                _reveal_rescue_diagnostics["candidates_scored"] += 1
                _reveal_rescue_diagnostics["sum_boost"] += rescue_score
                _reveal_rescue_diagnostics["max_boost"] = max(
                    _reveal_rescue_diagnostics["max_boost"], rescue_score
                )
                if rescue_score > 0:
                    _reveal_rescue_diagnostics["candidates_boosted"] += 1
            scored.append(item)

        original_top10 = [
            p.get("product_id")
            for p in sorted(scored, key=lambda x: (-(x.get("base_rerank_score", x["rerank_score"])), x.get("scope_rank", 9)))[:top_k]
        ]
        scored.sort(key=lambda x: (-x["rerank_score"], x.get("scope_rank", 9)))
        if TRIPLET_RERANKER_ENABLED and triplet_context:
            scored = self._apply_triplet_rank_fusion(scored, triplet_context)
        if TOP5_DIVERSITY_RERANK_ENABLED:
            scored = self._apply_top5_diversity(scored, slots=query_context, top_k=top_k)
        if REVEAL_RERANK_RESCUE_ENABLED and reveal_context:
            rescued_top10 = [p.get("product_id") for p in scored[:top_k]]
            if rescued_top10 != original_top10:
                _reveal_rescue_diagnostics["top10_changed_count"] += 1
        return scored[:top_k]

    def _apply_triplet_rank_fusion(self, scored: list, triplet_context: str) -> list:
        state = _load_triplet_state()
        if not state or not scored:
            return scored

        context = " ".join(str(triplet_context or "").split())
        if not context:
            return scored

        try:
            context_emb = _triplet_encode([context], state, is_query=True)[0]
            product_texts = state["product_texts"]
            to_encode = []
            encode_ids = []
            product_embs = []
            for item in scored:
                pid = str(item.get("product_id") or "")
                if pid in _TRIPLET_PRODUCT_EMBEDDINGS:
                    product_embs.append(_TRIPLET_PRODUCT_EMBEDDINGS[pid])
                    continue
                text = product_texts.get(pid) or _rescue_text(item)
                encode_ids.append(pid)
                to_encode.append(text)
                product_embs.append(None)

            if to_encode:
                encoded = _triplet_encode(to_encode, state)
                encoded_iter = iter(encoded)
                for i, current in enumerate(product_embs):
                    if current is None:
                        emb = next(encoded_iter)
                        pid = encode_ids.pop(0)
                        if pid:
                            _TRIPLET_PRODUCT_EMBEDDINGS[pid] = emb
                        product_embs[i] = emb

            for item, emb in zip(scored, product_embs):
                item["base_rerank_score"] = item.get("base_rerank_score", item.get("rerank_score", 0.0))
                item["triplet_score"] = round(float((context_emb * emb).sum().item()), 4)

            rule_order = sorted(scored, key=lambda x: (-x.get("base_rerank_score", x.get("rerank_score", 0.0)), x.get("scope_rank", 9)))
            triplet_order = sorted(scored, key=lambda x: (-x.get("triplet_score", 0.0), x.get("scope_rank", 9)))
            n = max(len(scored), 1)
            rule_rank = {id(item): idx for idx, item in enumerate(rule_order)}
            triplet_rank = {id(item): idx for idx, item in enumerate(triplet_order)}
            w = TRIPLET_BLEND_WEIGHT
            rule_values = [float(item.get("base_rerank_score", item.get("rerank_score", 0.0)) or 0.0) for item in scored]
            triplet_values = [float(item.get("triplet_score", 0.0) or 0.0) for item in scored]

            def _normalize(values):
                if not values:
                    return []
                lo = min(values)
                hi = max(values)
                if hi <= lo:
                    return [1.0 for _ in values]
                return [(value - lo) / (hi - lo) for value in values]

            rule_norm = _normalize(rule_values)
            triplet_norm = _normalize(triplet_values)
            for idx, item in enumerate(scored):
                rr = rule_rank.get(id(item), n - 1)
                tr = triplet_rank.get(id(item), n - 1)
                rule_rank_score = 1.0 - (rr / max(n - 1, 1))
                triplet_rank_score = 1.0 - (tr / max(n - 1, 1))
                if TRIPLET_FUSION_MODE == "score":
                    rule_component = rule_norm[idx]
                    triplet_component = triplet_norm[idx]
                else:
                    rule_component = rule_rank_score
                    triplet_component = triplet_rank_score
                blend = ((1.0 - w) * rule_component) + (w * triplet_component)
                item["triplet_rank"] = tr + 1
                item["rule_rank"] = rr + 1
                item["triplet_fusion_mode"] = TRIPLET_FUSION_MODE
                item["triplet_blend_score"] = round(blend, 4)
                item["rerank_score"] = round(blend, 4)
            scored.sort(key=lambda x: (-x["triplet_blend_score"], x.get("scope_rank", 9)))
        except Exception as exc:  # pragma: no cover - experiment fallback
            for item in scored:
                item["triplet_error"] = f"{type(exc).__name__}: {exc}"
            scored.sort(key=lambda x: (-x.get("base_rerank_score", x.get("rerank_score", 0.0)), x.get("scope_rank", 9)))
        return scored

    def _apply_top5_diversity(self, scored: list, slots: dict = None, top_k: int = 10) -> list:
        if not scored:
            return scored
        selected = []
        delayed = []
        token_cache = {}
        requested_brand = str((slots or {}).get("brand") or "").lower().strip()

        def tokens(item):
            key = id(item)
            if key not in token_cache:
                token_cache[key] = _diversity_token_set(item)
            return token_cache[key]

        for item in scored:
            if len(selected) >= min(5, top_k):
                selected.append(item)
                continue
            brand = _product_brand(item)
            similar_count = 0
            for kept in selected[:5]:
                same_brand = brand and brand == _product_brand(kept)
                if not same_brand:
                    continue
                threshold = TOP5_DIVERSITY_THRESHOLD
                brand_was_requested = requested_brand and (requested_brand in brand or brand in requested_brand)
                if brand_was_requested:
                    threshold = TOP5_DIVERSITY_BRAND_REQUESTED_THRESHOLD
                if _token_jaccard(tokens(item), tokens(kept)) >= threshold:
                    similar_count += 1
            if similar_count >= TOP5_DIVERSITY_MAX_SIMILAR:
                item["top5_diversity_delayed"] = True
                delayed.append(item)
            else:
                selected.append(item)

        reordered = []
        for item in selected + delayed:
            if id(item) not in {id(x) for x in reordered}:
                reordered.append(item)
        for item in scored:
            if id(item) not in {id(x) for x in reordered}:
                reordered.append(item)
        for idx, item in enumerate(reordered, start=1):
            item["top5_diversity_rank"] = idx
        return reordered

    def _passes_reveal_rescue_slot_gate(self, product: dict, slots: dict, weights: dict) -> bool:
        """Keep reveal rescue from bypassing active hard/near-hard constraints."""
        if _exclusion_fit(product, slots) <= 0.0:
            return False
        if _has_slot_price_contradiction(product, slots):
            return False

        components, _ = self._compute_components(product, slots, weights)
        guarded_thresholds = {
            "brand_fit": 0.5,
            "feature_fit": 0.34,
            "use_case_fit": 0.5,
            "gender_fit": 0.5,
            "compat_fit": 0.5,
            "screen_size_fit": 0.5,
            "age_fit": 0.5,
            "color_fit": 0.5,
            "ram_fit": 0.5,
            "storage_fit": 0.5,
            "gpu_fit": 0.5,
            "processor_fit": 0.5,
            "diet_fit": 0.5,
            "life_stage_fit": 0.5,
            "wet_or_dry_fit": 0.5,
            "protein_fit": 0.5,
            "skin_type_fit": 0.5,
            "spf_fit": 0.5,
            "fragrance_free_fit": 0.5,
            "benefit_fit": 0.5,
            "style_fit": 0.5,
            "material_fit": 0.5,
            "size_fit": 0.5,
        }
        for component, threshold in guarded_thresholds.items():
            if component in weights and components.get(component, 0.5) < threshold:
                return False
        return True

    def _build_query_context(self, slots: dict, node_config: dict = None) -> dict:
        query_context = dict(slots or {})
        if node_config:
            # Normalized metadata and controlled category fit both need the
            # routed category as a soft signal, never as a hard filter.
            query_context["category"] = {"l1": node_config.get("l1"), "l2": node_config.get("l2")}
            query_context["_rerank_category_l1"] = node_config.get("l1")
            query_context["_rerank_category_l2"] = node_config.get("l2")
        return query_context

    def _build_weights(self, slots: dict) -> dict:
        weights = dict(BASE_WEIGHTS)
        if CONTROLLED_RERANKER_V1_ENABLED:
            weights.update(CONTROLLED_RERANKER_WEIGHTS)
        if PRODUCT_TYPE_GUARD_ENABLED and _slot_filled(slots, "product_type"):
            weights.update(PRODUCT_TYPE_GUARD_WEIGHTS)
        for component, (slot_keys, weight) in SLOT_GATED_WEIGHTS.items():
            if any(_slot_filled(slots, k) for k in slot_keys):
                weights[component] = weight

        if NORMALIZED_METADATA_ENABLED:
            # Rating/review/quality are generic signals (like the always-on
            # rating_score above) — included whenever the feature layer is
            # on. Brand/category/attribute overlap only add weight when the
            # user actually gave a comparable signal, same convention as
            # SLOT_GATED_WEIGHTS: an unfilled slot's weight would just blend
            # in a meaningless neutral 0.5, diluting the real signal.
            weights["norm_rating_score"] = NORMALIZED_METADATA_WEIGHTS["norm_rating_score"]
            weights["norm_review_score"] = NORMALIZED_METADATA_WEIGHTS["norm_review_score"]
            weights["norm_metadata_quality"] = NORMALIZED_METADATA_WEIGHTS["norm_metadata_quality"]
            if _slot_filled(slots, "brand") or _slot_filled(slots, "exclude_brands"):
                weights["norm_brand_match"] = NORMALIZED_METADATA_WEIGHTS["norm_brand_match"]
            if _slot_filled(slots, "product_type"):
                weights["norm_category_overlap"] = NORMALIZED_METADATA_WEIGHTS["norm_category_overlap"]
            if _collect_keyword_terms(slots):
                weights["norm_attribute_overlap"] = NORMALIZED_METADATA_WEIGHTS["norm_attribute_overlap"]

        return weights

    def _normalized_metadata_features(self, product: dict, slots: dict) -> dict:
        """Empty dict (no-op) when the feature flag is off — zero behavior change."""
        if not NORMALIZED_METADATA_ENABLED:
            return {}
        product_id = product.get("product_id")
        record = get_normalized_product(product_id)
        return compute_normalized_metadata_features(slots, product_id, record)

    def _compute_components(self, product: dict, slots: dict, weights: dict) -> dict:
        components = {
            "semantic_score":      product.get("semantic_score", product.get("score", 0.0)),
            "price_fit":           _price_fit(product, slots),
            "rating_score":        _rating_score(product),
            "keyword_match_score": _keyword_match_score(product, slots),
        }
        norm_features = self._normalized_metadata_features(product, slots)
        for key in NORMALIZED_METADATA_WEIGHTS:
            if key in norm_features:
                components[key] = norm_features[key]
        for component in weights:
            if component in components:
                continue
            scorer = _COMPONENT_SCORERS.get(component)
            if scorer:
                components[component] = scorer(product, slots)
        return components, norm_features.get("norm_soft_penalty", 0.0)

    def _score(self, product: dict, slots: dict, weights: dict) -> float:
        components, soft_penalty = self._compute_components(product, slots, weights)
        weight_sum = sum(weights.values()) or 1.0
        total = sum((w / weight_sum) * components.get(comp, 0.5) for comp, w in weights.items())
        scope_mult = SCOPE_SCORE_MULTIPLIER.get(product.get("scope_rank", 0), 0.85)
        # norm_soft_penalty is a graded subtraction outside the weighted
        # average (same pattern as exclusion_fit, but soft/capped rather
        # than a hard multiplicative kill) — capped at MAX_SOFT_PENALTY in
        # normalized_metadata.py, so it can never dominate the score.
        return max(0.0, total * scope_mult * _exclusion_fit(product, slots) + soft_penalty)

    def explain(self, product: dict, slots: dict, node_config: dict = None) -> dict:
        """
        Debug helper: per-component scores, weights, and each component's
        contribution to the final rerank_score, for one product/slots pair.
        Used by the reranker failure audit — not called in the hot path.
        """
        query_context = self._build_query_context(slots, node_config)
        weights = self._build_weights(query_context)
        components, soft_penalty = self._compute_components(product, query_context, weights)
        weight_sum = sum(weights.values()) or 1.0
        exclusion = _exclusion_fit(product, slots)

        contributions = {
            comp: round((w / weight_sum) * components.get(comp, 0.5), 4)
            for comp, w in weights.items()
        }
        return {
            "product_id": product.get("product_id"),
            "title": product.get("title"),
            "controlled_reranker_v1_enabled": CONTROLLED_RERANKER_V1_ENABLED,
            "triplet_reranker_enabled": TRIPLET_RERANKER_ENABLED,
            "triplet_blend_weight": TRIPLET_BLEND_WEIGHT,
            "top5_diversity_rerank_enabled": TOP5_DIVERSITY_RERANK_ENABLED,
            "top5_diversity_threshold": TOP5_DIVERSITY_THRESHOLD,
            "top5_diversity_brand_requested_threshold": TOP5_DIVERSITY_BRAND_REQUESTED_THRESHOLD,
            "top5_diversity_max_similar": TOP5_DIVERSITY_MAX_SIMILAR,
            "weights": weights,
            "components": {k: round(v, 4) for k, v in components.items()},
            "contributions": contributions,
            "exclusion_fit": exclusion,
            "norm_soft_penalty": soft_penalty,
            "final_score": round(max(0.0, sum(contributions.values()) * exclusion + soft_penalty), 4),
        }
