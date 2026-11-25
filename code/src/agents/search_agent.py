"""
Search Agent
-------------
Scoped semantic search using FAISS IndexFlatIP indexes.
Searches the category node index first; falls back up the hierarchy.

Hierarchy:
  L2 index → L1 index → global index

Slot-based pre-filters applied AFTER vector search (post-filter):
  - price_max, price_min → filter by metadata price
  - brand → filter by brand match
  - exclude_brands → remove products from excluded brands
  - gender → filter by gender tag (if present in metadata)
  - pet_type → filter by pet_type tag

Returns top-K product dicts with similarity score.
"""

import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.taxonomy.text_norm import STOPWORDS, tokenize
from src.agents.multi_view_retrieval import (
    MULTI_VIEW_RETRIEVAL_ENABLED,
    apply_multi_view_retrieval,
)

ROOT          = Path(__file__).resolve().parent.parent.parent
INDEX_ROOT    = ROOT / "indexes"
METADATA_PATH = ROOT / "data" / "product_base_table.json"
VECTOR_TEXT_PATH = ROOT / "data" / "product_base_retrieval" / "product_base_retrieval_text.json"
CATEGORY_MAP_PATH = ROOT / "data" / "product_base_category_map.json"
BM25_INDEX_PATH = ROOT / "data" / "product_base_retrieval" / "bm25_index.json"
MODEL_NAME    = os.getenv("SEARCH_DENSE_MODEL_NAME", "all-MiniLM-L6-v2")
INDEX_BASE_DIR = os.getenv("SEARCH_DENSE_INDEX_BASE", "indexes").strip() or "indexes"
TOP_K         = 100   # retrieve this many before post-filter
RETURN_K      = 50    # return this many after post-filter
MIN_RETRIEVE_PER_INDEX = 120
RETRIEVE_OVERSAMPLE = 4


def _env_float(name: str, default: float, minimum: float = None, maximum: float = None) -> float:
    """Read a float env knob without letting a typo crash module import."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        print(f"[SearchAgent] Invalid {name}={raw!r}; using {default}.")
        return default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_int(name: str, default: int, minimum: int = None, maximum: int = None) -> int:
    """Read an int env knob without letting a typo crash module import."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"[SearchAgent] Invalid {name}={raw!r}; using {default}.")
        return default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _load_json_file(path: Path, expected_type=None):
    """Load JSON with actionable errors for generated artifact problems."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact is missing: {path}. "
            "Rebuild the catalog/index artifacts before running search."
        )
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact at {path}: {exc}") from exc
    if expected_type is not None and not isinstance(data, expected_type):
        raise ValueError(
            f"Invalid artifact shape at {path}: expected "
            f"{expected_type.__name__}, got {type(data).__name__}."
        )
    return data


# DEFAULT 0.7 (keyword-dominant), matching the reference MACPRS system.
HYBRID_KEYWORD_ALPHA = _env_float("SEARCH_HYBRID_KEYWORD_ALPHA", 0.7, minimum=0.0, maximum=1.0)
HYBRID_BOTH_SOURCE_BONUS = 0.03
TAXONOMY_BACKUP_RETURN_K = _env_int("SEARCH_TAXONOMY_BACKUP_RETURN_K", 10, minimum=0)
BM25_RECALL_SLICE = _env_int("SEARCH_BM25_RECALL_SLICE", 8, minimum=0)

# "fixed": DEFAULT (2026-07-06) — BM25 + dense blend at HYBRID_KEYWORD_ALPHA.
#          A fixed blend was previously measured net-NEGATIVE, but that was
#          under taxonomy-scoped search, where BM25 did lexical matching
#          inside an already-wrong small index (amplifying noise). Under
#          SEARCH_SCOPE_MODE=global_soft (now default) BM25 searches the whole
#          catalog and can surface exact title/product-type matches from
#          anywhere — the opposite regime. Re-A/B tested under global_soft:
#          robustly improves the RETRIEVAL stage on both splits
#          (test recall@50 0.365->0.392, search_hit_rate 0.403->0.442,
#          search_recall 0.347->0.390; val a clean sweep incl. hit@10
#          0.158->0.193). alpha=0.7 chosen: val prefers it clearly over 0.5,
#          test is ~tied, and it matches the reference system. Caveat: top-3
#          precision (hit@3/hit@5) dips slightly on test but rises on val — a
#          split-inconsistent reranking-position effect, next addressed by a
#          learned reranker, not a retrieval regression.
# "off"  : dense-only (pre-hybrid baseline, kept for comparison).
# "conditional": per-query alpha via get_hybrid_weights().
HYBRID_MODE = os.getenv("SEARCH_HYBRID_MODE", "fixed").strip().lower()
if HYBRID_MODE not in {"fixed", "off", "conditional"}:
    print(f"[SearchAgent] Invalid SEARCH_HYBRID_MODE={HYBRID_MODE!r}; using 'fixed'.")
    HYBRID_MODE = "fixed"

# "global_soft": DEFAULT (2026-07-06) — always search the whole catalog;
#                category becomes a soft score boost (CATEGORY_SOFT_BOOST_*)
#                instead of a hard filter. Mirrors a reference implementation
#                that never hard-scopes to a taxonomy node and reports much
#                higher search recall/hit rate. A wrong category guess here
#                just slightly mis-weights scoring instead of hard-excluding
#                the correct answer. A/B tested against "taxonomy" on both
#                splits: test recall@50 0.293->0.365, hit@10 0.127->0.155,
#                search_hit_rate 0.326->0.403, ZERO regressions (0 lost /
#                5 gained on hit@10); val showed the same broad improvement.
#                Promoted from opt-in experiment to default after that result.
# "taxonomy"   : legacy — scope search to the routed L2/L1/global index
#                hierarchy. The retrieval failure audit found this was the
#                dominant failure cause: a wrong category guess makes the
#                target product literally unreachable, no matter how good
#                reranking is, and ~10% of the catalog has no category tag
#                at all so hard-scoping guaranteed those were unreachable
#                too. Kept available for comparison, not for regular use.
SEARCH_SCOPE_MODE = os.getenv("SEARCH_SCOPE_MODE", "global_soft").strip().lower()
if SEARCH_SCOPE_MODE not in {"global_soft", "taxonomy"}:
    print(f"[SearchAgent] Invalid SEARCH_SCOPE_MODE={SEARCH_SCOPE_MODE!r}; using 'global_soft'.")
    SEARCH_SCOPE_MODE = "global_soft"
CATEGORY_SOFT_BOOST_L1 = _env_float("SEARCH_CATEGORY_BOOST_L1", 0.08, minimum=0.0)
CATEGORY_SOFT_BOOST_L2 = _env_float("SEARCH_CATEGORY_BOOST_L2", 0.08, minimum=0.0)


NOISY_QUERY_PATTERNS = [
    r"\bspecific product picture\b",
    r"\bshow me\b",
    r"\bcan you show\b",
    r"\bplease show\b",
    r"\bi like (this|that|it|one)\b",
    r"\bi'll take\b",
    r"\bi will take\b",
    r"\blooks good\b",
]

NOISY_QUERY_TERMS = {
    "yes", "yeah", "yep", "ok", "okay", "fine", "good", "great", "thanks",
    "thank", "you", "please", "sure", "cool", "nice", "no", "maybe",
    "a", "an", "the", "for", "my", "me", "this", "that", "these", "those",
    "to", "of", "and",
    "one", "ones", "first", "second", "third", "last", "neither",
    "anything", "something", "them", "choice", "look", "change",
    "recommend", "more", "few", "brand", "price", "quality", "normal",
    "hi", "hello", "there", "i", "am", "m", "want", "wanna", "would",
    "like", "buy", "market", "in", "is", "are", "anything", "better",
    "wonderful", "not", "don", "t", "good-looking",
    "do", "sell", "have", "need", "looking", "searching", "find", "get",
    "could", "can", "help", "pick", "choose", "recently", "preferably",
    "prefer", "hope", "with", "cool", "lovely", "discount", "bigger", "real",
}

CORE_SLOT_KEYS = (
    "product_type", "brand", "color", "use_case", "compatibility",
)

SPEC_SLOT_KEYS = (
    "ram", "storage", "processor", "gpu", "os", "battery_life",
    "screen_size", "diet_type", "wet_or_dry", "life_stage", "protein_source",
    "size", "style", "material", "occasion", "skin_type", "benefit", "spf",
)

NOISY_PRODUCT_TYPES = {
    "for me", "options", "option", "the third", "third", "cheaper",
    "few more", "specific product picture", "product picture", "picture",
    "type", "the first", "the second", "the one",
    "unscented", "wireless", "portable", "foldable", "first", "second",
    "last", "neither", "anything", "something", "them", "cool", "lovely",
    "discount", "bigger", "real",
}

NOISY_PRODUCT_TYPE_PREFIXES = (
    "specific product picture",
    "please",
    "wonderful",
    "well",
    "sure",
    "yes",
    "no ",
    "few more",
)

COLOR_WORDS = {
    "black", "white", "blue", "red", "green", "yellow", "pink", "purple",
    "grey", "gray", "brown", "silver", "gold", "beige", "orange",
}


# ── Singleton embedding model ──────────────────────────────────────────────────

_embedder: Optional[SentenceTransformer] = None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"[SearchAgent] Loading sentence-transformer '{MODEL_NAME}'...")
        _embedder = SentenceTransformer(MODEL_NAME)
        print("[SearchAgent] Model ready.")
    return _embedder


# ── Product metadata cache ─────────────────────────────────────────────────────

_metadata: Optional[dict] = None
_vector_texts: Optional[dict] = None
_category_map: Optional[dict] = None
_lexical_index: Optional[dict] = None

def _parse_price(product: dict) -> float | None:
    """
    Extract price from ranking_metadata / all_original_attributes.
    The catalog stores it under lowercase key 'price' as a plain numeric
    string (e.g. '136.00'), not '$136.00'. Only ~39% of products have it.
    """
    import re
    for src_key in ("ranking_metadata", "all_original_attributes"):
        src = product.get(src_key) or {}
        for key in ("price", "Price", "List Price"):
            val = src.get(key, "")
            if val:
                m = re.search(r"([\d,]+\.?\d*)", str(val))
                if m:
                    try:
                        return float(m.group(1).replace(",", ""))
                    except ValueError:
                        pass
    return None


def _parse_rating(product: dict) -> float | None:
    """Extract numeric rating from Customer Reviews string."""
    import re
    rm = product.get("ranking_metadata") or {}
    rev = str(rm.get("Customer Reviews", ""))
    m = re.search(r"([\d.]+)\s+out of 5", rev)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _get_metadata() -> dict:
    global _metadata
    if _metadata is None:
        print(f"[SearchAgent] Loading product metadata...")
        raw = _load_json_file(METADATA_PATH)
        vector_texts = _get_vector_texts()
        category_map = _get_category_map()

        # Metadata can be dict (keyed by product_id) or list
        if isinstance(raw, dict):
            products = raw.values()
        elif isinstance(raw, list):
            products = raw
        else:
            raise ValueError(
                f"Invalid artifact shape at {METADATA_PATH}: expected dict or list, "
                f"got {type(raw).__name__}."
            )

        _metadata = {}
        for p in products:
            pid = p.get("product_id")
            if not pid:
                continue
            category = category_map.get(pid, {})
            _metadata[pid] = {
                "product_id": pid,
                "title":      p.get("title", ""),
                "brand":      p.get("brand", ""),
                "color":      p.get("color", ""),
                "price":      _parse_price(p),
                "rating":     _parse_rating(p),
                "description": vector_texts.get(pid, ""),
                # optional enrichment fields (may not exist)
                "gender":     p.get("gender"),
                "pet_type":   p.get("pet_type"),
                "l1":         p.get("l1") or category.get("l1"),
                "l2":         p.get("l2") or category.get("l2"),
                "category":   p.get("category") or _format_category(category),
            }

        print(f"[SearchAgent] Metadata loaded: {len(_metadata):,} products.")
    return _metadata


def _get_vector_texts() -> dict:
    """Load generated product text used to build the FAISS indexes."""
    global _vector_texts
    if _vector_texts is None:
        print("[SearchAgent] Loading generated product descriptions...")
        raw = _load_json_file(VECTOR_TEXT_PATH, list)
        _vector_texts = {
            p["product_id"]: p.get("text", "")
            for p in raw
            if isinstance(p, dict) and p.get("product_id")
        }
        print(f"[SearchAgent] Descriptions loaded: {len(_vector_texts):,} products.")
    return _vector_texts


def _get_category_map() -> dict:
    """Load product taxonomy assignments keyed by product_id."""
    global _category_map
    if _category_map is None:
        print("[SearchAgent] Loading product category map...")
        _category_map = _load_json_file(CATEGORY_MAP_PATH, dict)
        print(f"[SearchAgent] Category map loaded: {len(_category_map):,} products.")
    return _category_map


def _format_category(category: dict) -> str | None:
    """Format category map entry as 'L1 > L2' when available."""
    l1 = category.get("l1")
    l2 = category.get("l2")
    if l1 and l2:
        return f"{l1} > {l2}"
    return l1 or l2


# ── Lexical retrieval cache ────────────────────────────────────────────────────

def _lex_tokens(text: str) -> list[str]:
    """Simple lexical tokenizer for BM25-style matching."""
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _product_lexical_text(product: dict) -> str:
    """
    Lexical document text. Title/brand/category are repeated because exact
    product-type words are more reliable there than in long descriptions.
    """
    title = str(product.get("title") or "")
    brand = str(product.get("brand") or "")
    category = str(product.get("category") or "")
    description = str(product.get("description") or "")
    return " ".join([title, title, brand, brand, category, description])


def _get_lexical_index(metadata: dict) -> dict:
    """Load the prebuilt BM25-style inverted index from disk.

    The MACPRS product-base experiment builds BM25 offline from
    data/product_base_retrieval/product_base_retrieval_text.json. Runtime
    should load that artifact instead of rebuilding lexical postings.
    """
    global _lexical_index
    if _lexical_index is not None:
        return _lexical_index

    print("[SearchAgent] Loading prebuilt BM25 lexical index...")
    raw = _load_json_file(BM25_INDEX_PATH, dict)

    n_docs = int(raw.get("n_docs") or 0)
    ids_path = BM25_INDEX_PATH.parent / "ids.json"
    ids = _load_json_file(ids_path, list)
    if n_docs and len(ids) < n_docs:
        raise ValueError(
            f"BM25/id mismatch: {BM25_INDEX_PATH} has {n_docs} docs "
            f"but {ids_path} has {len(ids)} ids."
        )

    doc_len_raw = raw.get("doc_len") or []
    if isinstance(doc_len_raw, list):
        doc_len = {str(ids[i]): length for i, length in enumerate(doc_len_raw) if i < len(ids)}
    elif isinstance(doc_len_raw, dict):
        doc_len = {str(pid): length for pid, length in doc_len_raw.items()}
    else:
        raise ValueError(f"Invalid BM25 doc_len shape in {BM25_INDEX_PATH}: {type(doc_len_raw).__name__}")

    postings = {}
    for token, entries in (raw.get("postings") or {}).items():
        converted = []
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            doc_ref, tf = entry
            if isinstance(doc_ref, int):
                if doc_ref >= len(ids):
                    continue
                pid = str(ids[doc_ref])
            else:
                pid = str(doc_ref)
            converted.append((pid, tf))
        postings[token] = converted

    _lexical_index = {
        "postings": postings,
        "doc_len": doc_len,
        "avg_len": float(raw.get("avg_doc_len") or raw.get("avg_len") or 1.0),
        "idf": raw.get("idf") or {},
        "n_docs": n_docs or len(ids),
    }
    print(f"[SearchAgent] Prebuilt BM25 ready: {_lexical_index['n_docs']:,} products.")
    return _lexical_index


def _normalize_scores(scores: dict) -> dict:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi <= lo:
        return {pid: 1.0 for pid in scores}
    return {pid: (score - lo) / (hi - lo) for pid, score in scores.items()}


def _bm25_search(query: str, scope_ids: list, metadata: dict, k: int) -> dict:
    """Return {product_id: bm25_score} for the current taxonomy scope."""
    tokens = _lex_tokens(query)
    if not tokens:
        return {}

    lex = _get_lexical_index(metadata)
    postings = lex["postings"]
    doc_len = lex["doc_len"]
    idf = lex["idf"]
    avg_len = lex["avg_len"] or 1.0
    allowed = set(scope_ids)
    scores = defaultdict(float)
    k1 = 1.5
    b = 0.75

    for token in tokens:
        token_idf = idf.get(token)
        if token_idf is None:
            continue
        for pid, tf in postings.get(token, []):
            if pid not in allowed:
                continue
            product = metadata.get(pid)
            if product is None:
                continue
            dl = doc_len.get(pid, avg_len)
            denom = tf + k1 * (1.0 - b + b * dl / avg_len)
            scores[pid] += token_idf * (tf * (k1 + 1.0)) / denom

    if not scores:
        return {}
    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
    return dict(top)


def get_hybrid_weights(query: str, slots: dict) -> tuple:
    """
    Per-query (dense_weight, bm25_weight), summing to 1.0.

    Conditional blend weights used only when HYBRID_MODE=conditional.

    The active default is HYBRID_MODE=fixed with a separate BM25 recall
    reserve (see module-level HYBRID_MODE and BM25_RECALL_SLICE). This helper
    remains for experiments; do not read it as the default retrieval policy.
    """
    content_tokens = [t for t in tokenize(query, bigrams=False) if t.lower() not in STOPWORDS]
    has_brand = bool(slots.get("brand"))
    if len(content_tokens) <= 3 or has_brand:
        return (0.5, 0.5)
    return (0.85, 0.15)


# ── FAISS index cache ──────────────────────────────────────────────────────────

_index_cache: dict = {}

def _load_index(index_path: str):
    """Load (faiss_index, id_list) from disk, cached in memory."""
    if INDEX_BASE_DIR != "indexes" and index_path.startswith("indexes/"):
        index_path = INDEX_BASE_DIR.rstrip("/") + index_path[len("indexes"):]
    if index_path in _index_cache:
        return _index_cache[index_path]

    import faiss

    abs_path = ROOT / index_path
    faiss_bin = abs_path / "faiss.bin"
    ids_json  = abs_path / "ids.json"

    if not faiss_bin.exists():
        return None, []
    if not ids_json.exists():
        raise FileNotFoundError(
            f"Index ids file is missing for {index_path}: {ids_json}. "
            "Rebuild indexes so faiss.bin and ids.json stay in sync."
        )

    index = faiss.read_index(str(faiss_bin))
    ids = _load_json_file(ids_json, list)
    if len(ids) < index.ntotal:
        raise ValueError(
            f"Index/id mismatch for {index_path}: faiss has {index.ntotal} vectors "
            f"but ids.json has {len(ids)} ids."
        )

    _index_cache[index_path] = (index, ids)
    return index, ids


# ── Post-filter helpers ────────────────────────────────────────────────────────

def _passes_filter(product: dict, slots: dict) -> bool:
    """Return True if product satisfies slot constraints."""
    # Price filter
    price_max = slots.get("price_max")
    price_min = slots.get("price_min")
    product_price = product.get("price")

    if product_price is not None:
        if price_max is not None and product_price > price_max:
            return False
        if price_min is not None and product_price < price_min:
            return False

    # Brand include filter — if user specified a brand, prefer it (soft filter only)
    # Hard filter would eliminate too many products; handled by reranker instead.

    # Brand exclude filter — hard exclude
    exclude_brands = slots.get("exclude_brands") or []
    if exclude_brands:
        product_brand = (product.get("brand") or "").lower()
        if any(eb.lower() in product_brand for eb in exclude_brands):
            return False

    exclude_features = slots.get("exclude_features") or []
    if exclude_features:
        product_text = (
            (product.get("title") or "") + " " +
            (product.get("description") or "")
        ).lower()
        if any(feature.lower() in product_text for feature in exclude_features):
            return False

    # Gender filter (if product has gender tag)
    gender_slot = slots.get("gender")
    product_gender = (product.get("gender") or "").lower()
    if gender_slot and product_gender and product_gender not in (gender_slot, "unisex"):
        return False

    # Pet type filter
    pet_type_slot = slots.get("pet_type")
    product_pet = (product.get("pet_type") or "").lower()
    if pet_type_slot and product_pet and pet_type_slot.lower() not in product_pet:
        return False

    return True


def _normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for query checks."""
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _clean_user_message(user_message: str) -> str:
    """Remove conversational filler that hurts retrieval embeddings."""
    text = _normalize_text(user_message)
    if not text:
        return ""
    for pattern in NOISY_QUERY_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"[^a-z0-9.+#&/\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    useful = [t.strip(".,!?;:") for t in tokens if t.strip(".,!?;:") not in NOISY_QUERY_TERMS]
    useful = [t for t in useful if t]
    if len(useful) < 2:
        return ""
    return " ".join(useful)


def _clean_product_type(value) -> str:
    """Drop conversational artifacts that NER sometimes stores as product_type."""
    text = _normalize_text(value)
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9.+#&/\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or text in NOISY_PRODUCT_TYPES or text in COLOR_WORDS:
        return ""
    if any(text.startswith(prefix) for prefix in NOISY_PRODUCT_TYPE_PREFIXES):
        return ""
    useful = [t for t in text.split() if t not in NOISY_QUERY_TERMS]
    if not useful:
        return ""
    return text


def _as_terms(value, slot_key: str = "") -> list:
    """Normalize a scalar/list slot value into search terms."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        terms = []
        for item in value:
            terms.extend(_as_terms(item, slot_key=slot_key))
        return terms
    if slot_key == "product_type":
        text = _clean_product_type(value)
        return [text] if text else []
    text = str(value).strip()
    return [text] if text else []


def _dedupe_terms(parts: list) -> list:
    """Keep first occurrence of each normalized term."""
    seen = set()
    out = []
    for part in parts:
        text = str(part or "").strip()
        if not text:
            continue
        key = _normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _dedupe_words(text: str) -> str:
    """
    Collapse repeated words across the WHOLE final query, not just whole
    duplicate parts. NER's product_type extraction is often a rough
    substring of the raw message ("gaming mouse along" from "...a gaming
    mouse along with a matching mouse pad..."), so the same words show up
    once as the product_type part and again inside the cleaned message
    part — _dedupe_terms only catches exact-duplicate whole parts, not this
    word-level overlap. Found via retrieval failure audit
    (scripts/13_retrieval_failure_audit.py): queries like "gaming mouse
    along gaming gaming mouse along matching mouse pad any recommendations"
    waste embedding capacity repeating the same words instead of adding new
    distinguishing signal.
    """
    seen = set()
    out = []
    for word in text.split():
        key = word.lower().strip(".,!?;:")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(word)
    return " ".join(out)


def _slot_terms(slots: dict, keys: tuple) -> list:
    parts = []
    for key in keys:
        parts.extend(_as_terms(slots.get(key), slot_key=key))
    return _dedupe_terms(parts)


def _feature_terms(slots: dict) -> list:
    return _dedupe_terms(_as_terms(slots.get("features"))[:8])


def _bool_terms(slots: dict) -> list:
    parts = []
    if slots.get("fragrance_free"):
        parts.append("fragrance free")
    return parts


def _build_query(slots: dict, user_message: str) -> str:
    """Build one robust retrieval query from trusted slots + cleaned message."""
    parts = []

    parts.extend(_as_terms(slots.get("product_type"), slot_key="product_type"))
    parts.extend(_as_terms(slots.get("brand")))
    parts.extend(_feature_terms(slots))
    parts.extend(_slot_terms(slots, SPEC_SLOT_KEYS))
    parts.extend(_bool_terms(slots))

    use_case = slots.get("use_case")
    if use_case:
        parts.append(use_case)

    compatibility = slots.get("compatibility")
    if compatibility:
        # Raw term, not a "compatible with X" phrase — filler words like
        # "compatible"/"with" dilute BM25 term overlap against corpus text
        # that just says the platform name (e.g. "iOS", "Android").
        parts.append(compatibility)

    screen_size = slots.get("screen_size")
    if screen_size:
        parts.append(screen_size)

    color = slots.get("color")
    if color:
        parts.append(color)

    clean_message = _clean_user_message(user_message)
    if clean_message:
        parts.append(clean_message)

    # DECISION (keep, not gated): word-level dedup was measured net -3 test /
    # +val conversations when it first landed, before QueryPlanner's
    # unconditional bare-query enrichment existed. Re-measured after that
    # enrichment shipped (see eval/results/baseline_post_task1/): dedup-on vs
    # dedup-off now produces ZERO flips on the 198-conversation test split
    # and +1/-0 on val. The original regression doesn't replicate — most
    # queries now go through QueryPlanner's own dedup/novelty logic
    # (_novel_message_terms + _dedupe), so this fallback path (used only when
    # QueryPlanner's `applied` is False) carries much less weight than it did
    # pre-enrichment. No gating rule needed. Don't re-litigate this without
    # new flip evidence — see eval/results/dedup_off_per_conversation_{test,val}.json
    # for the comparison data this decision is based on.
    return _dedupe_words(" ".join(_dedupe_terms(parts)).strip()) or "product"


# ── Public agent ───────────────────────────────────────────────────────────────

class SearchAgent:
    """
    Scoped FAISS semantic search.

    Usage:
        agent = SearchAgent()
        results = agent.search(
            node_config={...},    # from CategoryRouter.get_node_config()
            slots={...},          # current session slots
            user_message="...",   # raw user turn for embedding
            top_k=20,
        )
        # results: list of {product_id, title, score, ...metadata fields}
    """

    def __init__(self):
        self.last_query = ""
        self.last_queries = []
        self.last_scope_counts = {}

    def search_many(
        self,
        node_config: dict,
        slots: dict,
        queries: list,
        user_message: str = "",
        top_k: int = RETURN_K,
    ) -> list:
        """
        Multi-query retrieval: run each query variant, merge the candidate
        pools keeping the best score per product. Proven the single biggest
        retrieval lever — an oracle test (PSCoN gold keywords over our index)
        showed single-query recall 0.57 vs multi-query 0.74, closing almost
        the entire gap to the reference MACPRS system (0.78). Firing the
        product-type and each salient term as SEPARATE searches and merging
        catches products the joined query ranks too low to surface (e.g.
        "coffee beans" and "China" separately, not just "coffee beans China").
        """
        queries = [q for q in dict.fromkeys(q.strip() for q in queries if q and q.strip())]
        if not queries:
            queries = [_build_query(slots, user_message)]
        self.last_queries = queries
        self.last_query = queries[0]

        # Run each variant; give later (narrower) variants a smaller pool.
        per_query_k = max(top_k, RETURN_K)
        merged = {}
        for i, q in enumerate(queries):
            variant_k = per_query_k if i == 0 else max(per_query_k // 2, 20)
            for r in self.search(
                node_config=node_config, slots=slots, planned_query=q, top_k=variant_k,
            ):
                pid = r["product_id"]
                if pid not in merged or r.get("score", 0.0) > merged[pid].get("score", 0.0):
                    merged[pid] = r
        results = sorted(merged.values(), key=lambda r: -r.get("score", 0.0))[:top_k]

        # MultiViewRetrievalV1 (MACPRS_USE_MULTI_VIEW_RETRIEVAL, default off):
        # fuse the existing semantic+BM25 hybrid results above ("semantic
        # view", untouched) with title/attribute/category/compact_metadata
        # BM25 views over the same catalog. Additive only — never a
        # replacement for the search above, and never touched when the flag
        # is off (zero behavior change, see src/agents/multi_view_retrieval.py).
        if MULTI_VIEW_RETRIEVAL_ENABLED:
            fusion_query = " ".join(queries)
            results = apply_multi_view_retrieval(
                query=fusion_query,
                metadata=_get_metadata(),
                semantic_results=results,
                top_k=top_k,
            )
            print(f"[SearchAgent] Multi-view retrieval: {len(results)} fused results.")

        self.last_scope_counts = {0: len(results)}
        print(f"[SearchAgent] Multi-query ({len(queries)}) returning {len(results)} merged results.")
        return results

    def search(
        self,
        node_config: dict,
        slots: dict,
        user_message: str = "",
        planned_query: str = None,
        top_k: int = RETURN_K,
    ) -> list:
        embedder = _get_embedder()

        query = (planned_query or "").strip() or _build_query(slots, user_message)
        self.last_query = query
        print(f"[SearchAgent] Query: '{query[:80]}'")

        # Embed before loading catalog metadata to keep peak memory low.
        q_vec = embedder.encode([query], normalize_embeddings=True).astype("float32")

        metadata = _get_metadata()

        if SEARCH_SCOPE_MODE == "global_soft":
            results = self._search_global_soft(q_vec, query, slots, metadata, node_config, top_k)
            self.last_scope_counts = {0: len(results)}
            print(f"[SearchAgent] Returning {len(results)} results.")
            return results

        # Attempt hierarchy: L2 → L1 → global
        index_path = node_config.get("index_path", "indexes/_global")

        # Try the node's own index first
        primary_results = self._search_one_index(
            index_path,
            q_vec,
            query,
            slots,
            metadata,
            TOP_K,
            scope_rank=0,
            query_name="single",
        )

        backup_results = []
        l1 = node_config.get("l1")
        l2 = node_config.get("l2")
        is_specific_l2 = bool(l1 and l2 and not index_path.endswith("/_l1"))

        # For specific L2 routes, always reserve a small L1 backup slice.
        # This recovers L1-only catalog products without turning taxonomy
        # into a hard filter that silently kills recall.
        if l1 and (is_specific_l2 or len(primary_results) < top_k):
            l1_path = f"indexes/{l1}/_l1"
            extra = self._search_one_index(
                l1_path,
                q_vec,
                query,
                slots,
                metadata,
                TOP_K,
                scope_rank=1,
                query_name="single",
            )
            seen = {r["product_id"] for r in primary_results}
            for r in extra:
                if r["product_id"] not in seen:
                    backup_results.append(r)
                    seen.add(r["product_id"])

        if is_specific_l2 and backup_results and len(primary_results) >= top_k:
            backup_k = min(TAXONOMY_BACKUP_RETURN_K, top_k - 1)
            primary_k = top_k - backup_k
            results = primary_results[:primary_k] + backup_results[:backup_k]
        else:
            results = primary_results[:]
            seen = {r["product_id"] for r in results}
            for r in backup_results:
                if len(results) >= top_k:
                    break
                if r["product_id"] not in seen:
                    results.append(r)
                    seen.add(r["product_id"])

        # Fall back to global if still short
        if len(results) < top_k:
            extra = self._search_one_index(
                "indexes/_global",
                q_vec,
                query,
                slots,
                metadata,
                TOP_K,
                scope_rank=2,
                query_name="single",
            )
            seen = {r["product_id"] for r in results}
            for r in extra:
                if r["product_id"] not in seen:
                    results.append(r)
                    seen.add(r["product_id"])

        # Primary and backup slices are intentionally composed above; keep
        # that allocation instead of sorting all backup candidates away.
        results = results[:top_k]
        self.last_scope_counts = {}
        for r in results:
            scope = r.get("scope_rank", 9)
            self.last_scope_counts[scope] = self.last_scope_counts.get(scope, 0) + 1

        print(f"[SearchAgent] Returning {len(results)} results.")
        return results

    def _search_global_soft(
        self,
        q_vec: np.ndarray,
        query: str,
        slots: dict,
        metadata: dict,
        node_config: dict,
        top_k: int,
    ) -> list:
        """
        SEARCH_SCOPE_MODE=global_soft: always search the whole catalog and
        treat the routed category as a soft score boost, not a hard scope.
        A wrong CategoryRouter guess (confirmed the dominant cause of misses
        in the retrieval failure audit) can no longer make the target
        unreachable — it can only be scored slightly less favourably than
        it would with a correct guess.
        """
        results = self._search_one_index(
            "indexes/_global", q_vec, query, slots, metadata,
            max(top_k * RETRIEVE_OVERSAMPLE, MIN_RETRIEVE_PER_INDEX),
            scope_rank=0, query_name="single",
        )

        l1 = node_config.get("l1")
        l2 = node_config.get("l2")
        if l1:
            for r in results:
                boost = 0.0
                if r.get("l1") == l1:
                    boost += CATEGORY_SOFT_BOOST_L1
                    if l2 and r.get("l2") == l2:
                        boost += CATEGORY_SOFT_BOOST_L2
                if boost:
                    r["score"] = r.get("score", 0.0) + boost
                    r["category_boost"] = round(boost, 4)
            results.sort(key=lambda r: -r["score"])

        if BM25_RECALL_SLICE > 0:
            reserve_k = min(BM25_RECALL_SLICE, top_k)
            primary_k = max(top_k - reserve_k, 0)
            reserved = [r for r in results if r.get("bm25_reserved")]
            primary = [r for r in results if not r.get("bm25_reserved")]
            merged = primary[:primary_k] + reserved[:reserve_k] + primary[primary_k:]
            seen = set()
            deduped = []
            for r in merged:
                pid = r.get("product_id")
                if pid in seen:
                    continue
                seen.add(pid)
                deduped.append(r)
            return deduped[:top_k]

        return results[:top_k]

    # ── Internal ───────────────────────────────────────────────────────────────

    def _search_one_index(
        self,
        index_path: str,
        q_vec: np.ndarray,
        query: str,
        slots: dict,
        metadata: dict,
        k: int,
        scope_rank: int = 0,
        query_name: str = "",
    ) -> list:
        index, ids = _load_index(index_path)
        if index is None or index.ntotal == 0:
            return []

        q_vec = np.ascontiguousarray(q_vec, dtype="float32")

        # Retrieve more than needed to absorb post-filter losses
        k_retrieve = min(max(k * RETRIEVE_OVERSAMPLE, MIN_RETRIEVE_PER_INDEX), index.ntotal)
        scores, idxs = index.search(q_vec, k_retrieve)

        if HYBRID_MODE == "off":
            results = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0:
                    continue
                pid = ids[idx]
                product = metadata.get(pid)
                if product is None:
                    continue
                if not _passes_filter(product, slots):
                    continue
                results.append({
                    "product_id": pid,
                    "title":      product.get("title", ""),
                    "brand":      product.get("brand", ""),
                    "price":      product.get("price"),
                    "rating":     product.get("rating"),
                    "description": product.get("description", ""),
                    "score":      float(score),
                    "semantic_score": float(score),
                    "keyword_score":  0.0,
                    "hybrid_score":   float(score),
                    "hybrid_alpha":   0.0,
                    "scope_rank": scope_rank,
                    "best_query": query_name,
                    **{k: product.get(k) for k in ["category", "l1", "l2", "gender", "pet_type", "color"]
                       if product.get(k) is not None},
                })
                if len(results) >= k:
                    break
            return results

        semantic_raw = {}
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            pid = ids[idx]
            product = metadata.get(pid)
            if product is None:
                continue
            if not _passes_filter(product, slots):
                continue
            semantic_raw[pid] = float(score)
            if len(semantic_raw) >= k:
                break

        bm25_raw = _bm25_search(query, ids, metadata, k_retrieve)
        bm25_raw = {
            pid: score for pid, score in bm25_raw.items()
            if metadata.get(pid) is not None and _passes_filter(metadata[pid], slots)
        }

        semantic_norm = _normalize_scores(semantic_raw)
        bm25_norm = _normalize_scores(bm25_raw)
        candidate_ids = set(semantic_norm) | set(bm25_norm)

        if HYBRID_MODE == "conditional":
            dense_weight, keyword_weight = get_hybrid_weights(query, slots)
            alpha = keyword_weight
        else:
            alpha = min(max(HYBRID_KEYWORD_ALPHA, 0.0), 1.0)
        fused = {}
        for pid in candidate_ids:
            keyword_score = bm25_norm.get(pid, 0.0)
            semantic_score = semantic_norm.get(pid, 0.0)
            fused[pid] = (
                alpha * keyword_score
                + (1.0 - alpha) * semantic_score
                + (HYBRID_BOTH_SOURCE_BONUS if pid in bm25_norm and pid in semantic_norm else 0.0)
            )

        fused_order = [pid for pid, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)]
        if BM25_RECALL_SLICE > 0 and bm25_raw:
            bm25_order = [pid for pid, _ in sorted(bm25_raw.items(), key=lambda item: item[1], reverse=True)]
            reserve_k = min(BM25_RECALL_SLICE, k)
            primary_k = max(k - reserve_k, 0)
            ordered_ids = _dedupe_terms(fused_order[:primary_k] + bm25_order[:reserve_k] + fused_order[primary_k:])
        else:
            ordered_ids = fused_order

        results = []
        reserved_ids = set(bm25_order[:reserve_k]) if BM25_RECALL_SLICE > 0 and bm25_raw else set()
        for pid in ordered_ids:
            hybrid_score = fused.get(pid, bm25_norm.get(pid, 0.0))
            product = metadata.get(pid)
            if product is None:
                continue
            results.append({
                "product_id": pid,
                "title":      product.get("title", ""),
                "brand":      product.get("brand", ""),
                "price":      product.get("price"),
                "rating":     product.get("rating"),
                "description": product.get("description", ""),
                "score":      float(hybrid_score),
                "semantic_score": float(semantic_norm.get(pid, 0.0)),
                "keyword_score":  float(bm25_norm.get(pid, 0.0)),
                "hybrid_score":   float(hybrid_score),
                "hybrid_alpha":   alpha,
                "bm25_reserved":  pid in reserved_ids,
                "scope_rank": scope_rank,
                "best_query": query_name,
                **{k: product.get(k) for k in ["category", "l1", "l2", "gender", "pet_type", "color"]
                   if product.get(k) is not None},
            })
            if len(results) >= k:
                break

        return results
