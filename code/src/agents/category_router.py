"""
Category Router
----------------
Maps the user's expressed product type + context to a taxonomy node (L1/L2).

Resolution order (first match wins):
  1. Exact L2 label match against product_type          (confidence 1.0)
  2. L2 hint / alias match                               (confidence 0.95)
  3. Hybrid keyword+semantic scoring over every node     (confidence = score margin)
  4. Global fallback ("_global")                          (confidence 0.0)

Step 3 is the generalisation layer: keyword and L1 alias lookups only cover
product types someone thought to enumerate. The hybrid scorer works for any
query by comparing it against a per-node corpus derived from the actual
catalog (see scripts/10_build_route_index.py), weighted mostly toward
keyword/BM25 match (product-type words are usually explicit) with a smaller
semantic component to catch paraphrases the keyword side misses.

Returns:
  {
    "l1": str,          # e.g. "Electronics"
    "l2": str | None,   # e.g. "Traditional Laptop Computers" (or None)
    "node_key": str,    # e.g. "Electronics/Traditional Laptop Computers"
    "source": str,      # "exact_l2" | "fuzzy_l2" | "hybrid_taxonomy" | "global"
    "confidence": float,     # 0-1, see ROUTE_LOCK_THRESHOLD
    "alternatives": list,    # runner-up [{"node_key": ..., "score": ...}, ...]
  }
"""

import json
from pathlib import Path

import numpy as np
import os

from src.taxonomy.text_norm import tokenize as _tokenize

ROOT       = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "src" / "taxonomy" / "node_config.json"
ROUTE_INDEX_PATH = ROOT / "src" / "taxonomy" / "route_index.json"
ROUTE_EMB_PATH   = ROOT / "src" / "taxonomy" / "route_embeddings.npy"
ROUTE_EMB_KEYS_PATH = ROOT / "src" / "taxonomy" / "route_embeddings_keys.json"

# ── Hybrid scoring weights ─────────────────────────────────────────────────────
# Keyword-favored: product-type words are usually explicit ("mouse", "dog
# food", "steam eye mask"), so keyword/BM25 match should anchor the decision
# with semantic similarity as a smaller corrective signal.
KEYWORD_WEIGHT   = 0.60
SEMANTIC_WEIGHT  = 0.40

# Confidence = (top1 - top2) score margin. Below this, don't hard-lock the
# category — keep it soft so the pipeline re-routes as more slots arrive.


def _env_float(name: str, default: float, minimum: float = None, maximum: float = None) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        print(f"[CategoryRouter] Invalid {name}={raw!r}; using {default}.")
        return default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _load_json_file(path: Path, expected_type=None):
    if not path.exists():
        raise FileNotFoundError(
            f"Required taxonomy artifact is missing: {path}. "
            "Run the taxonomy/index build scripts before routing."
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


# Tuned on the val split (0.05/0.20/0.35 grid; 0.20 gave best hit@10/ndcg@10
# without over-restricting valid confident routes). Override via env for
# further tuning without editing this file.
ROUTE_LOCK_THRESHOLD = _env_float("ROUTE_LOCK_THRESHOLD", 0.20, minimum=0.0, maximum=1.0)
# Below this absolute top1 score, treat as no real match at all -> global.
ROUTE_MIN_SCORE = 0.05


# ── L2 keyword hints — common product phrases → specific L2 label fragments ───
L2_HINTS = {
    "gaming laptop": "traditional laptop",
    "laptop computer": "traditional laptop",
    "notebook": "traditional laptop",
    "laptop": "traditional laptop",
    "computer monitor": "computer monitor",
    "gaming monitor": "computer monitor",
    "monitor": "computer monitor",
    "screen": "computer monitor",
    "display": "computer monitor",
    "smartwatch": "smartwatch",
    "smart watch": "smartwatch",
    "headphone": "headphone",
    "earphone": "headphone",
    "earbud": "earbud",
    "true wireless": "earbud",
    "tws": "earbud",
    "running shoe": "shoes & footwear",
    "shoe": "shoes & footwear",
    "sneaker": "shoes & footwear",
    "dog food": "dog food",
    "cat food": "cat food",
    "electric bike": "electric bicycl",
    "e-bike": "electric bicycl",
}


class CategoryRouter:
    """
    Routes a product type string to the best taxonomy node.
    Loads node_config.json once and caches it.

    Usage:
        router = CategoryRouter()
        node = router.route("gaming laptop")
        # → {"l1": "Electronics", "l2": "Traditional Laptop Computers",
        #     "node_key": "Electronics/Traditional Laptop Computers", "source": "fuzzy_l2"}
    """

    def __init__(self):
        self._config: dict = {}
        self._l2_keys: list = []   # all "L1/L2" keys in lowercase for fuzzy search
        self._l1_keys: list = []   # all L1 keys
        self._loaded = False

        # Hybrid routing artifacts (lazy-loaded — see scripts/10_build_route_index.py)
        self._route_idf: dict = {}
        self._route_nodes: dict = {}     # node_key -> {"terms": {...}, "doc_len": int}
        self._route_emb_keys: list = []
        self._route_emb: np.ndarray = None
        self._embedder = None
        self._hybrid_loaded = False

    def _load(self):
        if self._loaded:
            return
        self._config = _load_json_file(CONFIG_PATH, dict)
        if "_global" not in self._config:
            raise ValueError(f"Taxonomy config is missing required '_global' node: {CONFIG_PATH}")

        for key in self._config:
            if key == "_global":
                continue
            if "/" in key:
                self._l2_keys.append(key)
            else:
                self._l1_keys.append(key)

        self._loaded = True

    def _load_hybrid(self):
        if self._hybrid_loaded:
            return
        route_index = _load_json_file(ROUTE_INDEX_PATH, dict)
        self._route_idf = route_index.get("idf")
        self._route_nodes = route_index.get("nodes")
        if not isinstance(self._route_idf, dict) or not isinstance(self._route_nodes, dict):
            raise ValueError(
                f"Invalid route index at {ROUTE_INDEX_PATH}: expected 'idf' and 'nodes' dicts."
            )

        if not ROUTE_EMB_PATH.exists():
            raise FileNotFoundError(f"Required route embedding artifact is missing: {ROUTE_EMB_PATH}")
        self._route_emb = np.load(ROUTE_EMB_PATH)
        self._route_emb_keys = _load_json_file(ROUTE_EMB_KEYS_PATH, list)
        if len(self._route_emb_keys) != len(self._route_emb):
            raise ValueError(
                f"Route embedding/key mismatch: {ROUTE_EMB_PATH} has {len(self._route_emb)} rows "
                f"but {ROUTE_EMB_KEYS_PATH} has {len(self._route_emb_keys)} keys."
            )

        self._hybrid_loaded = True

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def route(self, product_type: str = None, slots: dict = None) -> dict:
        """
        Main routing method.

        Args:
            product_type: The product_type slot extracted by NER (or None).
            slots: Full slot dict — used for pet_type hint etc.

        Returns:
            Node dict with l1, l2, node_key, source, confidence, alternatives.
        """
        self._load()
        slots = slots or {}
        query = (product_type or "").lower().strip()

        # ── Pet type hint → Pet Supplies ──────────────────────────────────────
        pet_type = slots.get("pet_type")
        if pet_type and not query:
            query = f"{pet_type} supplies"

        if not query:
            return self._global()

        # ── 1. Exact L2 label match ───────────────────────────────────────────
        for key in self._l2_keys:
            l2_label = key.split("/", 1)[1].lower()
            if query == l2_label:
                return self._make_result(key, "exact_l2", confidence=1.0)

        # ── 2. L2 hint lookup ─────────────────────────────────────────────────
        for hint, l2_fragment in sorted(L2_HINTS.items(), key=lambda x: -len(x[0])):
            if hint in query:
                # Prefer exact L2 labels for broad hints such as "dog food";
                # otherwise substring matches may choose siblings like canned food.
                for key in self._l2_keys:
                    if key.split("/", 1)[1].lower() == l2_fragment:
                        return self._apply_sibling_override(
                            self._make_result(key, "fuzzy_l2", confidence=0.95), query
                        )
                for key in self._l2_keys:
                    if l2_fragment in key.split("/", 1)[1].lower():
                        return self._apply_sibling_override(
                            self._make_result(key, "fuzzy_l2", confidence=0.95), query
                        )

        # ── 3. Hybrid keyword + semantic scoring over every node ──────────────
        return self._apply_sibling_override(self._hybrid_route(query), query)

    def _apply_sibling_override(self, result: dict, query: str) -> dict:
        """
        Confirmed keyword-overlap sibling confusions (see
        SIBLING_DISAMBIGUATION_RULES) — some queries land on the wrong side
        of a specific known pair because of a generic term shared by both
        categories' corpora (e.g. "spray" appears in both hair-care and
        fragrance titles), REGARDLESS of which routing step (alias lookup or
        hybrid scoring) produced the pick — an earlier alias step can match
        before the hybrid scorer ever runs, so this is applied uniformly as
        a post-process rather than only inside the hybrid path. Deliberately
        NOT a global confidence-threshold change: genuine ambiguous pairs
        (Office/Home chairs, etc.) are untouched — only the exact pairs
        proven wrong by the retrieval failure audit get a keyword override,
        and only when the query actually contains a disambiguating word for
        the OTHER side (an underspecified query like bare "spray" with no
        further context has nothing to disambiguate with and is left as-is).
        """
        l2 = result.get("l2")
        if not l2:
            return result

        q = query.lower()
        for pair, rule in SIBLING_DISAMBIGUATION_RULES.items():
            if l2 not in pair:
                continue
            other_l2 = next(label for label in pair if label != l2)
            chosen_match = any(kw in q for kw in rule.get(l2, []))
            other_match = any(kw in q for kw in rule.get(other_l2, []))
            if other_match and not chosen_match:
                other_key = next(
                    (key for key in self._l2_keys if key.split("/", 1)[1] == other_l2), None
                )
                if other_key:
                    overridden = self._make_result(other_key, "hybrid_taxonomy", confidence=result.get("confidence", 1.0))
                    overridden["alternatives"] = result.get("alternatives", [])
                    return overridden
        return result

    def get_node_config(self, node_key: str) -> dict:
        """Retrieve the full node config dict for a given node_key."""
        self._load()
        return self._config.get(node_key, self._config.get("_global", {}))

    # ── Hybrid scoring ────────────────────────────────────────────────────────

    def _keyword_scores(self, query_tokens: list) -> dict:
        """
        idf-weighted relative term frequency of the query against every
        node's term corpus. Uses tf/doc_len (a proportion) rather than raw
        tf, so a 30-product node and a 3000-product node are compared fairly
        — raw-count BM25 systematically favours whichever side the count
        scale happens to suit, which isn't meaningful when "documents" here
        are node corpora of wildly different sizes.
        """
        scores = {}
        for node_key, node_data in self._route_nodes.items():
            terms = node_data["terms"]
            doc_len = node_data.get("doc_len", 1) or 1
            score = 0.0
            for tok in query_tokens:
                tf = terms.get(tok, 0)
                if tf == 0:
                    continue
                idf = self._route_idf.get(tok, 0.0)
                score += idf * (tf / doc_len)
            scores[node_key] = score
        return scores

    def _hybrid_route(self, query: str) -> dict:
        self._load_hybrid()
        query_tokens = _tokenize(query)
        if not query_tokens:
            return self._global()

        kw_scores = self._keyword_scores(query_tokens)

        embedder = self._get_embedder()
        q_vec = embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
        sem_scores_arr = self._route_emb @ q_vec  # cosine sim, embeddings pre-normalised
        sem_scores = {
            key: float(sem_scores_arr[i]) for i, key in enumerate(self._route_emb_keys)
        }

        # Min-max normalise each side across candidate nodes so the two scales
        # (unbounded BM25 vs. roughly [-1, 1] cosine) combine meaningfully.
        kw_norm = _min_max_normalise(kw_scores)
        sem_norm = _min_max_normalise(sem_scores)

        combined = {}
        for node_key in kw_scores:
            combined[node_key] = (
                KEYWORD_WEIGHT * kw_norm.get(node_key, 0.0)
                + SEMANTIC_WEIGHT * sem_norm.get(node_key, 0.0)
            )

        ranked = sorted(combined.items(), key=lambda kv: -kv[1])
        top_key, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_score - second_score

        alternatives = [
            {"node_key": k, "score": round(s, 4)} for k, s in ranked[1:4]
        ]

        if top_score < ROUTE_MIN_SCORE:
            return self._global()

        confidence = round(min(margin, 1.0), 4)
        result = self._make_result(top_key, "hybrid_taxonomy", confidence=confidence)
        result["alternatives"] = alternatives
        result["top_score"] = round(top_score, 4)
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_result(self, key: str, source: str, confidence: float = 1.0) -> dict:
        parts = key.split("/", 1)
        l1 = parts[0]
        l2 = parts[1] if len(parts) > 1 else None
        return {
            "l1": l1, "l2": l2, "node_key": key, "source": source,
            "confidence": confidence, "alternatives": [],
        }

    def _global(self) -> dict:
        return {
            "l1": None, "l2": None, "node_key": "_global", "source": "global",
            "confidence": 0.0, "alternatives": [],
        }


def _min_max_normalise(scores: dict) -> dict:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


# Confirmed via retrieval failure audit (scripts/13_retrieval_failure_audit.py,
# then manually verified case-by-case): specific sibling L2 pairs where the
# hybrid scorer confidently locks the WRONG side because of a generic term
# shared by both categories' corpora — not genuine ambiguity. Deliberately
# small and pair-specific rather than a global confidence-threshold change,
# which would risk softening correct locks elsewhere. Most of the 24
# "hard-locked wrong sibling" cases found by the audit were EXCLUDED from
# this table after inspection: many had no ground-truth category at all
# (uncategorized target, not a routing bug), and several were genuine
# taxonomy overlap (e.g. "snack"->Snacks marked wrong because GT said
# Grocery General, while "sweet"->Candy & Sweets was ALSO marked wrong
# because GT said Snacks — contradictory, meaning those categories
# genuinely overlap; not a bug a keyword table can resolve). Only add a
# pair here once you've confirmed via CategoryRouter.route()'s
# "alternatives" output that the correct side scores as a strong runner-up,
# not just a plausible one.
SIBLING_DISAMBIGUATION_RULES = {
    frozenset({"Fragrance & Body", "Hair Care"}): {
        "Hair Care": ["shampoo", "conditioner", "hair spray", "dry shampoo"],
        "Fragrance & Body": ["perfume", "cologne", "body spray", "eau de"],
    },
    frozenset({"Earbud & In Ear Headphones", "Over Ear Headphones"}): {
        "Over Ear Headphones": ["over ear", "over-ear", "overear"],
        "Earbud & In Ear Headphones": ["earbud", "in ear", "in-ear", "earphone"],
    },
    frozenset({"Camping Hand Warmers", "Soccer Balls"}): {
        "Soccer Balls": ["football", "soccer ball", "soccer", "futbol"],
        "Camping Hand Warmers": ["hand warmer", "warmer packet"],
    },
}
