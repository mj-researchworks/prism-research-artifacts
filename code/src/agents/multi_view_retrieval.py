"""
Multi-View Retrieval V1
------------------------
Additional lightweight keyword (BM25-style) retrieval views merged on top of
SearchAgent's existing semantic+BM25 hybrid search ("semantic_view"). Gated
behind MACPRS_USE_MULTI_VIEW_RETRIEVAL (default "0" — off). When off,
SearchAgent.search_many() behaves exactly as before this module existed:
apply_multi_view_retrieval() is never called, and this module builds no
indexes and loads no extra data.

Motivation (MACPRS overall system audit,
outputs/macprs_overall_system_audit_report.md): retrieval miss is the
single largest measured failure bucket (94/178 Reveal conversations with
ground truth), bigger than rerank loss or constraint failure. This module
targets candidate RECALL, not reranking — it adds more ways for the right
product to enter the candidate pool before the (unmodified) reranker runs.

Views (all soft/additive — none of them hard-filter by taxonomy, category,
or product_type; they only add candidates and a soft fused score):
  - semantic_view:        SearchAgent's existing FAISS + BM25 hybrid search,
                           passed in unchanged, not reimplemented here.
  - title_view:            BM25 over product title text only — exact
                            catalog-item recall (e.g. "Logitech M705").
  - attribute_view:        BM25 over normalized product_type_norm/brand_norm/
                            color_norm/materials/features/compatibility/
                            key_specs (outputs/normalized_product_index.json)
                            — recovers spec/attribute-driven queries.
  - category_view:         BM25 over product_type_norm/category_candidates
                            only — soft category/product-type recall. Given
                            the LOWEST fusion weight of the four extra views
                            so a broad category match can never outrank an
                            exact title/attribute match; this is recall
                            help, not a taxonomy filter or category-drift
                            risk.
  - compact_metadata_view: BM25 over a short title+brand+product_type_norm+
                            key_specs string — a balanced lexical view.

Merging: weighted reciprocal-rank fusion (RRF) across all views' ranked
lists. RRF is used (rather than min-max normalizing raw scores across BM25
corpora of very different vocabularies and a cosine-similarity semantic
score) because it only needs each view's RANK ORDER, which is comparable
across arbitrarily different scoring functions/corpora — the standard
reason RRF is preferred for merging heterogeneous retrieval systems.

Every candidate keeps: retrieval_views (provenance), view_scores (raw
per-view score), and fused_retrieval_score (post-fusion score, aliased to
`score` in the final result dict).
"""

import math
import os
from collections import Counter, defaultdict

from src.agents.normalized_metadata import load_normalized_metadata

MULTI_VIEW_RETRIEVAL_ENABLED = os.getenv(
    "MACPRS_USE_MULTI_VIEW_RETRIEVAL", "0"
).strip().lower() in {"1", "true", "yes", "on"}

# Suggested initial weights (task spec) — category_view deliberately lowest
# so it can only help recall, never dominate an exact title/attribute hit.
VIEW_WEIGHTS = {
    "semantic": 1.0,
    "title": 0.9,
    "attribute": 0.8,
    "category": 0.5,
    "compact_metadata": 0.7,
}
RRF_K = 60  # standard reciprocal-rank-fusion damping constant
EXTRA_VIEW_NAMES = ("title", "attribute", "category", "compact_metadata")


def _lex_tokens(text) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _as_text(value) -> str:
    if not value:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(v) for v in value if v)
    return str(value)


def _flatten_key_specs(key_specs) -> str:
    if not key_specs:
        return ""
    if isinstance(key_specs, dict):
        return " ".join(str(v) for v in key_specs.values() if v)
    return str(key_specs)


def _build_bm25_index(id_to_text: dict) -> dict:
    """Same BM25 formulation as SearchAgent._get_lexical_index (k1=1.5,
    b=0.75) — a parallel, independent index so this module never touches
    SearchAgent's own lexical index or FAISS indexes."""
    postings = defaultdict(list)
    doc_len = {}
    n_docs = 0
    total_len = 0

    for pid, text in id_to_text.items():
        tokens = _lex_tokens(text)
        if not tokens:
            continue
        counts = Counter(tokens)
        length = sum(counts.values())
        doc_len[pid] = length
        total_len += length
        n_docs += 1
        for token, tf in counts.items():
            postings[token].append((pid, tf))

    avg_len = total_len / n_docs if n_docs else 1.0
    idf = {
        token: math.log(1.0 + (n_docs - len(items) + 0.5) / (len(items) + 0.5))
        for token, items in postings.items()
    }
    return {
        "postings": dict(postings),
        "doc_len": doc_len,
        "avg_len": avg_len,
        "idf": idf,
        "n_docs": n_docs,
    }


def _bm25_query(index: dict, query: str, k: int) -> dict:
    """Return {product_id: bm25_score} for this view's index."""
    tokens = _lex_tokens(query)
    if not tokens or not index["n_docs"]:
        return {}
    postings = index["postings"]
    doc_len = index["doc_len"]
    idf = index["idf"]
    avg_len = index["avg_len"] or 1.0
    scores = defaultdict(float)
    k1, b = 1.5, 0.75

    for token in tokens:
        token_idf = idf.get(token)
        if token_idf is None:
            continue
        for pid, tf in postings.get(token, []):
            dl = doc_len.get(pid, avg_len)
            denom = tf + k1 * (1.0 - b + b * dl / avg_len)
            scores[pid] += token_idf * (tf * (k1 + 1.0)) / denom

    if not scores:
        return {}
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return dict(top)


# ── View index cache ─────────────────────────────────────────────────────────

_view_indexes: dict | None = None


def reset_view_indexes():
    """Test-only helper — clears the cached indexes so tests can build a
    fresh small index over a fake product set."""
    global _view_indexes
    _view_indexes = None


def build_view_indexes(metadata: dict) -> dict:
    """
    Build (once, cached) the title/attribute/category/compact_metadata BM25
    indexes from `metadata` (SearchAgent's existing per-product dict, already
    loaded from product_metadata_for_ranking.json) plus
    outputs/normalized_product_index.json (loaded via the existing
    normalized_metadata loader — no duplicate file read). No FAISS index is
    built or touched.
    """
    global _view_indexes
    if _view_indexes is not None:
        return _view_indexes

    normalized = load_normalized_metadata()

    title_texts = {}
    attribute_texts = {}
    category_texts = {}
    compact_texts = {}

    for pid, product in metadata.items():
        title = product.get("title") or ""
        title_texts[pid] = title

        norm = normalized.get(pid) or {}
        attribute_texts[pid] = " ".join(filter(None, [
            _as_text(norm.get("product_type_norm")),
            _as_text(norm.get("brand_norm")),
            _as_text(norm.get("color_norm")),
            _as_text(norm.get("materials")),
            _as_text(norm.get("features")),
            _as_text(norm.get("compatibility")),
            _flatten_key_specs(norm.get("key_specs")),
        ]))
        category_texts[pid] = " ".join(filter(None, [
            _as_text(norm.get("product_type_norm")),
            _as_text(norm.get("category_candidates")),
        ]))
        compact_texts[pid] = " ".join(filter(None, [
            title,
            product.get("brand") or "",
            _as_text(norm.get("product_type_norm")),
            _flatten_key_specs(norm.get("key_specs")),
        ]))

    _view_indexes = {
        "title": _build_bm25_index(title_texts),
        "attribute": _build_bm25_index(attribute_texts),
        "category": _build_bm25_index(category_texts),
        "compact_metadata": _build_bm25_index(compact_texts),
    }
    _diagnostics["products_loaded"] = len(metadata)
    return _view_indexes


# ── Diagnostics ──────────────────────────────────────────────────────────────

_diagnostics = {
    "products_loaded": 0,
    "calls": 0,
    "candidates_returned_total": 0,
    "views_per_candidate_total": 0,
    "view_hit_counts": Counter(),
    "unique_gain_total": 0,
    "gold_view_hits": Counter(),
}


def reset_diagnostics():
    global _diagnostics
    _diagnostics = {
        "products_loaded": 0,
        "calls": 0,
        "candidates_returned_total": 0,
        "views_per_candidate_total": 0,
        "view_hit_counts": Counter(),
        "unique_gain_total": 0,
        "gold_view_hits": Counter(),
    }


def record_gold_hit(retrieval_views: list):
    """
    Called by eval code (not runtime) when a gold/relevant product is found
    among the raw candidate pool and multi-view retrieval was enabled —
    tallies which view(s) actually surfaced that gold product. Kept here
    (not duplicated in eval/evaluate.py) so all multi-view diagnostics share
    one counter/reset lifecycle.
    """
    for view in retrieval_views or []:
        _diagnostics["gold_view_hits"][view] += 1


def get_multi_view_diagnostics() -> dict:
    calls = _diagnostics["calls"] or 1
    candidates_total = _diagnostics["candidates_returned_total"] or 1
    return {
        "multi_view_retrieval_enabled": MULTI_VIEW_RETRIEVAL_ENABLED,
        "multi_view_products_loaded": _diagnostics["products_loaded"],
        "multi_view_candidates_returned": _diagnostics["candidates_returned_total"],
        "multi_view_avg_views_per_candidate": round(
            _diagnostics["views_per_candidate_total"] / candidates_total, 4
        ),
        "multi_view_view_hit_counts": dict(_diagnostics["view_hit_counts"]),
        "multi_view_unique_candidate_gain_avg": round(
            _diagnostics["unique_gain_total"] / calls, 4
        ),
        "multi_view_gold_view_hits": dict(_diagnostics["gold_view_hits"]),
    }


def _record_call_diagnostics(results: list, new_candidate_count: int):
    _diagnostics["calls"] += 1
    _diagnostics["candidates_returned_total"] += len(results)
    _diagnostics["unique_gain_total"] += new_candidate_count
    for r in results:
        views = r.get("retrieval_views") or []
        _diagnostics["views_per_candidate_total"] += len(views)
        for view in views:
            _diagnostics["view_hit_counts"][view] += 1


# ── Public entry point ───────────────────────────────────────────────────────

def apply_multi_view_retrieval(
    query: str,
    metadata: dict,
    semantic_results: list,
    top_k: int,
) -> list:
    """
    Fuse `semantic_results` (SearchAgent's existing hybrid search output —
    treated as the "semantic" view, unmodified) with title/attribute/
    category/compact_metadata BM25 views, via weighted reciprocal-rank
    fusion. Returns a new list of result dicts (same base schema as
    SearchAgent's normal output — product_id/title/brand/price/rating/
    description/score/semantic_score/...), each additionally carrying:
      - retrieval_views: sorted list of view names that surfaced this candidate
      - view_scores: {view_name: raw_score_in_that_view}
      - fused_retrieval_score: the RRF-fused score (also aliased to `score`)

    No hard filtering by taxonomy/category/product_type is performed here —
    every view only ADDS candidates and a soft score contribution.

    Callers must guard this with `if MULTI_VIEW_RETRIEVAL_ENABLED:` — this
    function itself does not check the flag, so it stays trivially testable
    regardless of the flag's env-var state (same convention as
    interpret_route_v1.build_interpret_context / reveal_route_v1.build_reveal_context).
    """
    indexes = build_view_indexes(metadata)
    per_view_k = max(top_k * 2, 100)

    view_ranked_ids = {"semantic": [r["product_id"] for r in semantic_results]}
    view_raw_scores = {
        "semantic": {
            r["product_id"]: r.get("semantic_score", r.get("score", 0.0))
            for r in semantic_results
        }
    }

    for view_name in EXTRA_VIEW_NAMES:
        raw_scores = _bm25_query(indexes[view_name], query, per_view_k)
        ranked = [pid for pid, _ in sorted(raw_scores.items(), key=lambda kv: kv[1], reverse=True)]
        view_ranked_ids[view_name] = ranked
        view_raw_scores[view_name] = raw_scores

    fused_scores = defaultdict(float)
    candidate_views = defaultdict(list)
    candidate_view_scores = defaultdict(dict)

    for view_name, ranked_ids in view_ranked_ids.items():
        weight = VIEW_WEIGHTS.get(view_name, 0.5)
        for rank, pid in enumerate(ranked_ids, start=1):
            fused_scores[pid] += weight / (RRF_K + rank)
            candidate_views[pid].append(view_name)
            candidate_view_scores[pid][view_name] = round(
                float(view_raw_scores[view_name].get(pid, 0.0)), 6
            )

    base_by_pid = {r["product_id"]: r for r in semantic_results}
    fused_sorted = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    results = []
    new_candidate_count = 0
    for pid, fused_score in fused_sorted:
        product = metadata.get(pid)
        if product is None:
            continue
        base = base_by_pid.get(pid)
        if base is not None:
            result = dict(base)
        else:
            new_candidate_count += 1
            result = {
                "product_id": pid,
                "title": product.get("title", ""),
                "brand": product.get("brand", ""),
                "price": product.get("price"),
                "rating": product.get("rating"),
                "description": product.get("description", ""),
                "score": 0.0,
                "semantic_score": 0.0,
                "keyword_score": 0.0,
                "hybrid_score": 0.0,
                "hybrid_alpha": 0.0,
                "scope_rank": 0,
                "best_query": "multi_view",
                **{
                    k: product.get(k)
                    for k in ("category", "l1", "l2", "gender", "pet_type", "color")
                    if product.get(k) is not None
                },
            }
        result["score"] = round(fused_score, 6)
        result["fused_retrieval_score"] = round(fused_score, 6)
        result["retrieval_views"] = sorted(set(candidate_views[pid]))
        result["view_scores"] = candidate_view_scores[pid]
        results.append(result)

    _record_call_diagnostics(results, new_candidate_count)
    return results
