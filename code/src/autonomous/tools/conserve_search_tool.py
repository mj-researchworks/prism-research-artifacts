"""Bounded adapter from the autonomous planner to existing ConServe search."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.autonomous.schemas import ComponentResult


ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = ROOT / "data" / "product_base_table.json"


class ConServeSearchTool:
    name = "conserve_search_tool"
    FALLBACK_BACKENDS = {"fallback", "catalog", "local_safe", "fallback_catalog"}
    FULL_STACK_BACKENDS = {"full", "conserve_stack", "existing_stack"}

    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.backend = os.getenv("AUTONOMOUS_RETRIEVAL_BACKEND", "auto").lower()
        self._stack: _ConServeStack | None = None
        self._cache: dict[str, dict[str, Any]] = {}

    def run(self, state: dict) -> ComponentResult:
        request = state.get("final_search_request")
        if not request:
            return ComponentResult(
                name=self.name,
                patch={"security_flags": ["missing_search_request"]},
            )

        slots = _request_to_slots(request)
        query = _request_value(request, "query_text") or state.get("latest_user_message", "")
        retrieval_plan = state.get("retrieval_plan") or {}
        query_variants = list(retrieval_plan.get("query_variants") or [])
        planned_top_k = _planned_top_k(retrieval_plan, self.top_k)
        search_depth = _planned_search_depth(retrieval_plan)
        adapter_diagnostics = _adapter_diagnostics(
            request=request,
            slots=slots,
            retrieval_plan=retrieval_plan,
            query_variants=query_variants,
            planned_top_k=planned_top_k,
            search_depth=search_depth,
        )
        cache_key = _cache_key(
            backend=self.backend,
            query=query,
            slots=slots,
            request=request,
            retrieval_plan=retrieval_plan,
            query_variants=query_variants,
            planned_top_k=planned_top_k,
            search_depth=search_depth,
            history=state.get("history") or [],
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return self._result_from_cache(
                cached=cached,
                query=query,
                slots=slots,
                retrieval_plan=retrieval_plan,
                adapter_diagnostics=adapter_diagnostics,
            )

        reused = self._try_reuse_previous_candidates(
            state=state,
            request=request,
            query=query,
            slots=slots,
            retrieval_plan=retrieval_plan,
            adapter_diagnostics=adapter_diagnostics,
        )
        if reused is not None:
            self._cache[cache_key] = {
                "candidates": deepcopy(reused.patch["raw_candidates"]),
                "verified": deepcopy(reused.patch["verified_candidates"]),
                "diagnostics": deepcopy(reused.patch["candidate_ledger"]["diagnostics"]),
                "request": _normalize_for_cache(request),
            }
            return reused

        if self.backend in self.FALLBACK_BACKENDS:
            candidates, verified, diagnostics = self._fallback_catalog_search(query, slots)
            diagnostics.update({
                "requested_backend": self.backend,
                "actual_backend": "fallback_catalog",
                "fallback_used": True,
                "fallback_reason": "fallback backend explicitly requested",
            })
        else:
            try:
                candidates, verified, diagnostics = self._run_existing_stack(
                    query,
                    slots,
                    state=state,
                    query_variants=query_variants,
                    top_k=planned_top_k,
                    search_depth=search_depth,
                )
                diagnostics.update({
                    "requested_backend": self.backend,
                    "actual_backend": diagnostics.get("backend") or "conserve_stack",
                    "fallback_used": False,
                })
            except Exception as exc:
                if self.backend in self.FULL_STACK_BACKENDS:
                    return self._failed_full_stack_result(
                        query=query,
                        slots=slots,
                        retrieval_plan=retrieval_plan,
                        adapter_diagnostics=adapter_diagnostics,
                        error=exc,
                    )
                candidates, verified, diagnostics = self._fallback_catalog_search(query, slots)
                diagnostics.update({
                    "requested_backend": self.backend,
                    "actual_backend": "fallback_catalog",
                    "fallback_used": True,
                    "fallback_reason": f"{type(exc).__name__}: {exc}",
                })

        candidates, candidate_exclusion_counts = _apply_autonomous_exclusions(candidates, request)
        verified, verified_exclusion_counts = _apply_autonomous_exclusions(verified, request)
        diagnostics["adapter"] = adapter_diagnostics
        diagnostics["adapter"]["post_filter_counts"] = {
            "raw": candidate_exclusion_counts,
            "verified": verified_exclusion_counts,
        }

        result = self._result(
            query=query,
            request=request,
            slots=slots,
            retrieval_plan=retrieval_plan,
            candidates=candidates,
            verified=verified,
            diagnostics=diagnostics,
            adapter_diagnostics=adapter_diagnostics,
            cache_hit=False,
        )
        self._cache[cache_key] = {
            "candidates": deepcopy(candidates),
            "verified": deepcopy(verified),
            "diagnostics": deepcopy(diagnostics),
            "request": _normalize_for_cache(request),
        }
        return result

    def _try_reuse_previous_candidates(
        self,
        *,
        state: dict,
        request: Any,
        query: str,
        slots: dict,
        retrieval_plan: dict,
        adapter_diagnostics: dict,
    ) -> ComponentResult | None:
        if not _candidate_reuse_enabled():
            return None

        previous_ledger = state.get("last_candidate_ledger") or {}
        previous_request = previous_ledger.get("request") or {}
        previous_slots = previous_ledger.get("slots") or {}
        previous_raw = list(state.get("last_raw_candidates") or [])
        previous_verified = list(state.get("last_verified_candidates") or [])
        previous_diagnostics = previous_ledger.get("diagnostics") or {}

        if not previous_raw or not previous_verified:
            return None
        if previous_diagnostics.get("actual_backend") in {"conserve_stack_failed"}:
            return None
        if not _candidate_reuse_request_compatible(
            previous_request=previous_request,
            previous_slots=previous_slots,
            current_request=request,
            current_slots=slots,
        ):
            return None

        candidates, candidate_exclusion_counts = _apply_autonomous_exclusions(deepcopy(previous_raw), request)
        verified, verified_exclusion_counts = _apply_autonomous_exclusions(deepcopy(previous_verified), request)
        candidates = _apply_current_hard_filters(candidates, request)
        verified = _apply_current_hard_filters(verified, request)
        minimum_verified = _candidate_reuse_min_verified(self.top_k)
        if len(verified) < minimum_verified:
            return None

        diagnostics = {
            "backend": "candidate_reuse",
            "requested_backend": self.backend,
            "actual_backend": "candidate_reuse",
            "fallback_used": False,
            "candidate_reuse_used": True,
            "candidate_reuse_reason": "compatible request reused previous verified candidate pool",
            "source_backend": previous_diagnostics.get("actual_backend") or previous_diagnostics.get("backend"),
            "source_raw_count": len(previous_raw),
            "source_verified_count": len(previous_verified),
            "adapter": adapter_diagnostics,
        }
        diagnostics["adapter"]["post_filter_counts"] = {
            "raw": candidate_exclusion_counts,
            "verified": verified_exclusion_counts,
        }
        diagnostics["adapter"]["candidate_reuse_checked"] = True
        diagnostics["adapter"]["candidate_reuse_used"] = True

        return self._result(
            query=query,
            request=request,
            slots=slots,
            retrieval_plan=retrieval_plan,
            candidates=candidates,
            verified=verified,
            diagnostics=diagnostics,
            adapter_diagnostics=adapter_diagnostics,
            cache_hit=False,
        )

    def _result_from_cache(
        self,
        *,
        cached: dict[str, Any],
        query: str,
        slots: dict,
        retrieval_plan: dict,
        adapter_diagnostics: dict,
    ) -> ComponentResult:
        candidates = deepcopy(cached["candidates"])
        verified = deepcopy(cached["verified"])
        diagnostics = deepcopy(cached["diagnostics"])
        diagnostics["cache_hit"] = True
        diagnostics["cache_size"] = len(self._cache)
        diagnostics["adapter"] = adapter_diagnostics
        adapter_diagnostics["cache_hit"] = True
        return self._result(
            query=query,
            request=cached.get("request"),
            slots=slots,
            retrieval_plan=retrieval_plan,
            candidates=candidates,
            verified=verified,
            diagnostics=diagnostics,
            adapter_diagnostics=adapter_diagnostics,
            cache_hit=True,
        )

    def _result(
        self,
        *,
        query: str,
        request: Any,
        slots: dict,
        retrieval_plan: dict,
        candidates: list,
        verified: list,
        diagnostics: dict,
        adapter_diagnostics: dict,
        cache_hit: bool,
    ) -> ComponentResult:
        diagnostics["cache_hit"] = cache_hit
        diagnostics["cache_size"] = len(self._cache)
        adapter_diagnostics["cache_hit"] = cache_hit
        return ComponentResult(
            name=self.name,
            patch={
                "raw_candidates": candidates,
                "verified_candidates": verified,
                "candidate_ledger": {
                    "query": query,
                    "request": _normalize_for_cache(request),
                    "slots": slots,
                    "retrieval_plan": retrieval_plan,
                    "raw_count": len(candidates),
                    "verified_count": len(verified),
                    "diagnostics": diagnostics,
                    "adapter_diagnostics": adapter_diagnostics,
                },
                "tool_observation": {
                    "component": self.name,
                    "summary": f"retrieved={len(candidates)} verified={len(verified)} backend={diagnostics.get('actual_backend')} cache_hit={cache_hit}",
                    "status": "success" if verified else "empty",
                },
            },
        )

    def _failed_full_stack_result(
        self,
        *,
        query: str,
        slots: dict,
        retrieval_plan: dict,
        adapter_diagnostics: dict,
        error: Exception,
    ) -> ComponentResult:
        diagnostics = {
            "backend": "conserve_stack",
            "requested_backend": self.backend,
            "actual_backend": "conserve_stack_failed",
            "fallback_used": False,
            "error": f"{type(error).__name__}: {error}",
            "adapter": adapter_diagnostics,
        }
        return ComponentResult(
            name=self.name,
            status="failed",
            patch={
                "raw_candidates": [],
                "verified_candidates": [],
                "security_flags": ["retrieval_tool_failed"],
                "candidate_ledger": {
                    "query": query,
                    "request": _normalize_for_cache(None),
                    "slots": slots,
                    "retrieval_plan": retrieval_plan,
                    "raw_count": 0,
                    "verified_count": 0,
                    "diagnostics": diagnostics,
                    "adapter_diagnostics": adapter_diagnostics,
                },
                "tool_observation": {
                    "component": self.name,
                    "summary": f"full ConServe retrieval failed: {diagnostics['error']}",
                    "status": "failed",
                },
            },
        )

    def _run_existing_stack(
        self,
        query: str,
        slots: dict,
        *,
        state: dict,
        query_variants: list[str] | None = None,
        top_k: int | None = None,
        search_depth: int = 50,
    ) -> tuple[list, list, dict]:
        if self._stack is None:
            self._stack = _ConServeStack()
        return self._stack.search(
            query=query,
            slots=slots,
            top_k=top_k or self.top_k,
            query_variants=query_variants or [],
            history=state.get("history") or [],
            search_depth=search_depth,
        )

    def _fallback_catalog_search(self, query: str, slots: dict) -> tuple[list, list, dict]:
        catalog = _load_catalog()
        terms = _tokens(query) + _tokens(slots.get("product_type")) + _tokens(slots.get("use_case"))
        price_max = slots.get("price_max")
        scored = []

        for product in catalog:
            text = _product_text(product)
            lexical_score = sum(1 for term in terms if term in text)
            if price_max is not None and _price(product) is not None and _price(product) > price_max:
                lexical_score -= 3
            if lexical_score <= 0:
                continue
            item = dict(product)
            item["score"] = float(lexical_score)
            item["rerank_score"] = float(lexical_score)
            item["final_score"] = float(lexical_score)
            scored.append(item)

        scored.sort(key=lambda p: (-p.get("final_score", 0.0), _price(p) or 10**9))
        product_type = slots.get("product_type")
        verified = [
            p for p in scored
            if _passes_hard_budget(p, price_max) and _passes_product_type_gate(p, product_type)
        ][: self.top_k]
        return scored[:50], verified, {"backend": "fallback_catalog"}


class _ConServeStack:
    def __init__(self):
        from src.agents.category_router import CategoryRouter
        from src.agents.evidence_agent import EvidenceScorer, VerifierAgent
        from src.agents.reranker import Reranker
        from src.agents.search_agent import SearchAgent

        self.router = CategoryRouter()
        self.search_agent = SearchAgent()
        self.reranker = Reranker()
        self.evidence = EvidenceScorer()
        self.verifier = VerifierAgent()

    def search(
        self,
        query: str,
        slots: dict,
        top_k: int,
        query_variants: list[str] | None = None,
        history: list | None = None,
        search_depth: int = 50,
    ) -> tuple[list, list, dict]:
        route = self.router.route(
            product_type=slots.get("product_type"),
            slots=slots,
        )
        node_key = route.get("node_key") or "_global"
        node_config = self.router.get_node_config(node_key)
        variants = list(query_variants or [])
        if query not in variants:
            variants.insert(0, query)
        if slots.get("product_type"):
            variants.append(str(slots["product_type"]))
        if slots.get("use_case"):
            variants.append(str(slots["use_case"]))
        variants = list(dict.fromkeys(value for value in variants if value))

        candidates = self.search_agent.search_many(
            node_config=node_config,
            slots=slots,
            queries=variants,
            user_message=query,
            top_k=search_depth,
        )
        review_k = max(30, top_k * 5, min(search_depth, 100))
        triplet_context = _triplet_context_for_reranker(query, history or [])
        ranked = self.reranker.rank(
            candidates=candidates,
            slots=slots,
            node_config=node_config,
            top_k=review_k,
            triplet_context=triplet_context,
        )
        evidenced = self.evidence.score_products(ranked, slots=slots, query=query)
        verified_result = self.verifier.verify(evidenced, slots=slots, top_k=review_k)
        verified_products = [
            p for p in verified_result.get("products", [])
            if _passes_hard_budget(p, slots.get("price_max"))
            and _passes_configured_product_type_gate(p, slots.get("product_type"))
        ][:top_k]
        return candidates, verified_products, {
            "backend": "conserve_stack",
            "node_key": node_key,
            "route": route,
            "variants": variants,
            "search_depth": search_depth,
            "review_k": review_k,
            "triplet_context": triplet_context,
            "triplet_context_mode": os.getenv("AUTONOMOUS_TRIPLET_CONTEXT", "query").strip().lower(),
            "rejected": verified_result.get("rejected", []),
            "warnings": verified_result.get("warnings", []),
        }


def _request_to_slots(request: Any) -> dict:
    category = _request_value(request, "category")
    hard_constraints = _request_value(request, "hard_constraints") or {}
    soft_preferences = _request_value(request, "soft_preferences") or {}

    slots = {}
    if category:
        slots["product_type"] = category
    if hard_constraints.get("budget_min") is not None:
        slots["price_min"] = hard_constraints["budget_min"]
    if hard_constraints.get("budget_max") is not None:
        slots["price_max"] = hard_constraints["budget_max"]
    if soft_preferences.get("use_case"):
        slots["use_case"] = soft_preferences["use_case"]
        slots["features"] = [soft_preferences["use_case"]]

    preferred = _request_value(request, "preferred") or {}
    if preferred.get("brands"):
        slots["brand"] = preferred["brands"][0]
    preferred_features = _rankable_features(preferred.get("features") or [])
    if preferred_features:
        slots["features"] = list(dict.fromkeys((slots.get("features") or []) + preferred_features))

    exclusions = _request_value(request, "exclusions") or {}
    if exclusions.get("brands"):
        slots["exclude_brands"] = exclusions["brands"]
    if exclusions.get("features"):
        slots["exclude_features"] = exclusions["features"]
    if exclusions.get("product_types"):
        slots["exclude_product_types"] = exclusions["product_types"]
    return slots


def _rankable_features(features: list) -> list[str]:
    vague = {
        "quality",
        "good quality",
        "normal",
        "nice",
        "good",
        "best",
        "cute",
    }
    result = []
    for feature in features or []:
        cleaned = " ".join(str(feature or "").lower().split())
        if not cleaned or cleaned in vague:
            continue
        result.append(str(feature))
    return result


def _request_value(request: Any, key: str):
    if isinstance(request, dict):
        return request.get(key)
    return getattr(request, key, None)


def _build_triplet_context(query: str, history: list) -> str:
    turns = []
    for turn in (history or [])[-8:]:
        if not isinstance(turn, dict):
            continue
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        role = turn.get("role")
        if role == "user":
            turns.append(f"User: {content}")
        elif role in {"assistant", "system"}:
            turns.append(f"Assistant clarification: {content}")
    if query and (not turns or turns[-1] != f"User: {query}"):
        turns.append(f"User: {query}")
    return "\n".join(turns[-8:]) or str(query or "")


def _triplet_context_for_reranker(query: str, history: list) -> str:
    mode = os.getenv("AUTONOMOUS_TRIPLET_CONTEXT", "query").strip().lower()
    if mode in {"conversation", "history", "structured"}:
        return _build_triplet_context(query, history)
    return str(query or "")


def _planned_top_k(retrieval_plan: dict, default: int) -> int:
    try:
        depth = int(retrieval_plan.get("search_depth") or 0)
    except (TypeError, ValueError):
        return default
    if depth >= 75:
        return max(default, 15)
    if depth >= 60:
        return max(default, 12)
    return default


def _planned_search_depth(retrieval_plan: dict, default: int = 50) -> int:
    try:
        depth = int(retrieval_plan.get("search_depth") or default)
    except (TypeError, ValueError):
        return default
    return max(25, min(depth, 100))


def _adapter_diagnostics(
    *,
    request: Any,
    slots: dict,
    retrieval_plan: dict,
    query_variants: list[str],
    planned_top_k: int,
    search_depth: int,
) -> dict:
    hard_constraints = _request_value(request, "hard_constraints") or {}
    soft_preferences = _request_value(request, "soft_preferences") or {}
    preferred = _request_value(request, "preferred") or {}
    exclusions = _request_value(request, "exclusions") or {}

    return {
        "query_text": _request_value(request, "query_text"),
        "category": _request_value(request, "category"),
        "mapped_slot_keys": sorted(slots.keys()),
        "mapped_hard_constraints": {
            key: hard_constraints.get(key)
            for key in ("budget_min", "budget_max")
            if hard_constraints.get(key) is not None
        },
        "mapped_soft_preferences": {
            key: soft_preferences.get(key)
            for key in ("use_case", "occasion", "recipient")
            if soft_preferences.get(key)
        },
        "mapped_features": list(slots.get("features") or []),
        "mapped_brand": slots.get("brand"),
        "mapped_exclude_brands": list(slots.get("exclude_brands") or []),
        "mapped_exclude_features": list(slots.get("exclude_features") or []),
        "mapped_exclude_product_types": list(slots.get("exclude_product_types") or []),
        "request_preferred_seen": {
            "brands": list(preferred.get("brands") or []),
            "features": list(preferred.get("features") or []),
        },
        "request_exclusions_seen": {
            "brands": list(exclusions.get("brands") or []),
            "products": list(exclusions.get("products") or []),
            "product_types": list(exclusions.get("product_types") or []),
            "features": list(exclusions.get("features") or []),
        },
        "retrieval_strategy": retrieval_plan.get("strategy"),
        "query_variants_from_plan": list(query_variants),
        "query_variant_count": len(query_variants),
        "planned_top_k": planned_top_k,
        "search_depth": search_depth,
        "triplet_context_mode": os.getenv("AUTONOMOUS_TRIPLET_CONTEXT", "query").strip().lower(),
    }


def _cache_key(
    *,
    backend: str,
    query: str,
    slots: dict,
    request: Any,
    retrieval_plan: dict,
    query_variants: list[str],
    planned_top_k: int,
    search_depth: int,
    history: list,
) -> str:
    triplet_mode = os.getenv("AUTONOMOUS_TRIPLET_CONTEXT", "query").strip().lower()
    history_key = _build_triplet_context(query, history) if triplet_mode in {"conversation", "history", "structured"} else ""
    payload = {
        "backend": backend,
        "query": query,
        "slots": slots,
        "request": _normalize_for_cache(request),
        "retrieval_strategy": retrieval_plan.get("strategy"),
        "query_variants": list(query_variants),
        "planned_top_k": planned_top_k,
        "search_depth": search_depth,
        "triplet_context_mode": triplet_mode,
        "history_key": history_key,
    }
    return json.dumps(_normalize_for_cache(payload), sort_keys=True, separators=(",", ":"))


def _normalize_for_cache(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_for_cache(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_cache(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _normalize_for_cache(value.__dict__)
    return value


def _apply_autonomous_exclusions(products: list[dict], request: Any) -> tuple[list[dict], dict[str, int]]:
    exclusions = _request_value(request, "exclusions") or {}
    excluded_products = {_norm(value) for value in exclusions.get("products") or [] if _norm(value)}
    excluded_brands = [_norm(value) for value in exclusions.get("brands") or [] if _norm(value)]
    excluded_features = [_norm(value) for value in exclusions.get("features") or [] if _norm(value)]
    excluded_product_types = _active_product_type_exclusions(request, exclusions)
    counts = {"products": 0, "brands": 0, "features": 0, "product_types": 0}
    kept = []

    for product in products:
        product_id = _norm(product.get("product_id"))
        brand = _norm(product.get("brand"))
        text = _product_text(product)
        if product_id and product_id in excluded_products:
            counts["products"] += 1
            continue
        if brand and any(excluded == brand or excluded in brand for excluded in excluded_brands):
            counts["brands"] += 1
            continue
        if any(term and term in text for term in excluded_features):
            counts["features"] += 1
            continue
        if any(term and term in text for term in excluded_product_types):
            counts["product_types"] += 1
            continue
        kept.append(product)

    return kept, counts


def _candidate_reuse_enabled() -> bool:
    value = os.getenv("AUTONOMOUS_CANDIDATE_REUSE", "0").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _candidate_reuse_min_verified(default_top_k: int) -> int:
    default_min = str(max(3, min(default_top_k, 5)))
    raw_value = os.getenv("AUTONOMOUS_CANDIDATE_REUSE_MIN_VERIFIED", default_min).strip()
    try:
        return max(3, int(raw_value))
    except ValueError:
        return max(3, min(default_top_k, 5))


def _candidate_reuse_request_compatible(
    *,
    previous_request: Any,
    previous_slots: dict,
    current_request: Any,
    current_slots: dict,
) -> bool:
    previous_category = _norm(_request_value(previous_request, "category") or previous_slots.get("product_type"))
    current_category = _norm(_request_value(current_request, "category") or current_slots.get("product_type"))
    if not previous_category or previous_category != current_category:
        return False

    comparable_keys = ("hard_constraints", "soft_preferences", "preferred")
    for key in comparable_keys:
        if _normalize_for_cache(_request_value(previous_request, key) or {}) != _normalize_for_cache(_request_value(current_request, key) or {}):
            return False

    previous_exclusions = _normalize_exclusions(_request_value(previous_request, "exclusions") or {})
    current_exclusions = _normalize_exclusions(_request_value(current_request, "exclusions") or {})
    return _exclusions_are_same_or_superset(previous_exclusions, current_exclusions)


def _normalize_exclusions(exclusions: dict) -> dict[str, set[str]]:
    return {
        key: {_norm(value) for value in exclusions.get(key) or [] if _norm(value)}
        for key in ("brands", "products", "product_types", "features")
    }


def _exclusions_are_same_or_superset(previous: dict[str, set[str]], current: dict[str, set[str]]) -> bool:
    for key, previous_values in previous.items():
        current_values = current.get(key, set())
        if not previous_values.issubset(current_values):
            return False
    return True


def _apply_current_hard_filters(products: list[dict], request: Any) -> list[dict]:
    hard_constraints = _request_value(request, "hard_constraints") or {}
    price_max = hard_constraints.get("budget_max")
    product_type = _request_value(request, "category")
    return [
        product
        for product in products
        if _passes_hard_budget(product, price_max)
        and _passes_configured_product_type_gate(product, product_type)
    ]


def _passes_configured_product_type_gate(product: dict, product_type) -> bool:
    mode = os.getenv("AUTONOMOUS_PRODUCT_TYPE_GATE", "hard").strip().lower()
    if mode in {"0", "false", "no", "off", "none"}:
        return True
    return _passes_product_type_gate(product, product_type)


def _active_product_type_exclusions(request: Any, exclusions: dict) -> list[str]:
    query_context = " ".join(
        str(value or "").lower()
        for value in (
            _request_value(request, "query_text"),
            _request_value(request, "category"),
        )
    )
    active = []
    for value in exclusions.get("product_types") or []:
        term = _norm(value)
        if not term:
            continue
        if re.search(rf"\b{re.escape(term)}\b", query_context):
            continue
        active.append(term)
    return active


def _load_catalog() -> list:
    with open(CATALOG_PATH, encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else list(data.values())


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _tokens(value: Any) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", str(value or "").lower()) if len(t) > 2]


def _product_text(product: dict) -> str:
    fields = ("title", "brand", "description", "category", "l1", "l2")
    return " ".join(str(product.get(field) or "").lower() for field in fields)


def _passes_product_type_gate(product: dict, product_type) -> bool:
    if not product_type:
        return True
    product_type = str(product_type).lower().strip()
    title = str(product.get("title") or "").lower()
    text = _product_text(product)
    if product_type == "laptop":
        blocked = (
            "backpack", "stand", "sleeve", "case", "charger", "adapter",
            "monitor", "mouse", "bag", "headset", "headphone", "earbuds",
            "external cd", "external dvd", "cd/dvd", "dvd drive", "cd drive",
            "dock", "docking", "keyboard", "cooling pad", "cable", "aux",
            "speaker", "hub", "usb hub", "screen protector", "screen cleaner",
            "cleaner", "wipe", "microfiber", "spray", "desk", "table",
            "cart", "mount", "holder",
        )
        return "laptop" in title and not any(term in title for term in blocked)
    if product_type == "headphones":
        return any(term in text for term in ("headphone", "headphones", "earbuds", "ear buds", "headset"))
    if product_type in text:
        return True
    tokens = _category_tokens(product_type)
    if not tokens:
        return True
    matched = sum(1 for token in tokens if token in text)
    required = _required_category_token_matches(product_type, tokens)
    return matched >= required


def _required_category_token_matches(product_type: str, tokens: list[str]) -> int:
    packaging_terms = {
        "pack", "packs", "packet", "packets", "set", "sets", "kit", "kits",
        "bundle", "bundles", "box", "boxes", "bag", "bags",
    }
    raw_tokens = set(re.findall(r"[a-z0-9]+", str(product_type or "").lower()))
    if raw_tokens & packaging_terms:
        return 1
    return 1 if len(tokens) == 1 else min(2, len(tokens))


def _category_tokens(value: str) -> list[str]:
    stopwords = {
        "for", "with", "and", "or", "the", "that", "this", "some", "pair",
        "good", "quality", "best", "nice", "cute", "cheap", "cheaper",
        "portable", "lightweight", "wireless", "electric", "general",
        "pack", "packs", "packet", "packets", "set", "sets", "kit", "kits",
        "bundle", "bundles", "box", "boxes", "bag", "bags",
    }
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) > 2 and token not in stopwords
    ]


def _passes_hard_budget(product: dict, price_max) -> bool:
    if price_max is None:
        return True
    price = _price(product)
    return price is not None and price <= float(price_max)


def _price(product: dict) -> float | None:
    value = product.get("price")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
