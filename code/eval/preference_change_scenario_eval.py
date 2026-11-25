#!/usr/bin/env python3
"""Controlled preference-change scenario evaluation for PRISM.

This evaluator is intentionally separate from PRISM's runtime code. It runs
small controlled scenarios through the normal AutonomousRunner and scores
whether the stored/search state and displayed recommendations respect known
preference changes.

The first version is PRISM-only. It supports changed/control variants and
immediate/delayed checkpoints. It does not implement the optional corrected
state branch yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SCENARIOS = ROOT / "eval" / "scenarios" / "preference_change_pilot.jsonl"
DEFAULT_OUT_DIR = ROOT / "eval" / "results" / "preference_change_scenarios"
DEFAULT_PRODUCTS = ROOT / "data" / "product_base_table.json"
DEFAULT_PRODUCT_TEXT = ROOT / "data" / "product_base_retrieval" / "product_base_retrieval_text.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--product-text", type=Path, default=DEFAULT_PRODUCT_TEXT)
    parser.add_argument("--planner-mode", default=os.getenv("AUTONOMOUS_PLANNER_MODE", "auto"))
    parser.add_argument("--execution-scope", default="full")
    parser.add_argument("--retrieval-backend", default=os.getenv("AUTONOMOUS_RETRIEVAL_BACKEND", "conserve_stack"))
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument("--skip-delayed", action="store_true")
    args = parser.parse_args()

    os.environ["AUTONOMOUS_EXECUTION_SCOPE"] = args.execution_scope
    os.environ["AUTONOMOUS_RETRIEVAL_BACKEND"] = args.retrieval_backend
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from src.autonomous.eval_logger import EvalLogger
    from src.autonomous.runner import AutonomousRunner

    scenarios = load_scenarios(args.scenarios)
    if args.max_scenarios:
        scenarios = scenarios[: args.max_scenarios]
    products = load_products(args.products, args.product_text)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    started = time.time()
    for index, scenario in enumerate(scenarios, start=1):
        print(f"[{index:3}/{len(scenarios)}] {scenario['scenario_id']} {scenario['event_type']}", flush=True)
        variants = [("changed", scenario["turns"], scenario)]
        if not args.skip_control:
            variants.append(("control", scenario["control_turns"], {"control_expected": scenario["control_expected"]}))

        for variant, base_turns, expected_source in variants:
            checkpoints = [("immediate", base_turns, expected_for(scenario, variant, "immediate"))]
            if not args.skip_delayed:
                checkpoints.append((
                    "delayed",
                    base_turns + list(scenario.get("delayed_turns") or []),
                    expected_for(scenario, variant, "delayed"),
                ))
            for checkpoint, turns, expected in checkpoints:
                runner = AutonomousRunner(
                    planner_mode=args.planner_mode,
                    execution_scope=args.execution_scope,
                    eval_logger=EvalLogger(enabled=False),
                )
                response = ""
                state: dict[str, Any] = {}
                latency_ms = 0
                failed = ""
                try:
                    for turn in turns:
                        turn_started = time.perf_counter()
                        response, state = runner.run_turn(str(turn))
                        latency_ms += int((time.perf_counter() - turn_started) * 1000)
                except Exception as exc:
                    failed = f"{type(exc).__name__}: {exc}"
                    print(f"  failed {variant}/{checkpoint}: {failed}", flush=True)

                rows.append(score_run(
                    scenario=scenario,
                    variant=variant,
                    checkpoint=checkpoint,
                    turns=turns,
                    response=response,
                    state=state,
                    expected=expected,
                    products=products,
                    latency_ms=latency_ms,
                    error=failed,
                ))

    summary = summarize(rows, args, elapsed=time.time() - started)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows_path = args.out_dir / f"preference_change_scenarios_{timestamp}.csv"
    summary_path = args.out_dir / f"preference_change_scenarios_summary_{timestamp}.json"
    write_csv(rows_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nPreference-change scenario evaluation complete")
    print(f"Rows    -> {rows_path}")
    print(f"Summary -> {summary_path}")
    print(json.dumps(summary["overall"], indent=2))
    return 0


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))
    return scenarios


def expected_for(scenario: dict[str, Any], variant: str, checkpoint: str) -> dict[str, Any]:
    if variant == "control":
        return dict(scenario["control_expected"])
    key = "expected_delayed" if checkpoint == "delayed" else "expected_immediate"
    return dict(scenario[key])


def load_products(metadata_path: Path, text_path: Path) -> dict[str, dict[str, Any]]:
    products: dict[str, dict[str, Any]] = {}
    with metadata_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        for pid, product in raw.items():
            if isinstance(product, dict):
                products[str(pid)] = dict(product)
    elif isinstance(raw, list):
        for product in raw:
            if isinstance(product, dict) and product.get("product_id"):
                products[str(product["product_id"])] = dict(product)

    with text_path.open(encoding="utf-8") as handle:
        text_rows = json.load(handle)
    if isinstance(text_rows, list):
        for row in text_rows:
            if isinstance(row, dict) and row.get("product_id"):
                pid = str(row["product_id"])
                products.setdefault(pid, {"product_id": pid})
                products[pid]["retrieval_text"] = str(row.get("text") or "")
    return products


def score_run(
    *,
    scenario: dict[str, Any],
    variant: str,
    checkpoint: str,
    turns: list[str],
    response: str,
    state: dict[str, Any],
    expected: dict[str, Any],
    products: dict[str, dict[str, Any]],
    latency_ms: int,
    error: str,
) -> dict[str, Any]:
    request = serialize(state.get("final_search_request"))
    recommended_ids = extract_recommended_ids(state)
    recommended_texts = [product_search_text(products.get(pid, {"product_id": pid})) for pid in recommended_ids]
    state_text = state_to_text(state, request)
    active = flatten_values(expected.get("active"))
    unchanged = flatten_values(expected.get("unchanged"))
    exclusions = flatten_values(expected.get("hard_exclusions"))
    constraints = flatten_values(expected.get("hard_constraints"))

    active_hits = matched_terms(active, state_text)
    unchanged_hits = matched_terms(unchanged, state_text)
    exclusion_state_hits = matched_terms(exclusions, state_text)
    product_exclusion_hits = product_violations(exclusions, recommended_ids, recommended_texts)
    product_constraint_unknowns = unknown_constraint_checks(constraints, recommended_ids, recommended_texts)

    state_exact = int(bool(active) and len(active_hits) == len(unique_norm_terms(active)) and not exclusion_state_hits)
    retained = len(unchanged_hits) / len(unique_norm_terms(unchanged)) if unique_norm_terms(unchanged) else None
    recommendation_found = int(bool(recommended_ids))
    violation = int(bool(product_exclusion_hits))
    unknown_rate = (
        len(product_constraint_unknowns) / max(1, len(recommended_ids) * max(1, len(unique_norm_terms(constraints))))
        if recommended_ids and constraints else 0.0
    )

    return {
        "scenario_id": scenario.get("scenario_id"),
        "event_type": scenario.get("event_type"),
        "variant": variant,
        "checkpoint": checkpoint,
        "turn_count": len(turns),
        "last_user_message": turns[-1] if turns else "",
        "eval_error": error,
        "recommendation_found": recommendation_found,
        "state_exact": state_exact,
        "retained": retained,
        "violation": violation,
        "unknown_rate": unknown_rate,
        "latency_ms": latency_ms,
        "recommended_ids": " ".join(recommended_ids),
        "recommended_titles": " || ".join(str(products.get(pid, {}).get("title") or "")[:120] for pid in recommended_ids),
        "active_expected": " | ".join(active),
        "unchanged_expected": " | ".join(unchanged),
        "exclusions_expected": " | ".join(exclusions),
        "constraints_expected": " | ".join(constraints),
        "active_state_hits": " | ".join(active_hits),
        "unchanged_state_hits": " | ".join(unchanged_hits),
        "exclusion_state_hits": " | ".join(exclusion_state_hits),
        "product_exclusion_hits": " | ".join(product_exclusion_hits),
        "unknown_constraint_checks": " | ".join(product_constraint_unknowns[:30]),
        "query_text": str(request.get("query_text") or ""),
        "category": str(request.get("category") or ""),
        "hard_constraints": json.dumps(request.get("hard_constraints") or {}, ensure_ascii=False),
        "soft_preferences": json.dumps(request.get("soft_preferences") or {}, ensure_ascii=False),
        "preferred": json.dumps(request.get("preferred") or {}, ensure_ascii=False),
        "exclusions": json.dumps(request.get("exclusions") or {}, ensure_ascii=False),
        "preference_events": json.dumps(state.get("preference_events") or [], ensure_ascii=False),
        "recent_memory_events": json.dumps(state.get("recent_memory_events") or [], ensure_ascii=False),
        "response": response,
    }


def extract_recommended_ids(state: dict[str, Any]) -> list[str]:
    for key in ("last_recommendations", "recommendation_decisions", "verified_candidates", "raw_candidates"):
        ids = []
        for item in state.get(key) or []:
            if isinstance(item, str):
                ids.append(item)
            elif isinstance(item, dict):
                pid = item.get("product_id") or item.get("id") or item.get("asin")
                if pid:
                    ids.append(str(pid))
        if ids:
            return list(dict.fromkeys(ids[:5]))
    return []


def product_search_text(product: dict[str, Any]) -> str:
    attrs = product.get("attributes") or {}
    if isinstance(attrs, dict):
        attr_text = " ".join(f"{k} {v}" for k, v in attrs.items())
    else:
        attr_text = str(attrs or "")
    return " ".join(
        str(part or "")
        for part in [
            product.get("product_id"),
            product.get("title"),
            product.get("brand"),
            product.get("color"),
            product.get("category"),
            product.get("description"),
            attr_text,
            product.get("retrieval_text"),
        ]
    )


def state_to_text(state: dict[str, Any], request: dict[str, Any]) -> str:
    parts = [
        request.get("query_text"),
        request.get("category"),
        json.dumps(request.get("hard_constraints") or {}, ensure_ascii=False),
        json.dumps(request.get("soft_preferences") or {}, ensure_ascii=False),
        json.dumps(request.get("preferred") or {}, ensure_ascii=False),
        json.dumps(request.get("exclusions") or {}, ensure_ascii=False),
        json.dumps(serialize(state.get("preferences") or {}), ensure_ascii=False),
        json.dumps(state.get("recent_memory_events") or [], ensure_ascii=False),
    ]
    return norm(" ".join(str(part or "") for part in parts))


def matched_terms(terms: list[str], text: str) -> list[str]:
    normalized_text = norm(text)
    return [term for term in unique_norm_terms(terms) if term and term in normalized_text]


def product_violations(exclusions: list[str], ids: list[str], texts: list[str]) -> list[str]:
    out: list[str] = []
    exclusion_terms = unique_norm_terms(exclusions)
    for pid, text in zip(ids, texts):
        normalized_text = norm(text)
        for term in exclusion_terms:
            if term and term in normalized_text:
                out.append(f"{pid}:{term}")
    return out


def unknown_constraint_checks(constraints: list[str], ids: list[str], texts: list[str]) -> list[str]:
    out: list[str] = []
    constraint_terms = unique_norm_terms(constraints)
    for pid, text in zip(ids, texts):
        normalized_text = norm(text)
        for term in constraint_terms:
            if term and term not in normalized_text:
                out.append(f"{pid}:{term}")
    return out


def flatten_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [str(v) for v in value.values() if v not in (None, "")]
    if isinstance(value, list):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)]


def unique_norm_terms(terms: list[str]) -> list[str]:
    out: list[str] = []
    for term in terms:
        normalized = norm(term)
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: serialize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [serialize(item) for item in value]
    return value if value is not None else {}


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, *, elapsed: float) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for key in ("overall",):
        groups[key] = summarize_group(rows)
    for event_type in sorted(set(str(row["event_type"]) for row in rows)):
        groups[f"event:{event_type}"] = summarize_group([row for row in rows if row["event_type"] == event_type])
    for checkpoint in sorted(set(str(row["checkpoint"]) for row in rows)):
        groups[f"checkpoint:{checkpoint}"] = summarize_group([row for row in rows if row["checkpoint"] == checkpoint])
    for variant in sorted(set(str(row["variant"]) for row in rows)):
        groups[f"variant:{variant}"] = summarize_group([row for row in rows if row["variant"] == variant])
    return {
        "experiment": "preference_change_scenarios_prism_only",
        "scenario_file": str(args.scenarios),
        "planner_mode": args.planner_mode,
        "retrieval_backend": args.retrieval_backend,
        "elapsed_seconds": round(elapsed, 2),
        **groups,
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    rec_rows = [row for row in rows if int(row.get("recommendation_found") or 0)]
    return {
        "cases": n,
        "coverage": mean(rows, "recommendation_found"),
        "state_exact": mean(rows, "state_exact"),
        "retained": mean_defined(rows, "retained"),
        "violation": mean(rec_rows, "violation") if rec_rows else None,
        "unknown_rate": mean(rec_rows, "unknown_rate") if rec_rows else None,
        "latency_ms_avg": mean(rows, "latency_ms"),
    }


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    values = [float(row.get(key) or 0) for row in rows]
    return round(sum(values) / len(values), 4)


def mean_defined(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) not in (None, "")]
    if not values:
        return None
    return round(sum(float(value) for value in values) / len(values), 4)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
