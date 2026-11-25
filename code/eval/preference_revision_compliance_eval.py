#!/usr/bin/env python3
"""Offline PRCR/SPVR analysis for PRISM turn logs.

This script does not rerun PRISM. It reads a saved turn-level CSV, extracts
explicit preference revision/rejection events, finds the next recommendation
after each event, and checks whether displayed products violate the new active
preference state.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRODUCTS = ROOT / "data" / "product_base_table.json"
DEFAULT_PRODUCT_TEXT = ROOT / "data" / "product_base_retrieval" / "product_base_retrieval_text.json"
DEFAULT_OUT_DIR = ROOT / "eval" / "results" / "preference_revision_compliance"

ELIGIBLE_EVENT_TYPES = {
    "brand_rejection",
    "feature_rejection",
    "product_rejection",
    "preference_revision",
    "preference_clear",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=Path, required=True)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--product-text", type=Path, default=DEFAULT_PRODUCT_TEXT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--all-future-recommendations",
        action="store_true",
        help="Score every later recommendation after a revision. Default scores only the next recommendation.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    products = load_products(args.products, args.product_text)
    rows = load_turns(args.turns)
    by_conv: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_conv[str(row["conversation_id"])].append(row)
    for conv_rows in by_conv.values():
        conv_rows.sort(key=lambda r: int(r["turn_index"]))

    eval_rows: list[dict[str, Any]] = []
    for conv_id, conv_rows in sorted(by_conv.items()):
        events = extract_eligible_events(conv_rows)
        for event in events:
            future_recs = [
                row for row in conv_rows
                if int(row["turn_index"]) > int(event["turn_id"]) and is_recommendation(row)
            ]
            if not future_recs:
                continue
            if not args.all_future_recommendations:
                future_recs = future_recs[:1]
            for rec_row in future_recs:
                eval_rows.append(score_event(conv_id, event, rec_row, products))

    summary = summarize(eval_rows)
    out_csv = args.out_dir / "preference_revision_compliance_events.csv"
    out_json = args.out_dir / "preference_revision_compliance_summary.json"
    write_csv(out_csv, eval_rows)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Preference revision compliance evaluation complete")
    print(f"Events CSV -> {out_csv}")
    print(f"Summary    -> {out_json}")
    print(json.dumps(summary, indent=2))


def load_turns(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def extract_eligible_events(conv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int, str, str, str]] = set()
    events: list[dict[str, Any]] = []
    for row in conv_rows:
        for event in parse_json_list(row.get("preference_events")):
            event_type = str(event.get("type") or "")
            if event_type not in ELIGIBLE_EVENT_TYPES:
                continue
            turn_id = int(event.get("turn_id") or row["turn_index"])
            old_value = clean_value(event.get("old_value"))
            new_value = clean_value(event.get("new_value"))
            slot = str(event.get("slot") or "")
            key = (event_type, turn_id, slot, old_value, new_value)
            if key in seen:
                continue
            seen.add(key)
            copied = dict(event)
            copied["turn_id"] = turn_id
            copied["old_value"] = old_value
            copied["new_value"] = new_value
            copied["slot"] = slot
            events.append(copied)
    return events


def score_event(conv_id: str, event: dict[str, Any], rec_row: dict[str, Any], products: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rec_ids = split_ids(rec_row.get("recommended_ids"))
    stale_terms = stale_violation_terms(event)
    active_terms = active_required_terms(event)
    product_evidence = []
    stale_hits: list[str] = []
    active_missing: list[str] = []

    for pid in rec_ids:
        text = product_search_text(products.get(pid, {"product_id": pid}))
        lowered = normalize_for_match(text)
        stale_match = [term for term in stale_terms if term and term in lowered]
        active_match = [term for term in active_terms if term and term in lowered]
        if stale_match:
            stale_hits.append(f"{pid}:{'|'.join(stale_match)}")
        if active_terms and not active_match:
            active_missing.append(pid)
        product_evidence.append({
            "product_id": pid,
            "title": str(products.get(pid, {}).get("title") or "")[:160],
            "stale_matches": stale_match,
            "active_matches": active_match,
        })

    stale_violation = bool(stale_hits)
    # Conservative compliance: no stale terms and, when the revision creates a
    # simple positive active requirement, every displayed item contains it.
    compliant = (not stale_violation) and (not active_missing)

    return {
        "conversation_id": conv_id,
        "revision_turn": event.get("turn_id"),
        "recommendation_turn": rec_row.get("turn_index"),
        "event_type": event.get("type"),
        "slot": event.get("slot"),
        "old_preference": event.get("old_value"),
        "new_active_preference": event.get("new_value"),
        "evidence_text": event.get("evidence_text"),
        "recommended_ids": " ".join(rec_ids),
        "recommended_titles": " || ".join(str(products.get(pid, {}).get("title") or "")[:120] for pid in rec_ids),
        "compliant": int(compliant),
        "stale_violation": int(stale_violation),
        "stale_terms_checked": " | ".join(stale_terms),
        "active_terms_checked": " | ".join(active_terms),
        "stale_hits": " ; ".join(stale_hits),
        "active_missing_product_ids": " ".join(active_missing),
        "product_evidence_json": json.dumps(product_evidence, ensure_ascii=False),
    }


def stale_violation_terms(event: dict[str, Any]) -> list[str]:
    event_type = str(event.get("type") or "")
    old_value = clean_value(event.get("old_value"))
    new_value = clean_value(event.get("new_value"))
    terms: list[str] = []

    if event_type in {"brand_rejection", "feature_rejection", "product_rejection"}:
        terms.extend([old_value, strip_exclude(new_value)])
    elif event_type == "preference_revision":
        terms.append(old_value)
    elif event_type == "preference_clear":
        terms.append(old_value)
    return unique_terms(terms)


def active_required_terms(event: dict[str, Any]) -> list[str]:
    event_type = str(event.get("type") or "")
    new_value = clean_value(event.get("new_value"))
    if event_type == "preference_revision" and new_value and not new_value.lower().startswith("exclude:"):
        return unique_terms([new_value])
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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    compliant = sum(int(r["compliant"]) for r in rows)
    stale = sum(int(r["stale_violation"]) for r in rows)
    by_type: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["event_type"])
        item = by_type.setdefault(key, {"n": 0, "compliant": 0, "stale_violation": 0})
        item["n"] += 1
        item["compliant"] += int(row["compliant"])
        item["stale_violation"] += int(row["stale_violation"])
    for item in by_type.values():
        item["prcr"] = item["compliant"] / item["n"] if item["n"] else None
        item["spvr"] = item["stale_violation"] / item["n"] if item["n"] else None
    return {
        "eligible_post_revision_recommendation_events": total,
        "compliant_events": compliant,
        "stale_preference_violation_events": stale,
        "PRCR": compliant / total if total else None,
        "SPVR": stale / total if total else None,
        "by_event_type": by_type,
        "notes": (
            "Rule-based offline scorer. It checks explicit rejection/revision terms "
            "against displayed product metadata/text. Ambiguous semantic cases should "
            "be manually audited before paper reporting."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_json_list(raw: str | None) -> list[dict[str, Any]]:
    if not raw or raw == "[]":
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def split_ids(raw: str | None) -> list[str]:
    return [part for part in str(raw or "").split() if part]


def clean_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(v) for v in value if v).strip()
    return str(value).strip()


def strip_exclude(value: str) -> str:
    return re.sub(r"^exclude\\s*:\\s*", "", value or "", flags=re.I).strip()


def normalize_for_match(text: str) -> str:
    return re.sub(r"\\s+", " ", str(text or "").lower())


def unique_terms(terms: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        term = normalize_for_match(strip_exclude(clean_value(term))).strip()
        if not term or term in {"none", "null"}:
            continue
        if term not in seen:
            seen.add(term)
            out.append(term)
    return out


def is_recommendation(row: dict[str, Any]) -> bool:
    return str(row.get("recommended") or "").lower() in {"1", "1.0", "true", "yes"} and bool(split_ids(row.get("recommended_ids")))


if __name__ == "__main__":
    main()
