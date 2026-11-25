"""Offline Active Preference State F1 evaluation for PRISM turn traces.

This script does not run PRISM, retrieval, or any LLM/API. It reads saved
turn-level CSV traces and evaluates whether the system's active preference
state matches a reference active-state annotation.

Recommended paper-grade use:
  1. Create a reference CSV from dialogue review with columns:
       conversation_id, turn_index, slot, value, polarity
     where polarity is one of active, preferred, excluded, hard, soft.
  2. Run this script with --reference-csv.

Debug/silver use:
  If --reference-csv is omitted, the script derives a silver reference from
  the saved preference_events column. This is useful for regression checking,
  but it is not an independent human label because it comes from PRISM's own
  reflection trace.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

Fact = tuple[str, str, str]  # polarity, slot, normalized value

LIST_KEYS = {"brands", "features", "products", "product_types"}
POLARITY_ALIASES = {
    "active": "active",
    "preferred": "preferred",
    "preference": "preferred",
    "exclude": "excluded",
    "excluded": "excluded",
    "rejected": "excluded",
    "hard": "hard",
    "constraint": "hard",
    "soft": "soft",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", nargs="+", required=True, help="Saved turn CSV path(s) or glob(s).")
    parser.add_argument("--reference-csv", default=None, help="Optional human/reference active-state CSV.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--include-empty-reference",
        action="store_true",
        help="Include rows with no reference facts as true negatives/empty-state checks.",
    )
    return parser.parse_args()


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    out: list[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def safe_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def norm_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^exclude\s*:\s*", "", text)
    text = re.sub(r"[^a-z0-9$%.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def singular_slot(slot: str) -> str:
    slot = norm_text(slot).replace(" ", "_")
    aliases = {
        "brands": "brand",
        "brand": "brand",
        "features": "feature",
        "feature": "feature",
        "products": "product",
        "product": "product",
        "product_types": "product_type",
        "product_type": "product_type",
        "budget_max": "budget_max",
        "price_max": "budget_max",
        "color": "color",
        "material": "material",
        "use_case": "use_case",
        "recipient": "recipient",
        "category": "category",
    }
    return aliases.get(slot, slot)


def add_fact(facts: set[Fact], polarity: str, slot: str, value: Any) -> None:
    polarity = POLARITY_ALIASES.get(norm_text(polarity), norm_text(polarity))
    slot = singular_slot(slot)
    if isinstance(value, list):
        for item in value:
            add_fact(facts, polarity, slot, item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            add_fact(facts, polarity, key, item)
        return
    normalized = norm_text(value)
    if normalized:
        facts.add((polarity, slot, normalized))


def facts_from_mapping(mapping: dict[str, Any], polarity: str) -> set[Fact]:
    facts: set[Fact] = set()
    for key, value in (mapping or {}).items():
        if key in LIST_KEYS and isinstance(value, list):
            slot = singular_slot(key)
            for item in value:
                add_fact(facts, polarity, slot, item)
        elif isinstance(value, list):
            for item in value:
                add_fact(facts, polarity, key, item)
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                add_fact(facts, polarity, subkey, subvalue)
        else:
            add_fact(facts, polarity, key, value)
    return facts


def predicted_facts(row: dict[str, str]) -> set[Fact]:
    facts: set[Fact] = set()
    hard = safe_json(row.get("hard_constraints"), {})
    soft = safe_json(row.get("soft_preferences"), {})
    preferred = safe_json(row.get("preferred"), {})
    exclusions = safe_json(row.get("exclusions"), {})

    facts |= facts_from_mapping(hard, "hard")
    facts |= facts_from_mapping(soft, "soft")
    facts |= facts_from_mapping(preferred, "preferred")
    facts |= facts_from_mapping(exclusions, "excluded")

    category = row.get("category")
    if category:
        add_fact(facts, "active", "category", category)
    query_text = row.get("query_text")
    # Query text is intentionally not treated as a fact; it is a retrieval artifact.
    return facts


def silver_reference_from_events(row: dict[str, str]) -> set[Fact]:
    facts: set[Fact] = set()
    events = safe_json(row.get("preference_events"), [])
    if not isinstance(events, list):
        return facts
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = norm_text(event.get("type"))
        slot = event.get("slot") or "preference"
        new_value = event.get("new_value")
        old_value = event.get("old_value")
        if event_type in {"brand_rejection", "feature_rejection", "product_rejection"}:
            value = new_value or old_value
            add_fact(facts, "excluded", slot, value)
        elif event_type in {"preference_clear", "clear", "preference_removal"}:
            # Clearing/removing a preference means the old value should not be active.
            continue
        elif event_type in {"preference_revision", "preference_acceptance", "new_preference"}:
            add_fact(facts, "active", slot, new_value)
        elif new_value:
            add_fact(facts, "active", slot, new_value)
    return facts


def row_key(row: dict[str, str]) -> tuple[str, str]:
    cid = str(row.get("conversation_id") or row.get("Conversation ID") or row.get("conv_id") or "").strip()
    turn = str(row.get("turn_index") or row.get("Turn") or row.get("turn") or "").strip()
    return cid, turn


def load_reference(path: Path) -> dict[tuple[str, str], set[Fact]]:
    refs: dict[tuple[str, str], set[Fact]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"conversation_id", "turn_index", "slot", "value"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Reference CSV missing required columns: {sorted(missing)}")
        for row in reader:
            cid = str(row.get("conversation_id") or "").strip()
            turn = str(row.get("turn_index") or "").strip()
            polarity = row.get("polarity") or row.get("status") or "active"
            add_fact(refs[(cid, turn)], polarity, row.get("slot") or "preference", row.get("value"))
    return refs


def prf(predicted: set[Fact], reference: set[Fact]) -> dict[str, float | int]:
    tp = len(predicted & reference)
    fp = len(predicted - reference)
    fn = len(reference - predicted)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not reference else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def fmt(value: float | int | str | None) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else str(value)


def main() -> None:
    args = parse_args()
    turns_paths = expand_paths(args.turns)
    rows = load_rows(turns_paths)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_by_key = load_reference(Path(args.reference_csv)) if args.reference_csv else None
    label_source = "manual_reference_csv" if reference_by_key is not None else "silver_from_preference_events"

    scored_rows: list[dict[str, Any]] = []
    all_tp = all_fp = all_fn = 0
    by_slot: dict[str, Counter] = defaultdict(Counter)
    f1_values: list[float] = []

    for row in rows:
        key = row_key(row)
        pred = predicted_facts(row)
        ref = reference_by_key.get(key, set()) if reference_by_key is not None else silver_reference_from_events(row)
        if not ref and not args.include_empty_reference:
            continue
        scores = prf(pred, ref)
        all_tp += int(scores["tp"])
        all_fp += int(scores["fp"])
        all_fn += int(scores["fn"])
        f1_values.append(float(scores["f1"]))

        for fact in pred & ref:
            by_slot[fact[1]]["tp"] += 1
        for fact in pred - ref:
            by_slot[fact[1]]["fp"] += 1
        for fact in ref - pred:
            by_slot[fact[1]]["fn"] += 1

        scored_rows.append({
            "conversation_id": key[0],
            "turn_index": key[1],
            "label_source": label_source,
            "predicted_fact_count": len(pred),
            "reference_fact_count": len(ref),
            "tp": scores["tp"],
            "fp": scores["fp"],
            "fn": scores["fn"],
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "predicted_facts": json.dumps(sorted(pred), ensure_ascii=False),
            "reference_facts": json.dumps(sorted(ref), ensure_ascii=False),
            "missing_reference_facts": json.dumps(sorted(ref - pred), ensure_ascii=False),
            "extra_predicted_facts": json.dumps(sorted(pred - ref), ensure_ascii=False),
            "source_file": row.get("_source_file", ""),
        })

    micro_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) else 0.0
    micro_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0
    macro_f1 = mean(f1_values) if f1_values else 0.0
    macro_f1_sample_sd = stdev(f1_values) if len(f1_values) > 1 else 0.0

    summary = {
        "label_source": label_source,
        "turn_files": [str(path) for path in turns_paths],
        "input_rows": len(rows),
        "scored_rows": len(scored_rows),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_turn_f1": macro_f1,
        "macro_turn_f1_sample_sd": macro_f1_sample_sd,
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn,
        "note": "Manual reference CSV is required for paper-grade State F1. Silver mode is a trace consistency audit only.",
    }

    detail_path = out_dir / "active_preference_state_f1_rows.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "conversation_id", "turn_index", "label_source", "predicted_fact_count", "reference_fact_count",
            "tp", "fp", "fn", "precision", "recall", "f1", "predicted_facts", "reference_facts",
            "missing_reference_facts", "extra_predicted_facts", "source_file",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_rows)

    slot_rows = []
    for slot, counts in sorted(by_slot.items()):
        slot_scores = prf(
            {("x", slot, f"tp{i}") for i in range(counts["tp"])} | {("x", slot, f"fp{i}") for i in range(counts["fp"])},
            {("x", slot, f"tp{i}") for i in range(counts["tp"])} | {("x", slot, f"fn{i}") for i in range(counts["fn"])},
        )
        slot_rows.append({
            "slot": slot,
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "precision": slot_scores["precision"],
            "recall": slot_scores["recall"],
            "f1": slot_scores["f1"],
        })

    slot_path = out_dir / "active_preference_state_f1_by_slot.csv"
    with slot_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slot", "tp", "fp", "fn", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(slot_rows)

    summary_path = out_dir / "active_preference_state_f1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    main_table_path = out_dir / "active_preference_state_f1_main_table.csv"
    with main_table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Metric", "Value"])
        writer.writeheader()
        for key in [
            "label_source", "input_rows", "scored_rows", "micro_precision", "micro_recall",
            "micro_f1", "macro_turn_f1", "macro_turn_f1_sample_sd", "tp", "fp", "fn", "note",
        ]:
            writer.writerow({"Metric": key, "Value": fmt(summary[key])})

    print("Saved:")
    print(summary_path)
    print(main_table_path)
    print(detail_path)
    print(slot_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
