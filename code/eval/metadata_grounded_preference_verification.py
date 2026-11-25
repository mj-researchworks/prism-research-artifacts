"""
Metadata-grounded preference verification for PRISM PRCR/SPVR outputs.

This is a post-hoc evaluation script. It does not modify or run the PRISM
recommendation pipeline. Given an event-level PRCR/SPVR table and product
metadata, it checks whether displayed products are supported, contradicted, or
unverifiable with respect to the active preference text.

Typical Colab usage:

python -m eval.metadata_grounded_preference_verification \
  --events /content/drive/MyDrive/prism/eval/results/prcr_spvr_semantic_4fold_corrected/advisor_clarified_outputs/advisor_event_level_prcr_spvr_with_missed_revisions_FINAL.csv \
  --product-metadata /content/drive/MyDrive/prism/data/product_base_table.json \
  --retrieval-text /content/drive/MyDrive/prism/data/product_base_retrieval/product_base_retrieval_text.json \
  --out-dir /content/drive/MyDrive/prism/eval/results/metadata_grounded_preference_verification
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EVENTS = (
    "eval/results/prcr_spvr_semantic_4fold_corrected/advisor_clarified_outputs/"
    "advisor_event_level_prcr_spvr_with_missed_revisions_FINAL.csv"
)
DEFAULT_PRODUCT_METADATA = "data/product_base_table.json"
DEFAULT_RETRIEVAL_TEXT = "data/product_base_retrieval/product_base_retrieval_text.json"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "but",
    "by",
    "do",
    "does",
    "for",
    "from",
    "have",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "one",
    "show",
    "some",
    "that",
    "the",
    "this",
    "to",
    "too",
    "want",
    "wants",
    "with",
    "without",
}

NEGATION_PATTERNS = [
    r"\bdo(?:es)?\s+not\s+want\b",
    r"\bdon'?t\s+want\b",
    r"\bno\s+longer\s+want\b",
    r"\bnot\s+this\b",
    r"\bnot\s+these\b",
    r"\bexclude\b",
    r"\bavoid\b",
    r"\bwithout\b",
]

SUBJECTIVE_TERMS = {
    "beautiful",
    "cool",
    "cute",
    "elegant",
    "exquisite",
    "feminine",
    "good",
    "nice",
    "pretty",
    "stylish",
    "ugly",
}


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", norm(text))
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}


def split_product_ids(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return re.findall(r"[A-Z0-9]{8,12}|\b\d{9,13}\b", str(value))


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_product_metadata(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    if isinstance(data, list):
        out = {}
        for item in data:
            if isinstance(item, dict) and item.get("product_id"):
                out[str(item["product_id"])] = item
        return out
    raise ValueError(f"Unsupported metadata format: {path}")


def load_retrieval_text(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    data = load_json(path)
    out = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("product_id"):
                out[str(item["product_id"])] = str(item.get("text") or "")
    elif isinstance(data, dict):
        for pid, value in data.items():
            if isinstance(value, dict):
                out[str(pid)] = str(value.get("text") or "")
            else:
                out[str(pid)] = str(value or "")
    return out


def flatten_product_text(product: dict[str, Any] | None, retrieval_text: str = "") -> str:
    if not product:
        return retrieval_text
    parts = [
        product.get("product_id"),
        product.get("title"),
        product.get("brand"),
        product.get("color"),
        retrieval_text,
    ]
    attrs = product.get("attributes")
    if isinstance(attrs, dict):
        for key, value in attrs.items():
            parts.append(key)
            parts.append(value)
    return " | ".join(str(p) for p in parts if p)


def is_negated_preference(text: str) -> bool:
    lowered = norm(text)
    return any(re.search(pattern, lowered) for pattern in NEGATION_PATTERNS)


def has_price_constraint(text: str) -> bool:
    lowered = norm(text)
    return bool(re.search(r"(\$|under|below|less than|cheaper|expensive|affordable)", lowered))


def has_dimension_constraint(text: str) -> bool:
    lowered = norm(text)
    return bool(re.search(r"\b(size|height|wide|width|tall|short|long|inch|inches|cm|mm)\b", lowered))


def metadata_has_field(product_text: str, field_words: set[str]) -> bool:
    text = norm(product_text)
    return any(word in text for word in field_words)


def overlap_score(pref_text: str, product_text: str) -> tuple[float, list[str]]:
    pref_tokens = tokenize(pref_text)
    product_tokens = tokenize(product_text)
    if not pref_tokens:
        return 0.0, []
    hits = sorted(pref_tokens & product_tokens)
    return len(hits) / len(pref_tokens), hits


def verify_product(
    product_id: str,
    product: dict[str, Any] | None,
    retrieval_text: str,
    row: pd.Series,
) -> dict[str, Any]:
    product_text = flatten_product_text(product, retrieval_text)
    active_pref = norm(row.get("New Active Preference"))
    old_pref = norm(row.get("Old Preference"))
    event_type = norm(row.get("Event Type"))
    user_message = norm(row.get("User Message"))

    active_score, active_hits = overlap_score(active_pref, product_text)
    old_score, old_hits = overlap_score(old_pref, product_text)

    reasons: list[str] = []
    label = "unverifiable"

    if not product and not retrieval_text:
        return {
            "product_id": product_id,
            "metadata_label": "unverifiable",
            "metadata_reason": "product metadata not found",
            "active_overlap": 0.0,
            "old_overlap": 0.0,
            "active_hits": "",
            "old_hits": "",
        }

    active_is_negative = is_negated_preference(active_pref) or is_negated_preference(user_message)
    # A replacement can legitimately preserve overlap with the old broad class
    # ("phone accessory" -> "phone protective case"). Treat old-preference
    # matches as stale only for explicit rejection/clear/negative cases.
    stale_check_applicable = (
        "rejection" in event_type
        or "clear" in event_type
        or active_is_negative
        or any(p in user_message for p in ("don't want", "do not want", "not this", "not these", "wrong thing"))
    )

    if active_is_negative and active_score > 0:
        label = "contradicted"
        reasons.append("displayed product matches explicitly negated active preference")

    if stale_check_applicable and old_pref and old_score >= 0.5:
        if old_pref != active_pref:
            label = "contradicted"
            reasons.append("displayed product strongly matches superseded old preference")

    if label != "contradicted":
        if active_pref and active_score >= 0.5:
            label = "supported"
            reasons.append("metadata/title overlaps active preference")
        elif active_pref and active_score > 0 and len(active_hits) >= 2:
            label = "supported"
            reasons.append("metadata/title contains multiple active preference terms")

    if label == "unverifiable":
        if has_price_constraint(active_pref) and not metadata_has_field(product_text, {"price", "$"}):
            reasons.append("price constraint cannot be verified from available metadata")
        elif has_dimension_constraint(active_pref) and not metadata_has_field(
            product_text, {"dimension", "dimensions", "inch", "inches", "height", "size"}
        ):
            reasons.append("dimension/size constraint cannot be verified from available metadata")
        elif tokenize(active_pref) & SUBJECTIVE_TERMS:
            reasons.append("subjective preference cannot be verified objectively from metadata")
        else:
            reasons.append("available metadata is insufficient for this preference")

    return {
        "product_id": product_id,
        "metadata_label": label,
        "metadata_reason": "; ".join(dict.fromkeys(reasons)),
        "active_overlap": round(active_score, 4),
        "old_overlap": round(old_score, 4),
        "active_hits": " ".join(active_hits),
        "old_hits": " ".join(old_hits),
    }


def event_label(product_labels: list[str], recommendation_found: bool) -> str:
    if not recommendation_found or not product_labels:
        return "no_followup"
    if "contradicted" in product_labels:
        return "contradicted"
    if product_labels and all(label == "supported" for label in product_labels):
        return "supported"
    return "unverifiable"


def rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def int_flag(value: Any) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def build_targeted_adjudication(event_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize metadata verification only where it can adjudicate PRCR/SPVR."""
    observed = event_df[event_df["Metadata Verification Label"] != "no_followup"].copy()
    observed["Original Unknown"] = observed["Unknown/Unverifiable (0/1)"].map(int_flag)
    observed["Original Stale Violation"] = observed["Stale Violation (0/1)"].map(int_flag)
    observed["Original Compliant"] = observed["Compliant (0/1)"].map(int_flag)
    observed["Original Noncompliant"] = 1 - observed["Original Compliant"]

    targeted_mask = (
        (observed["Original Unknown"] == 1)
        | (observed["Original Stale Violation"] == 1)
        | (observed["Original Noncompliant"] == 1)
    )
    targeted = observed[targeted_mask].copy()

    def adjudication(row: pd.Series) -> str:
        metadata_label = row["Metadata Verification Label"]
        if metadata_label == "contradicted":
            if int_flag(row["Original Stale Violation"]) == 1:
                return "metadata_confirms_stale_or_constraint_violation"
            if int_flag(row["Original Unknown"]) == 1:
                return "metadata_resolves_unknown_as_contradiction"
            return "metadata_confirms_noncompliance"
        if metadata_label == "supported":
            if int_flag(row["Original Unknown"]) == 1:
                return "metadata_resolves_unknown_as_supported"
            if int_flag(row["Original Stale Violation"]) == 1:
                return "metadata_does_not_confirm_stale_violation"
            return "metadata_supports_active_preference"
        if int_flag(row["Original Unknown"]) == 1:
            return "still_unverifiable_after_metadata"
        if int_flag(row["Original Stale Violation"]) == 1:
            return "metadata_does_not_confirm_stale_violation"
        return "metadata_unverifiable_for_noncompliance"

    if not targeted.empty:
        targeted["Metadata Adjudication"] = targeted.apply(adjudication, axis=1)
        targeted["Targeted Reason"] = targeted.apply(
            lambda row: "; ".join(
                reason
                for reason, active in [
                    ("original_unknown", int_flag(row["Original Unknown"]) == 1),
                    ("original_stale_violation", int_flag(row["Original Stale Violation"]) == 1),
                    ("original_noncompliant", int_flag(row["Original Noncompliant"]) == 1),
                ]
                if active
            ),
            axis=1,
        )
    else:
        targeted["Metadata Adjudication"] = []
        targeted["Targeted Reason"] = []

    def count_where(df: pd.DataFrame, condition) -> int:
        if df.empty:
            return 0
        return int(condition(df).sum())

    original_unknown = observed[observed["Original Unknown"] == 1]
    original_stale = observed[observed["Original Stale Violation"] == 1]
    original_noncompliant = observed[observed["Original Noncompliant"] == 1]

    rows = [
        {
            "Metric": "Observed post-revision recommendation events",
            "Count": len(observed),
            "Rate": "",
            "Interpretation": "Same denominator used by observed PRCR/SPVR.",
        },
        {
            "Metric": "Original unknown/unverifiable events",
            "Count": len(original_unknown),
            "Rate": rate(len(original_unknown), len(observed)),
            "Interpretation": "Events where title-level PRCR/SPVR evidence was insufficient.",
        },
        {
            "Metric": "Unknown resolved as supported by metadata",
            "Count": count_where(original_unknown, lambda df: df["Metadata Verification Label"] == "supported"),
            "Rate": rate(count_where(original_unknown, lambda df: df["Metadata Verification Label"] == "supported"), len(original_unknown)),
            "Interpretation": "Metadata gives positive evidence for the active preference.",
        },
        {
            "Metric": "Unknown resolved as contradiction by metadata",
            "Count": count_where(original_unknown, lambda df: df["Metadata Verification Label"] == "contradicted"),
            "Rate": rate(count_where(original_unknown, lambda df: df["Metadata Verification Label"] == "contradicted"), len(original_unknown)),
            "Interpretation": "Metadata converts an unknown case into a visible conflict.",
        },
        {
            "Metric": "Unknown still unverifiable after metadata",
            "Count": count_where(original_unknown, lambda df: df["Metadata Verification Label"] == "unverifiable"),
            "Rate": rate(count_where(original_unknown, lambda df: df["Metadata Verification Label"] == "unverifiable"), len(original_unknown)),
            "Interpretation": "Even metadata lacks enough evidence; not counted as confirmed failure.",
        },
        {
            "Metric": "Original stale-violation events",
            "Count": len(original_stale),
            "Rate": rate(len(original_stale), len(observed)),
            "Interpretation": "Events previously flagged as visible stale preference conflicts.",
        },
        {
            "Metric": "Stale violations confirmed by metadata",
            "Count": count_where(original_stale, lambda df: df["Metadata Verification Label"] == "contradicted"),
            "Rate": rate(count_where(original_stale, lambda df: df["Metadata Verification Label"] == "contradicted"), len(original_stale)),
            "Interpretation": "Metadata supports the stale/conflict label.",
        },
        {
            "Metric": "Stale violations not confirmed by metadata",
            "Count": count_where(original_stale, lambda df: df["Metadata Verification Label"] != "contradicted"),
            "Rate": rate(count_where(original_stale, lambda df: df["Metadata Verification Label"] != "contradicted"), len(original_stale)),
            "Interpretation": "Metadata does not prove the stale label; inspect manually if used as example.",
        },
        {
            "Metric": "Original noncompliant events",
            "Count": len(original_noncompliant),
            "Rate": rate(len(original_noncompliant), len(observed)),
            "Interpretation": "Events with Compliant=0 in the PRCR/SPVR table.",
        },
        {
            "Metric": "Noncompliance confirmed by metadata contradiction",
            "Count": count_where(original_noncompliant, lambda df: df["Metadata Verification Label"] == "contradicted"),
            "Rate": rate(count_where(original_noncompliant, lambda df: df["Metadata Verification Label"] == "contradicted"), len(original_noncompliant)),
            "Interpretation": "Metadata-visible conflict among noncompliant cases.",
        },
        {
            "Metric": "Clear metadata-visible contradictions overall",
            "Count": int((observed["Metadata Verification Label"] == "contradicted").sum()),
            "Rate": rate(int((observed["Metadata Verification Label"] == "contradicted").sum()), len(observed)),
            "Interpretation": "Conservative contradiction-only rate over all observed recommendation events.",
        },
    ]
    return targeted, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default=DEFAULT_EVENTS)
    parser.add_argument("--product-metadata", default=DEFAULT_PRODUCT_METADATA)
    parser.add_argument("--retrieval-text", default=DEFAULT_RETRIEVAL_TEXT)
    parser.add_argument("--out-dir", default="eval/results/metadata_grounded_preference_verification")
    args = parser.parse_args()

    events_path = Path(args.events)
    product_metadata_path = Path(args.product_metadata)
    retrieval_text_path = Path(args.retrieval_text) if args.retrieval_text else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(events_path)
    products = load_product_metadata(product_metadata_path)
    retrieval_text = load_retrieval_text(retrieval_text_path)

    product_rows = []
    event_rows = []

    for event_index, row in events.iterrows():
        product_ids = split_product_ids(row.get("Recommended Product IDs"))
        recommendation_found = bool(int(row.get("Recommendation Found") or 0)) if not pd.isna(row.get("Recommendation Found")) else bool(product_ids)
        labels = []

        for product_id in product_ids:
            result = verify_product(
                product_id=product_id,
                product=products.get(product_id),
                retrieval_text=retrieval_text.get(product_id, ""),
                row=row,
            )
            labels.append(result["metadata_label"])
            product_rows.append(
                {
                    "event_index": event_index,
                    "Fold": row.get("Fold"),
                    "Conversation ID": row.get("Conversation ID"),
                    "Revision Turn": row.get("Revision Turn"),
                    "Recommendation Turn": row.get("Recommendation Turn"),
                    "Event Type": row.get("Event Type"),
                    "Slot": row.get("Slot"),
                    "New Active Preference": row.get("New Active Preference"),
                    "Old Preference": row.get("Old Preference"),
                    "product_id": product_id,
                    "product_title": (products.get(product_id) or {}).get("title"),
                    **result,
                }
            )

        label = event_label(labels, recommendation_found)
        event_rows.append(
            {
                **row.to_dict(),
                "Metadata Verification Label": label,
                "Metadata Supported Products": labels.count("supported"),
                "Metadata Contradicted Products": labels.count("contradicted"),
                "Metadata Unverifiable Products": labels.count("unverifiable"),
                "Metadata Product Count": len(product_ids),
            }
        )

    product_df = pd.DataFrame(product_rows)
    event_df = pd.DataFrame(event_rows)

    observed = event_df[event_df["Metadata Verification Label"] != "no_followup"]
    overall_counts = Counter(observed["Metadata Verification Label"])
    summary_rows = [
        {
            "Scope": "pooled_observed_recommendations",
            "Events": len(observed),
            "Supported": overall_counts.get("supported", 0),
            "Contradicted": overall_counts.get("contradicted", 0),
            "Unverifiable": overall_counts.get("unverifiable", 0),
            "Supported Rate": rate(overall_counts.get("supported", 0), len(observed)),
            "Contradicted Rate": rate(overall_counts.get("contradicted", 0), len(observed)),
            "Unverifiable Rate": rate(overall_counts.get("unverifiable", 0), len(observed)),
        }
    ]

    fold_rows = []
    for fold, fold_df in event_df.groupby("Fold", dropna=False):
        fold_observed = fold_df[fold_df["Metadata Verification Label"] != "no_followup"]
        counts = Counter(fold_observed["Metadata Verification Label"])
        fold_rows.append(
            {
                "Fold": fold,
                "Explicit Revision Events": len(fold_df),
                "Observed Recommendation Events": len(fold_observed),
                "No Follow-up Events": int((fold_df["Metadata Verification Label"] == "no_followup").sum()),
                "Supported": counts.get("supported", 0),
                "Contradicted": counts.get("contradicted", 0),
                "Unverifiable": counts.get("unverifiable", 0),
                "Supported Rate": rate(counts.get("supported", 0), len(fold_observed)),
                "Contradicted Rate": rate(counts.get("contradicted", 0), len(fold_observed)),
                "Unverifiable Rate": rate(counts.get("unverifiable", 0), len(fold_observed)),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    if not fold_df.empty:
        summary_rows.extend(
            [
                {
                    "Scope": "fold_mean_sample_sd",
                    "Events": "",
                    "Supported": "",
                    "Contradicted": "",
                    "Unverifiable": "",
                    "Supported Rate": f"{fold_df['Supported Rate'].mean():.4f} ± {fold_df['Supported Rate'].std(ddof=1):.4f}",
                    "Contradicted Rate": f"{fold_df['Contradicted Rate'].mean():.4f} ± {fold_df['Contradicted Rate'].std(ddof=1):.4f}",
                    "Unverifiable Rate": f"{fold_df['Unverifiable Rate'].mean():.4f} ± {fold_df['Unverifiable Rate'].std(ddof=1):.4f}",
                }
            ]
        )
    summary_df = pd.DataFrame(summary_rows)
    targeted_df, targeted_summary_df = build_targeted_adjudication(event_df)

    event_out = out_dir / "metadata_grounded_event_verification.csv"
    product_out = out_dir / "metadata_grounded_product_evidence.csv"
    fold_out = out_dir / "metadata_grounded_fold_summary.csv"
    summary_out = out_dir / "metadata_grounded_main_summary.csv"
    targeted_out = out_dir / "metadata_targeted_adjudication_events.csv"
    targeted_summary_out = out_dir / "metadata_targeted_adjudication_summary.csv"
    json_out = out_dir / "metadata_grounded_summary.json"

    event_df.to_csv(event_out, index=False)
    product_df.to_csv(product_out, index=False)
    fold_df.to_csv(fold_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    targeted_df.to_csv(targeted_out, index=False)
    targeted_summary_df.to_csv(targeted_summary_out, index=False)

    summary_json = {
        "inputs": {
            "events": str(events_path),
            "product_metadata": str(product_metadata_path),
            "retrieval_text": str(retrieval_text_path) if retrieval_text_path else None,
        },
        "events_total": int(len(event_df)),
        "observed_recommendation_events": int(len(observed)),
        "pooled_counts": dict(overall_counts),
        "folds": fold_rows,
        "outputs": {
            "event_verification": str(event_out),
            "product_evidence": str(product_out),
            "fold_summary": str(fold_out),
            "main_summary": str(summary_out),
            "targeted_adjudication_events": str(targeted_out),
            "targeted_adjudication_summary": str(targeted_summary_out),
        },
    }
    json_out.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    print("Saved:")
    print(event_out)
    print(product_out)
    print(fold_out)
    print(summary_out)
    print(targeted_out)
    print(targeted_summary_out)
    print(json_out)
    print("\nMain summary:")
    print(summary_df.to_string(index=False))
    print("\nFold summary:")
    print(fold_df.to_string(index=False))
    print("\nTargeted adjudication summary:")
    print(targeted_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
