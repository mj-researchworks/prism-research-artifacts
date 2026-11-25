"""Offline PSCon ground-truth protocol audit for PRISM outputs.

This script does not run PRISM or call any LLM/API. It reads saved turn-level
recommendation CSVs and the PSCon conversation file, then reports complementary
reference protocols:

A. Historical candidate-reference agreement: search_results U recommended_products.
B. Historical human-recommendation agreement: recommended_products only.
C. Observed user-rating agreement: explicit liked/disliked labels only.

Unjudged products are reported as unjudged, never treated as disliked.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PRODUCT_ID_RE = re.compile(r"\b[A-Z0-9]{8,12}\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", nargs="+", required=True, help="Turn CSV path(s) or glob(s).")
    parser.add_argument("--conversations", required=True, help="PSCon conversation_en.json path.")
    parser.add_argument("--category-map", default=None, help="Optional product_category_map.json for reference filtering.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    return parser.parse_args()


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))
    unique: list[Path] = []
    seen = set()
    for path in paths:
        resolved = str(path)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def product_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, dict):
                pid = item.get("product_id") or item.get("asin") or item.get("id")
                if pid:
                    out.append(str(pid))
            elif item:
                out.append(str(item))
        return dedupe(out)
    if isinstance(value, dict):
        pid = value.get("product_id") or value.get("asin") or value.get("id")
        return [str(pid)] if pid else []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        decoded = json.loads(text)
    except Exception:
        decoded = None
    if decoded is not None:
        return product_ids(decoded)
    return dedupe(PRODUCT_ID_RE.findall(text))


def dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for value in values:
        key = str(value).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def load_turn_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def build_protocol_references(conversations: list[dict[str, Any]], category_map: dict[str, Any] | None) -> dict[str, dict[str, set[str]]]:
    refs: dict[str, dict[str, set[str]]] = {}
    valid_ids = set(category_map) if category_map else None
    for conv in conversations:
        cid = str(conv.get("conv_id"))
        search_ids: set[str] = set()
        human_rec_ids: set[str] = set()
        liked_ids: set[str] = set()
        disliked_ids: set[str] = set()
        rated_ids: set[str] = set()

        for turn in conv.get("conversation", []):
            if turn.get("role") == "system" and turn.get("action") == "Recommend":
                for pid in product_ids(turn.get("search_results")):
                    if valid_ids is None or pid in valid_ids:
                        search_ids.add(pid)
                for pid in product_ids(turn.get("recommended_products")):
                    if valid_ids is None or pid in valid_ids:
                        human_rec_ids.add(pid)
            for item in turn.get("user_rating") or []:
                if not isinstance(item, dict):
                    continue
                pid = str(item.get("product_id") or "").strip()
                if not pid:
                    continue
                if valid_ids is not None and pid not in valid_ids:
                    continue
                label = str(item.get("product_rate") or "").lower()
                rated_ids.add(pid)
                if "(liked)" in label or label.endswith("liked"):
                    liked_ids.add(pid)
                if "(disliked)" in label or label.endswith("disliked"):
                    disliked_ids.add(pid)

        refs[cid] = {
            "historical_union": search_ids | human_rec_ids,
            "historical_search": search_ids,
            "historical_human_recommended": human_rec_ids,
            "liked": liked_ids,
            "disliked": disliked_ids,
            "rated": rated_ids,
        }
    return refs


def dcg(binary_relevance: list[int]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(binary_relevance))


def hit_at(ids: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    return 1.0 if any(pid in relevant for pid in ids[:k]) else 0.0


def ndcg_at(ids: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    actual = [1 if pid in relevant else 0 for pid in ids[:k]]
    ideal = [1] * min(len(relevant), k) + [0] * max(0, k - len(relevant))
    denom = dcg(ideal)
    return dcg(actual) / denom if denom else None


def mrr(ids: list[str], relevant: set[str]) -> float | None:
    if not relevant:
        return None
    for index, pid in enumerate(ids, start=1):
        if pid in relevant:
            return 1.0 / index
    return 0.0


def mean_defined(values: list[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    return mean(defined) if defined else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def row_conv_id(row: dict[str, str]) -> str:
    return str(row.get("conversation_id") or row.get("case_conversation_id") or row.get("conv_id") or "").strip()


def score_reference_protocol(rows: list[dict[str, str]], refs: dict[str, dict[str, set[str]]], ref_key: str, list_key: str) -> dict[str, Any]:
    hits3: list[float | None] = []
    hits5: list[float | None] = []
    hits10: list[float | None] = []
    ndcgs3: list[float | None] = []
    ndcgs10: list[float | None] = []
    mrrs: list[float | None] = []
    eligible = 0
    with_recs = 0
    for row in rows:
        cid = row_conv_id(row)
        relevant = refs.get(cid, {}).get(ref_key, set())
        ranked = product_ids(row.get(list_key))
        if ranked:
            with_recs += 1
        if not relevant:
            continue
        eligible += 1
        hits3.append(hit_at(ranked, relevant, 3))
        hits5.append(hit_at(ranked, relevant, 5))
        hits10.append(hit_at(ranked, relevant, 10))
        ndcgs3.append(ndcg_at(ranked, relevant, 3))
        ndcgs10.append(ndcg_at(ranked, relevant, 10))
        mrrs.append(mrr(ranked, relevant))
    return {
        "reference": ref_key,
        "ranked_list": list_key,
        "rows": len(rows),
        "rows_with_recommendations": with_recs,
        "eligible_rows_with_reference": eligible,
        "hit@3": mean_defined(hits3),
        "hit@5": mean_defined(hits5),
        "hit@10": mean_defined(hits10),
        "ndcg@3": mean_defined(ndcgs3),
        "ndcg@10": mean_defined(ndcgs10),
        "mrr": mean_defined(mrrs),
    }


def observed_rating_summary(rows: list[dict[str, str]], refs: dict[str, dict[str, set[str]]], list_key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total_displayed = liked_displayed = disliked_displayed = judged_displayed = unjudged_displayed = 0
    rows_with_recs = rows_with_any_judged = rows_with_like_ref = rows_with_dislike_ref = 0
    like_hit_rows = dislike_violation_rows = 0
    audit_rows: list[dict[str, Any]] = []

    for row in rows:
        cid = row_conv_id(row)
        ranked = product_ids(row.get(list_key))
        labels = refs.get(cid, {})
        liked = labels.get("liked", set())
        disliked = labels.get("disliked", set())
        rated = labels.get("rated", set())
        if ranked:
            rows_with_recs += 1
        if liked and ranked:
            rows_with_like_ref += 1
            if any(pid in liked for pid in ranked[:3]):
                like_hit_rows += 1
        if disliked and ranked:
            rows_with_dislike_ref += 1
            if any(pid in disliked for pid in ranked[:3]):
                dislike_violation_rows += 1
        judged_in_row = False
        for rank, pid in enumerate(ranked[:10], start=1):
            total_displayed += 1
            label = "unjudged"
            if pid in liked:
                liked_displayed += 1
                judged_displayed += 1
                judged_in_row = True
                label = "liked"
            elif pid in disliked:
                disliked_displayed += 1
                judged_displayed += 1
                judged_in_row = True
                label = "disliked"
            elif pid in rated:
                judged_displayed += 1
                judged_in_row = True
                label = "rated_other"
            else:
                unjudged_displayed += 1
            audit_rows.append({
                "source_file": row.get("_source_file", ""),
                "conversation_id": cid,
                "turn_index": row.get("turn_index", ""),
                "ranked_list": list_key,
                "rank": rank,
                "product_id": pid,
                "observed_rating_label": label,
            })
        if judged_in_row:
            rows_with_any_judged += 1

    judged_den = max(judged_displayed, 1)
    total_den = max(total_displayed, 1)
    return {
        "reference": "observed_user_rating",
        "ranked_list": list_key,
        "rows": len(rows),
        "rows_with_recommendations": rows_with_recs,
        "rows_with_any_judged_displayed_product": rows_with_any_judged,
        "rows_with_liked_reference_and_recs": rows_with_like_ref,
        "like_hit@3_over_rows_with_likes": like_hit_rows / rows_with_like_ref if rows_with_like_ref else None,
        "rows_with_disliked_reference_and_recs": rows_with_dislike_ref,
        "disliked_violation@3_over_rows_with_dislikes": dislike_violation_rows / rows_with_dislike_ref if rows_with_dislike_ref else None,
        "displayed_products@10": total_displayed,
        "judged_displayed_products@10": judged_displayed,
        "liked_displayed_products@10": liked_displayed,
        "disliked_displayed_products@10": disliked_displayed,
        "unjudged_displayed_products@10": unjudged_displayed,
        "judgment_coverage@10": judged_displayed / total_den,
        "liked_rate_among_judged@10": liked_displayed / judged_den,
        "disliked_rate_among_judged@10": disliked_displayed / judged_den,
        "unjudged_rate@10": unjudged_displayed / total_den,
    }, audit_rows


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in summaries for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: fmt(value) if isinstance(value, float) else value for key, value in row.items()})


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dataset_label_summary(refs: dict[str, dict[str, set[str]]]) -> dict[str, Any]:
    convs = list(refs.values())
    liked_ids = set().union(*(r["liked"] for r in convs)) if convs else set()
    disliked_ids = set().union(*(r["disliked"] for r in convs)) if convs else set()
    rated_ids = set().union(*(r["rated"] for r in convs)) if convs else set()
    return {
        "conversations": len(convs),
        "conversations_with_historical_union": sum(bool(r["historical_union"]) for r in convs),
        "conversations_with_human_recommendations": sum(bool(r["historical_human_recommended"]) for r in convs),
        "conversations_with_likes": sum(bool(r["liked"]) for r in convs),
        "conversations_with_dislikes": sum(bool(r["disliked"]) for r in convs),
        "unique_rated_ids": len(rated_ids),
        "unique_liked_ids": len(liked_ids),
        "unique_disliked_ids": len(disliked_ids),
        "liked_disliked_conflict_ids": len(liked_ids & disliked_ids),
    }


def main() -> None:
    args = parse_args()
    turn_paths = expand_paths(args.turns)
    missing = [str(path) for path in turn_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing turn CSV(s): {missing}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conversations = load_json(args.conversations)
    category_map = load_json(args.category_map) if args.category_map else None
    refs = build_protocol_references(conversations, category_map)
    rows = load_turn_rows(turn_paths)

    summaries: list[dict[str, Any]] = []
    for list_key in ("recommended_ids", "verified_candidate_ids"):
        for ref_key in ("historical_union", "historical_human_recommended", "liked"):
            summaries.append(score_reference_protocol(rows, refs, ref_key, list_key))
        rating_summary, rating_rows = observed_rating_summary(rows, refs, list_key)
        summaries.append(rating_summary)
        write_rows(out_dir / f"observed_rating_product_audit_{list_key}.csv", rating_rows)

    label_summary = dataset_label_summary(refs)
    write_summary_csv(out_dir / "protocol_summary.csv", summaries)
    (out_dir / "protocol_summary.json").write_text(json.dumps({"dataset_labels": label_summary, "protocols": summaries}, indent=2), encoding="utf-8")

    print("Saved:")
    print(out_dir / "protocol_summary.csv")
    print(out_dir / "protocol_summary.json")
    print(out_dir / "observed_rating_product_audit_recommended_ids.csv")
    print(out_dir / "observed_rating_product_audit_verified_candidate_ids.csv")
    print("\nDataset label summary:")
    print(json.dumps(label_summary, indent=2))
    print("\nProtocol summaries:")
    for row in summaries:
        compact = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
        print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
