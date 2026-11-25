#!/usr/bin/env python3
"""Export readable autonomous workflow trajectories from eval CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "autonomous_full_recommendation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns-csv", type=Path, default=None)
    parser.add_argument("--conversations-csv", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    turns_path = args.turns_csv or latest_csv(f"turns_{args.split}_*.csv")
    conversations_path = args.conversations_csv or latest_csv(f"conversations_{args.split}_*.csv")
    out_path = args.out or RESULTS_DIR / f"trajectory_examples_{args.split}.md"

    turns = read_csv(turns_path)
    conversations = read_csv(conversations_path)
    by_conversation: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in turns:
        by_conversation[row["conversation_id"]].append(row)

    selected = select_examples(conversations, args.limit)
    lines = [
        "# Autonomous Workflow Trajectory Examples",
        "",
        f"Turn rows: `{turns_path}`",
        f"Conversation rows: `{conversations_path}`",
        "",
    ]

    for index, conv in enumerate(selected, start=1):
        cid = conv["conversation_id"]
        rows = by_conversation.get(cid, [])
        lines.extend(render_conversation(index, conv, rows))

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} trajectory examples -> {out_path}")


def latest_csv(pattern: str) -> Path:
    paths = sorted(RESULTS_DIR.glob(pattern))
    if not paths:
        raise SystemExit(f"No CSV found for pattern: {RESULTS_DIR / pattern}")
    return paths[-1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_examples(conversations: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    successful = [row for row in conversations if as_float(row.get("hit@3")) > 0]
    failures = [row for row in conversations if as_float(row.get("hit@3")) == 0 and as_int(row.get("n_relevant")) > 0]
    no_gt = [row for row in conversations if as_int(row.get("n_relevant")) == 0]
    pool = successful[: max(limit - 3, 0)] + failures[:2] + no_gt[:1]
    if len(pool) < limit:
        seen = {row["conversation_id"] for row in pool}
        pool.extend(row for row in conversations if row["conversation_id"] not in seen)
    return pool[:limit]


def render_conversation(index: int, conv: dict[str, str], rows: list[dict[str, str]]) -> list[str]:
    lines = [
        f"## Example {index}: Conversation {conv['conversation_id']}",
        "",
        (
            f"- Final query: `{conv.get('final_query', '')}`; "
            f"category: `{conv.get('final_category', '')}`"
        ),
        (
            f"- Metrics: Hit@3={conv.get('hit@3', '')}, Hit@10={conv.get('hit@10', '')}, "
            f"NDCG@10={conv.get('ndcg@10', '')}, MRR={conv.get('mrr', '')}"
        ),
        f"- Displayed recommendation ids: `{conv.get('final_recommended_ids', '')}`",
        "",
    ]
    for row in rows:
        if not row.get("user_message"):
            continue
        lines.append(f"### Turn {row.get('turn_index')}: {row.get('user_message')}")
        lines.append(f"- Expected dataset action: `{row.get('expected_action', '')}`")
        lines.append(f"- System response: {compact(row.get('response', ''), 320)}")
        lines.append(f"- Query after turn: `{row.get('query_text', '')}`; category: `{row.get('category', '')}`")
        lines.append(f"- Critic: `{row.get('critic_decision', '')}` -> `{row.get('critic_next_action', '')}`")
        lines.append(f"- Components: `{', '.join(parse_json(row.get('components_called'), []))}`")
        trace = parse_json(row.get("planner_step_trace"), [])
        if trace:
            step_text = " -> ".join(
                f"{item.get('target') or item.get('action')}[{item.get('backend', '')}]"
                for item in trace
            )
            lines.append(f"- Planner path: `{step_text}`")
        if row.get("recommended_ids"):
            lines.append(f"- Recommended ids: `{row.get('recommended_ids')}`")
        flags = parse_json(row.get("preference_flags"), []) + parse_json(row.get("control_flags"), [])
        if flags:
            lines.append(f"- Flags: `{', '.join(flags)}`")
        lines.append("")
    return lines


def parse_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return fallback
    return parsed


def compact(text: str, limit: int) -> str:
    one_line = " ".join(str(text or "").split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3].rstrip() + "..."


def as_float(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def as_int(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
