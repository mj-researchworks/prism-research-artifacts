"""Trace logger placeholder for later experiment/security analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_trace(state: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(state, default=str, indent=2), encoding="utf-8")

