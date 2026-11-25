"""Typed contracts for the autonomous ConServe control loop.

This module intentionally uses only the Python standard library so the first
autonomous skeleton stays easy to run and inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


PreferenceStatus = Literal[
    "new",
    "stable",
    "uncertain",
    "revised",
    "superseded",
    "conflicting",
]


@dataclass
class PreferenceItem:
    slot: str
    value: Any
    status: PreferenceStatus = "new"
    confidence: float = 1.0
    source_turn: int = 0
    previous_value: Any | None = None
    reason: str | None = None


@dataclass
class PlannerDecision:
    action: Literal["CALL_AGENT", "CALL_TOOL", "RESPOND", "STOP"]
    target: str | None = None
    reason: str = ""


@dataclass
class ComponentResult:
    name: str
    patch: dict[str, Any] = field(default_factory=dict)
    message: str | None = None
    status: Literal["success", "failed"] = "success"
    duration_ms: int | None = None


@dataclass
class SearchRequest:
    query_text: str
    category: str | None = None
    hard_constraints: dict[str, Any] = field(default_factory=dict)
    soft_preferences: dict[str, Any] = field(default_factory=dict)
    preferred: dict[str, Any] = field(default_factory=dict)
    exclusions: dict[str, Any] = field(default_factory=dict)
    control_flags: list[str] = field(default_factory=list)
    preference_flags: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    unclassified_flags: list[str] = field(default_factory=list)
    security_flags: list[str] = field(default_factory=list)
