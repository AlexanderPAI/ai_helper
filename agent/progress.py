"""Transport-independent progress events emitted by agent orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

AgentProgressKind = Literal[
    "planning",
    "plan_ready",
    "tool_started",
    "tool_completed",
    "finalizing",
]


@dataclass(frozen=True, slots=True)
class AgentProgressEvent:
    """A semantic lifecycle event without interface-specific presentation."""

    kind: AgentProgressKind
    tool_name: str | None = None
    step_number: int | None = None
    total_steps: int | None = None


AgentProgressCallback = Callable[[AgentProgressEvent], Awaitable[None]]


__all__ = ["AgentProgressCallback", "AgentProgressEvent", "AgentProgressKind"]
