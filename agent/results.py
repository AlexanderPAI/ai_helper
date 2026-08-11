"""Provider- and tool-independent results returned by the agent."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MediaResult:
    """A remote media resource ready for delivery by an interface adapter."""

    id: int | str
    url: str
    media_type: str


@dataclass(frozen=True, slots=True)
class StructuredToolResult:
    """Machine-readable tool result with deterministic user-facing Markdown."""

    kind: str
    markdown: str
    data: Mapping[str, Any]


ToolResult = str | MediaResult | StructuredToolResult


__all__ = ["MediaResult", "StructuredToolResult", "ToolResult"]
