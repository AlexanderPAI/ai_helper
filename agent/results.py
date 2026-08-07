"""Provider- and tool-independent results returned by the agent."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MediaResult:
    """A remote media resource ready for delivery by an interface adapter."""

    id: int | str
    url: str
    media_type: str


ToolResult = str | MediaResult


__all__ = ["MediaResult", "ToolResult"]
