"""Agent-related services."""

from typing import Any

from .calendar_tools import (
    CancelCalendarEventTool,
    CreateCalendarEventTool,
    GetCalendarEventTool,
    ListCalendarEventsTool,
    UpdateCalendarEventTool,
    calendar_tools,
)
from .context import AgentRuntimeContext
from .providers import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    OpenRouterProvider,
    ToolCall,
)
from .results import MediaResult, StructuredToolResult, ToolResult
from .settings import HumorAPISettings, OpenRouterSettings
from .tools import AgentTool, AgentToolError, SendMemeTool

__all__ = [
    "Agent",
    "AgentRuntimeContext",
    "AgentTool",
    "AgentToolError",
    "CancelCalendarEventTool",
    "CreateCalendarEventTool",
    "GetCalendarEventTool",
    "HumorAPISettings",
    "LLMProvider",
    "LLMProviderError",
    "LLMResponse",
    "ListCalendarEventsTool",
    "MediaResult",
    "OpenRouterProvider",
    "OpenRouterSettings",
    "SendMemeTool",
    "StructuredToolResult",
    "ToolCall",
    "ToolResult",
    "UpdateCalendarEventTool",
    "calendar_tools",
]


def __getattr__(name: str) -> Any:
    """Load agent classes lazily so ``python -m agent.agent`` runs cleanly."""
    if name == "Agent":
        from .agent import Agent

        return Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
