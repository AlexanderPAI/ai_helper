"""Agent-related services."""

from typing import Any

from .calendar_tools import (
    AddCalendarRemindersTool,
    CancelCalendarEventTool,
    CreateCalendarEventTool,
    GetCalendarEventTool,
    ListCalendarEventsTool,
    UpdateCalendarEventTool,
    calendar_tools,
)
from .context import AgentRuntimeContext
from .place_tools import SearchPlacesTool
from .providers import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    OpenRouterProvider,
    ToolCall,
    UrlCitation,
)
from .results import MediaResult, StructuredToolResult, ToolResult
from .settings import (
    HumorAPISettings,
    OpenRouterSettings,
    OpenRouterWebSearchSettings,
)
from .tools import AgentTool, AgentToolError, SendMemeTool

__all__ = [
    "AddCalendarRemindersTool",
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
    "OpenRouterWebSearchSettings",
    "SearchPlacesTool",
    "SendMemeTool",
    "StructuredToolResult",
    "ToolCall",
    "ToolResult",
    "UpdateCalendarEventTool",
    "UrlCitation",
    "calendar_tools",
]


def __getattr__(name: str) -> Any:
    """Load agent classes lazily so ``python -m agent.agent`` runs cleanly."""
    if name == "Agent":
        from .agent import Agent

        return Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
