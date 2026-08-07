"""Agent-related services."""

from typing import Any

from .providers import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    OpenRouterProvider,
    ToolCall,
)
from .results import MediaResult, ToolResult
from .settings import HumorAPISettings, OpenRouterSettings
from .tools import AgentTool, AgentToolError, SendMemeTool

__all__ = [
    "TELEGRAM_MESSAGE_LIMIT",
    "Agent",
    "AgentTool",
    "AgentToolError",
    "HumorAPISettings",
    "LLMProvider",
    "LLMProviderError",
    "LLMResponse",
    "MediaResult",
    "OpenRouterProvider",
    "OpenRouterSettings",
    "SendMemeTool",
    "ToolCall",
    "ToolResult",
]


def __getattr__(name: str) -> Any:
    """Load agent classes lazily so ``python -m agent.agent`` runs cleanly."""
    if name in {"Agent", "TELEGRAM_MESSAGE_LIMIT"}:
        from .agent import TELEGRAM_MESSAGE_LIMIT, Agent

        exports = {
            "Agent": Agent,
            "TELEGRAM_MESSAGE_LIMIT": TELEGRAM_MESSAGE_LIMIT,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
