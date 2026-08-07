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
    if name == "Agent":
        from .agent import Agent

        return Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
