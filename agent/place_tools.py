"""Tools for discovering physical places."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .context import AgentRuntimeContext
from .openrouter_tools import openrouter_web_search_tool
from .prompts import ToolPrompt, load_tool_prompt
from .providers import LLMProvider
from .settings import OpenRouterWebSearchSettings


class SearchPlacesTool:
    """Placeholder for a future OpenRouter-backed place search workflow.

    The provider is injected now so the eventual implementation can run a
    specialized, grounded LLM request without coupling the tool to a concrete
    OpenRouter client or creating a second provider instance.
    """

    name = "search_places"

    def __init__(
        self,
        search_provider: LLMProvider,
        web_search_settings: OpenRouterWebSearchSettings,
    ) -> None:
        self.search_provider = search_provider
        self.web_search_tool = openrouter_web_search_tool(web_search_settings)
        self.prompt: ToolPrompt = load_tool_prompt(self.name)

    @property
    def description(self) -> str:
        """Return the model-facing tool description loaded from YAML."""
        return self.prompt.description

    @property
    def schema(self) -> Mapping[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 500,
                            "description": self.prompt.parameter_descriptions["query"],
                        },
                        "location": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 200,
                            "description": self.prompt.parameter_descriptions[
                                "location"
                            ],
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 5,
                            "description": self.prompt.parameter_descriptions["limit"],
                        },
                    },
                    "required": ["query", "location"],
                    "additionalProperties": False,
                },
            },
        }

    def invoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> str:
        """Return a stable placeholder until place discovery is implemented."""
        return "Поиск мест пока не подключён."

    async def ainvoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> str:
        """Return a stable placeholder without performing network requests."""
        return self.invoke(arguments, context=context)


__all__ = ["SearchPlacesTool"]
