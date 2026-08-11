"""Definitions for server-side tools provided by OpenRouter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .settings import OpenRouterWebSearchSettings


def openrouter_web_search_tool(
    settings: OpenRouterWebSearchSettings,
) -> Mapping[str, Any]:
    """Build a tool definition without performing a network operation."""
    return {
        "type": "openrouter:web_search",
        "parameters": {
            "engine": settings.engine,
            "max_results": settings.max_results,
            "max_total_results": settings.max_total_results,
            "max_characters": settings.max_characters,
        },
    }


__all__ = ["openrouter_web_search_tool"]
