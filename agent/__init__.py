"""Agent-related services."""

from typing import Any

from .providers import LLMProvider, OpenRouterProvider
from .settings import OpenRouterSettings

__all__ = [
    "TELEGRAM_MESSAGE_LIMIT",
    "Agent",
    "LLMProvider",
    "OpenRouterProvider",
    "OpenRouterSettings",
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
