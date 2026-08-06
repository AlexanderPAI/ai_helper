"""Tools available to the conversational agent."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

MEME_PLACEHOLDER = "Скоро здесь будет приходить мем"


class AgentTool(Protocol):
    """Contract for a tool that can be selected by the LLM."""

    name: str
    description: str

    @property
    def schema(self) -> Mapping[str, Any]: ...

    def invoke(self, arguments: Mapping[str, Any]) -> str: ...


class SendMemeTool:
    """Placeholder for sending a contextually appropriate meme to the chat."""

    name = "send_meme"
    description = (
        "Отправить в чат уместный мем, когда он хорошо подходит к контексту "
        "диалога. Не вызывай инструмент без повода и не вызывай его вместо "
        "обычного содержательного ответа на вопрос пользователя."
    )

    @property
    def schema(self) -> Mapping[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }

    def invoke(self, arguments: Mapping[str, Any]) -> str:
        """Return a temporary message until image delivery is implemented."""
        if arguments:
            raise ValueError("send_meme does not accept arguments")
        return MEME_PLACEHOLDER


DEFAULT_TOOLS: tuple[AgentTool, ...] = (SendMemeTool(),)
