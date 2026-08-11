"""LLM providers used by the application."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self

from openai import AsyncOpenAI, OpenAI, OpenAIError
from openai.types.chat import ChatCompletion

from .settings import OpenRouterSettings


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Provider-independent request to invoke an agent tool."""

    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class UrlCitation:
    """A web source attached to a provider response."""

    url: str
    title: str
    content: str | None = None


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Text and tool calls returned by an LLM provider."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    citations: tuple[UrlCitation, ...] = ()


class LLMProvider(Protocol):
    """Contract implemented by LLM providers used by the agent."""

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> LLMResponse: ...

    async def agenerate(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> LLMResponse: ...


class LLMProviderError(RuntimeError):
    """Provider-independent failure while communicating with an LLM service."""


class OpenRouterProvider(LLMProvider):
    """Service for accessing models through the OpenRouter API.

    Configuration can be passed explicitly or read from these environment
    variables: ``OPENROUTER_API_KEY``, ``OPENROUTER_MODEL``,
    ``OPENROUTER_SITE_URL`` and ``OPENROUTER_APP_NAME``.
    """

    def __init__(
        self,
        *,
        settings: OpenRouterSettings | None = None,
    ) -> None:
        self.settings = settings or OpenRouterSettings()  # type: ignore[call-arg]

        headers = self._build_headers(
            site_url=str(self.settings.site_url) if self.settings.site_url else None,
            app_name=self.settings.app_name,
        )
        client_options = {
            "api_key": self.settings.api_key.get_secret_value(),
            "base_url": str(self.settings.base_url),
            "default_headers": headers,
            "timeout": self.settings.timeout,
        }
        self.client = OpenAI(**client_options)
        self.async_client = AsyncOpenAI(**client_options)

    @staticmethod
    def _build_headers(*, site_url: str | None, app_name: str | None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
        return headers

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> ChatCompletion:
        """Create a chat completion using the configured model."""
        try:
            return self.client.chat.completions.create(
                model=self.settings.model,
                messages=list(messages),  # type: ignore[arg-type]
                **options,
            )
        except OpenAIError as error:
            raise LLMProviderError("LLM provider request failed") from error

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> LLMResponse:
        """Return a provider-independent model response."""
        return self._build_response(self.chat(messages, **options))

    async def achat(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> ChatCompletion:
        """Asynchronously create a chat completion."""
        try:
            return await self.async_client.chat.completions.create(
                model=self.settings.model,
                messages=list(messages),  # type: ignore[arg-type]
                **options,
            )
        except OpenAIError as error:
            raise LLMProviderError("LLM provider request failed") from error

    async def agenerate(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> LLMResponse:
        """Asynchronously return a provider-independent model response."""
        return self._build_response(await self.achat(messages, **options))

    @staticmethod
    def _build_response(completion: ChatCompletion) -> LLMResponse:
        message = completion.choices[0].message
        tool_calls = tuple(
            ToolCall(
                name=tool_call.function.name,
                arguments=OpenRouterProvider._load_tool_arguments(
                    tool_call.function.arguments
                ),
            )
            for tool_call in message.tool_calls or ()
        )
        citations = OpenRouterProvider._load_citations(
            getattr(message, "annotations", None)
        )
        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            citations=citations,
        )

    @staticmethod
    def _load_citations(annotations: Any) -> tuple[UrlCitation, ...]:
        citations: list[UrlCitation] = []
        for annotation in annotations or ():
            if getattr(annotation, "type", None) != "url_citation":
                continue
            citation = getattr(annotation, "url_citation", None)
            url = getattr(citation, "url", None)
            title = getattr(citation, "title", None)
            content = getattr(citation, "content", None)
            if not isinstance(url, str) or not isinstance(title, str):
                continue
            citations.append(
                UrlCitation(
                    url=url,
                    title=title,
                    content=content if isinstance(content, str) else None,
                )
            )
        return tuple(citations)

    @staticmethod
    def _load_tool_arguments(arguments: str) -> Mapping[str, Any]:
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise TypeError("tool arguments must be a JSON object")
        return parsed

    def close(self) -> None:
        """Close resources owned by the synchronous client."""
        self.client.close()

    async def aclose(self) -> None:
        """Close resources owned by the asynchronous client."""
        await self.async_client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
