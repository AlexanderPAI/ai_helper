"""LLM providers used by the application."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from .settings import OpenRouterSettings


class OpenRouterProvider:
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
        return self.client.chat.completions.create(
            model=self.settings.model,
            messages=list(messages),  # type: ignore[arg-type]
            **options,
        )

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> str:
        """Return only the model text expected by the agent."""
        completion = self.chat(messages, **options)
        return completion.choices[0].message.content or ""

    async def achat(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> ChatCompletion:
        """Asynchronously create a chat completion."""
        return await self.async_client.chat.completions.create(
            model=self.settings.model,
            messages=list(messages),  # type: ignore[arg-type]
            **options,
        )

    async def agenerate(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> str:
        """Asynchronously return only the model text expected by the agent."""
        completion = await self.achat(messages, **options)
        return completion.choices[0].message.content or ""

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
