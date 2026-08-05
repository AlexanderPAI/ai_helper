"""LLM providers used by the application."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class OpenRouterProvider:
    """Service for accessing models through the OpenRouter API.

    Configuration can be passed explicitly or read from these environment
    variables: ``OPENROUTER_API_KEY``, ``OPENROUTER_MODEL``,
    ``OPENROUTER_SITE_URL`` and ``OPENROUTER_APP_NAME``.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL")

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. Pass api_key or set "
                "OPENROUTER_API_KEY."
            )
        if not self.model:
            raise ValueError(
                "OpenRouter model is required. Pass model or set OPENROUTER_MODEL."
            )

        headers = self._build_headers(
            site_url=site_url or os.getenv("OPENROUTER_SITE_URL"),
            app_name=app_name or os.getenv("OPENROUTER_APP_NAME"),
        )
        client_options = {
            "api_key": self.api_key,
            "base_url": self.BASE_URL,
            "default_headers": headers,
            "timeout": timeout,
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
            model=self.model,
            messages=list(messages),  # type: ignore[arg-type]
            **options,
        )

    async def achat(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> ChatCompletion:
        """Asynchronously create a chat completion."""
        return await self.async_client.chat.completions.create(
            model=self.model,
            messages=list(messages),  # type: ignore[arg-type]
            **options,
        )

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
