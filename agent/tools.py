"""Tools available to the conversational agent."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Mapping
from typing import Any, Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .context import AgentRuntimeContext
from .prompts import ToolPrompt, load_tool_prompt
from .results import MediaResult, ToolResult
from .settings import HumorAPISettings

logger = logging.getLogger(__name__)


class AgentToolError(RuntimeError):
    """Raised when an agent tool cannot complete its operation."""


class EmptyMemeSearchError(AgentToolError):
    """Raised internally when a valid meme search has no results."""


class AgentTool(Protocol):
    """Contract for a tool that can be selected by the LLM."""

    name: str
    description: str

    @property
    def schema(self) -> Mapping[str, Any]: ...

    def invoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> ToolResult: ...

    async def ainvoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> ToolResult: ...


class SendMemeTool:
    """Fetch a contextually appropriate ready-made meme from Humor API."""

    name = "send_meme"

    def __init__(self, settings: HumorAPISettings) -> None:
        self.settings = settings
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
                        "mode": {
                            "type": "string",
                            "enum": ["random", "search"],
                            "description": self.prompt.parameter_descriptions["mode"],
                        },
                        "keywords": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 50,
                            "pattern": "^[A-Za-z][A-Za-z0-9_-]*$",
                            "description": self.prompt.parameter_descriptions[
                                "keywords"
                            ],
                        },
                    },
                    "required": ["mode"],
                    "additionalProperties": False,
                },
            },
        }

    def invoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> MediaResult:
        """Fetch exactly one meme using the mode selected by the agent."""
        mode, keywords = self._parse_arguments(arguments)
        payload = self._request(mode, keywords)
        try:
            return self._parse_response(payload, mode)
        except EmptyMemeSearchError:
            logger.info(
                "Humor API search returned no memes; falling back to random "
                "keywords=%r",
                keywords,
            )
            return self._parse_response(self._request("random", None), "random")

    def _request(
        self,
        mode: Literal["random", "search"],
        keywords: str | None,
    ) -> Any:
        url = self._request_url(mode)
        query: dict[str, str | int] = {}
        if keywords:
            query["keywords"] = keywords
        if mode == "search":
            query["number"] = 1

        request = Request(
            f"{url}?{urlencode(query)}" if query else url,
            headers={
                "Accept": "application/json",
                "User-Agent": self.settings.user_agent,
                "x-api-key": self.settings.api_key.get_secret_value(),
            },
        )
        try:
            with urlopen(request, timeout=self.settings.timeout) as response:
                payload = json.load(response)
        except HTTPError as error:
            raise AgentToolError(f"Humor API returned HTTP {error.code}") from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise AgentToolError("Humor API request failed") from error

        return payload

    async def ainvoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> MediaResult:
        """Fetch one meme without blocking the agent event loop."""
        return await asyncio.to_thread(self.invoke, arguments, context=context)

    def _request_url(self, mode: Literal["random", "search"]) -> str:
        if mode == "random":
            return str(self.settings.random_url)
        return str(self.settings.search_url)

    @staticmethod
    def _parse_arguments(
        arguments: Mapping[str, Any],
    ) -> tuple[Literal["random", "search"], str | None]:
        mode = arguments.get("mode")
        if mode not in {"random", "search"}:
            raise AgentToolError("send_meme mode must be random or search")

        raw_keywords = arguments.get("keywords")
        if raw_keywords is not None and not isinstance(raw_keywords, str):
            raise AgentToolError("send_meme keywords must be a string")
        keywords = raw_keywords.strip() if raw_keywords else None
        if mode == "search" and not keywords:
            raise AgentToolError("send_meme search mode requires keywords")
        if keywords:
            english_words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", keywords)
            if not english_words:
                logger.info("send_meme received no English keywords; using random mode")
                return "random", None
            keywords = english_words[0]
        return mode, keywords

    @staticmethod
    def _parse_response(payload: Any, mode: Literal["random", "search"]) -> MediaResult:
        if not isinstance(payload, dict):
            raise AgentToolError("Humor API returned an invalid response")

        meme: Any
        if mode == "search":
            memes = payload.get("memes")
            if not isinstance(memes, list) or not memes:
                raise EmptyMemeSearchError("Humor API did not find a matching meme")
            meme = memes[0]
        else:
            meme = payload

        if not isinstance(meme, dict):
            raise AgentToolError("Humor API returned an invalid meme")
        meme_id = meme.get("id")
        url = meme.get("url")
        media_type = meme.get("type")
        if (
            not isinstance(meme_id, int)
            or not isinstance(url, str)
            or not url.startswith(("https://", "http://"))
            or not isinstance(media_type, str)
            or not media_type.startswith("image/")
        ):
            raise AgentToolError("Humor API returned an invalid meme")
        return MediaResult(id=meme_id, url=url, media_type=media_type)
