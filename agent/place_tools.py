"""Tools for discovering physical places."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Mapping
from datetime import timedelta
from typing import Any
from zoneinfo import ZoneInfo

from calendar_app import CalendarError, CalendarEvent, CalendarService

from .context import AgentRuntimeContext
from .openrouter_tools import openrouter_web_search_tool
from .prompts import ToolPrompt, load_tool_prompt
from .providers import LLMProvider, LLMResponse
from .settings import OpenRouterWebSearchSettings
from .tools import AgentToolError

_AMBIGUOUS_LOCATIONS = {
    "здесь",
    "рядом",
    "рядом со мной",
    "поблизости",
    "nearby",
    "near me",
}

_RECENT_EVENT_WINDOW = timedelta(hours=24)
_VENUE_EVENT_MARKERS = (
    "бар",
    "кафе",
    "кальян",
    "кофейн",
    "паб",
    "пицц",
    "ресторан",
    "столов",
    "встреч",
    "завтрак",
    "обед",
    "ужин",
    "свидан",
    "посид",
    "праздн",
    "день рождения",
    "др ",
)

logger = logging.getLogger(__name__)

_SEARCH_SYSTEM_PROMPT = """
Ты выполняешь специализированный поиск физических мест для посещения через
доступный инструмент веб-поиска. Обязательно выполни веб-поиск и отвечай на
русском языке.

Ищи реальные действующие заведения в указанном городе. Для России проверяй
официальные страницы заведений, свежие страницы надёжных каталогов и городских
изданий. Не считай старую подборку доказательством того, что заведение продолжает
работать.

Верни не больше запрошенного числа вариантов. Для каждого укажи название, адрес,
почему место подходит, подтверждённый режим работы и ценовой ориентир, если они
найдены. Не показывай ссылки, URL, список источников, внутренние поисковые запросы
и технические сведения. Не придумывай отсутствующие данные и отмечай то, что не
удалось подтвердить.
""".strip()


class SearchPlacesTool:
    """Discover physical places through OpenRouter's server-side web search."""

    name = "search_places"

    def __init__(
        self,
        search_provider: LLMProvider,
        web_search_settings: OpenRouterWebSearchSettings,
        calendar_service: CalendarService | None = None,
    ) -> None:
        self.search_provider = search_provider
        self.web_search_tool = openrouter_web_search_tool(web_search_settings)
        self.calendar_service = calendar_service
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
                        "is_food_service": {
                            "type": "boolean",
                            "description": self.prompt.parameter_descriptions[
                                "is_food_service"
                            ],
                        },
                    },
                    "required": ["query", "location", "is_food_service"],
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
        """Search for places with a grounded provider request."""
        parsed = self._parse_arguments(arguments)
        if parsed is None:
            return self._location_question()
        query, location, limit, is_food_service = parsed
        response = self.search_provider.generate(
            self._messages(query, location, limit, context),
            tools=[self.web_search_tool],
        )
        recent_event = (
            asyncio.run(self._recent_venue_event(context)) if is_food_service else None
        )
        return self._format_response(
            response,
            is_food_service=is_food_service,
            recent_event=recent_event,
            context=context,
        )

    async def ainvoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> str:
        """Search for places without blocking the agent event loop."""
        parsed = self._parse_arguments(arguments)
        if parsed is None:
            return self._location_question()
        query, location, limit, is_food_service = parsed
        response = await self.search_provider.agenerate(
            self._messages(query, location, limit, context),
            tools=[self.web_search_tool],
        )
        recent_event = (
            await self._recent_venue_event(context) if is_food_service else None
        )
        return self._format_response(
            response,
            is_food_service=is_food_service,
            recent_event=recent_event,
            context=context,
        )

    @staticmethod
    def _parse_arguments(
        arguments: Mapping[str, Any],
    ) -> tuple[str, str, int, bool] | None:
        query = arguments.get("query")
        location = arguments.get("location")
        limit = arguments.get("limit", 5)
        is_food_service = arguments.get("is_food_service")
        if not isinstance(query, str) or not query.strip():
            raise AgentToolError("search_places query must be a non-empty string")
        if not isinstance(location, str) or not location.strip():
            return None
        if location.strip().casefold() in _AMBIGUOUS_LOCATIONS:
            return None
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 10
        ):
            raise AgentToolError("search_places limit must be an integer from 1 to 10")
        if not isinstance(is_food_service, bool):
            raise AgentToolError("search_places is_food_service must be a boolean")
        return query.strip(), location.strip(), limit, is_food_service

    @staticmethod
    def _messages(
        query: str,
        location: str,
        limit: int,
        context: AgentRuntimeContext | None,
    ) -> list[dict[str, str]]:
        current_time = "не указано"
        if context is not None:
            current_time = context.current_time.astimezone(
                ZoneInfo(context.timezone)
            ).isoformat()
        return [
            {"role": "system", "content": _SEARCH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Локация: {location}\n"
                    f"Что найти: {query}\n"
                    f"Максимум вариантов: {limit}\n"
                    f"Текущие локальные дата и время: {current_time}"
                ),
            },
        ]

    @staticmethod
    def _format_response(
        response: LLMResponse,
        *,
        is_food_service: bool,
        recent_event: CalendarEvent | None,
        context: AgentRuntimeContext | None,
    ) -> str:
        if not isinstance(response, LLMResponse):
            raise TypeError("place search provider must return an LLMResponse")
        content = response.content.strip()
        if not content:
            raise AgentToolError("place search returned an empty response")
        content = SearchPlacesTool._remove_links(content)
        if is_food_service:
            if recent_event is not None and context is not None:
                local_start = recent_event.starts_at.astimezone(
                    ZoneInfo(context.timezone)
                )
                event_label = local_start.strftime("%d.%m.%Y в %H:%M")
                content = (
                    f"{content}\n\nНедавно вы создали событие "
                    f"«{recent_event.title}» на {event_label}. Хотите добавить "
                    "одно из найденных мест в это событие?"
                )
            else:
                content = (
                    f"{content}\n\nХотите запланировать посещение одного из этих "
                    "заведений в календаре?"
                )
        return content

    async def _recent_venue_event(
        self,
        context: AgentRuntimeContext | None,
    ) -> CalendarEvent | None:
        if self.calendar_service is None or context is None:
            return None
        try:
            page = await self.calendar_service.list_event_page(
                context.chat_id,
                starts_from=context.current_time,
                limit=20,
            )
        except CalendarError:
            logger.warning("Could not inspect recent calendar events", exc_info=True)
            return None

        cutoff = context.current_time - _RECENT_EVENT_WINDOW
        candidates = [
            event
            for event in page.events
            if event.created_at >= cutoff and self._resembles_venue_visit(event)
        ]
        return max(candidates, key=lambda event: event.created_at, default=None)

    @staticmethod
    def _resembles_venue_visit(event: CalendarEvent) -> bool:
        description = event.description or ""
        if "Название:" in description and "Адрес:" in description:
            return False
        text = f"{event.title} {description}".casefold()
        return any(marker in text for marker in _VENUE_EVENT_MARKERS)

    @staticmethod
    def _remove_links(content: str) -> str:
        content = re.sub(
            r"(?ims)\n+#{0,3}\s*\**источники\**\s*:?\s*\n.*\Z",
            "",
            content,
        )
        content = re.sub(r"\[([^\]]+)]\(https?://[^)]+\)", r"\1", content)
        content = re.sub(r"https?://\S+", "", content)
        return re.sub(r"[ \t]+(?=\n|$)", "", content).strip()

    @staticmethod
    def _location_question() -> str:
        return "В каком городе искать места?"


__all__ = ["SearchPlacesTool"]
