"""Tools for discovering physical places."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from zoneinfo import ZoneInfo

from .context import AgentRuntimeContext
from .openrouter_tools import openrouter_web_search_tool
from .prompts import ToolPrompt, load_tool_prompt
from .providers import LLMProvider, LLMResponse, UrlCitation
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

_SEARCH_SYSTEM_PROMPT = """
Ты выполняешь специализированный поиск физических мест для посещения через
доступный инструмент веб-поиска. Обязательно выполни веб-поиск и отвечай на
русском языке.

Ищи реальные действующие заведения в указанном городе. Для России отдавай
приоритет официальным сайтам и официальным страницам заведений, затем свежим
страницам надёжных каталогов и городских изданий. Не считай старую подборку
доказательством того, что заведение продолжает работать.

Верни не больше запрошенного числа вариантов. Для каждого укажи название, адрес
и прямую ссылку на официальный сайт или официальную страницу заведения, а также
почему место подходит, подтверждённый режим работы и ценовой ориентир, если они
найдены. Оформляй название, адрес и сайт явно, чтобы выбранный вариант можно было
добавить в календарь. Не придумывай отсутствующие данные и отмечай то, что не
удалось подтвердить. Не показывай внутренние поисковые запросы и технические
сведения.
""".strip()


class SearchPlacesTool:
    """Discover physical places through OpenRouter's server-side web search."""

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
        return self._format_response(response, is_food_service=is_food_service)

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
        return self._format_response(response, is_food_service=is_food_service)

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
    def _format_response(response: LLMResponse, *, is_food_service: bool) -> str:
        if not isinstance(response, LLMResponse):
            raise TypeError("place search provider must return an LLMResponse")
        content = response.content.strip()
        if not content:
            raise AgentToolError("place search returned an empty response")

        unseen = [
            citation for citation in response.citations if citation.url not in content
        ]
        if unseen:
            sources = "\n".join(
                f"- [{SearchPlacesTool._citation_title(citation)}]({citation.url})"
                for citation in unseen
            )
            content = f"{content}\n\n**Источники**\n\n{sources}"
        if is_food_service:
            content = (
                f"{content}\n\nХотите запланировать посещение одного из этих "
                "заведений в календаре?"
            )
        return content

    @staticmethod
    def _citation_title(citation: UrlCitation) -> str:
        return citation.title.replace("[", "").replace("]", "").strip() or citation.url

    @staticmethod
    def _location_question() -> str:
        return "В каком городе искать места?"


__all__ = ["SearchPlacesTool"]
