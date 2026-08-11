"""Agent tools adapting structured model arguments to CalendarService use cases."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
from collections.abc import Awaitable, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, TypeVar
from urllib.parse import urlsplit
from uuid import UUID
from zoneinfo import ZoneInfo

from calendar_app import (
    UNSET,
    AddEventReminders,
    CalendarConflictError,
    CalendarError,
    CalendarEvent,
    CalendarEventCursor,
    CalendarNotFoundError,
    CalendarReminderStatus,
    CalendarService,
    CreateEvent,
    ReminderDraft,
    UpdateEvent,
)

from .context import AgentRuntimeContext
from .prompts import ToolPrompt, load_tool_prompt
from .results import StructuredToolResult
from .tools import AgentToolError

_T = TypeVar("_T")


class _CalendarTool:
    name: str

    def __init__(self, service: CalendarService) -> None:
        self.service = service
        self.prompt: ToolPrompt = load_tool_prompt(self.name)

    @property
    def description(self) -> str:
        return self.prompt.description

    def _parameter(self, name: str) -> str:
        return self.prompt.parameter_descriptions[name]

    @staticmethod
    def _context(context: AgentRuntimeContext | None) -> AgentRuntimeContext:
        if context is None:
            raise AgentToolError("calendar tools require trusted runtime context")
        return context

    @staticmethod
    def _sync(awaitable: Awaitable[StructuredToolResult]) -> StructuredToolResult:
        return asyncio.run(awaitable)

    async def _execute(
        self,
        operation: str,
        awaitable: Awaitable[_T],
        formatter: Any,
    ) -> StructuredToolResult:
        try:
            value = await awaitable
        except CalendarNotFoundError:
            return StructuredToolResult(
                kind=f"calendar_{operation}_not_found",
                markdown=(
                    "Событие не найдено в календаре этого чата. "
                    "Сначала запросите актуальный список событий."
                ),
                data={"operation": operation, "status": "not_found"},
            )
        except CalendarConflictError:
            return StructuredToolResult(
                kind=f"calendar_{operation}_conflict",
                markdown=(
                    "Событие уже изменилось. Получите его актуальную версию и "
                    "повторите действие."
                ),
                data={"operation": operation, "status": "conflict"},
            )
        except CalendarError as error:
            return StructuredToolResult(
                kind=f"calendar_{operation}_invalid",
                markdown=f"Не удалось выполнить действие: {_escape(str(error))}.",
                data={
                    "operation": operation,
                    "status": "invalid",
                    "reason": str(error),
                },
            )
        return formatter(value)


class CreateCalendarEventTool(_CalendarTool):
    name = "create_calendar_event"

    @property
    def schema(self) -> Mapping[str, Any]:
        return _function_schema(
            self.name,
            self.description,
            {
                "title": _string(self._parameter("title"), max_length=255),
                "description": _string(self._parameter("description"), max_length=4000),
                "place": _place_schema(self._parameter("place")),
                "starts_at_local": _string(self._parameter("starts_at_local")),
                "timezone": _string(self._parameter("timezone"), max_length=64),
                "reminders": _reminders_schema(self._parameter("reminders")),
            },
            required=("title", "starts_at_local"),
        )

    def invoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> StructuredToolResult:
        return self._sync(self.ainvoke(arguments, context=context))

    async def ainvoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> StructuredToolResult:
        runtime = self._context(context)
        timezone = _optional_string(arguments, "timezone") or runtime.timezone
        request = CreateEvent(
            chat_id=runtime.chat_id,
            title=_required_string(arguments, "title"),
            description=_description_with_place(
                _optional_string(arguments, "description"), arguments
            ),
            starts_at=_local_datetime(arguments, "starts_at_local"),
            timezone=timezone,
            created_by_user_id=runtime.user_id,
            created_by_display_name=runtime.user_display_name,
            reminders=_reminders(arguments),
        )
        return await self._execute(
            "created",
            self.service.create_event(request),
            lambda event: _event_result("created", event),
        )


class ListCalendarEventsTool(_CalendarTool):
    name = "list_calendar_events"

    @property
    def schema(self) -> Mapping[str, Any]:
        return _function_schema(
            self.name,
            self.description,
            {
                "starts_from_local": _string(self._parameter("starts_from_local")),
                "starts_until_local": _string(self._parameter("starts_until_local")),
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": self._parameter("limit"),
                },
                "cursor": _string(self._parameter("cursor")),
                "keywords": {
                    "type": "array",
                    "maxItems": 5,
                    "items": {"type": "string", "minLength": 1, "maxLength": 64},
                    "description": self._parameter("keywords"),
                },
            },
        )

    def invoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> StructuredToolResult:
        return self._sync(self.ainvoke(arguments, context=context))

    async def ainvoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> StructuredToolResult:
        runtime = self._context(context)
        zone = ZoneInfo(runtime.timezone)
        cursor_token = _optional_string(arguments, "cursor")
        if cursor_token is None:
            starts_from = _optional_boundary(arguments, "starts_from_local", zone)
            starts_until = _optional_boundary(arguments, "starts_until_local", zone)
            search_terms = _optional_string_list(arguments, "keywords", maximum=5)
            after = None
            if starts_from is None:
                starts_from = runtime.current_time.astimezone(UTC)
        else:
            starts_from, starts_until, search_terms, after = _decode_list_cursor(
                cursor_token
            )
        limit = _optional_int(arguments, "limit", default=10)
        if not 1 <= limit <= 20:
            raise AgentToolError("limit must be between 1 and 20")
        return await self._execute(
            "listed",
            self.service.list_event_page(
                runtime.chat_id,
                starts_from=starts_from,
                starts_until=starts_until,
                search_terms=search_terms,
                after=after,
                limit=limit,
            ),
            lambda page: _list_result(
                page.events,
                runtime.timezone,
                selection_mode=bool(search_terms),
                next_cursor=(
                    _encode_list_cursor(
                        page.next_cursor,
                        starts_from,
                        starts_until,
                        search_terms,
                    )
                    if page.next_cursor is not None
                    else None
                ),
            ),
        )


class GetCalendarEventTool(_CalendarTool):
    name = "get_calendar_event"

    @property
    def schema(self) -> Mapping[str, Any]:
        return _function_schema(
            self.name,
            self.description,
            {"event_id": _string(self._parameter("event_id"))},
            required=("event_id",),
        )

    def invoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> StructuredToolResult:
        return self._sync(self.ainvoke(arguments, context=context))

    async def ainvoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> StructuredToolResult:
        runtime = self._context(context)
        return await self._execute(
            "retrieved",
            self.service.get_event(
                runtime.chat_id, _required_uuid(arguments, "event_id")
            ),
            lambda event: _event_result("retrieved", event),
        )


class AddCalendarRemindersTool(_CalendarTool):
    name = "add_calendar_reminders"

    @property
    def schema(self) -> Mapping[str, Any]:
        return _function_schema(
            self.name,
            self.description,
            {
                "event_id": _string(self._parameter("event_id")),
                "expected_version": {
                    "type": "integer",
                    "minimum": 1,
                    "description": self._parameter("expected_version"),
                },
                "reminders": _reminders_schema(self._parameter("reminders")),
            },
            required=("event_id", "expected_version", "reminders"),
        )

    def invoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> StructuredToolResult:
        return self._sync(self.ainvoke(arguments, context=context))

    async def ainvoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> StructuredToolResult:
        runtime = self._context(context)
        request = AddEventReminders(
            chat_id=runtime.chat_id,
            event_id=_required_uuid(arguments, "event_id"),
            expected_version=_required_int(arguments, "expected_version"),
            reminders=_reminders(arguments),
        )
        return await self._execute(
            "reminders_added",
            self.service.add_event_reminders(request),
            lambda event: _event_result("reminders_added", event),
        )


class UpdateCalendarEventTool(_CalendarTool):
    name = "update_calendar_event"

    @property
    def schema(self) -> Mapping[str, Any]:
        return _function_schema(
            self.name,
            self.description,
            {
                "event_id": _string(self._parameter("event_id")),
                "expected_version": {
                    "type": "integer",
                    "minimum": 1,
                    "description": self._parameter("expected_version"),
                },
                "title": _string(self._parameter("title"), max_length=255),
                "description": {
                    "anyOf": [{"type": "string", "maxLength": 4000}, {"type": "null"}],
                    "description": self._parameter("description"),
                },
                "place": _place_schema(self._parameter("place")),
                "starts_at_local": _string(self._parameter("starts_at_local")),
                "timezone": _string(self._parameter("timezone"), max_length=64),
                "reminders": _reminders_schema(self._parameter("reminders")),
            },
            required=("event_id", "expected_version"),
        )

    def invoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> StructuredToolResult:
        return self._sync(self.ainvoke(arguments, context=context))

    async def ainvoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> StructuredToolResult:
        runtime = self._context(context)
        changes = {
            "title",
            "description",
            "place",
            "starts_at_local",
            "timezone",
            "reminders",
        }
        if not changes.intersection(arguments):
            raise AgentToolError("update_calendar_event requires a changed field")
        request = UpdateEvent(
            chat_id=runtime.chat_id,
            event_id=_required_uuid(arguments, "event_id"),
            expected_version=_required_int(arguments, "expected_version"),
            title=(
                _required_string(arguments, "title") if "title" in arguments else UNSET
            ),
            description=_updated_description(arguments),
            starts_at=(
                _local_datetime(arguments, "starts_at_local")
                if "starts_at_local" in arguments
                else UNSET
            ),
            timezone=(
                _required_string(arguments, "timezone")
                if "timezone" in arguments
                else UNSET
            ),
            reminders=_reminders(arguments) if "reminders" in arguments else UNSET,
        )
        return await self._execute(
            "updated",
            self.service.update_event(request),
            lambda event: _event_result("updated", event),
        )


class CancelCalendarEventTool(_CalendarTool):
    name = "cancel_calendar_event"

    @property
    def schema(self) -> Mapping[str, Any]:
        return _function_schema(
            self.name,
            self.description,
            {
                "event_id": _string(self._parameter("event_id")),
                "expected_version": {
                    "type": "integer",
                    "minimum": 1,
                    "description": self._parameter("expected_version"),
                },
            },
            required=("event_id", "expected_version"),
        )

    def invoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> StructuredToolResult:
        return self._sync(self.ainvoke(arguments, context=context))

    async def ainvoke(
        self,
        arguments: Mapping[str, Any],
        *,
        context: AgentRuntimeContext | None = None,
    ) -> StructuredToolResult:
        runtime = self._context(context)
        return await self._execute(
            "cancelled",
            self.service.cancel_event(
                runtime.chat_id,
                _required_uuid(arguments, "event_id"),
                expected_version=_required_int(arguments, "expected_version"),
            ),
            lambda event: _event_result("cancelled", event),
        )


def calendar_tools(service: CalendarService) -> tuple[_CalendarTool, ...]:
    """Build the complete calendar tool set backed by one service."""
    return (
        CreateCalendarEventTool(service),
        ListCalendarEventsTool(service),
        GetCalendarEventTool(service),
        AddCalendarRemindersTool(service),
        UpdateCalendarEventTool(service),
        CancelCalendarEventTool(service),
    )


def _function_schema(
    name: str,
    description: str,
    properties: Mapping[str, Any],
    *,
    required: Sequence[str] = (),
) -> Mapping[str, Any]:
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": dict(properties),
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = list(required)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _string(description: str, *, max_length: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "description": description}
    if max_length is not None:
        schema["maxLength"] = max_length
    return schema


def _reminders_schema(description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "maxItems": 5,
        "description": description,
        "items": {
            "type": "object",
            "properties": {
                "offset_minutes": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "За сколько минут до события отправить сообщение.",
                },
                "message_text": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 4096,
                    "description": (
                        "Полный итоговый текст напоминания со всеми обязательными "
                        "упоминаниями пользователя."
                    ),
                },
            },
            "required": ["offset_minutes", "message_text"],
            "additionalProperties": False,
        },
    }


def _place_schema(description: str) -> dict[str, Any]:
    return {
        "type": "object",
        "description": description,
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 255},
            "address": {"type": "string", "minLength": 1, "maxLength": 500},
            "website": {
                "type": "string",
                "format": "uri",
                "minLength": 1,
                "maxLength": 1000,
            },
        },
        "required": ["name", "address", "website"],
        "additionalProperties": False,
    }


def _required_string(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise AgentToolError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_string(arguments: Mapping[str, Any], name: str) -> str | None:
    if name not in arguments or arguments[name] is None:
        return None
    return _required_string(arguments, name)


def _optional_string_list(
    arguments: Mapping[str, Any], name: str, *, maximum: int
) -> tuple[str, ...]:
    value = arguments.get(name, [])
    if not isinstance(value, list) or len(value) > maximum:
        raise AgentToolError(f"{name} must be an array with at most {maximum} items")
    return tuple(_required_string({name: item}, name) for item in value)


def _nullable_string(arguments: Mapping[str, Any], name: str) -> str | None:
    if arguments.get(name) is None:
        return None
    return _required_string(arguments, name)


def _updated_description(arguments: Mapping[str, Any]) -> Any:
    if "place" in arguments:
        if "description" not in arguments:
            raise AgentToolError(
                "adding a place requires the current description or null"
            )
        return _description_with_place(
            _nullable_string(arguments, "description"), arguments
        )
    if "description" in arguments:
        return _nullable_string(arguments, "description")
    return UNSET


def _description_with_place(
    description: str | None,
    arguments: Mapping[str, Any],
) -> str | None:
    if "place" not in arguments:
        return description
    raw_place = arguments["place"]
    if not isinstance(raw_place, Mapping):
        raise AgentToolError("place must be an object")
    name = _required_string(raw_place, "name")
    address = _required_string(raw_place, "address")
    website = _required_string(raw_place, "website")
    parsed_website = urlsplit(website)
    if parsed_website.scheme not in {"http", "https"} or not parsed_website.netloc:
        raise AgentToolError("place website must be an HTTP(S) URL")

    place_block = f"Название: {name}\nАдрес: {address}\nСайт: {website}"
    combined = (
        f"{description.rstrip()}\n\n{place_block}" if description else place_block
    )
    if len(combined) > 4000:
        raise AgentToolError("description with place must not exceed 4000 characters")
    return combined


def _required_int(arguments: Mapping[str, Any], name: str) -> int:
    value = arguments.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AgentToolError(f"{name} must be an integer")
    return value


def _optional_int(arguments: Mapping[str, Any], name: str, *, default: int) -> int:
    if name not in arguments:
        return default
    return _required_int(arguments, name)


def _required_uuid(arguments: Mapping[str, Any], name: str) -> UUID:
    try:
        return UUID(_required_string(arguments, name))
    except ValueError as error:
        raise AgentToolError(f"{name} must be a UUID") from error


def _local_datetime(arguments: Mapping[str, Any], name: str) -> datetime:
    value = _required_string(arguments, name)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise AgentToolError(f"{name} must be an ISO local datetime") from error
    if parsed.tzinfo is not None:
        raise AgentToolError(f"{name} must not contain a UTC offset")
    return parsed


def _optional_boundary(
    arguments: Mapping[str, Any], name: str, timezone: ZoneInfo
) -> datetime | None:
    value = _optional_string(arguments, name)
    if value is None:
        return None
    try:
        if "T" in value:
            local = datetime.fromisoformat(value)
        else:
            local = datetime.combine(date.fromisoformat(value), time.min)
    except ValueError as error:
        raise AgentToolError(f"{name} must be an ISO date or local datetime") from error
    if local.tzinfo is not None:
        raise AgentToolError(f"{name} must not contain a UTC offset")
    return local.replace(tzinfo=timezone).astimezone(UTC)


def _reminders(arguments: Mapping[str, Any]) -> tuple[ReminderDraft, ...]:
    value = arguments.get("reminders", [])
    if not isinstance(value, list):
        raise AgentToolError("reminders must be an array")
    reminders = []
    for item in value:
        if not isinstance(item, Mapping):
            raise AgentToolError("each reminder must be an object")
        offset = _required_int(item, "offset_minutes")
        reminders.append(
            ReminderDraft(
                offset=timedelta(minutes=offset),
                message_text=_required_string(item, "message_text"),
            )
        )
    return tuple(reminders)


def _event_result(operation: str, event: CalendarEvent) -> StructuredToolResult:
    labels = {
        "created": "✅ Событие сохранено",
        "retrieved": "📅 Событие",
        "reminders_added": "✅ Напоминание добавлено",
        "updated": "✅ Событие обновлено",
        "cancelled": "✅ Событие отменено",
    }
    return StructuredToolResult(
        kind=f"calendar_event_{operation}",
        markdown=_event_markdown(labels[operation], event),
        data={"operation": operation, "event": _event_data(event)},
    )


def _list_result(
    events: Sequence[CalendarEvent],
    timezone: str,
    *,
    selection_mode: bool = False,
    next_cursor: str | None = None,
) -> StructuredToolResult:
    ordered_events = sorted(events, key=lambda event: event.starts_at)
    if not ordered_events:
        markdown = "В календаре этого чата нет событий за выбранный период."
    else:
        heading = (
            "## 🔎 Подходящие события" if selection_mode else "## 📅 Ближайшие события"
        )
        lines = [heading, ""]
        for position, event in enumerate(ordered_events, start=1):
            local = event.starts_at.astimezone(ZoneInfo(timezone))
            lines.extend((f"### {position}. {_escape(event.title)}", ""))
            lines.append(f"🗓 **Когда:** {local:%d.%m.%Y в %H:%M}")
            lines.append(f"🌍 **Часовой пояс:** {_escape(timezone)}")
            if event.description:
                lines.append(f"📝 **Описание:** {_escape(event.description)}")

            open_reminders = sorted(
                (
                    reminder
                    for reminder in event.reminders
                    if reminder.status
                    in (
                        CalendarReminderStatus.PENDING,
                        CalendarReminderStatus.PROCESSING,
                    )
                ),
                key=lambda reminder: reminder.remind_at,
            )
            if open_reminders:
                lines.extend(("", "🔔 **Напоминания:**"))
                for reminder in open_reminders:
                    reminder_local = reminder.remind_at.astimezone(ZoneInfo(timezone))
                    lines.append(
                        f"- {reminder_local:%d.%m.%Y в %H:%M} — "
                        f"{_escape(reminder.message_text)}"
                    )
            else:
                lines.extend(("", "🔕 Напоминания не установлены"))

            if position < len(ordered_events):
                lines.extend(("", "---", ""))
        if next_cursor is not None:
            lines.extend(
                (
                    "",
                    "_Есть ещё события. Скажите «покажи следующие»._",
                )
            )
        if selection_mode:
            lines.append("")
            if len(ordered_events) == 1:
                lines.append("**Это событие вы имели в виду?**")
            else:
                lines.append(
                    "**Подходит несколько событий. Какое именно вы имели в виду?**"
                )
        markdown = "\n".join(lines)
    return StructuredToolResult(
        kind="calendar_events_listed",
        markdown=markdown,
        data={
            "operation": "listed",
            "timezone": timezone,
            "events": [_event_data(event) for event in ordered_events],
            "has_more": next_cursor is not None,
            "next_cursor": next_cursor,
            "selection_required": selection_mode,
        },
    )


def _encode_list_cursor(
    cursor: CalendarEventCursor,
    starts_from: datetime | None,
    starts_until: datetime | None,
    search_terms: Sequence[str],
) -> str:
    payload = json.dumps(
        {
            "after": cursor.starts_at.isoformat(),
            "event_id": str(cursor.event_id),
            "from": starts_from.isoformat() if starts_from is not None else None,
            "until": starts_until.isoformat() if starts_until is not None else None,
            "keywords": list(search_terms),
        },
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_list_cursor(
    token: str,
) -> tuple[datetime | None, datetime | None, tuple[str, ...], CalendarEventCursor]:
    try:
        padding = "=" * (-len(token) % 4)
        payload = json.loads(base64.urlsafe_b64decode(token + padding))
        starts_at = datetime.fromisoformat(payload["after"])
        starts_from = (
            datetime.fromisoformat(payload["from"])
            if payload["from"] is not None
            else None
        )
        starts_until = (
            datetime.fromisoformat(payload["until"])
            if payload["until"] is not None
            else None
        )
        cursor = CalendarEventCursor(starts_at, UUID(payload["event_id"]))
        keywords = payload["keywords"]
        if not isinstance(keywords, list) or not all(
            isinstance(item, str) and item for item in keywords
        ):
            raise ValueError
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        binascii.Error,
        json.JSONDecodeError,
    ) as error:
        raise AgentToolError("cursor is invalid or expired") from error
    if (
        starts_at.tzinfo is None
        or (starts_from is not None and starts_from.tzinfo is None)
        or (starts_until is not None and starts_until.tzinfo is None)
    ):
        raise AgentToolError("cursor is invalid or expired")
    return starts_from, starts_until, tuple(keywords), cursor


def _event_markdown(heading: str, event: CalendarEvent) -> str:
    zone = ZoneInfo(event.source_timezone)
    local = event.starts_at.astimezone(zone)
    lines = [
        f"## {heading}",
        "",
        f"### {_escape(event.title)}",
        "",
        f"🗓 **Когда:** {local:%d.%m.%Y в %H:%M}",
        f"🌍 **Часовой пояс:** {_escape(event.source_timezone)}",
    ]
    if event.description:
        lines.append(f"📝 **Описание:** {_escape(event.description)}")
    open_reminders = [
        reminder
        for reminder in event.reminders
        if reminder.status
        in (CalendarReminderStatus.PENDING, CalendarReminderStatus.PROCESSING)
    ]
    if open_reminders:
        lines.extend(("", "🔔 **Напоминания:**"))
        for reminder in open_reminders:
            reminder_local = reminder.remind_at.astimezone(zone)
            lines.append(
                f"- {reminder_local:%d.%m.%Y в %H:%M} — "
                f"{_escape(reminder.message_text)}"
            )
    else:
        lines.extend(("", "🔕 Напоминания не установлены"))
    return "\n".join(lines)


def _event_data(event: CalendarEvent) -> dict[str, Any]:
    return {
        "id": str(event.id),
        "chat_id": event.chat_id,
        "title": event.title,
        "description": event.description,
        "starts_at": event.starts_at.isoformat(),
        "source_timezone": event.source_timezone,
        "status": event.status.value,
        "version": event.version,
        "reminders": [
            {
                "id": str(reminder.id),
                "remind_at": reminder.remind_at.isoformat(),
                "message_text": reminder.message_text,
                "status": reminder.status.value,
            }
            for reminder in event.reminders
        ],
    }


def _escape(value: str) -> str:
    for character in "\\`*_[]()~>#+-=|{}.!":
        value = value.replace(character, f"\\{character}")
    return value


__all__ = [
    "AddCalendarRemindersTool",
    "CancelCalendarEventTool",
    "CreateCalendarEventTool",
    "GetCalendarEventTool",
    "ListCalendarEventsTool",
    "UpdateCalendarEventTool",
    "calendar_tools",
]
