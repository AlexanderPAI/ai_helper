from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from agent import AgentRuntimeContext, StructuredToolResult
from agent.calendar_tools import (
    AddCalendarRemindersTool,
    CancelCalendarEventTool,
    CreateCalendarEventTool,
    GetCalendarEventTool,
    ListCalendarEventsTool,
    UpdateCalendarEventTool,
    calendar_tools,
)
from calendar_app import (
    CalendarEvent,
    CalendarEventCursor,
    CalendarEventPage,
    CalendarEventStatus,
    CalendarReminder,
    CalendarReminderStatus,
)

NOW = datetime(2026, 8, 7, 9, 0, tzinfo=UTC)


def runtime_context(chat_id: int = -100123) -> AgentRuntimeContext:
    return AgentRuntimeContext(
        chat_id=chat_id,
        user_id=42,
        user_display_name="Александр",
        message_id=99,
        current_time=NOW,
        timezone="Europe/Moscow",
    )


def event(*, chat_id: int = -100123, version: int = 1) -> CalendarEvent:
    event_id = uuid4()
    reminder = CalendarReminder(
        id=uuid4(),
        event_id=event_id,
        remind_at=datetime(2026, 8, 8, 11, 0, tzinfo=UTC),
        message_text="Взять договор и позвонить Анне",
        status=CalendarReminderStatus.PENDING,
        attempts=0,
        next_attempt_at=datetime(2026, 8, 8, 11, 0, tzinfo=UTC),
        locked_at=None,
        locked_by=None,
        sent_at=None,
        telegram_message_id=None,
        last_error=None,
        created_at=NOW,
        updated_at=NOW,
    )
    return CalendarEvent(
        id=event_id,
        chat_id=chat_id,
        title="Встреча по договору",
        description="Продление договора",
        starts_at=datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
        source_timezone="Europe/Moscow",
        status=CalendarEventStatus.ACTIVE,
        created_by_user_id=42,
        created_by_display_name="Александр",
        created_at=NOW,
        updated_at=NOW,
        cancelled_at=None,
        version=version,
        reminders=(reminder,),
    )


class CalendarToolSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = Mock()

    def test_complete_tool_set_has_no_model_controlled_chat_or_user_ids(self) -> None:
        tools = calendar_tools(self.service)

        self.assertEqual(
            {tool.name for tool in tools},
            {
                "create_calendar_event",
                "add_calendar_reminders",
                "list_calendar_events",
                "get_calendar_event",
                "update_calendar_event",
                "cancel_calendar_event",
            },
        )
        for tool in tools:
            properties = tool.schema["function"]["parameters"]["properties"]
            self.assertNotIn("chat_id", properties)
            self.assertNotIn("user_id", properties)


class CalendarToolExecutionTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_uses_trusted_metadata_and_preserves_custom_text(self) -> None:
        service = Mock()
        saved = event()
        service.create_event = AsyncMock(return_value=saved)
        tool = CreateCalendarEventTool(service)

        result = await tool.ainvoke(
            {
                "title": "Встреча по договору",
                "description": "Продление договора",
                "starts_at_local": "2026-08-08T15:00:00",
                "reminders": [
                    {
                        "offset_minutes": 60,
                        "message_text": "Взять договор и позвонить Анне",
                    }
                ],
            },
            context=runtime_context(),
        )

        request = service.create_event.await_args.args[0]
        self.assertEqual(request.chat_id, -100123)
        self.assertEqual(request.created_by_user_id, 42)
        self.assertEqual(request.created_by_display_name, "Александр")
        self.assertEqual(request.timezone, "Europe/Moscow")
        self.assertEqual(request.reminders[0].offset, timedelta(hours=1))
        self.assertEqual(
            request.reminders[0].message_text,
            "Взять договор и позвонить Анне",
        )
        self.assertIsInstance(result, StructuredToolResult)
        self.assertEqual(result.kind, "calendar_event_created")
        self.assertEqual(result.data["event"]["chat_id"], -100123)
        self.assertIn("Взять договор и позвонить Анне", result.markdown)
        self.assertNotIn(str(saved.id), result.markdown)
        self.assertNotIn("версия", result.markdown.casefold())

    async def test_create_appends_place_after_custom_description(self) -> None:
        service = Mock()
        service.create_event = AsyncMock(return_value=event())
        tool = CreateCalendarEventTool(service)

        await tool.ainvoke(
            {
                "title": "Ужин",
                "description": "Встреча с друзьями",
                "starts_at_local": "2026-08-08T19:00:00",
                "place": {
                    "name": "Кафе Север",
                    "address": "Москва, Тверская улица, 1",
                    "website": "https://example.test/cafe",
                },
            },
            context=runtime_context(),
        )

        request = service.create_event.await_args.args[0]
        self.assertEqual(
            request.description,
            "Встреча с друзьями\n\n"
            "Название: Кафе Север\n"
            "Адрес: Москва, Тверская улица, 1\n"
            "Сайт: https://example.test/cafe",
        )

    async def test_list_is_chronological_and_hides_internal_fields_from_user(
        self,
    ) -> None:
        service = Mock()
        later = event(version=7)
        sooner = replace(
            event(version=3),
            title="Завтрак",
            starts_at=datetime(2026, 8, 8, 10, 0, tzinfo=UTC),
        )
        service.list_event_page = AsyncMock(
            return_value=CalendarEventPage(
                events=(later, sooner),
                next_cursor=CalendarEventCursor(later.starts_at, later.id),
            )
        )
        tool = ListCalendarEventsTool(service)

        result = await tool.ainvoke(
            {"keywords": ["договор"]}, context=runtime_context()
        )

        self.assertEqual(service.list_event_page.await_args.args[0], -100123)
        self.assertEqual(service.list_event_page.await_args.kwargs["starts_from"], NOW)
        self.assertEqual(service.list_event_page.await_args.kwargs["limit"], 10)
        self.assertEqual(
            service.list_event_page.await_args.kwargs["search_terms"],
            ("договор",),
        )
        self.assertEqual(result.kind, "calendar_events_listed")
        self.assertEqual(result.data["events"][0]["id"], str(sooner.id))
        self.assertLess(
            result.markdown.index("Завтрак"),
            result.markdown.index("Встреча по договору"),
        )
        self.assertIn("📝 **Описание:** Продление договора", result.markdown)
        self.assertIn("🔔 **Напоминания:**", result.markdown)
        self.assertNotIn("ID:", result.markdown)
        self.assertNotIn("версия", result.markdown.casefold())
        self.assertTrue(result.data["has_more"])
        self.assertIsInstance(result.data["next_cursor"], str)
        self.assertIn("покажи следующие", result.markdown)
        self.assertIn("Какое именно вы имели в виду?", result.markdown)
        self.assertTrue(result.data["selection_required"])

        service.list_event_page.reset_mock()
        service.list_event_page.return_value = CalendarEventPage((later,), None)
        next_result = await tool.ainvoke(
            {"cursor": result.data["next_cursor"]}, context=runtime_context()
        )

        next_call = service.list_event_page.await_args.kwargs
        self.assertEqual(next_call["after"].starts_at, later.starts_at)
        self.assertEqual(next_call["after"].event_id, later.id)
        self.assertEqual(next_call["starts_from"], NOW)
        self.assertEqual(next_call["search_terms"], ("договор",))
        self.assertFalse(next_result.data["has_more"])
        self.assertIsNone(next_result.data["next_cursor"])

    async def test_get_update_and_cancel_are_scoped_to_runtime_chat(self) -> None:
        service = Mock()
        current = event(version=3)
        service.get_event = AsyncMock(return_value=current)
        service.update_event = AsyncMock(return_value=current)
        service.cancel_event = AsyncMock(return_value=current)
        context = runtime_context(chat_id=-777)

        get_result = await GetCalendarEventTool(service).ainvoke(
            {"event_id": str(current.id)}, context=context
        )
        update_result = await UpdateCalendarEventTool(service).ainvoke(
            {
                "event_id": str(current.id),
                "expected_version": 3,
                "title": "Новое название",
            },
            context=context,
        )
        cancel_result = await CancelCalendarEventTool(service).ainvoke(
            {"event_id": str(current.id), "expected_version": 3},
            context=context,
        )

        service.get_event.assert_awaited_once_with(-777, current.id)
        update_request = service.update_event.await_args.args[0]
        self.assertEqual(update_request.chat_id, -777)
        self.assertEqual(update_request.expected_version, 3)
        self.assertEqual(update_request.title, "Новое название")
        service.cancel_event.assert_awaited_once_with(
            -777, current.id, expected_version=3
        )
        self.assertEqual(get_result.kind, "calendar_event_retrieved")
        self.assertEqual(update_result.kind, "calendar_event_updated")
        self.assertEqual(cancel_result.kind, "calendar_event_cancelled")
        for result in (get_result, update_result, cancel_result):
            self.assertNotIn(str(current.id), result.markdown)
            self.assertNotIn("версия", result.markdown.casefold())

    async def test_update_appends_place_after_existing_description(self) -> None:
        service = Mock()
        current = event(version=3)
        service.update_event = AsyncMock(return_value=current)

        await UpdateCalendarEventTool(service).ainvoke(
            {
                "event_id": str(current.id),
                "expected_version": 3,
                "description": "Продление договора",
                "place": {
                    "name": "Кафе Север",
                    "address": "Москва, Тверская улица, 1",
                    "website": "https://example.test/cafe",
                },
            },
            context=runtime_context(),
        )

        request = service.update_event.await_args.args[0]
        self.assertEqual(
            request.description,
            "Продление договора\n\n"
            "Название: Кафе Север\n"
            "Адрес: Москва, Тверская улица, 1\n"
            "Сайт: https://example.test/cafe",
        )

    async def test_add_reminder_uses_dedicated_service_operation(self) -> None:
        service = Mock()
        current = event(version=4)
        service.add_event_reminders = AsyncMock(return_value=current)

        result = await AddCalendarRemindersTool(service).ainvoke(
            {
                "event_id": str(current.id),
                "expected_version": 3,
                "reminders": [
                    {
                        "offset_minutes": 60,
                        "message_text": "Отвести Фикса к окулисту",
                    }
                ],
            },
            context=runtime_context(chat_id=-777),
        )

        request = service.add_event_reminders.await_args.args[0]
        self.assertEqual(request.chat_id, -777)
        self.assertEqual(request.event_id, current.id)
        self.assertEqual(request.expected_version, 3)
        self.assertEqual(request.reminders[0].offset, timedelta(hours=1))
        self.assertEqual(
            request.reminders[0].message_text,
            "Отвести Фикса к окулисту",
        )
        self.assertEqual(result.kind, "calendar_event_reminders_added")
        self.assertIn("Напоминание добавлено", result.markdown)
        self.assertNotIn(str(current.id), result.markdown)
