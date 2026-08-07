from __future__ import annotations

import os
import unittest
from datetime import UTC, datetime, timedelta
from typing import Never
from uuid import UUID, uuid4

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from calendar_app import (
    CalendarConflictError,
    CalendarEventStatus,
    CalendarNotFoundError,
    CalendarReminderStatus,
    CalendarService,
    CalendarUnitOfWorkFactory,
    CalendarValidationError,
    CreateEvent,
    ReminderDraft,
    UpdateEvent,
)
from database.calendar import SqlAlchemyCalendarUnitOfWorkFactory
from database.models import CalendarChat, CalendarEvent, EventReminder

FIXED_NOW = datetime(2026, 8, 7, 9, 0, tzinfo=UTC)


def local_datetime(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    """Build wall-clock input whose IANA timezone is supplied separately."""
    return datetime(year, month, day, hour, minute)  # noqa: DTZ001


class CalendarBusinessRulesTest(unittest.IsolatedAsyncioTestCase):
    def service(self, now: datetime = FIXED_NOW) -> CalendarService:
        return CalendarService(
            _forbidden_unit_of_work_factory,
            now=lambda: now,
        )

    async def test_rejects_past_event_before_opening_transaction(self) -> None:
        with self.assertRaisesRegex(CalendarValidationError, "future"):
            await self.service().create_event(
                CreateEvent(
                    chat_id=1,
                    title="Past",
                    starts_at=local_datetime(2026, 8, 7, 10, 0),
                    timezone="Europe/Moscow",
                )
            )

    async def test_rejects_reminder_whose_time_is_in_the_past(self) -> None:
        with self.assertRaisesRegex(CalendarValidationError, "reminder time"):
            await self.service().create_event(
                CreateEvent(
                    chat_id=1,
                    title="Soon",
                    starts_at=local_datetime(2026, 8, 7, 12, 10),
                    timezone="Europe/Moscow",
                    reminders=(ReminderDraft(timedelta(hours=1), "Prepare"),),
                )
            )

    async def test_rejects_blank_custom_reminder_text(self) -> None:
        with self.assertRaisesRegex(CalendarValidationError, "cannot be blank"):
            await self.service().create_event(
                CreateEvent(
                    chat_id=1,
                    title="Meeting",
                    starts_at=local_datetime(2026, 8, 8, 15, 0),
                    timezone="Europe/Moscow",
                    reminders=(ReminderDraft(timedelta(hours=1), "  "),),
                )
            )

    async def test_rejects_ambiguous_local_time(self) -> None:
        with self.assertRaisesRegex(CalendarValidationError, "ambiguous"):
            await self.service().create_event(
                CreateEvent(
                    chat_id=1,
                    title="DST overlap",
                    starts_at=local_datetime(2026, 11, 1, 1, 30),
                    timezone="America/New_York",
                )
            )

    async def test_rejects_nonexistent_local_time(self) -> None:
        with self.assertRaisesRegex(CalendarValidationError, "does not exist"):
            await self.service(datetime(2026, 1, 1, tzinfo=UTC)).create_event(
                CreateEvent(
                    chat_id=1,
                    title="DST gap",
                    starts_at=local_datetime(2026, 3, 8, 2, 30),
                    timezone="America/New_York",
                )
            )


class _ForbiddenUnitOfWorkFactory:
    def __call__(self) -> Never:
        raise AssertionError("validation should finish before opening a unit of work")


_forbidden_unit_of_work_factory: CalendarUnitOfWorkFactory = (
    _ForbiddenUnitOfWorkFactory()
)


@unittest.skipUnless(
    os.environ.get("DATABASE_TEST_URL"),
    "DATABASE_TEST_URL is required for PostgreSQL integration tests",
)
class CalendarServicePostgreSQLTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(os.environ["DATABASE_TEST_URL"])
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.service = CalendarService(
            SqlAlchemyCalendarUnitOfWorkFactory(self.sessions), now=lambda: FIXED_NOW
        )
        self.chat_id = -(uuid4().int % 9_000_000_000 + 1)

    async def asyncTearDown(self) -> None:
        async with self.sessions() as session, session.begin():
            event_ids = (
                await session.scalars(
                    CalendarEvent.__table__.select()
                    .with_only_columns(CalendarEvent.id)
                    .where(CalendarEvent.chat_id == self.chat_id)
                )
            ).all()
            if event_ids:
                await session.execute(
                    delete(EventReminder).where(EventReminder.event_id.in_(event_ids))
                )
                await session.execute(
                    delete(CalendarEvent).where(CalendarEvent.id.in_(event_ids))
                )
            await session.execute(
                delete(CalendarChat).where(CalendarChat.chat_id == self.chat_id)
            )
        await self.engine.dispose()

    async def test_full_calendar_lifecycle_is_scoped_and_transactional(self) -> None:
        event = await self.service.create_event(
            CreateEvent(
                chat_id=self.chat_id,
                title="  Contract meeting  ",
                description=" Discuss the renewal ",
                starts_at=local_datetime(2026, 8, 8, 15, 0),
                timezone="Europe/Moscow",
                created_by_user_id=42,
                created_by_display_name=" Alexander ",
                reminders=(
                    ReminderDraft(
                        timedelta(hours=1),
                        "Bring the contract and call Anna",
                    ),
                ),
            )
        )

        self.assertEqual(event.title, "Contract meeting")
        self.assertEqual(event.description, "Discuss the renewal")
        self.assertEqual(event.starts_at, datetime(2026, 8, 8, 12, 0, tzinfo=UTC))
        self.assertEqual(event.source_timezone, "Europe/Moscow")
        self.assertEqual(event.version, 1)
        self.assertEqual(len(event.reminders), 1)
        self.assertEqual(
            event.reminders[0].message_text,
            "Bring the contract and call Anna",
        )
        self.assertEqual(
            event.reminders[0].remind_at,
            datetime(2026, 8, 8, 11, 0, tzinfo=UTC),
        )

        settings = await self.service.get_settings(self.chat_id)
        self.assertIsNotNone(settings)
        self.assertEqual(settings.timezone, "Europe/Moscow")
        listed = await self.service.list_events(self.chat_id)
        self.assertEqual([item.id for item in listed], [event.id])

        with self.assertRaises(CalendarNotFoundError):
            await self.service.get_event(self.chat_id - 1, event.id)

        updated = await self.service.update_event(
            UpdateEvent(
                chat_id=self.chat_id,
                event_id=event.id,
                expected_version=event.version,
                starts_at=local_datetime(2026, 8, 9, 16, 0),
            )
        )
        self.assertEqual(updated.version, 2)
        self.assertEqual(updated.starts_at, datetime(2026, 8, 9, 13, 0, tzinfo=UTC))
        self.assertEqual(len(updated.reminders), 2)
        statuses = {reminder.status for reminder in updated.reminders}
        self.assertEqual(
            statuses,
            {CalendarReminderStatus.CANCELLED, CalendarReminderStatus.PENDING},
        )
        pending = next(
            reminder
            for reminder in updated.reminders
            if reminder.status is CalendarReminderStatus.PENDING
        )
        self.assertEqual(pending.remind_at, updated.starts_at - timedelta(hours=1))
        self.assertEqual(pending.message_text, event.reminders[0].message_text)

        with self.assertRaises(CalendarConflictError):
            await self.service.update_event(
                UpdateEvent(
                    chat_id=self.chat_id,
                    event_id=event.id,
                    expected_version=1,
                    title="Stale title",
                )
            )
        unchanged = await self.service.get_event(self.chat_id, event.id)
        self.assertEqual(unchanged.title, "Contract meeting")
        self.assertEqual(unchanged.version, 2)

        cancelled = await self.service.cancel_event(
            self.chat_id,
            event.id,
            expected_version=updated.version,
        )
        self.assertEqual(cancelled.status, CalendarEventStatus.CANCELLED)
        self.assertEqual(cancelled.version, 3)
        self.assertTrue(
            all(
                reminder.status is CalendarReminderStatus.CANCELLED
                for reminder in cancelled.reminders
            )
        )
        self.assertEqual(await self.service.list_events(self.chat_id), [])

    async def test_can_replace_reminder_text_at_the_same_time(self) -> None:
        event = await self.service.create_event(
            CreateEvent(
                chat_id=self.chat_id,
                title="Meeting",
                starts_at=local_datetime(2026, 8, 8, 15, 0),
                timezone="Europe/Moscow",
                reminders=(ReminderDraft(timedelta(hours=1), "Old text"),),
            )
        )

        updated = await self.service.update_event(
            UpdateEvent(
                chat_id=self.chat_id,
                event_id=event.id,
                expected_version=event.version,
                reminders=(ReminderDraft(timedelta(hours=1), "New custom text"),),
            )
        )

        self.assertEqual(len(updated.reminders), 2)
        self.assertEqual(
            {
                (reminder.status, reminder.message_text)
                for reminder in updated.reminders
            },
            {
                (CalendarReminderStatus.CANCELLED, "Old text"),
                (CalendarReminderStatus.PENDING, "New custom text"),
            },
        )

    async def test_failed_creation_does_not_create_chat_settings(self) -> None:
        with self.assertRaises(CalendarValidationError):
            await self.service.create_event(
                CreateEvent(
                    chat_id=self.chat_id,
                    title="Meeting",
                    starts_at=local_datetime(2026, 8, 8, 15, 0),
                    timezone="Europe/Moscow",
                    reminders=(ReminderDraft(timedelta(hours=1), ""),),
                )
            )

        self.assertIsNone(await self.service.get_settings(self.chat_id))

    async def test_timezone_settings_can_be_created_and_updated(self) -> None:
        created = await self.service.set_timezone(self.chat_id, "Europe/Moscow")
        updated = await self.service.set_timezone(self.chat_id, "Asia/Tokyo")

        self.assertEqual(created.timezone, "Europe/Moscow")
        self.assertEqual(updated.timezone, "Asia/Tokyo")

    async def test_missing_event_is_not_confused_with_another_chat(self) -> None:
        with self.assertRaises(CalendarNotFoundError):
            await self.service.get_event(self.chat_id, UUID(int=0))
