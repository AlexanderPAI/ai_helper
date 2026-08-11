from __future__ import annotations

import asyncio
import os
import unittest
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Self
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from calendar_app import CalendarService, CreateEvent, ReminderDraft
from database import (
    SqlAlchemyCalendarUnitOfWorkFactory,
    SqlAlchemyReminderUnitOfWorkFactory,
)
from database.models import CalendarChat, CalendarEvent, EventReminder
from reminder_app import (
    PermanentReminderDeliveryError,
    ReminderDelivery,
    ReminderWorker,
    ReminderWorkerOptions,
    TemporaryReminderDeliveryError,
)

NOW = datetime(2026, 8, 7, 9, 0, tzinfo=UTC)


def delivery(*, attempts: int = 1) -> ReminderDelivery:
    return ReminderDelivery(
        reminder_id=uuid4(),
        event_id=uuid4(),
        chat_id=-100123,
        event_title="Встреча",
        event_starts_at=NOW + timedelta(hours=1),
        event_timezone="Europe/Moscow",
        message_text="Взять договор",
        remind_at=NOW,
        attempts=attempts,
    )


class _MemoryQueue:
    def __init__(self, reminders: list[ReminderDelivery]) -> None:
        self.reminders = reminders
        self.recovered = 0
        self.sent: list[tuple[UUID, int]] = []
        self.rescheduled: list[tuple[UUID, datetime, str]] = []
        self.failed: list[tuple[UUID, str]] = []
        self.polled = asyncio.Event()

    async def recover_expired(self, **kwargs) -> int:
        return self.recovered

    async def reserve_due(self, **kwargs) -> list[ReminderDelivery]:
        self.polled.set()
        reminders, self.reminders = self.reminders, []
        return reminders

    async def mark_sent(
        self, reminder_id: UUID, *, external_message_id: int, **kwargs
    ) -> bool:
        self.sent.append((reminder_id, external_message_id))
        return True

    async def reschedule(
        self,
        reminder_id: UUID,
        *,
        next_attempt_at: datetime,
        error: str,
        **kwargs,
    ) -> bool:
        self.rescheduled.append((reminder_id, next_attempt_at, error))
        return True

    async def mark_failed(self, reminder_id: UUID, *, error: str, **kwargs) -> bool:
        self.failed.append((reminder_id, error))
        return True


class _MemoryUnitOfWork:
    def __init__(self, queue: _MemoryQueue) -> None:
        self.queue = queue

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class _MemoryUnitOfWorkFactory:
    def __init__(self, queue: _MemoryQueue) -> None:
        self.queue = queue

    def __call__(self) -> _MemoryUnitOfWork:
        return _MemoryUnitOfWork(self.queue)


class _Sender:
    def __init__(self, *outcomes: int | BaseException) -> None:
        self.outcomes = list(outcomes)
        self.deliveries: list[ReminderDelivery] = []

    async def send(self, reminder: ReminderDelivery) -> int:
        self.deliveries.append(reminder)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class ReminderWorkerTest(unittest.IsolatedAsyncioTestCase):
    async def test_success_marks_reserved_reminder_sent(self) -> None:
        reminder = delivery()
        queue = _MemoryQueue([reminder])
        worker = ReminderWorker(
            _MemoryUnitOfWorkFactory(queue),
            _Sender(777),
            worker_id="test-worker",
            now=lambda: NOW,
        )

        report = await worker.run_once()

        self.assertEqual(report.reserved, 1)
        self.assertEqual(report.sent, 1)
        self.assertEqual(queue.sent, [(reminder.reminder_id, 777)])

    async def test_temporary_error_uses_backoff_and_retry_after(self) -> None:
        reminder = delivery(attempts=2)
        queue = _MemoryQueue([reminder])
        worker = ReminderWorker(
            _MemoryUnitOfWorkFactory(queue),
            _Sender(TemporaryReminderDeliveryError("rate limit", retry_after=90)),
            worker_id="test-worker",
            options=ReminderWorkerOptions(retry_jitter_ratio=0),
            now=lambda: NOW,
        )

        report = await worker.run_once()

        self.assertEqual(report.rescheduled, 1)
        self.assertEqual(queue.rescheduled[0][1], NOW + timedelta(seconds=90))
        self.assertEqual(queue.rescheduled[0][2], "rate limit")

    async def test_permanent_error_and_exhausted_retry_are_failed(self) -> None:
        permanent = delivery()
        exhausted = delivery(attempts=5)
        queue = _MemoryQueue([permanent, exhausted])
        sender = _Sender(
            PermanentReminderDeliveryError("blocked"),
            TemporaryReminderDeliveryError("timeout"),
        )
        worker = ReminderWorker(
            _MemoryUnitOfWorkFactory(queue),
            sender,
            worker_id="test-worker",
            options=ReminderWorkerOptions(max_attempts=5),
            now=lambda: NOW,
        )

        report = await worker.run_once()

        self.assertEqual(report.failed, 2)
        self.assertEqual(
            {item[0] for item in queue.failed},
            {permanent.reminder_id, exhausted.reminder_id},
        )

    async def test_stop_interrupts_long_poll_wait(self) -> None:
        queue = _MemoryQueue([])
        worker = ReminderWorker(
            _MemoryUnitOfWorkFactory(queue),
            _Sender(),
            worker_id="test-worker",
            options=ReminderWorkerOptions(poll_interval=3600),
            now=lambda: NOW,
        )
        task = asyncio.create_task(worker.run())
        await asyncio.wait_for(queue.polled.wait(), timeout=1)

        worker.request_stop()
        await asyncio.wait_for(task, timeout=1)

        self.assertTrue(task.done())


@unittest.skipUnless(
    os.environ.get("DATABASE_TEST_URL"),
    "DATABASE_TEST_URL is required for PostgreSQL integration tests",
)
class ReminderWorkerPostgreSQLTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(os.environ["DATABASE_TEST_URL"])
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.calendar_service = CalendarService(
            SqlAlchemyCalendarUnitOfWorkFactory(self.sessions), now=lambda: NOW
        )
        self.reminder_uow = SqlAlchemyReminderUnitOfWorkFactory(self.sessions)
        self.chat_id = -(uuid4().int % 9_000_000_000 + 1)

    async def asyncTearDown(self) -> None:
        async with self.sessions() as session, session.begin():
            event_ids = (
                await session.scalars(
                    select(CalendarEvent.id).where(
                        CalendarEvent.chat_id == self.chat_id
                    )
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

    async def create_due_reminder(self) -> UUID:
        event = await self.calendar_service.create_event(
            CreateEvent(
                chat_id=self.chat_id,
                title="Встреча",
                starts_at=datetime(2026, 8, 7, 13, 0),  # noqa: DTZ001
                timezone="Europe/Moscow",
                reminders=(ReminderDraft(timedelta(hours=1), "Взять договор"),),
            )
        )
        return event.reminders[0].id

    async def reminder_record(self, reminder_id: UUID) -> EventReminder:
        async with self.sessions() as session:
            record = await session.get(EventReminder, reminder_id)
            if record is None:
                raise AssertionError("reminder record not found")
            session.expunge(record)
            return record

    async def test_successful_delivery_persists_sent_state(self) -> None:
        reminder_id = await self.create_due_reminder()
        sender = _Sender(501)
        worker = ReminderWorker(
            self.reminder_uow,
            sender,
            worker_id="worker-a",
            now=lambda: NOW,
        )

        report = await worker.run_once()
        record = await self.reminder_record(reminder_id)

        self.assertEqual(report.sent, 1)
        self.assertEqual(record.status, "sent")
        self.assertEqual(record.attempts, 1)
        self.assertEqual(record.telegram_message_id, 501)
        self.assertIsNotNone(record.sent_at)
        self.assertIsNone(record.locked_by)

    async def test_live_worker_delivers_reminder_when_it_becomes_due(self) -> None:
        realtime_service = CalendarService(
            SqlAlchemyCalendarUnitOfWorkFactory(self.sessions)
        )
        starts_at = datetime.now(UTC) + timedelta(seconds=2)
        event = await realtime_service.create_event(
            CreateEvent(
                chat_id=self.chat_id,
                title="Событие по живому таймеру",
                starts_at=starts_at.replace(tzinfo=None),
                timezone="UTC",
                reminders=(ReminderDraft(timedelta(seconds=1), "Пора"),),
            )
        )
        reminder_id = event.reminders[0].id
        sender = _Sender(599)
        worker = ReminderWorker(
            self.reminder_uow,
            sender,
            worker_id="live-worker",
            options=ReminderWorkerOptions(poll_interval=0.05),
        )
        task = asyncio.create_task(worker.run())
        try:
            async with asyncio.timeout(4):
                while not sender.deliveries:
                    await asyncio.sleep(0.05)
        finally:
            worker.request_stop()
            await task

        record = await self.reminder_record(reminder_id)
        self.assertEqual(record.status, "sent")
        self.assertEqual(record.telegram_message_id, 599)

    async def test_restart_recovers_expired_lease_and_delivers(self) -> None:
        reminder_id = await self.create_due_reminder()
        async with self.reminder_uow() as unit_of_work:
            reserved = await unit_of_work.queue.reserve_due(
                now=NOW,
                limit=10,
                worker_id="crashed-worker",
            )
        self.assertEqual([item.reminder_id for item in reserved], [reminder_id])

        restarted_at = NOW + timedelta(minutes=6)
        worker = ReminderWorker(
            self.reminder_uow,
            _Sender(502),
            worker_id="replacement-worker",
            options=ReminderWorkerOptions(lease_timeout=timedelta(minutes=5)),
            now=lambda: restarted_at,
        )

        report = await worker.run_once()
        record = await self.reminder_record(reminder_id)

        self.assertEqual(report.recovered, 1)
        self.assertEqual(report.sent, 1)
        self.assertEqual(record.status, "sent")
        self.assertEqual(record.attempts, 2)
        self.assertEqual(record.telegram_message_id, 502)

    async def test_temporary_failure_is_retried_after_backoff(self) -> None:
        reminder_id = await self.create_due_reminder()
        clock = [NOW]
        sender = _Sender(TemporaryReminderDeliveryError("network"), 503)
        worker = ReminderWorker(
            self.reminder_uow,
            sender,
            worker_id="worker-a",
            options=ReminderWorkerOptions(
                retry_base_delay=timedelta(seconds=30),
                retry_jitter_ratio=0,
            ),
            now=lambda: clock[0],
        )

        first = await worker.run_once()
        pending = await self.reminder_record(reminder_id)
        clock[0] = NOW + timedelta(seconds=30)
        second = await worker.run_once()
        sent = await self.reminder_record(reminder_id)

        self.assertEqual(first.rescheduled, 1)
        self.assertEqual(pending.status, "pending")
        self.assertEqual(pending.next_attempt_at, clock[0])
        self.assertEqual(pending.last_error, "network")
        self.assertEqual(second.sent, 1)
        self.assertEqual(sent.status, "sent")
        self.assertEqual(sent.attempts, 2)

    async def test_permanent_failure_is_not_retried(self) -> None:
        reminder_id = await self.create_due_reminder()
        worker = ReminderWorker(
            self.reminder_uow,
            _Sender(PermanentReminderDeliveryError("bot blocked")),
            worker_id="worker-a",
            now=lambda: NOW,
        )

        report = await worker.run_once()
        record = await self.reminder_record(reminder_id)

        self.assertEqual(report.failed, 1)
        self.assertEqual(record.status, "failed")
        self.assertEqual(record.last_error, "bot blocked")

    async def test_skip_locked_prevents_duplicate_reservation(self) -> None:
        reminder_id = await self.create_due_reminder()
        first_reserved = asyncio.Event()
        release_first = asyncio.Event()

        async def reserve_first() -> list[ReminderDelivery]:
            async with self.reminder_uow() as unit_of_work:
                result = await unit_of_work.queue.reserve_due(
                    now=NOW,
                    limit=10,
                    worker_id="worker-a",
                )
                first_reserved.set()
                await release_first.wait()
                return result

        first_task = asyncio.create_task(reserve_first())
        await asyncio.wait_for(first_reserved.wait(), timeout=1)
        async with self.reminder_uow() as unit_of_work:
            second = await unit_of_work.queue.reserve_due(
                now=NOW,
                limit=10,
                worker_id="worker-b",
            )
        release_first.set()
        first = await asyncio.wait_for(first_task, timeout=1)

        self.assertEqual([item.reminder_id for item in first], [reminder_id])
        self.assertEqual(second, [])


if __name__ == "__main__":
    unittest.main()
