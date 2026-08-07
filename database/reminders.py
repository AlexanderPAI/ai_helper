"""SQLAlchemy adapters for reliable reminder queue operations."""

from __future__ import annotations

from datetime import datetime
from types import TracebackType
from typing import Self
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncSessionTransaction,
    async_sessionmaker,
)

from calendar_app.domain import CalendarEventStatus, CalendarReminderStatus
from reminder_app.domain import ReminderDelivery

from .models import CalendarEvent as CalendarEventRecord
from .models import EventReminder as EventReminderRecord


class SqlAlchemyReminderQueue:
    """PostgreSQL queue operations executed in a caller-owned transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def reserve_due(
        self,
        *,
        now: datetime,
        limit: int,
        worker_id: str,
    ) -> list[ReminderDelivery]:
        statement = (
            select(EventReminderRecord, CalendarEventRecord)
            .join(
                CalendarEventRecord,
                CalendarEventRecord.id == EventReminderRecord.event_id,
            )
            .where(
                EventReminderRecord.status == CalendarReminderStatus.PENDING.value,
                EventReminderRecord.next_attempt_at <= now,
                EventReminderRecord.remind_at <= now,
                CalendarEventRecord.status == CalendarEventStatus.ACTIVE.value,
            )
            .order_by(
                EventReminderRecord.next_attempt_at,
                EventReminderRecord.remind_at,
                EventReminderRecord.id,
            )
            .limit(limit)
            .with_for_update(skip_locked=True, of=EventReminderRecord)
        )
        rows = (await self.session.execute(statement)).all()
        deliveries = []
        for reminder, event in rows:
            reminder.status = CalendarReminderStatus.PROCESSING.value
            reminder.attempts += 1
            reminder.locked_at = now
            reminder.locked_by = worker_id
            deliveries.append(
                ReminderDelivery(
                    reminder_id=reminder.id,
                    event_id=event.id,
                    chat_id=event.chat_id,
                    event_title=event.title,
                    event_starts_at=event.starts_at,
                    event_timezone=event.source_timezone,
                    message_text=reminder.message_text,
                    remind_at=reminder.remind_at,
                    attempts=reminder.attempts,
                )
            )
        await self.session.flush()
        return deliveries

    async def recover_expired(
        self,
        *,
        now: datetime,
        locked_before: datetime,
    ) -> int:
        statement = (
            update(EventReminderRecord)
            .where(
                EventReminderRecord.status == CalendarReminderStatus.PROCESSING.value,
                EventReminderRecord.locked_at <= locked_before,
            )
            .values(
                status=CalendarReminderStatus.PENDING.value,
                next_attempt_at=now,
                locked_at=None,
                locked_by=None,
            )
        )
        result = await self.session.execute(statement)
        return result.rowcount  # type: ignore[attr-defined]

    async def mark_sent(
        self,
        reminder_id: UUID,
        *,
        worker_id: str,
        sent_at: datetime,
        external_message_id: int,
    ) -> bool:
        return await self._finish(
            reminder_id,
            worker_id=worker_id,
            values={
                "status": CalendarReminderStatus.SENT.value,
                "sent_at": sent_at,
                "telegram_message_id": external_message_id,
                "locked_at": None,
                "locked_by": None,
                "last_error": None,
            },
        )

    async def reschedule(
        self,
        reminder_id: UUID,
        *,
        worker_id: str,
        next_attempt_at: datetime,
        error: str,
    ) -> bool:
        return await self._finish(
            reminder_id,
            worker_id=worker_id,
            values={
                "status": CalendarReminderStatus.PENDING.value,
                "next_attempt_at": next_attempt_at,
                "locked_at": None,
                "locked_by": None,
                "last_error": error,
            },
        )

    async def mark_failed(
        self,
        reminder_id: UUID,
        *,
        worker_id: str,
        error: str,
    ) -> bool:
        return await self._finish(
            reminder_id,
            worker_id=worker_id,
            values={
                "status": CalendarReminderStatus.FAILED.value,
                "locked_at": None,
                "locked_by": None,
                "last_error": error,
            },
        )

    async def _finish(
        self,
        reminder_id: UUID,
        *,
        worker_id: str,
        values: dict[str, object],
    ) -> bool:
        statement = (
            update(EventReminderRecord)
            .where(
                EventReminderRecord.id == reminder_id,
                EventReminderRecord.status == CalendarReminderStatus.PROCESSING.value,
                EventReminderRecord.locked_by == worker_id,
            )
            .values(**values)
        )
        result = await self.session.execute(statement)
        return result.rowcount == 1  # type: ignore[attr-defined]


class SqlAlchemyReminderUnitOfWork:
    """One SQLAlchemy session and transaction for reminder queue state."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None
        self._transaction: AsyncSessionTransaction | None = None
        self.queue: SqlAlchemyReminderQueue

    async def __aenter__(self) -> Self:
        self._session = self._session_factory()
        await self._session.__aenter__()
        self._transaction = self._session.begin()
        await self._transaction.__aenter__()
        self.queue = SqlAlchemyReminderQueue(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._transaction is None or self._session is None:
            return None
        try:
            return await self._transaction.__aexit__(exc_type, exc_value, traceback)
        finally:
            await self._session.__aexit__(exc_type, exc_value, traceback)


class SqlAlchemyReminderUnitOfWorkFactory:
    """Create SQLAlchemy-backed reminder queue units of work."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    def __call__(self) -> SqlAlchemyReminderUnitOfWork:
        return SqlAlchemyReminderUnitOfWork(self._session_factory)


__all__ = [
    "SqlAlchemyReminderQueue",
    "SqlAlchemyReminderUnitOfWork",
    "SqlAlchemyReminderUnitOfWorkFactory",
]
