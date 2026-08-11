"""SQLAlchemy adapters for the calendar persistence ports."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from types import TracebackType
from typing import Self
from uuid import UUID

from sqlalchemy import Select, and_, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncSessionTransaction,
    async_sessionmaker,
)
from sqlalchemy.orm import selectinload

from calendar_app.domain import (
    CalendarEvent,
    CalendarEventCursor,
    CalendarEventStatus,
    CalendarReminder,
    CalendarReminderStatus,
    CalendarSettings,
)
from calendar_app.errors import CalendarConflictError
from calendar_app.ports import ReminderRow

from .models import CalendarChat as CalendarChatRecord
from .models import CalendarEvent as CalendarEventRecord
from .models import EventReminder as EventReminderRecord


class SqlAlchemyCalendarRepository:
    """PostgreSQL calendar operations executed in a caller-owned transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_settings(self, chat_id: int) -> CalendarSettings | None:
        record = await self.session.get(CalendarChatRecord, chat_id)
        return self._settings(record) if record is not None else None

    async def ensure_settings(self, chat_id: int, timezone: str) -> CalendarSettings:
        statement = (
            insert(CalendarChatRecord)
            .values(chat_id=chat_id, timezone=timezone)
            .on_conflict_do_nothing(index_elements=[CalendarChatRecord.chat_id])
        )
        await self.session.execute(statement)
        record = await self.session.get(CalendarChatRecord, chat_id)
        if record is None:  # pragma: no cover - guarded by the same transaction
            raise RuntimeError("calendar settings were not created")
        return self._settings(record)

    async def update_timezone(
        self, chat_id: int, timezone: str
    ) -> CalendarSettings | None:
        statement = (
            update(CalendarChatRecord)
            .where(CalendarChatRecord.chat_id == chat_id)
            .values(timezone=timezone)
            .returning(CalendarChatRecord)
        )
        record = (await self.session.execute(statement)).scalar_one_or_none()
        return self._settings(record) if record is not None else None

    async def create_event(
        self,
        *,
        event_id: UUID,
        chat_id: int,
        title: str,
        description: str | None,
        starts_at: datetime,
        source_timezone: str,
        created_by_user_id: int | None,
        created_by_display_name: str | None,
        reminders: Sequence[ReminderRow],
    ) -> CalendarEvent:
        record = CalendarEventRecord(
            id=event_id,
            chat_id=chat_id,
            title=title,
            description=description,
            starts_at=starts_at,
            source_timezone=source_timezone,
            status=CalendarEventStatus.ACTIVE.value,
            created_by_user_id=created_by_user_id,
            created_by_display_name=created_by_display_name,
        )
        record.reminders.extend(
            EventReminderRecord(
                id=reminder_id,
                event_id=event_id,
                remind_at=remind_at,
                next_attempt_at=remind_at,
                message_text=message_text,
                status=CalendarReminderStatus.PENDING.value,
            )
            for reminder_id, remind_at, message_text in reminders
        )
        self.session.add(record)
        await self.session.flush()
        await self.session.refresh(record, attribute_names=["reminders"])
        return self._event(record)

    async def get_event(self, chat_id: int, event_id: UUID) -> CalendarEvent | None:
        statement = self._event_query().where(
            CalendarEventRecord.chat_id == chat_id,
            CalendarEventRecord.id == event_id,
        )
        record = (await self.session.execute(statement)).scalar_one_or_none()
        return self._event(record) if record is not None else None

    async def list_events(
        self,
        chat_id: int,
        *,
        starts_from: datetime | None = None,
        starts_until: datetime | None = None,
        statuses: Sequence[CalendarEventStatus] = (CalendarEventStatus.ACTIVE,),
        search_terms: Sequence[str] = (),
        after: CalendarEventCursor | None = None,
        limit: int = 50,
    ) -> list[CalendarEvent]:
        statement = self._event_query().where(
            CalendarEventRecord.chat_id == chat_id,
            CalendarEventRecord.status.in_(status.value for status in statuses),
        )
        if starts_from is not None:
            statement = statement.where(CalendarEventRecord.starts_at >= starts_from)
        if starts_until is not None:
            statement = statement.where(CalendarEventRecord.starts_at < starts_until)
        for term in search_terms:
            pattern = f"%{self._escape_like(term)}%"
            statement = statement.where(
                or_(
                    CalendarEventRecord.title.ilike(pattern, escape="\\"),
                    CalendarEventRecord.description.ilike(pattern, escape="\\"),
                )
            )
        if after is not None:
            statement = statement.where(
                or_(
                    CalendarEventRecord.starts_at > after.starts_at,
                    and_(
                        CalendarEventRecord.starts_at == after.starts_at,
                        CalendarEventRecord.id > after.event_id,
                    ),
                )
            )
        statement = statement.order_by(
            CalendarEventRecord.starts_at, CalendarEventRecord.id
        ).limit(limit)
        records = (await self.session.execute(statement)).scalars().all()
        return [self._event(record) for record in records]

    @staticmethod
    def _escape_like(value: str) -> str:
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    async def update_event(
        self,
        chat_id: int,
        event_id: UUID,
        expected_version: int,
        values: dict[str, object],
    ) -> int:
        statement = (
            update(CalendarEventRecord)
            .where(
                CalendarEventRecord.chat_id == chat_id,
                CalendarEventRecord.id == event_id,
                CalendarEventRecord.version == expected_version,
            )
            .values(**values, version=CalendarEventRecord.version + 1)
            .returning(CalendarEventRecord.version)
        )
        new_version = (await self.session.execute(statement)).scalar_one_or_none()
        if new_version is None:
            raise CalendarConflictError("event version is stale")
        return new_version

    async def cancel_open_reminders(self, event_id: UUID) -> None:
        statement = (
            update(EventReminderRecord)
            .where(
                EventReminderRecord.event_id == event_id,
                EventReminderRecord.status.in_(
                    (
                        CalendarReminderStatus.PENDING.value,
                        CalendarReminderStatus.PROCESSING.value,
                    )
                ),
            )
            .values(
                status=CalendarReminderStatus.CANCELLED.value,
                locked_at=None,
                locked_by=None,
            )
        )
        await self.session.execute(statement)

    async def add_reminders(
        self, event_id: UUID, reminders: Sequence[ReminderRow]
    ) -> None:
        self.session.add_all(
            EventReminderRecord(
                id=reminder_id,
                event_id=event_id,
                remind_at=remind_at,
                next_attempt_at=remind_at,
                message_text=message_text,
                status=CalendarReminderStatus.PENDING.value,
            )
            for reminder_id, remind_at, message_text in reminders
        )
        await self.session.flush()

    @staticmethod
    def _event_query() -> Select[tuple[CalendarEventRecord]]:
        return (
            select(CalendarEventRecord)
            .options(selectinload(CalendarEventRecord.reminders))
            .execution_options(populate_existing=True)
        )

    @staticmethod
    def _settings(record: CalendarChatRecord) -> CalendarSettings:
        return CalendarSettings(
            chat_id=record.chat_id,
            timezone=record.timezone,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @classmethod
    def _event(cls, record: CalendarEventRecord) -> CalendarEvent:
        return CalendarEvent(
            id=record.id,
            chat_id=record.chat_id,
            title=record.title,
            description=record.description,
            starts_at=record.starts_at,
            source_timezone=record.source_timezone,
            status=CalendarEventStatus(record.status),
            created_by_user_id=record.created_by_user_id,
            created_by_display_name=record.created_by_display_name,
            created_at=record.created_at,
            updated_at=record.updated_at,
            cancelled_at=record.cancelled_at,
            version=record.version,
            reminders=tuple(
                cls._reminder(reminder)
                for reminder in sorted(
                    record.reminders, key=lambda item: item.remind_at
                )
            ),
        )

    @staticmethod
    def _reminder(record: EventReminderRecord) -> CalendarReminder:
        return CalendarReminder(
            id=record.id,
            event_id=record.event_id,
            remind_at=record.remind_at,
            message_text=record.message_text,
            status=CalendarReminderStatus(record.status),
            attempts=record.attempts,
            next_attempt_at=record.next_attempt_at,
            locked_at=record.locked_at,
            locked_by=record.locked_by,
            sent_at=record.sent_at,
            telegram_message_id=record.telegram_message_id,
            last_error=record.last_error,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class SqlAlchemyCalendarUnitOfWork:
    """One SQLAlchemy session and transaction for a calendar use case."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None
        self._transaction: AsyncSessionTransaction | None = None
        self.repository: SqlAlchemyCalendarRepository

    async def __aenter__(self) -> Self:
        self._session = self._session_factory()
        await self._session.__aenter__()
        self._transaction = self._session.begin()
        await self._transaction.__aenter__()
        self.repository = SqlAlchemyCalendarRepository(self._session)
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


class SqlAlchemyCalendarUnitOfWorkFactory:
    """Create SQLAlchemy-backed calendar units of work."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    def __call__(self) -> SqlAlchemyCalendarUnitOfWork:
        return SqlAlchemyCalendarUnitOfWork(self._session_factory)


__all__ = [
    "SqlAlchemyCalendarRepository",
    "SqlAlchemyCalendarUnitOfWork",
    "SqlAlchemyCalendarUnitOfWorkFactory",
]
