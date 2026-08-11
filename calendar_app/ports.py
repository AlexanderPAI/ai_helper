"""Infrastructure-independent ports required by calendar use cases."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from types import TracebackType
from typing import Protocol, Self
from uuid import UUID

from .domain import (
    CalendarEvent,
    CalendarEventCursor,
    CalendarEventStatus,
    CalendarSettings,
)

ReminderRow = tuple[UUID, datetime, str]


class CalendarRepository(Protocol):
    """Storage operations used by the calendar application layer."""

    async def get_settings(self, chat_id: int) -> CalendarSettings | None: ...

    async def ensure_settings(
        self, chat_id: int, timezone: str
    ) -> CalendarSettings: ...

    async def update_timezone(
        self, chat_id: int, timezone: str
    ) -> CalendarSettings | None: ...

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
    ) -> CalendarEvent: ...

    async def get_event(self, chat_id: int, event_id: UUID) -> CalendarEvent | None: ...

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
    ) -> list[CalendarEvent]: ...

    async def update_event(
        self,
        chat_id: int,
        event_id: UUID,
        expected_version: int,
        values: dict[str, object],
    ) -> int: ...

    async def cancel_open_reminders(self, event_id: UUID) -> None: ...

    async def add_reminders(
        self, event_id: UUID, reminders: Sequence[ReminderRow]
    ) -> None: ...


class CalendarUnitOfWork(Protocol):
    """Atomic persistence boundary supplied to one calendar use case."""

    repository: CalendarRepository

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class CalendarUnitOfWorkFactory(Protocol):
    """Create an isolated unit of work without exposing its infrastructure."""

    def __call__(self) -> CalendarUnitOfWork: ...


__all__ = [
    "CalendarRepository",
    "CalendarUnitOfWork",
    "CalendarUnitOfWorkFactory",
    "ReminderRow",
]
