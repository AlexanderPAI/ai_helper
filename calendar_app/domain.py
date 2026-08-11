"""Domain values used to manage calendars without Telegram or an LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final
from uuid import UUID


class CalendarEventStatus(StrEnum):
    ACTIVE = "active"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class CalendarReminderStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    SENT = "sent"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class CalendarSettings:
    chat_id: int
    timezone: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class CalendarReminder:
    id: UUID
    event_id: UUID
    remind_at: datetime
    message_text: str
    status: CalendarReminderStatus
    attempts: int
    next_attempt_at: datetime
    locked_at: datetime | None
    locked_by: str | None
    sent_at: datetime | None
    telegram_message_id: int | None
    last_error: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    id: UUID
    chat_id: int
    title: str
    description: str | None
    starts_at: datetime
    source_timezone: str
    status: CalendarEventStatus
    created_by_user_id: int | None
    created_by_display_name: str | None
    created_at: datetime
    updated_at: datetime
    cancelled_at: datetime | None
    version: int
    reminders: tuple[CalendarReminder, ...] = ()


@dataclass(frozen=True, slots=True)
class CalendarEventCursor:
    starts_at: datetime
    event_id: UUID


@dataclass(frozen=True, slots=True)
class CalendarEventPage:
    events: tuple[CalendarEvent, ...]
    next_cursor: CalendarEventCursor | None


@dataclass(frozen=True, slots=True)
class ReminderDraft:
    """A reminder requested at an offset before an event."""

    offset: timedelta
    message_text: str


@dataclass(frozen=True, slots=True)
class CreateEvent:
    chat_id: int
    title: str
    starts_at: datetime
    timezone: str
    description: str | None = None
    created_by_user_id: int | None = None
    created_by_display_name: str | None = None
    reminders: tuple[ReminderDraft, ...] = field(default_factory=tuple)


class _UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final = _UnsetType()


@dataclass(frozen=True, slots=True)
class UpdateEvent:
    """Partial event update; UNSET leaves nullable fields unchanged."""

    chat_id: int
    event_id: UUID
    expected_version: int
    title: str | _UnsetType = UNSET
    description: str | None | _UnsetType = UNSET
    starts_at: datetime | _UnsetType = UNSET
    timezone: str | _UnsetType = UNSET
    reminders: tuple[ReminderDraft, ...] | _UnsetType = UNSET


@dataclass(frozen=True, slots=True)
class AddEventReminders:
    chat_id: int
    event_id: UUID
    expected_version: int
    reminders: tuple[ReminderDraft, ...]
