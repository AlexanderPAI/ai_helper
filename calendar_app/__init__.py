"""Application-independent chat calendar domain and persistence API."""

from .domain import (
    UNSET,
    CalendarEvent,
    CalendarEventStatus,
    CalendarReminder,
    CalendarReminderStatus,
    CalendarSettings,
    CreateEvent,
    ReminderDraft,
    UpdateEvent,
)
from .errors import (
    CalendarConflictError,
    CalendarError,
    CalendarNotFoundError,
    CalendarValidationError,
)
from .repository import CalendarRepository
from .service import CalendarService, CalendarServiceOptions

__all__ = [
    "UNSET",
    "CalendarConflictError",
    "CalendarError",
    "CalendarEvent",
    "CalendarEventStatus",
    "CalendarNotFoundError",
    "CalendarReminder",
    "CalendarReminderStatus",
    "CalendarRepository",
    "CalendarService",
    "CalendarServiceOptions",
    "CalendarSettings",
    "CalendarValidationError",
    "CreateEvent",
    "ReminderDraft",
    "UpdateEvent",
]
