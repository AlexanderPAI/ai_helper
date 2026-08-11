"""Domain values required to deliver a persisted reminder."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ReminderDelivery:
    """One atomically reserved reminder ready for an external sender."""

    reminder_id: UUID
    event_id: UUID
    chat_id: int
    event_title: str
    event_starts_at: datetime
    event_timezone: str
    message_text: str
    remind_at: datetime
    attempts: int


__all__ = ["ReminderDelivery"]
