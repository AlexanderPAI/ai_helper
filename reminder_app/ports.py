"""Infrastructure-independent ports used by the reminder worker."""

from __future__ import annotations

from datetime import datetime
from types import TracebackType
from typing import Protocol, Self
from uuid import UUID

from .domain import ReminderDelivery


class ReminderDeliveryError(RuntimeError):
    """Base error raised by an external reminder sender."""


class TemporaryReminderDeliveryError(ReminderDeliveryError):
    """A delivery failure that may succeed on a later attempt."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PermanentReminderDeliveryError(ReminderDeliveryError):
    """A delivery failure that must not be retried."""


class ReminderSender(Protocol):
    """Deliver one reminder through an external messaging system."""

    async def send(self, reminder: ReminderDelivery) -> int: ...


class ReminderQueue(Protocol):
    """Atomic persistence operations required by the worker."""

    async def reserve_due(
        self,
        *,
        now: datetime,
        limit: int,
        worker_id: str,
    ) -> list[ReminderDelivery]: ...

    async def recover_expired(
        self,
        *,
        now: datetime,
        locked_before: datetime,
    ) -> int: ...

    async def mark_sent(
        self,
        reminder_id: UUID,
        *,
        worker_id: str,
        sent_at: datetime,
        external_message_id: int,
    ) -> bool: ...

    async def reschedule(
        self,
        reminder_id: UUID,
        *,
        worker_id: str,
        next_attempt_at: datetime,
        error: str,
    ) -> bool: ...

    async def mark_failed(
        self,
        reminder_id: UUID,
        *,
        worker_id: str,
        error: str,
    ) -> bool: ...


class ReminderUnitOfWork(Protocol):
    """One atomic reminder queue transaction."""

    queue: ReminderQueue

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class ReminderUnitOfWorkFactory(Protocol):
    """Create an isolated reminder queue transaction."""

    def __call__(self) -> ReminderUnitOfWork: ...


__all__ = [
    "PermanentReminderDeliveryError",
    "ReminderDeliveryError",
    "ReminderQueue",
    "ReminderSender",
    "ReminderUnitOfWork",
    "ReminderUnitOfWorkFactory",
    "TemporaryReminderDeliveryError",
]
