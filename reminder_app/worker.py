"""Reliable polling worker for due calendar reminders."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .domain import ReminderDelivery
from .ports import (
    PermanentReminderDeliveryError,
    ReminderSender,
    ReminderUnitOfWorkFactory,
    TemporaryReminderDeliveryError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReminderWorkerOptions:
    poll_interval: float = 10.0
    batch_size: int = 20
    lease_timeout: timedelta = timedelta(minutes=5)
    max_attempts: int = 5
    retry_base_delay: timedelta = timedelta(seconds=30)
    retry_max_delay: timedelta = timedelta(hours=1)
    retry_jitter_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.lease_timeout <= timedelta(0):
            raise ValueError("lease_timeout must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.retry_base_delay <= timedelta(0):
            raise ValueError("retry_base_delay must be positive")
        if self.retry_max_delay < self.retry_base_delay:
            raise ValueError("retry_max_delay must not be less than base delay")
        if not 0 <= self.retry_jitter_ratio <= 1:
            raise ValueError("retry_jitter_ratio must be between zero and one")


@dataclass(frozen=True, slots=True)
class ReminderWorkerRun:
    """Counts produced by one polling iteration."""

    recovered: int = 0
    reserved: int = 0
    sent: int = 0
    rescheduled: int = 0
    failed: int = 0


class ReminderWorker:
    """Reserve, deliver and persist reminder outcomes without chat context."""

    def __init__(
        self,
        unit_of_work_factory: ReminderUnitOfWorkFactory,
        sender: ReminderSender,
        *,
        worker_id: str,
        options: ReminderWorkerOptions | None = None,
        now: Callable[[], datetime] | None = None,
        random_value: Callable[[], float] | None = None,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be blank")
        self.unit_of_work_factory = unit_of_work_factory
        self.sender = sender
        self.worker_id = worker_id.strip()
        self.options = options or ReminderWorkerOptions()
        self._now = now or (lambda: datetime.now(UTC))
        self._random_value = random_value or random.random
        self._stop_requested = asyncio.Event()

    async def run(self) -> None:
        """Poll until graceful shutdown is requested."""
        logger.info("Reminder worker started worker_id=%s", self.worker_id)
        try:
            while not self._stop_requested.is_set():
                try:
                    report = await self.run_once()
                    if report.reserved or report.recovered:
                        logger.info(
                            "Reminder worker iteration worker_id=%s recovered=%d "
                            "reserved=%d sent=%d rescheduled=%d failed=%d",
                            self.worker_id,
                            report.recovered,
                            report.reserved,
                            report.sent,
                            report.rescheduled,
                            report.failed,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Reminder worker iteration failed worker_id=%s",
                        self.worker_id,
                    )
                await self._wait_for_next_poll()
        finally:
            logger.info("Reminder worker stopped worker_id=%s", self.worker_id)

    def request_stop(self) -> None:
        """Ask the polling loop to finish after its current batch."""
        self._stop_requested.set()

    async def run_once(self) -> ReminderWorkerRun:
        """Recover leases, reserve one batch and deliver it."""
        now = self._utc_now()
        async with self.unit_of_work_factory() as unit_of_work:
            recovered = await unit_of_work.queue.recover_expired(
                now=now,
                locked_before=now - self.options.lease_timeout,
            )
            reminders = await unit_of_work.queue.reserve_due(
                now=now,
                limit=self.options.batch_size,
                worker_id=self.worker_id,
            )

        sent = 0
        rescheduled = 0
        failed = 0
        for reminder in reminders:
            outcome = await self._deliver(reminder)
            sent += outcome == "sent"
            rescheduled += outcome == "rescheduled"
            failed += outcome == "failed"
        return ReminderWorkerRun(
            recovered=recovered,
            reserved=len(reminders),
            sent=sent,
            rescheduled=rescheduled,
            failed=failed,
        )

    async def _deliver(self, reminder: ReminderDelivery) -> str:
        logger.info(
            "Reminder delivery started reminder_id=%s event_id=%s chat_id=%s "
            "attempt=%d scheduled_at=%s",
            reminder.reminder_id,
            reminder.event_id,
            reminder.chat_id,
            reminder.attempts,
            reminder.remind_at.isoformat(),
        )
        try:
            external_message_id = await self.sender.send(reminder)
        except asyncio.CancelledError:
            raise
        except PermanentReminderDeliveryError as error:
            logger.error(
                "Reminder delivery permanently rejected reminder_id=%s error=%s",
                reminder.reminder_id,
                self._error_text(error),
            )
            await self._mark_failed(reminder, error)
            return "failed"
        except TemporaryReminderDeliveryError as error:
            logger.warning(
                "Reminder delivery temporarily failed reminder_id=%s error=%s",
                reminder.reminder_id,
                self._error_text(error),
            )
            return await self._retry_or_fail(reminder, error)
        except Exception as error:
            logger.exception(
                "Unexpected reminder sender error reminder_id=%s",
                reminder.reminder_id,
            )
            return await self._retry_or_fail(
                reminder,
                TemporaryReminderDeliveryError(str(error) or type(error).__name__),
            )

        async with self.unit_of_work_factory() as unit_of_work:
            changed = await unit_of_work.queue.mark_sent(
                reminder.reminder_id,
                worker_id=self.worker_id,
                sent_at=self._utc_now(),
                external_message_id=external_message_id,
            )
        if not changed:
            logger.warning(
                "Reminder delivery result lost its lease reminder_id=%s worker_id=%s",
                reminder.reminder_id,
                self.worker_id,
            )
        else:
            logger.info(
                "Reminder delivery completed reminder_id=%s telegram_message_id=%s",
                reminder.reminder_id,
                external_message_id,
            )
        return "sent"

    async def _retry_or_fail(
        self,
        reminder: ReminderDelivery,
        error: TemporaryReminderDeliveryError,
    ) -> str:
        if reminder.attempts >= self.options.max_attempts:
            await self._mark_failed(reminder, error)
            return "failed"

        delay = self._backoff(reminder.attempts, error.retry_after)
        async with self.unit_of_work_factory() as unit_of_work:
            await unit_of_work.queue.reschedule(
                reminder.reminder_id,
                worker_id=self.worker_id,
                next_attempt_at=self._utc_now() + delay,
                error=self._error_text(error),
            )
        return "rescheduled"

    async def _mark_failed(
        self,
        reminder: ReminderDelivery,
        error: BaseException,
    ) -> None:
        async with self.unit_of_work_factory() as unit_of_work:
            await unit_of_work.queue.mark_failed(
                reminder.reminder_id,
                worker_id=self.worker_id,
                error=self._error_text(error),
            )

    def _backoff(self, attempts: int, retry_after: float | None) -> timedelta:
        exponent = max(attempts - 1, 0)
        base_seconds = self.options.retry_base_delay.total_seconds() * (2**exponent)
        capped_seconds = min(
            base_seconds,
            self.options.retry_max_delay.total_seconds(),
        )
        jitter = capped_seconds * self.options.retry_jitter_ratio * self._random_value()
        seconds = capped_seconds + jitter
        if retry_after is not None:
            seconds = max(seconds, retry_after)
        return timedelta(seconds=seconds)

    async def _wait_for_next_poll(self) -> None:
        if self._stop_requested.is_set():
            return
        try:
            await asyncio.wait_for(
                self._stop_requested.wait(), timeout=self.options.poll_interval
            )
        except TimeoutError:
            pass

    @staticmethod
    def _error_text(error: BaseException) -> str:
        return (str(error) or type(error).__name__)[:4000]

    def _utc_now(self) -> datetime:
        value = self._now()
        if value.tzinfo is None or value.utcoffset() is None:
            raise RuntimeError("reminder clock must return a timezone-aware datetime")
        return value.astimezone(UTC)


__all__ = ["ReminderWorker", "ReminderWorkerOptions", "ReminderWorkerRun"]
