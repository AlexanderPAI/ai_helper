"""Transactional calendar use cases and business rules."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .domain import (
    UNSET,
    CalendarEvent,
    CalendarEventStatus,
    CalendarReminderStatus,
    CalendarSettings,
    CreateEvent,
    ReminderDraft,
    UpdateEvent,
)
from .errors import CalendarNotFoundError, CalendarValidationError
from .repository import CalendarRepository


@dataclass(frozen=True, slots=True)
class CalendarServiceOptions:
    max_title_length: int = 255
    max_description_length: int = 4000
    max_reminder_text_length: int = 4096
    max_reminders_per_event: int = 5
    max_list_limit: int = 100


class CalendarService:
    """Manage chat calendars through atomic, validated use cases."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        options: CalendarServiceOptions | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.options = options or CalendarServiceOptions()
        self._now = now or (lambda: datetime.now(UTC))

    async def get_settings(self, chat_id: int) -> CalendarSettings | None:
        self._validate_chat_id(chat_id)
        async with self.session_factory() as session:
            return await CalendarRepository(session).get_settings(chat_id)

    async def set_timezone(self, chat_id: int, timezone: str) -> CalendarSettings:
        self._validate_chat_id(chat_id)
        timezone = self._timezone(timezone).key
        async with self.session_factory() as session, session.begin():
            repository = CalendarRepository(session)
            current = await repository.get_settings(chat_id)
            if current is None:
                return await repository.ensure_settings(chat_id, timezone)
            updated = await repository.update_timezone(chat_id, timezone)
            if updated is None:  # pragma: no cover - protected by transaction
                raise CalendarNotFoundError("calendar settings not found")
            return updated

    async def create_event(self, request: CreateEvent) -> CalendarEvent:
        self._validate_chat_id(request.chat_id)
        now = self._utc_now()
        timezone = self._timezone(request.timezone)
        starts_at = self._to_utc(request.starts_at, timezone)
        if starts_at <= now:
            raise CalendarValidationError("event must start in the future")

        title = self._title(request.title)
        description = self._description(request.description)
        reminder_rows = self._reminder_rows(starts_at, request.reminders, now)
        event_id = uuid4()

        async with self.session_factory() as session, session.begin():
            repository = CalendarRepository(session)
            await repository.ensure_settings(request.chat_id, timezone.key)
            return await repository.create_event(
                event_id=event_id,
                chat_id=request.chat_id,
                title=title,
                description=description,
                starts_at=starts_at,
                source_timezone=timezone.key,
                created_by_user_id=request.created_by_user_id,
                created_by_display_name=self._display_name(
                    request.created_by_display_name
                ),
                reminders=reminder_rows,
            )

    async def get_event(self, chat_id: int, event_id: UUID) -> CalendarEvent:
        self._validate_chat_id(chat_id)
        async with self.session_factory() as session:
            event = await CalendarRepository(session).get_event(chat_id, event_id)
        if event is None:
            raise CalendarNotFoundError("event not found in this chat")
        return event

    async def list_events(
        self,
        chat_id: int,
        *,
        starts_from: datetime | None = None,
        starts_until: datetime | None = None,
        statuses: Sequence[CalendarEventStatus] = (CalendarEventStatus.ACTIVE,),
        limit: int = 50,
    ) -> list[CalendarEvent]:
        self._validate_chat_id(chat_id)
        if not 1 <= limit <= self.options.max_list_limit:
            raise CalendarValidationError(
                f"limit must be between 1 and {self.options.max_list_limit}"
            )
        if not statuses:
            raise CalendarValidationError("at least one event status is required")
        normalized_from = self._require_aware_utc(starts_from, "starts_from")
        normalized_until = self._require_aware_utc(starts_until, "starts_until")
        if (
            normalized_from is not None
            and normalized_until is not None
            and normalized_from >= normalized_until
        ):
            raise CalendarValidationError("starts_from must be before starts_until")

        async with self.session_factory() as session:
            return await CalendarRepository(session).list_events(
                chat_id,
                starts_from=normalized_from,
                starts_until=normalized_until,
                statuses=statuses,
                limit=limit,
            )

    async def update_event(self, request: UpdateEvent) -> CalendarEvent:
        self._validate_chat_id(request.chat_id)
        if request.expected_version < 1:
            raise CalendarValidationError("expected_version must be positive")
        now = self._utc_now()

        async with self.session_factory() as session, session.begin():
            repository = CalendarRepository(session)
            current = await repository.get_event(request.chat_id, request.event_id)
            if current is None:
                raise CalendarNotFoundError("event not found in this chat")
            if current.status is not CalendarEventStatus.ACTIVE:
                raise CalendarValidationError("only active events can be updated")

            values: dict[str, object] = {}
            if request.title is not UNSET:
                values["title"] = self._title(request.title)
            if request.description is not UNSET:
                values["description"] = self._description(request.description)

            starts_at = current.starts_at
            source_timezone = current.source_timezone
            start_changed = request.starts_at is not UNSET
            if request.timezone is not UNSET and not start_changed:
                raise CalendarValidationError(
                    "timezone can only be changed together with starts_at"
                )
            if start_changed:
                timezone_name = (
                    request.timezone
                    if request.timezone is not UNSET
                    else current.source_timezone
                )
                timezone = self._timezone(timezone_name)
                starts_at = self._to_utc(request.starts_at, timezone)
                if starts_at <= now:
                    raise CalendarValidationError("event must start in the future")
                source_timezone = timezone.key
                values["starts_at"] = starts_at
                values["source_timezone"] = source_timezone

            replace_reminders = request.reminders is not UNSET or start_changed
            reminder_drafts: Sequence[ReminderDraft]
            if request.reminders is not UNSET:
                reminder_drafts = request.reminders
            elif start_changed:
                reminder_drafts = self._open_reminder_drafts(current)
            else:
                reminder_drafts = ()

            reminder_rows = (
                self._reminder_rows(starts_at, reminder_drafts, now)
                if replace_reminders
                else []
            )
            await repository.update_event(
                request.chat_id,
                request.event_id,
                request.expected_version,
                values,
            )
            if replace_reminders:
                await repository.cancel_open_reminders(request.event_id)
                await repository.add_reminders(request.event_id, reminder_rows)

            updated = await repository.get_event(request.chat_id, request.event_id)
            if updated is None:  # pragma: no cover - protected by transaction
                raise CalendarNotFoundError("event not found in this chat")
            return updated

    async def cancel_event(
        self,
        chat_id: int,
        event_id: UUID,
        *,
        expected_version: int,
    ) -> CalendarEvent:
        self._validate_chat_id(chat_id)
        if expected_version < 1:
            raise CalendarValidationError("expected_version must be positive")
        now = self._utc_now()

        async with self.session_factory() as session, session.begin():
            repository = CalendarRepository(session)
            current = await repository.get_event(chat_id, event_id)
            if current is None:
                raise CalendarNotFoundError("event not found in this chat")
            if current.status is not CalendarEventStatus.ACTIVE:
                raise CalendarValidationError("only active events can be cancelled")
            await repository.update_event(
                chat_id,
                event_id,
                expected_version,
                {
                    "status": CalendarEventStatus.CANCELLED.value,
                    "cancelled_at": now,
                },
            )
            await repository.cancel_open_reminders(event_id)
            cancelled = await repository.get_event(chat_id, event_id)
            if cancelled is None:  # pragma: no cover - protected by transaction
                raise CalendarNotFoundError("event not found in this chat")
            return cancelled

    def _reminder_rows(
        self,
        starts_at: datetime,
        reminders: Sequence[ReminderDraft],
        now: datetime,
    ) -> list[tuple[UUID, datetime, str]]:
        if len(reminders) > self.options.max_reminders_per_event:
            raise CalendarValidationError(
                f"an event can have at most {self.options.max_reminders_per_event} "
                "reminders"
            )

        rows: list[tuple[UUID, datetime, str]] = []
        seen_times: set[datetime] = set()
        for reminder in reminders:
            if reminder.offset < timedelta(0):
                raise CalendarValidationError("reminder offset cannot be negative")
            remind_at = starts_at - reminder.offset
            if remind_at < now:
                raise CalendarValidationError("reminder time cannot be in the past")
            if remind_at in seen_times:
                raise CalendarValidationError("reminder times must be unique")
            seen_times.add(remind_at)
            rows.append(
                (uuid4(), remind_at, self._reminder_text(reminder.message_text))
            )
        return rows

    @staticmethod
    def _open_reminder_drafts(event: CalendarEvent) -> tuple[ReminderDraft, ...]:
        return tuple(
            ReminderDraft(
                offset=event.starts_at - reminder.remind_at,
                message_text=reminder.message_text,
            )
            for reminder in event.reminders
            if reminder.status
            in (CalendarReminderStatus.PENDING, CalendarReminderStatus.PROCESSING)
        )

    def _title(self, value: str) -> str:
        if not isinstance(value, str):
            raise CalendarValidationError("title must be a string")
        value = value.strip()
        if not value:
            raise CalendarValidationError("title cannot be blank")
        if len(value) > self.options.max_title_length:
            raise CalendarValidationError("title is too long")
        return value

    def _description(self, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise CalendarValidationError("description must be a string or null")
        value = value.strip()
        if len(value) > self.options.max_description_length:
            raise CalendarValidationError("description is too long")
        return value or None

    def _reminder_text(self, value: str) -> str:
        if not isinstance(value, str):
            raise CalendarValidationError("reminder text must be a string")
        value = value.strip()
        if not value:
            raise CalendarValidationError("reminder text cannot be blank")
        if len(value) > self.options.max_reminder_text_length:
            raise CalendarValidationError("reminder text is too long")
        return value

    @staticmethod
    def _display_name(value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise CalendarValidationError("display name must be a string or null")
        value = value.strip()
        return value[:255] or None

    @staticmethod
    def _validate_chat_id(chat_id: int) -> None:
        if isinstance(chat_id, bool) or not isinstance(chat_id, int):
            raise CalendarValidationError("chat_id must be an integer")

    @staticmethod
    def _timezone(value: str) -> ZoneInfo:
        if not isinstance(value, str) or not value.strip():
            raise CalendarValidationError("timezone must be an IANA timezone name")
        try:
            return ZoneInfo(value.strip())
        except ZoneInfoNotFoundError as error:
            raise CalendarValidationError("unknown IANA timezone") from error

    @staticmethod
    def _to_utc(value: datetime, timezone: ZoneInfo) -> datetime:
        if not isinstance(value, datetime):
            raise CalendarValidationError("starts_at must be a datetime")
        if value.tzinfo is not None:
            if value.utcoffset() is None:
                raise CalendarValidationError("starts_at has an invalid timezone")
            return value.astimezone(UTC)

        candidates = []
        for fold in (0, 1):
            candidate = value.replace(tzinfo=timezone, fold=fold)
            roundtrip = candidate.astimezone(UTC).astimezone(timezone)
            if roundtrip.replace(tzinfo=None) == value:
                candidates.append(candidate)
        if not candidates:
            raise CalendarValidationError("local event time does not exist")
        if len({candidate.utcoffset() for candidate in candidates}) > 1:
            raise CalendarValidationError("local event time is ambiguous")
        return candidates[0].astimezone(UTC)

    @staticmethod
    def _require_aware_utc(value: datetime | None, field_name: str) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise CalendarValidationError(f"{field_name} must include a timezone")
        return value.astimezone(UTC)

    def _utc_now(self) -> datetime:
        value = self._now()
        if value.tzinfo is None or value.utcoffset() is None:
            raise RuntimeError("calendar clock must return a timezone-aware datetime")
        return value.astimezone(UTC)
