"""SQLAlchemy models for chat calendars and reminder delivery state."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for application tables."""


class TimestampMixin:
    """Server-managed creation and update timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("CURRENT_TIMESTAMP"),
    )


class CalendarChat(TimestampMixin, Base):
    """Persistent calendar settings owned by one Telegram chat."""

    __tablename__ = "calendar_chats"

    chat_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=False
    )
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)

    events: Mapped[list[CalendarEvent]] = relationship(back_populates="chat")


class CalendarEvent(TimestampMixin, Base):
    """A calendar event isolated within its owning Telegram chat."""

    __tablename__ = "calendar_events"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'cancelled', 'completed')",
            name="ck_calendar_events_status",
        ),
        CheckConstraint("version >= 1", name="ck_calendar_events_version"),
        Index("ix_calendar_events_chat_starts_at", "chat_id", "starts_at"),
        Index(
            "ix_calendar_events_chat_status_starts_at",
            "chat_id",
            "status",
            "starts_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    chat_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("calendar_chats.chat_id", ondelete="RESTRICT"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'active'")
    )
    created_by_user_id: Mapped[int | None] = mapped_column(BigInteger)
    created_by_display_name: Mapped[str | None] = mapped_column(String(255))
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )

    chat: Mapped[CalendarChat] = relationship(back_populates="events")
    reminders: Mapped[list[EventReminder]] = relationship(
        back_populates="event", cascade="all, delete-orphan"
    )


class EventReminder(TimestampMixin, Base):
    """A scheduled Telegram notification with its final persisted text."""

    __tablename__ = "event_reminders"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'sent', 'failed', 'cancelled')",
            name="ck_event_reminders_status",
        ),
        CheckConstraint("attempts >= 0", name="ck_event_reminders_attempts"),
        CheckConstraint(
            "length(btrim(message_text)) > 0",
            name="ck_event_reminders_message_text_not_blank",
        ),
        Index(
            "ux_event_reminders_event_remind_at_open",
            "event_id",
            "remind_at",
            unique=True,
            postgresql_where=text("status IN ('pending', 'processing')"),
        ),
        Index(
            "ix_event_reminders_status_next_attempt_at",
            "status",
            "next_attempt_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    event_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("calendar_events.id", ondelete="CASCADE"),
        nullable=False,
    )
    remind_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    message_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'pending'")
    )
    attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    locked_by: Mapped[str | None] = mapped_column(String(255))
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    telegram_message_id: Mapped[int | None] = mapped_column(BigInteger)
    last_error: Mapped[str | None] = mapped_column(Text)

    event: Mapped[CalendarEvent] = relationship(back_populates="reminders")
