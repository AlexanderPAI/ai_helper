"""Telegram adapter for reminder delivery."""

from __future__ import annotations

from zoneinfo import ZoneInfo

from aiogram import Bot
from aiogram.exceptions import (
    TelegramAPIError,
    TelegramBadRequest,
    TelegramEntityTooLarge,
    TelegramForbiddenError,
    TelegramMigrateToChat,
    TelegramNetworkError,
    TelegramNotFound,
    TelegramRetryAfter,
    TelegramServerError,
    TelegramUnauthorizedError,
)

from reminder_app import (
    PermanentReminderDeliveryError,
    ReminderDelivery,
    TemporaryReminderDeliveryError,
)


class TelegramReminderSender:
    """Send one persisted reminder as a plain Telegram message."""

    def __init__(self, bot: Bot) -> None:
        self.bot = bot

    async def send(self, reminder: ReminderDelivery) -> int:
        try:
            message = await self.bot.send_message(
                chat_id=reminder.chat_id,
                text=self.format_message(reminder),
            )
        except TelegramRetryAfter as error:
            raise TemporaryReminderDeliveryError(
                "Telegram rate limit",
                retry_after=float(error.retry_after),
            ) from error
        except (TelegramNetworkError, TelegramServerError) as error:
            raise TemporaryReminderDeliveryError(
                "Telegram is temporarily unavailable"
            ) from error
        except (
            TelegramBadRequest,
            TelegramEntityTooLarge,
            TelegramForbiddenError,
            TelegramMigrateToChat,
            TelegramNotFound,
            TelegramUnauthorizedError,
        ) as error:
            raise PermanentReminderDeliveryError(
                "Telegram rejected reminder delivery"
            ) from error
        except TelegramAPIError as error:
            raise TemporaryReminderDeliveryError("Telegram API error") from error
        return message.message_id

    @staticmethod
    def format_message(reminder: ReminderDelivery) -> str:
        local_start = reminder.event_starts_at.astimezone(
            ZoneInfo(reminder.event_timezone)
        )
        return (
            "⏰ Напоминание\n\n"
            f"{reminder.event_title}\n"
            f"{local_start:%d.%m.%Y %H:%M} ({reminder.event_timezone})\n\n"
            f"{reminder.message_text}"
        )


__all__ = ["TelegramReminderSender"]
