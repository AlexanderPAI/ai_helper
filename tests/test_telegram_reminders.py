from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from aiogram.exceptions import TelegramForbiddenError, TelegramRetryAfter

from bot.reminders import TelegramReminderSender
from reminder_app import (
    PermanentReminderDeliveryError,
    ReminderDelivery,
    TemporaryReminderDeliveryError,
)


def reminder() -> ReminderDelivery:
    return ReminderDelivery(
        reminder_id=uuid4(),
        event_id=uuid4(),
        chat_id=-100123,
        event_title="Встреча по договору",
        event_starts_at=datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
        event_timezone="Europe/Moscow",
        message_text="Взять договор и позвонить Анне",
        remind_at=datetime(2026, 8, 8, 11, 0, tzinfo=UTC),
        attempts=1,
    )


class TelegramReminderSenderTest(unittest.IsolatedAsyncioTestCase):
    async def test_sends_custom_text_without_llm_or_chat_context(self) -> None:
        bot = Mock()
        bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=77))
        sender = TelegramReminderSender(bot)

        message_id = await sender.send(reminder())

        self.assertEqual(message_id, 77)
        self.assertEqual(bot.send_message.await_args.kwargs["chat_id"], -100123)
        text = bot.send_message.await_args.kwargs["text"]
        self.assertIn("08.08.2026 15:00 (Europe/Moscow)", text)
        self.assertIn("Взять договор и позвонить Анне", text)

    async def test_rate_limit_becomes_temporary_error_with_retry_after(self) -> None:
        bot = Mock()
        bot.send_message = AsyncMock(
            side_effect=TelegramRetryAfter(Mock(), "retry", retry_after=12)
        )
        sender = TelegramReminderSender(bot)

        with self.assertRaises(TemporaryReminderDeliveryError) as raised:
            await sender.send(reminder())

        self.assertEqual(raised.exception.retry_after, 12)

    async def test_forbidden_error_is_permanent(self) -> None:
        bot = Mock()
        bot.send_message = AsyncMock(
            side_effect=TelegramForbiddenError(Mock(), "blocked")
        )
        sender = TelegramReminderSender(bot)

        with self.assertRaises(PermanentReminderDeliveryError):
            await sender.send(reminder())


if __name__ == "__main__":
    unittest.main()
