"""Telegram progress messages for long-running requests."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from time import monotonic

from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message

logger = logging.getLogger(__name__)

STATUS_UPDATE_INTERVAL = 5
STATUS_MESSAGES = (
    "🔎 Анализирую вопрос…",
    "🧠 Обдумываю ответ…",
    "✍️ Формирую ответ…",
)


class TelegramProgressReporter:
    """Create and periodically update a request status message."""

    @asynccontextmanager
    async def track(self, message: Message) -> AsyncIterator[Message]:
        """Yield a status message and stop its updater on context exit."""
        started_at = monotonic()
        status_message = await message.reply("⏳ Получил вопрос. Готовлю ответ…")
        update_task = asyncio.create_task(self._update(status_message, started_at))
        try:
            yield status_message
        finally:
            update_task.cancel()
            with suppress(asyncio.CancelledError):
                await update_task

    @staticmethod
    async def _update(status_message: Message, started_at: float) -> None:
        update_index = 0
        while True:
            await asyncio.sleep(STATUS_UPDATE_INTERVAL)
            elapsed = monotonic() - started_at
            status = STATUS_MESSAGES[update_index % len(STATUS_MESSAGES)]
            update_index += 1

            try:
                await status_message.edit_text(
                    f"{status}\n\nПрошло: {elapsed:.0f} сек."
                )
            except TelegramAPIError:
                logger.warning(
                    "Failed to update status chat_id=%s",
                    status_message.chat.id,
                    exc_info=True,
                )

            logger.info(
                "LLM request still running chat_id=%s elapsed=%.1fs",
                status_message.chat.id,
                elapsed,
            )


__all__ = ["TelegramProgressReporter"]
