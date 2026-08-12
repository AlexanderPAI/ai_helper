"""Telegram progress messages for long-running agent requests."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from time import monotonic

from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message

from agent import AgentProgressEvent

logger = logging.getLogger(__name__)

STATUS_UPDATE_INTERVAL = 5

_TOOL_STATUSES = {
    "search_places": "🔎 Ищу подходящие места…",
    "send_meme": "🖼 Ищу подходящий мем…",
    "create_calendar_event": "📅 Создаю событие в календаре…",
    "list_calendar_events": "📋 Получаю события из календаря…",
    "get_calendar_event": "📅 Проверяю данные события…",
    "add_calendar_reminders": "⏰ Добавляю напоминания…",
    "update_calendar_event": "✏️ Обновляю событие в календаре…",
    "cancel_calendar_event": "🗑 Отменяю событие в календаре…",
}


@dataclass(slots=True)
class TelegramProgressSession:
    """Maintain the current user-facing stage and elapsed time."""

    message: Message
    started_at: float
    stage: str = "🔎 Анализирую запрос…"
    _last_text: str | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def report(self, event: AgentProgressEvent) -> None:
        """Translate an agent lifecycle event into a non-technical status."""
        self.stage = self._event_text(event)
        await self.refresh()

    async def refresh(self) -> None:
        """Update the Telegram message while preserving elapsed time."""
        elapsed = monotonic() - self.started_at
        text = f"{self.stage}\n\nПрошло: {elapsed:.0f} сек."
        async with self._lock:
            if text == self._last_text:
                return
            try:
                await self.message.edit_text(text)
                self._last_text = text
            except TelegramAPIError:
                logger.warning(
                    "Failed to update status chat_id=%s",
                    self.message.chat.id,
                    exc_info=True,
                )

    @staticmethod
    def _event_text(event: AgentProgressEvent) -> str:
        if event.kind == "planning":
            return "🔎 Анализирую запрос и намечаю шаги…"
        if event.kind == "plan_ready":
            if event.total_steps == 1:
                return "🧭 План готов. Выполняю задачу…"
            return f"🧭 План готов, шагов: {event.total_steps}. Начинаю…"
        if event.kind == "tool_started":
            status = _TOOL_STATUSES.get(event.tool_name or "", "⚙️ Выполняю задачу…")
            if event.step_number and event.total_steps and event.total_steps > 1:
                return f"Этап {event.step_number} из {event.total_steps}\n{status}"
            return status
        if event.kind == "tool_completed":
            if (
                event.step_number
                and event.total_steps
                and event.step_number < event.total_steps
            ):
                return (
                    f"✅ Этап {event.step_number} из {event.total_steps} готов. "
                    "Перехожу к следующему…"
                )
            return "✅ Действие выполнено. Проверяю результат…"
        return "✍️ Формирую итоговый ответ…"


class TelegramProgressReporter:
    """Create and periodically update a request status message."""

    @asynccontextmanager
    async def track(self, message: Message) -> AsyncIterator[TelegramProgressSession]:
        """Yield a progress session and stop its updater on context exit."""
        started_at = monotonic()
        status_message = await message.reply("⏳ Получил вопрос. Готовлю ответ…")
        session = TelegramProgressSession(status_message, started_at)
        update_task = asyncio.create_task(self._update(session))
        try:
            yield session
        finally:
            update_task.cancel()
            with suppress(asyncio.CancelledError):
                await update_task

    @staticmethod
    async def _update(session: TelegramProgressSession) -> None:
        while True:
            await asyncio.sleep(STATUS_UPDATE_INTERVAL)
            await session.refresh()
            logger.info(
                "LLM request still running chat_id=%s elapsed=%.1fs",
                session.message.chat.id,
                monotonic() - session.started_at,
            )


__all__ = ["TelegramProgressReporter", "TelegramProgressSession"]
