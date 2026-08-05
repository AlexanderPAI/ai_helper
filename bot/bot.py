"""Telegram interface for the conversational agent."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from contextlib import suppress
from time import monotonic
from uuid import UUID, uuid4

from aiogram import Bot, Dispatcher
from aiogram.enums import MessageEntityType
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message, MessageEntity
from openai import OpenAIError
from telegramify_markdown import convert, split_entities

from agent import Agent, OpenRouterProvider

from .settings import TelegramSettings

logger = logging.getLogger(__name__)

STATUS_UPDATE_INTERVAL = 5
STATUS_MESSAGES = (
    "🔎 Анализирую вопрос…",
    "🧠 Обдумываю ответ…",
    "✍️ Формирую ответ…",
)


class TelegramAgentBot:
    """Route allowed group mentions to shared per-chat agent sessions."""

    def __init__(
        self,
        *,
        agent: Agent,
        bot_username: str,
        allowed_chat_ids: frozenset[int],
    ) -> None:
        self.agent = agent
        self.bot_username = bot_username.casefold().lstrip("@")
        self.allowed_chat_ids = allowed_chat_ids
        self._session_ids: dict[int, UUID] = {}
        self._chat_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    def register(self, dispatcher: Dispatcher) -> None:
        """Register bot handlers on an aiogram dispatcher."""
        dispatcher.message.register(self.handle_message)

    async def handle_message(self, message: Message) -> None:
        """Handle an addressed text message from an allowed chat."""
        if message.chat.id not in self.allowed_chat_ids:
            logger.debug(
                "Ignoring message from unauthorized chat_id=%s", message.chat.id
            )
            return
        if not message.text:
            logger.debug("Ignoring non-text message in chat_id=%s", message.chat.id)
            return

        prompt = self._extract_prompt(message)
        if prompt is None:
            logger.debug(
                "Ignoring message without bot mention in chat_id=%s", message.chat.id
            )
            return

        async with self._chat_locks[message.chat.id]:
            session_id = self._session_ids.setdefault(message.chat.id, uuid4())
            logger.info(
                "Accepted message chat_id=%s message_id=%s session_id=%s",
                message.chat.id,
                message.message_id,
                session_id,
            )

            if prompt.casefold() == "/reset":
                logger.info(
                    "Resetting context chat_id=%s session_id=%s",
                    message.chat.id,
                    session_id,
                )
                await self.agent.areset_context(session_id=session_id)
                new_session_id = uuid4()
                self._session_ids[message.chat.id] = new_session_id
                await message.reply("Контекст чата сброшен.")
                logger.info(
                    "Context reset completed chat_id=%s old_session_id=%s "
                    "new_session_id=%s",
                    message.chat.id,
                    session_id,
                    new_session_id,
                )
                return

            if not prompt:
                await message.reply("Добавьте вопрос после упоминания бота.")
                return

            started_at = monotonic()
            status_message = await message.reply("⏳ Получил вопрос. Готовлю ответ…")
            logger.info("LLM request started chat_id=%s", message.chat.id)
            progress_task = asyncio.create_task(
                self._update_request_status(status_message, started_at)
            )
            try:
                response = await self.agent.ainvoke(prompt, session_id=session_id)
            except OpenAIError:
                logger.exception(
                    "LLM request failed chat_id=%s after %.1fs",
                    message.chat.id,
                    monotonic() - started_at,
                )
                await status_message.edit_text(
                    "❌ Не удалось получить ответ от модели."
                )
                return
            finally:
                progress_task.cancel()
                with suppress(asyncio.CancelledError):
                    await progress_task

            logger.info(
                "LLM response received chat_id=%s after %.1fs characters=%d",
                message.chat.id,
                monotonic() - started_at,
                len(response),
            )
            chunks_sent = await self._replace_status_with_response(
                message,
                status_message,
                response,
            )
            logger.info(
                "Response sent chat_id=%s chunks=%d total_time=%.1fs",
                message.chat.id,
                chunks_sent,
                monotonic() - started_at,
            )

    @staticmethod
    async def _update_request_status(
        status_message: Message,
        started_at: float,
    ) -> None:
        """Periodically update one Telegram message while the LLM is working."""
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

    @staticmethod
    async def _replace_status_with_response(
        request_message: Message,
        status_message: Message,
        markdown: str,
    ) -> int:
        """Replace the progress status with a Telegram-formatted response."""
        text, entities = convert(markdown)
        chunks = split_entities(text, entities, max_utf16_len=4096)

        for index, (chunk_text, chunk_entities) in enumerate(chunks):
            telegram_entities = [
                MessageEntity(**entity.to_dict()) for entity in chunk_entities
            ]
            if index == 0:
                await status_message.edit_text(
                    chunk_text,
                    entities=telegram_entities,
                )
            else:
                await request_message.answer(
                    chunk_text,
                    entities=telegram_entities,
                )
        return len(chunks)

    def _extract_prompt(self, message: Message) -> str | None:
        """Return text without this bot's @mention, or None if not mentioned."""
        text = message.text
        if text is None:
            return None

        own_mentions = [
            entity.extract_from(text)
            for entity in message.entities or []
            if entity.type == MessageEntityType.MENTION
            and entity.extract_from(text).casefold() == f"@{self.bot_username}"
        ]
        if not own_mentions:
            return None

        for mention in own_mentions:
            text = text.replace(mention, "", 1)
        return text.strip()


async def run_bot() -> None:
    """Build application services and start Telegram long polling."""
    logger.info("Loading bot configuration")
    settings = TelegramSettings()  # type: ignore[call-arg]
    logger.info("Configuration loaded allowed_chats=%d", len(settings.chat_ids))

    async with (
        Bot(token=settings.bot_token.get_secret_value()) as telegram_bot,
        OpenRouterProvider() as provider,
    ):
        logger.info("Connecting to Telegram")
        bot_user = await telegram_bot.get_me()
        if bot_user.username is None:
            raise RuntimeError("Telegram bot must have a username")

        application = TelegramAgentBot(
            agent=Agent(provider),
            bot_username=bot_user.username,
            allowed_chat_ids=settings.chat_ids,
        )
        dispatcher = Dispatcher()
        application.register(dispatcher)

        logger.info(
            "Starting @%s in %d allowed chats",
            bot_user.username,
            len(settings.chat_ids),
        )
        await dispatcher.start_polling(telegram_bot)


def main() -> None:
    """Run the Telegram bot."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
