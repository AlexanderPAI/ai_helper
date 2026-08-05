"""Telegram interface for the conversational agent."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from uuid import UUID, uuid4

from aiogram import Bot, Dispatcher
from aiogram.enums import MessageEntityType
from aiogram.types import Message, MessageEntity
from openai import OpenAIError
from telegramify_markdown import convert, split_entities

from agent import Agent, OpenRouterProvider

from .settings import TelegramSettings

logger = logging.getLogger(__name__)


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
        if message.chat.id not in self.allowed_chat_ids or not message.text:
            return

        prompt = self._extract_prompt(message)
        if prompt is None:
            return

        async with self._chat_locks[message.chat.id]:
            session_id = self._session_ids.setdefault(message.chat.id, uuid4())

            if prompt.casefold() == "/reset":
                await self.agent.areset_context(session_id=session_id)
                await message.reply("Контекст чата сброшен.")
                return

            if not prompt:
                await message.reply("Добавьте вопрос после упоминания бота.")
                return

            try:
                response = await self.agent.ainvoke(prompt, session_id=session_id)
            except OpenAIError:
                logger.exception("LLM request failed for chat %s", message.chat.id)
                await message.reply("Не удалось получить ответ от модели.")
                return

            await self._reply_with_markdown(message, response)

    @staticmethod
    async def _reply_with_markdown(message: Message, markdown: str) -> None:
        """Render model Markdown as Telegram-native formatted text."""
        text, entities = convert(markdown)
        chunks = split_entities(text, entities, max_utf16_len=4096)

        for index, (chunk_text, chunk_entities) in enumerate(chunks):
            telegram_entities = [
                MessageEntity(**entity.to_dict()) for entity in chunk_entities
            ]
            if index == 0:
                await message.reply(chunk_text, entities=telegram_entities)
            else:
                await message.answer(chunk_text, entities=telegram_entities)

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
    settings = TelegramSettings()  # type: ignore[call-arg]

    async with (
        Bot(token=settings.bot_token.get_secret_value()) as telegram_bot,
        OpenRouterProvider() as provider,
    ):
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
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
