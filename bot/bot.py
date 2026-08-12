"""Telegram interface for the conversational agent."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from contextlib import suppress
from datetime import UTC, datetime
from time import monotonic
from uuid import uuid4

from aiogram import Bot, Dispatcher
from aiogram.enums import MessageEntityType
from aiogram.types import Message
from sqlalchemy.exc import SQLAlchemyError

from agent import (
    Agent,
    AgentRuntimeContext,
    AgentToolError,
    HumorAPISettings,
    LLMProviderError,
    OpenRouterProvider,
    OpenRouterWebSearchSettings,
    SearchPlacesTool,
    SendMemeTool,
    calendar_tools,
)
from calendar_app import CalendarService
from database import (
    DatabaseSettings,
    SqlAlchemyCalendarUnitOfWorkFactory,
    SqlAlchemyReminderUnitOfWorkFactory,
    create_database_engine,
    create_session_factory,
)
from reminder_app import ReminderWorker

from .progress import TelegramProgressReporter
from .prompts import load_system_prompt
from .reminders import TelegramReminderSender
from .rendering import TelegramResponseRenderer
from .sessions import ChatSessionRegistry
from .settings import ReminderWorkerSettings, TelegramSettings

logger = logging.getLogger(__name__)


class TelegramAgentBot:
    """Route allowed group mentions to shared per-chat agent sessions."""

    def __init__(
        self,
        *,
        agent: Agent,
        bot_username: str,
        allowed_chat_ids: frozenset[int],
        calendar_service: CalendarService,
        calendar_default_timezone: str,
        sessions: ChatSessionRegistry | None = None,
        progress: TelegramProgressReporter | None = None,
        renderer: TelegramResponseRenderer | None = None,
    ) -> None:
        self.agent = agent
        self.bot_username = bot_username.casefold().lstrip("@")
        self.allowed_chat_ids = allowed_chat_ids
        self.calendar_service = calendar_service
        self.calendar_default_timezone = calendar_default_timezone
        self.sessions = sessions or ChatSessionRegistry()
        self.progress = progress or TelegramProgressReporter()
        self.renderer = renderer or TelegramResponseRenderer()

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

        async with self.sessions.lock(message.chat.id):
            session_id = self.sessions.session_id(message.chat.id)
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
                new_session_id = self.sessions.reset(message.chat.id)
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
            logger.info("LLM request started chat_id=%s", message.chat.id)
            async with self.progress.track(message) as progress_session:
                status_message = progress_session.message
                try:
                    runtime_context = await self._runtime_context(message)
                    response = await self.agent.ainvoke(
                        prompt,
                        session_id=session_id,
                        runtime_context=runtime_context,
                        progress_callback=progress_session.report,
                    )
                except LLMProviderError, AgentToolError, SQLAlchemyError:
                    logger.exception(
                        "LLM request failed chat_id=%s after %.1fs",
                        message.chat.id,
                        monotonic() - started_at,
                    )
                    await status_message.edit_text("❌ Не удалось подготовить ответ.")
                    return

            logger.info(
                "LLM response received chat_id=%s after %.1fs characters=%d",
                message.chat.id,
                monotonic() - started_at,
                len(response) if isinstance(response, str) else 0,
            )
            chunks_sent = await self.renderer.send(message, status_message, response)
            logger.info(
                "Response sent chat_id=%s chunks=%d total_time=%.1fs",
                message.chat.id,
                chunks_sent,
                monotonic() - started_at,
            )

    async def _runtime_context(self, message: Message) -> AgentRuntimeContext:
        """Build trusted tool metadata without exposing it to model arguments."""
        settings = await self.calendar_service.get_settings(message.chat.id)
        user = message.from_user
        return AgentRuntimeContext(
            chat_id=message.chat.id,
            user_id=user.id if user is not None else None,
            user_display_name=user.full_name if user is not None else None,
            message_id=message.message_id,
            current_time=datetime.now(UTC),
            timezone=(
                settings.timezone
                if settings is not None
                else self.calendar_default_timezone
            ),
        )

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
    humor_api_settings = HumorAPISettings()  # type: ignore[call-arg]
    web_search_settings = OpenRouterWebSearchSettings()
    database_settings = DatabaseSettings()  # type: ignore[call-arg]
    reminder_settings = ReminderWorkerSettings()
    logger.info("Configuration loaded allowed_chats=%d", len(settings.chat_ids))

    database_engine = create_database_engine(database_settings)
    calendar_service = CalendarService(
        SqlAlchemyCalendarUnitOfWorkFactory(create_session_factory(database_engine))
    )
    reminder_unit_of_work_factory = SqlAlchemyReminderUnitOfWorkFactory(
        create_session_factory(database_engine)
    )
    try:
        async with (
            Bot(token=settings.bot_token.get_secret_value()) as telegram_bot,
            OpenRouterProvider() as provider,
        ):
            logger.info("Connecting to Telegram")
            bot_user = await telegram_bot.get_me()
            if bot_user.username is None:
                raise RuntimeError("Telegram bot must have a username")

            reminder_worker = ReminderWorker(
                reminder_unit_of_work_factory,
                TelegramReminderSender(telegram_bot),
                worker_id=(f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"),
                options=reminder_settings.worker_options,
            )
            reminder_task = asyncio.create_task(
                reminder_worker.run(), name="reminder-worker"
            )
            polling_task: asyncio.Task[None] | None = None
            try:
                application = TelegramAgentBot(
                    agent=Agent(
                        provider,
                        additional_system_prompts=(load_system_prompt(),),
                        tools=(
                            SendMemeTool(humor_api_settings),
                            SearchPlacesTool(
                                provider, web_search_settings, calendar_service
                            ),
                            *calendar_tools(calendar_service),
                        ),
                    ),
                    bot_username=bot_user.username,
                    allowed_chat_ids=settings.chat_ids,
                    calendar_service=calendar_service,
                    calendar_default_timezone=settings.calendar_default_timezone,
                )
                dispatcher = Dispatcher()
                application.register(dispatcher)

                logger.info(
                    "Starting @%s in %d allowed chats",
                    bot_user.username,
                    len(settings.chat_ids),
                )
                polling_task = asyncio.create_task(
                    dispatcher.start_polling(telegram_bot),
                    name="telegram-polling",
                )
                done, _ = await asyncio.wait(
                    (polling_task, reminder_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if reminder_task in done:
                    await reminder_task
                    raise RuntimeError("reminder worker stopped unexpectedly")
                await polling_task
            finally:
                reminder_worker.request_stop()
                if polling_task is not None and not polling_task.done():
                    polling_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await polling_task
                try:
                    await asyncio.wait_for(
                        reminder_task,
                        timeout=reminder_settings.shutdown_timeout,
                    )
                except TimeoutError:
                    logger.warning("Reminder worker graceful shutdown timed out")
                    reminder_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await reminder_task
    finally:
        await database_engine.dispose()


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
