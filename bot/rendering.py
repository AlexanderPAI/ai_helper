"""Render provider-independent agent results through Telegram."""

from __future__ import annotations

from aiogram.types import Message, MessageEntity
from telegramify_markdown import convert, split_entities

from agent import MediaResult, ToolResult


class TelegramResponseRenderer:
    """Convert agent results to Telegram messages and media."""

    async def send(
        self,
        request_message: Message,
        status_message: Message,
        result: ToolResult,
    ) -> int:
        """Replace a progress status with the result and return chunks sent."""
        if isinstance(result, MediaResult):
            await status_message.delete()
            await request_message.answer_photo(result.url)
            return 1
        return await self._send_markdown(request_message, status_message, result)

    @staticmethod
    async def _send_markdown(
        request_message: Message,
        status_message: Message,
        markdown: str,
    ) -> int:
        text, entities = convert(markdown)
        chunks = split_entities(text, entities, max_utf16_len=4096)

        for index, (chunk_text, chunk_entities) in enumerate(chunks):
            telegram_entities = [
                MessageEntity(**entity.to_dict()) for entity in chunk_entities
            ]
            if index == 0:
                await status_message.edit_text(chunk_text, entities=telegram_entities)
            else:
                await request_message.answer(chunk_text, entities=telegram_entities)
        return len(chunks)


__all__ = ["TelegramResponseRenderer"]
