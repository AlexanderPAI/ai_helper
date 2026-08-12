from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agent import AgentProgressEvent
from bot.progress import TelegramProgressSession


class TelegramProgressSessionTest(unittest.IsolatedAsyncioTestCase):
    async def test_tool_name_is_rendered_as_human_readable_stage(self) -> None:
        message = SimpleNamespace(
            edit_text=AsyncMock(),
            chat=SimpleNamespace(id=123),
        )
        session = TelegramProgressSession(message=message, started_at=0)

        await session.report(
            AgentProgressEvent(
                kind="tool_started",
                tool_name="search_places",
                step_number=1,
                total_steps=2,
            )
        )

        text = message.edit_text.await_args.args[0]
        self.assertIn("Этап 1 из 2", text)
        self.assertIn("Ищу подходящие места", text)
        self.assertIn("Прошло:", text)
        self.assertNotIn("search_places", text)

    async def test_calendar_tool_has_non_technical_status(self) -> None:
        message = SimpleNamespace(
            edit_text=AsyncMock(),
            chat=SimpleNamespace(id=123),
        )
        session = TelegramProgressSession(message=message, started_at=0)

        await session.report(
            AgentProgressEvent(
                kind="tool_started",
                tool_name="list_calendar_events",
                step_number=1,
                total_steps=1,
            )
        )

        text = message.edit_text.await_args.args[0]
        self.assertIn("Получаю события из календаря", text)
        self.assertNotIn("list_calendar_events", text)


if __name__ == "__main__":
    unittest.main()
