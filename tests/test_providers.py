from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from openai import OpenAIError

from agent.providers import LLMProviderError, OpenRouterProvider


class OpenRouterProviderErrorTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.provider = object.__new__(OpenRouterProvider)
        self.provider.settings = SimpleNamespace(model="test-model")

    def test_chat_translates_openai_error(self) -> None:
        create = Mock(side_effect=OpenAIError("request failed"))
        self.provider.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with self.assertRaises(LLMProviderError) as raised:
            self.provider.chat([])

        self.assertIsInstance(raised.exception.__cause__, OpenAIError)

    async def test_achat_translates_openai_error(self) -> None:
        create = AsyncMock(side_effect=OpenAIError("request failed"))
        self.provider.async_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with self.assertRaises(LLMProviderError) as raised:
            await self.provider.achat([])

        self.assertIsInstance(raised.exception.__cause__, OpenAIError)


if __name__ == "__main__":
    unittest.main()
