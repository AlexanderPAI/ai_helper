from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from openai import OpenAIError

from agent.providers import LLMProviderError, OpenRouterProvider, UrlCitation


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

    def test_build_response_preserves_web_citations(self) -> None:
        message = SimpleNamespace(
            content="Ответ со ссылкой",
            tool_calls=None,
            annotations=[
                SimpleNamespace(
                    type="url_citation",
                    url_citation=SimpleNamespace(
                        url="https://example.test/place",
                        title="Example Place",
                        content="Открыто до 02:00",
                    ),
                )
            ],
        )
        completion = SimpleNamespace(choices=[SimpleNamespace(message=message)])

        response = OpenRouterProvider._build_response(completion)

        self.assertEqual(
            response.citations,
            (
                UrlCitation(
                    url="https://example.test/place",
                    title="Example Place",
                    content="Открыто до 02:00",
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
