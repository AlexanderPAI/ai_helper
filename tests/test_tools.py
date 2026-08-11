from __future__ import annotations

import json
import unittest
from io import BytesIO
from typing import Self
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from pydantic import AnyHttpUrl, SecretStr

from agent.place_tools import SearchPlacesTool
from agent.results import MediaResult
from agent.settings import HumorAPISettings
from agent.tools import SendMemeTool


class _UnusedProvider:
    def generate(self, messages, **options):
        raise AssertionError("placeholder must not call the provider")

    async def agenerate(self, messages, **options):
        raise AssertionError("placeholder must not call the provider")


class _Response(BytesIO):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class SendMemeToolTest(unittest.TestCase):
    def setUp(self) -> None:
        settings = HumorAPISettings(
            api_key=SecretStr("secret"),
            random_url=AnyHttpUrl("https://example.test/memes/random"),
            search_url=AnyHttpUrl("https://example.test/memes/search"),
            user_agent="ai-helper-test/0.1",
        )
        self.tool = SendMemeTool(settings)

    def test_model_instructions_are_loaded_from_yaml(self) -> None:
        schema = self.tool.schema["function"]

        self.assertEqual(schema["description"], self.tool.prompt.description)
        properties = schema["parameters"]["properties"]
        self.assertEqual(
            properties["mode"]["description"],
            self.tool.prompt.parameter_descriptions["mode"],
        )
        self.assertEqual(
            properties["keywords"]["description"],
            self.tool.prompt.parameter_descriptions["keywords"],
        )

    @patch("agent.tools.urlopen")
    def test_search_uses_agent_keywords_and_requests_one_result(
        self, mock_urlopen
    ) -> None:
        mock_urlopen.return_value = _Response(
            json.dumps(
                {
                    "memes": [
                        {
                            "id": 42,
                            "url": "https://images.example.test/meme.jpg",
                            "type": "image/jpeg",
                        }
                    ]
                }
            ).encode()
        )

        result = self.tool.invoke({"mode": "search", "keywords": "python bugs"})

        self.assertEqual(
            result,
            MediaResult(42, "https://images.example.test/meme.jpg", "image/jpeg"),
        )
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(
            parse_qs(urlparse(request.full_url).query),
            {"keywords": ["python"], "number": ["1"]},
        )
        self.assertEqual(request.get_header("X-api-key"), "secret")
        self.assertEqual(request.get_header("User-agent"), "ai-helper-test/0.1")

    @patch("agent.tools.urlopen")
    def test_random_returns_single_meme_without_number_parameter(
        self, mock_urlopen
    ) -> None:
        mock_urlopen.return_value = _Response(
            json.dumps(
                {
                    "id": 7,
                    "url": "https://images.example.test/random.png",
                    "type": "image/png",
                }
            ).encode()
        )

        result = self.tool.invoke({"mode": "random"})

        self.assertEqual(result.id, 7)
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(parse_qs(urlparse(request.full_url).query), {})

    @patch("agent.tools.urlopen")
    def test_search_uses_one_keyword_and_falls_back_when_empty(
        self, mock_urlopen
    ) -> None:
        mock_urlopen.side_effect = [
            _Response(json.dumps({"memes": [], "available": 0}).encode()),
            _Response(
                json.dumps(
                    {
                        "id": 8,
                        "url": "https://images.example.test/fallback.jpg",
                        "type": "image/jpeg",
                    }
                ).encode()
            ),
        ]

        result = self.tool.invoke(
            {"mode": "search", "keywords": "programmer coding bugs"}
        )

        self.assertEqual(result.id, 8)
        first_request = mock_urlopen.call_args_list[0].args[0]
        self.assertEqual(
            parse_qs(urlparse(first_request.full_url).query),
            {"keywords": ["programmer"], "number": ["1"]},
        )
        second_request = mock_urlopen.call_args_list[1].args[0]
        self.assertEqual(second_request.full_url, "https://example.test/memes/random")


class SearchPlacesToolTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = SearchPlacesTool(_UnusedProvider())

    def test_schema_is_loaded_from_yaml(self) -> None:
        schema = self.tool.schema["function"]

        self.assertEqual(schema["name"], "search_places")
        self.assertEqual(schema["description"], self.tool.prompt.description)
        self.assertEqual(set(schema["parameters"]["required"]), {"query", "location"})

    async def test_placeholder_does_not_call_provider(self) -> None:
        result = await self.tool.ainvoke(
            {"query": "спокойная кальянная", "location": "Москва"}
        )

        self.assertEqual(result, "Поиск мест пока не подключён.")


if __name__ == "__main__":
    unittest.main()
