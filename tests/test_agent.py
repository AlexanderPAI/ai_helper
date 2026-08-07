from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import Mock

from agent.agent import Agent
from agent.context import AgentRuntimeContext
from agent.providers import LLMResponse, ToolCall
from agent.results import StructuredToolResult


class AgentInterfaceIndependenceTest(unittest.TestCase):
    def test_additional_system_prompts_are_injected_after_core_prompt(self) -> None:
        agent = Agent(Mock(), additional_system_prompts=("Interface rules",))

        messages = agent._messages_for_provider(
            {"messages": [{"role": "user", "content": "Hello"}], "output": ""}
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertNotIn("Telegram", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "system", "content": "Interface rules"})
        self.assertEqual(messages[2], {"role": "user", "content": "Hello"})

    def test_assistant_history_is_not_limited_by_an_interface(self) -> None:
        response = "x" * 5000
        agent = Agent(Mock())

        message = agent._assistant_message(response)

        self.assertEqual(message["content"], response)


class AgentRuntimeContextTest(unittest.IsolatedAsyncioTestCase):
    async def test_context_reaches_tool_but_ids_are_not_added_to_arguments(
        self,
    ) -> None:
        context = AgentRuntimeContext(
            chat_id=-100500,
            user_id=42,
            user_display_name="Александр",
            message_id=7,
            current_time=datetime(2026, 8, 7, 9, 0, tzinfo=UTC),
            timezone="Europe/Moscow",
        )
        provider = _Provider()
        tool = _ContextCapturingTool()
        agent = Agent(provider, tools=(tool,))

        result = await agent.ainvoke("Создай событие", runtime_context=context)

        self.assertIs(tool.context, context)
        self.assertEqual(tool.arguments, {"title": "Встреча"})
        self.assertIsInstance(result, StructuredToolResult)
        runtime_prompt = provider.messages[-2]["content"]
        self.assertIn("2026-08-07T12:00:00+03:00", runtime_prompt)
        self.assertIn("Europe/Moscow", runtime_prompt)
        self.assertNotIn("-100500", runtime_prompt)


class _Provider:
    def __init__(self) -> None:
        self.messages = []

    async def agenerate(self, messages, **options):
        self.messages = messages
        return LLMResponse(tool_calls=(ToolCall("test_context", {"title": "Встреча"}),))


class _ContextCapturingTool:
    name = "test_context"
    description = "test"

    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }

    async def ainvoke(self, arguments, *, context=None):
        self.arguments = arguments
        self.context = context
        return StructuredToolResult("test", "Готово", {"ok": True})

    def invoke(self, arguments, *, context=None):
        raise NotImplementedError


if __name__ == "__main__":
    unittest.main()
