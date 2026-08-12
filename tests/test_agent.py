from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import Mock

from agent.agent import Agent
from agent.context import AgentRuntimeContext
from agent.planning import ActionPlanError
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

    def test_structured_data_is_not_embedded_in_visible_history(self) -> None:
        output = StructuredToolResult(
            "calendar_events_listed",
            "Красивый список без технических полей",
            {"event_id": "secret-id", "version": 3},
        )

        history = Agent._history_text(output)

        self.assertEqual(history, "Красивый список без технических полей")
        self.assertNotIn("secret-id", history)

    def test_structured_data_is_injected_as_separate_system_state(self) -> None:
        agent = Agent(Mock())

        messages = agent._messages_for_provider(
            {
                "messages": [{"role": "user", "content": "Да, это оно"}],
                "output": "",
                "internal_tool_data": {"event_id": "secret-id", "version": 3},
            }
        )

        self.assertEqual(messages[-2]["role"], "system")
        self.assertIn("secret-id", messages[-2]["content"])
        self.assertEqual(messages[-1]["content"], "Да, это оно")

    def test_legacy_internal_data_is_removed_from_model_output(self) -> None:
        content = (
            "Напоминание добавлено\n"
            '<internal_tool_data>{"event_id":"secret"}</internal_tool_data>'
        )

        sanitized = Agent._sanitize_model_output(content)

        self.assertEqual(sanitized, "Напоминание добавлено")
        self.assertNotIn("secret", sanitized)

    def test_tool_flow_marker_is_removed_from_visible_model_output(self) -> None:
        content = "Поиск завершён.\n\n**TOOL\\_FLOW\\_DONE**"

        sanitized = Agent._sanitize_model_output(content)

        self.assertEqual(sanitized, "Поиск завершён.")

    def test_markdown_tool_flow_marker_is_recognized(self) -> None:
        self.assertTrue(
            Agent._is_tool_flow_done("Понял, выполняю поиск.\n\n**TOOL\\_FLOW\\_DONE**")
        )


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
        runtime_prompt = next(
            message["content"]
            for message in provider.messages
            if message["role"] == "system"
            and "Доверенные данные текущего запроса" in message["content"]
        )
        self.assertIn("2026-08-07T12:00:00+03:00", runtime_prompt)
        self.assertIn("Europe/Moscow", runtime_prompt)
        self.assertNotIn("-100500", runtime_prompt)


class AgentToolOrchestrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_markdown_completion_marker_returns_actual_tool_result(self) -> None:
        provider = _SequencedProvider(
            _plan_response(("search_places", "Найти пивной бар")),
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        "search_places",
                        {
                            "query": "пивной бар на заводе имени Казакова",
                            "location": "Москва, метро Кутузовская",
                        },
                    ),
                )
            ),
            LLMResponse(
                content=(
                    "Понял, уточняю поиск. Выполняю поиск.\n\n**TOOL\\_FLOW\\_DONE**"
                )
            ),
        )
        search_result = "Найден бар «Тест» — Москва, улица Тестовая, 1."
        search = _RecordingTool("search_places", search_result)
        agent = Agent(provider, tools=(search,))

        result = await agent.ainvoke("Найди пивной бар на Кутузовской")

        self.assertEqual(result, search_result)
        self.assertNotIn("TOOL", result)

    async def test_empty_initial_response_is_retried_and_action_is_executed(
        self,
    ) -> None:
        provider = _SequencedProvider(
            LLMResponse(),
            _plan_response(("list_calendar_events", "Показать оставшиеся события")),
            LLMResponse(tool_calls=(ToolCall("list_calendar_events", {"limit": 10}),)),
            LLMResponse(content="__TOOL_FLOW_DONE__"),
        )
        events = StructuredToolResult(
            "calendar_events_listed",
            "Оставшиеся события: встреча в пятницу.",
            {"events": [{"event_id": "event-2", "version": 1}]},
        )
        list_events = _RecordingTool("list_calendar_events", events)
        agent = Agent(provider, tools=(list_events,))

        with self.assertLogs("agent.agent", level="WARNING") as logs:
            result = await agent.ainvoke("Покажи список оставшихся событий")

        self.assertIs(result, events)
        self.assertEqual(list_events.calls, [{"limit": 10}])
        self.assertIn("empty_initial_response retry=true", "\n".join(logs.output))
        self.assertIn(
            "Предыдущий ответ модели оказался пустым",
            provider.messages_by_call[1][-1]["content"],
        )

    async def test_repeated_empty_response_returns_visible_fallback(self) -> None:
        provider = _SequencedProvider(LLMResponse(), LLMResponse())
        agent = Agent(provider, tools=(_RecordingTool("search_places", "Бар"),))

        result = await agent.ainvoke("Покажи события")

        self.assertIn("повторите запрос", result)

    async def test_plain_conversation_does_not_enter_tool_execution(self) -> None:
        provider = _SequencedProvider(LLMResponse(content="Привет! Как ваши дела?"))
        tool = _RecordingTool("search_places", "Не должно вызываться")
        agent = Agent(provider, tools=(tool,))

        result = await agent.ainvoke("Привет")

        self.assertEqual(result, "Привет! Как ваши дела?")
        self.assertEqual(tool.calls, [])
        self.assertEqual(len(provider.messages_by_call), 1)
        planning_tools = provider.options_by_call[0]["tools"]
        self.assertEqual(planning_tools[0]["function"]["name"], "create_action_plan")

    async def test_dependent_tools_run_in_multiple_rounds(self) -> None:
        provider = _SequencedProvider(
            _plan_response(
                ("search_places", "Найти бар"),
                ("create_calendar_event", "Создать событие с местом и напоминанием"),
            ),
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        "search_places",
                        {
                            "query": "бар для дня рождения",
                            "location": "Москва, метро Павелецкая",
                        },
                    ),
                )
            ),
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        "create_calendar_event",
                        {
                            "title": "Празднование дня рождения",
                            "starts_at_local": "2026-08-18T19:00:00",
                            "place": {
                                "name": "Бар Тест",
                                "address": "Москва, Тестовая улица, 1",
                            },
                            "reminders": [
                                {
                                    "offset_minutes": 1440,
                                    "message_text": "Завтра празднование дня рождения",
                                }
                            ],
                        },
                    ),
                )
            ),
            LLMResponse(content="__TOOL_FLOW_DONE__"),
        )
        search = _RecordingTool(
            "search_places",
            "1. Бар Тест — Москва, Тестовая улица, 1",
        )
        created = StructuredToolResult(
            "calendar_created",
            "Событие создано с местом и напоминанием.",
            {"event_id": "event-1", "version": 1},
        )
        create = _RecordingTool("create_calendar_event", created)
        agent = Agent(provider, tools=(search, create))
        progress_events = []

        async def capture_progress(event) -> None:
            progress_events.append(event)

        result = await agent.ainvoke(
            "Запланируй день рождения и найди бар у метро Павелецкая",
            progress_callback=capture_progress,
        )

        self.assertIs(result, created)
        self.assertEqual(len(search.calls), 1)
        self.assertEqual(len(create.calls), 1)
        self.assertEqual(
            create.calls[0]["place"]["name"],
            "Бар Тест",
        )
        self.assertEqual(create.calls[0]["reminders"][0]["offset_minutes"], 1440)
        self.assertIn("Бар Тест", provider.messages_by_call[2][-1]["content"])
        first_execution_tools = {
            schema["function"]["name"]
            for schema in provider.options_by_call[1]["tools"]
        }
        second_execution_tools = {
            schema["function"]["name"]
            for schema in provider.options_by_call[2]["tools"]
        }
        self.assertEqual(first_execution_tools, {"search_places"})
        self.assertEqual(second_execution_tools, {"create_calendar_event"})
        self.assertEqual(
            [event.kind for event in progress_events],
            [
                "planning",
                "plan_ready",
                "tool_started",
                "tool_completed",
                "tool_started",
                "tool_completed",
                "finalizing",
            ],
        )
        self.assertEqual(progress_events[2].tool_name, "search_places")
        self.assertEqual(progress_events[4].tool_name, "create_calendar_event")

    async def test_standalone_tool_stops_without_calling_unrequested_tools(
        self,
    ) -> None:
        provider = _SequencedProvider(
            _plan_response(("search_places", "Найти и показать бары")),
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        "search_places",
                        {"query": "бары", "location": "Москва"},
                    ),
                )
            ),
            LLMResponse(content="__TOOL_FLOW_DONE__"),
        )
        search = _RecordingTool("search_places", "Найдено три бара")
        create = _RecordingTool("create_calendar_event", "Создано")
        agent = Agent(provider, tools=(search, create))

        result = await agent.ainvoke("Покажи бары в Москве")

        self.assertEqual(result, "Найдено три бара")
        self.assertEqual(len(search.calls), 1)
        self.assertEqual(create.calls, [])
        execution_tools = {
            schema["function"]["name"]
            for schema in provider.options_by_call[1]["tools"]
        }
        self.assertEqual(execution_tools, {"search_places"})

    async def test_tool_outside_current_plan_step_is_rejected(self) -> None:
        provider = _SequencedProvider(
            _plan_response(
                ("search_places", "Найти бар"),
                ("create_calendar_event", "Создать событие"),
            ),
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        "create_calendar_event",
                        {"title": "Слишком рано"},
                    ),
                )
            ),
        )
        search = _RecordingTool("search_places", "Бар")
        create = _RecordingTool("create_calendar_event", "Создано")
        agent = Agent(provider, tools=(search, create))

        with self.assertRaises(ActionPlanError):
            await agent.ainvoke("Найди бар и создай событие")

        self.assertEqual(search.calls, [])
        self.assertEqual(create.calls, [])


class _Provider:
    def __init__(self) -> None:
        self.messages = []
        self.call_count = 0

    async def agenerate(self, messages, **options):
        self.messages = messages
        self.call_count += 1
        if self.call_count == 1:
            return _plan_response(("test_context", "Создать встречу"))
        if self.call_count > 2:
            return LLMResponse(content="__TOOL_FLOW_DONE__")
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


class _SequencedProvider:
    def __init__(self, *responses: LLMResponse) -> None:
        self.responses = list(responses)
        self.messages_by_call = []
        self.options_by_call = []

    async def agenerate(self, messages, **options):
        self.messages_by_call.append([dict(message) for message in messages])
        self.options_by_call.append(options)
        return self.responses.pop(0)


class _RecordingTool:
    def __init__(self, name, result) -> None:
        self.name = name
        self.description = name
        self.result = result
        self.calls = []

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
        self.calls.append(arguments)
        return self.result

    def invoke(self, arguments, *, context=None):
        self.calls.append(arguments)
        return self.result


def _plan_response(*steps: tuple[str, str]) -> LLMResponse:
    return LLMResponse(
        tool_calls=(
            ToolCall(
                "create_action_plan",
                {
                    "goals": [purpose for _, purpose in steps],
                    "steps": [
                        {
                            "id": f"step_{index}",
                            "tool": tool,
                            "purpose": purpose,
                            "depends_on": ([] if index == 1 else [f"step_{index - 1}"]),
                        }
                        for index, (tool, purpose) in enumerate(steps, start=1)
                    ],
                },
            ),
        )
    )


if __name__ == "__main__":
    unittest.main()
