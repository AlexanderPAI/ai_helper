"""LangGraph-based conversational agent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from operator import add
from typing import Annotated, Any, TypedDict
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .context import AgentRuntimeContext
from .prompts import load_system_prompt
from .providers import LLMProvider, LLMResponse
from .results import MediaResult, StructuredToolResult, ToolResult
from .tools import AgentTool

Message = dict[str, str]


class AgentState(TypedDict):
    """Conversation history accumulated by the graph checkpointer."""

    messages: Annotated[list[Message], add]
    output: ToolResult


class Agent:
    """A minimal stateful agent whose LLM implementation is provider-agnostic.

    One instance can maintain multiple independent conversations identified by
    UUID session IDs.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        session_id: UUID | None = None,
        additional_system_prompts: Sequence[str] = (),
        provider_options: Mapping[str, Any] | None = None,
        tools: Sequence[AgentTool] = (),
    ) -> None:
        if session_id is not None and not isinstance(session_id, UUID):
            raise TypeError("session_id must be a UUID")
        if any(
            not isinstance(prompt, str) or not prompt.strip()
            for prompt in additional_system_prompts
        ):
            raise ValueError("additional system prompts must be non-empty strings")

        self.provider = provider
        self.session_id = session_id or uuid4()
        self.provider_options = dict(provider_options or {})
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("tool names must be unique")
        self.system_prompts = (
            load_system_prompt(),
            *(prompt.strip() for prompt in additional_system_prompts),
        )
        self.checkpointer = InMemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(AgentState)
        provider_node = RunnableLambda(self._call_provider, afunc=self._acall_provider)
        builder.add_node("llm", provider_node)
        builder.add_edge(START, "llm")
        builder.add_edge("llm", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _call_provider(self, state: AgentState, config: RunnableConfig) -> AgentState:
        context = self._runtime_context(config)
        response = self.provider.generate(
            self._messages_for_provider(state, context),
            **self._provider_options(),
        )
        output = self._response_output(response, context=context)
        return {
            "messages": [self._assistant_message(self._history_text(output))],
            "output": output,
        }

    async def _acall_provider(
        self, state: AgentState, config: RunnableConfig
    ) -> AgentState:
        context = self._runtime_context(config)
        response = await self.provider.agenerate(
            self._messages_for_provider(state, context),
            **self._provider_options(),
        )
        output = await self._aresponse_output(response, context=context)
        return {
            "messages": [self._assistant_message(self._history_text(output))],
            "output": output,
        }

    def _provider_options(self) -> dict[str, Any]:
        options = dict(self.provider_options)
        if self.tools:
            options["tools"] = [tool.schema for tool in self.tools.values()]
            options["tool_choice"] = "auto"
        return options

    def _response_output(
        self,
        response: LLMResponse,
        *,
        context: AgentRuntimeContext | None = None,
    ) -> ToolResult:
        if not isinstance(response, LLMResponse):
            raise TypeError("provider must return an LLMResponse")
        if not response.tool_calls:
            return response.content

        results: list[ToolResult] = []
        for tool_call in response.tool_calls:
            try:
                tool = self.tools[tool_call.name]
            except KeyError as error:
                raise ValueError(f"unknown tool requested: {tool_call.name}") from error
            results.append(tool.invoke(tool_call.arguments, context=context))
        return self._combine_tool_results(results)

    async def _aresponse_output(
        self,
        response: LLMResponse,
        *,
        context: AgentRuntimeContext | None = None,
    ) -> ToolResult:
        if not isinstance(response, LLMResponse):
            raise TypeError("provider must return an LLMResponse")
        if not response.tool_calls:
            return response.content

        results = []
        for tool_call in response.tool_calls:
            try:
                tool = self.tools[tool_call.name]
            except KeyError as error:
                raise ValueError(f"unknown tool requested: {tool_call.name}") from error
            results.append(await tool.ainvoke(tool_call.arguments, context=context))
        return self._combine_tool_results(results)

    @staticmethod
    def _combine_tool_results(results: list[ToolResult]) -> ToolResult:
        if len(results) == 1:
            return results[0]
        if any(isinstance(result, MediaResult) for result in results):
            raise ValueError("media tools cannot be combined with other tool calls")
        text_results = [
            result.markdown if isinstance(result, StructuredToolResult) else result
            for result in results
        ]
        return "\n\n".join(text_results)

    @staticmethod
    def _history_text(output: ToolResult) -> str:
        if isinstance(output, MediaResult):
            return f"[Отправлен медиафайл, id={output.id}]"
        if isinstance(output, StructuredToolResult):
            return output.markdown
        return output

    def _messages_for_provider(
        self,
        state: AgentState,
        context: AgentRuntimeContext | None = None,
    ) -> list[Message]:
        messages = [
            *({"role": "system", "content": prompt} for prompt in self.system_prompts),
        ]
        if context is not None:
            messages.append(
                {
                    "role": "system",
                    "content": self._runtime_prompt(context),
                }
            )
        return [*messages, *state["messages"]]

    def _assistant_message(self, response: str) -> Message:
        if not isinstance(response, str):
            raise TypeError("provider must return a string")
        return {
            "role": "assistant",
            "content": response,
        }

    def _config(
        self,
        session_id: UUID | None = None,
        runtime_context: AgentRuntimeContext | None = None,
    ) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": self._thread_id(session_id),
                "runtime_context": runtime_context,
            },
        }

    @staticmethod
    def _runtime_context(config: RunnableConfig) -> AgentRuntimeContext | None:
        context = config.get("configurable", {}).get("runtime_context")
        if context is not None and not isinstance(context, AgentRuntimeContext):
            raise TypeError("runtime_context must be an AgentRuntimeContext")
        return context

    @staticmethod
    def _runtime_prompt(context: AgentRuntimeContext) -> str:
        local_time = context.current_time.astimezone(ZoneInfo(context.timezone))
        return (
            "Доверенные данные текущего запроса: "
            f"сейчас {local_time.isoformat()}, "
            f"таймзона календаря {context.timezone}. "
            "Используй эти дату, время и таймзону для относительных выражений. "
            "Идентификаторы чата и пользователя инструменты получают напрямую "
            "от приложения — не спрашивай их и не передавай в аргументах."
        )

    def _thread_id(self, session_id: UUID | None = None) -> str:
        resolved_session_id = session_id or self.session_id
        if not isinstance(resolved_session_id, UUID):
            raise TypeError("session_id must be a UUID")
        return str(resolved_session_id)

    def invoke(
        self,
        prompt: str,
        *,
        session_id: UUID | None = None,
        runtime_context: AgentRuntimeContext | None = None,
    ) -> ToolResult:
        """Send a prompt within a session and return the agent result."""
        result = self.graph.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=self._config(session_id, runtime_context),
        )
        return result["output"]

    async def ainvoke(
        self,
        prompt: str,
        *,
        session_id: UUID | None = None,
        runtime_context: AgentRuntimeContext | None = None,
    ) -> ToolResult:
        """Asynchronously send a prompt within a session."""
        result = await self.graph.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=self._config(session_id, runtime_context),
        )
        return result["output"]

    def reset_context(self, *, session_id: UUID | None = None) -> None:
        """Delete all stored messages for one dialog session."""
        self.checkpointer.delete_thread(self._thread_id(session_id))

    async def areset_context(self, *, session_id: UUID | None = None) -> None:
        """Asynchronously delete all stored messages for one dialog session."""
        await self.checkpointer.adelete_thread(self._thread_id(session_id))
