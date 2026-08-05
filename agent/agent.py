"""LangGraph-based conversational agent."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Mapping, Sequence
from operator import add
from typing import Annotated, Any, Protocol, TypedDict
from uuid import UUID, uuid4

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from openai import OpenAIError
from pydantic import ValidationError

from .prompts import load_system_prompt

TELEGRAM_MESSAGE_LIMIT = 4096


class LLMProvider(Protocol):
    """Contract that any LLM provider used by :class:`Agent` must implement."""

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> str: ...

    async def agenerate(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> str: ...


Message = dict[str, str]


class AgentState(TypedDict):
    """Conversation history accumulated by the graph checkpointer."""

    messages: Annotated[list[Message], add]


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
        max_response_length: int = TELEGRAM_MESSAGE_LIMIT,
        provider_options: Mapping[str, Any] | None = None,
    ) -> None:
        if max_response_length < 1:
            raise ValueError("max_response_length must be greater than zero")
        if session_id is not None and not isinstance(session_id, UUID):
            raise TypeError("session_id must be a UUID")

        self.provider = provider
        self.session_id = session_id or uuid4()
        self.max_response_length = max_response_length
        self.provider_options = dict(provider_options or {})
        self.system_prompt = load_system_prompt()
        self.checkpointer = InMemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(AgentState)
        provider_node = RunnableLambda(self._call_provider, afunc=self._acall_provider)
        builder.add_node("llm", provider_node)
        builder.add_edge(START, "llm")
        builder.add_edge("llm", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _call_provider(self, state: AgentState) -> AgentState:
        response = self.provider.generate(
            self._messages_for_provider(state),
            **self.provider_options,
        )
        return {"messages": [self._assistant_message(response)]}

    async def _acall_provider(self, state: AgentState) -> AgentState:
        response = await self.provider.agenerate(
            self._messages_for_provider(state),
            **self.provider_options,
        )
        return {"messages": [self._assistant_message(response)]}

    def _messages_for_provider(self, state: AgentState) -> list[Message]:
        return [
            {"role": "system", "content": self.system_prompt},
            *state["messages"],
        ]

    def _assistant_message(self, response: str) -> Message:
        if not isinstance(response, str):
            raise TypeError("provider must return a string")
        return {
            "role": "assistant",
            "content": response[: self.max_response_length],
        }

    def _config(self, session_id: UUID | None = None) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": self._thread_id(session_id)},
        }

    def _thread_id(self, session_id: UUID | None = None) -> str:
        resolved_session_id = session_id or self.session_id
        if not isinstance(resolved_session_id, UUID):
            raise TypeError("session_id must be a UUID")
        return str(resolved_session_id)

    def invoke(self, prompt: str, *, session_id: UUID | None = None) -> str:
        """Send a prompt within a session and return a size-limited response."""
        result = self.graph.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=self._config(session_id),
        )
        return result["messages"][-1]["content"]

    async def ainvoke(self, prompt: str, *, session_id: UUID | None = None) -> str:
        """Asynchronously send a prompt within a session."""
        result = await self.graph.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=self._config(session_id),
        )
        return result["messages"][-1]["content"]

    def reset_context(self, *, session_id: UUID | None = None) -> None:
        """Delete all stored messages for one dialog session."""
        self.checkpointer.delete_thread(self._thread_id(session_id))

    async def areset_context(self, *, session_id: UUID | None = None) -> None:
        """Asynchronously delete all stored messages for one dialog session."""
        await self.checkpointer.adelete_thread(self._thread_id(session_id))


async def _run_console(session_id: UUID | None = None) -> None:
    """Run an interactive console dialog using the configured provider."""
    from .providers import OpenRouterProvider

    async with OpenRouterProvider() as provider:
        agent = Agent(provider, session_id=session_id)

        print(f"Session: {agent.session_id}")
        print("Commands: /reset — reset context, /exit — exit")

        while True:
            try:
                prompt = await asyncio.to_thread(input, "You: ")
            except EOFError, KeyboardInterrupt:
                print()
                break

            prompt = prompt.strip()
            if not prompt:
                continue
            if prompt in {"/exit", "/quit"}:
                break
            if prompt == "/reset":
                await agent.areset_context()
                print("Context reset.")
                continue

            try:
                response = await agent.ainvoke(prompt)
            except OpenAIError as error:
                print(f"Error: {error}")
                continue

            print(f"Agent: {response}")


def main() -> None:
    """Parse command-line arguments and start the interactive agent."""
    parser = argparse.ArgumentParser(description="Run the agent in the console")
    parser.add_argument(
        "--session-id",
        type=UUID,
        help="UUID to use for this conversation (a new UUID is created by default)",
    )
    arguments = parser.parse_args()
    try:
        asyncio.run(_run_console(arguments.session_id))
    except ValidationError as error:
        print(f"Failed to start agent: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
