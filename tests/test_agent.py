from __future__ import annotations

import unittest
from unittest.mock import Mock

from agent.agent import Agent


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


if __name__ == "__main__":
    unittest.main()
