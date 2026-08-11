"""Load prompts bundled with the agent package."""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import yaml

SYSTEM_PROMPT_FILE = files(__package__).joinpath("system.yaml")
TOOL_PROMPTS_FILE = files(__package__).joinpath("tools.yaml")


@dataclass(frozen=True, slots=True)
class ToolPrompt:
    """Model-facing description of a tool and its parameters."""

    description: str
    parameter_descriptions: Mapping[str, str]


def load_system_prompt() -> str:
    """Load and validate the agent system prompt from YAML."""
    content: Any = yaml.safe_load(SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise TypeError("system prompt YAML must contain a mapping")

    prompt = content.get("system_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")
    return prompt.strip()


def load_tool_prompt(tool_name: str) -> ToolPrompt:
    """Load and validate one tool's model-facing instructions from YAML."""
    content: Any = yaml.safe_load(TOOL_PROMPTS_FILE.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or not isinstance(content.get("tools"), dict):
        raise TypeError("tool prompts YAML must contain a tools mapping")

    tool = content["tools"].get(tool_name)
    if tool is None:
        raise KeyError(f"tool prompt is not configured: {tool_name}")
    if not isinstance(tool, dict):
        raise TypeError(f"tool prompt must be a mapping: {tool_name}")

    description = tool.get("description")
    parameters = tool.get("parameters")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"tool description must be a non-empty string: {tool_name}")
    if not isinstance(parameters, dict) or any(
        not isinstance(name, str) or not isinstance(value, str) or not value.strip()
        for name, value in parameters.items()
    ):
        raise ValueError(
            f"tool parameters must contain string descriptions: {tool_name}"
        )

    return ToolPrompt(
        description=description.strip(),
        parameter_descriptions={
            name: value.strip() for name, value in parameters.items()
        },
    )


__all__ = ["ToolPrompt", "load_system_prompt", "load_tool_prompt"]
