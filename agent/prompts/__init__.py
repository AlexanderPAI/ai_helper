"""Load prompts bundled with the agent package."""

from importlib.resources import files
from typing import Any

import yaml

SYSTEM_PROMPT_FILE = files(__package__).joinpath("system.yaml")


def load_system_prompt() -> str:
    """Load and validate the agent system prompt from YAML."""
    content: Any = yaml.safe_load(SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise TypeError("system prompt YAML must contain a mapping")

    prompt = content.get("system_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")
    return prompt.strip()


__all__ = ["load_system_prompt"]
