"""Validated action plans used by the agent orchestration layer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .tools import AgentToolError

PLAN_TOOL_NAME = "create_action_plan"


class ActionPlanError(AgentToolError):
    """Raised when the model returns an invalid or unsafe action plan."""


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One planned tool capability and its dependencies."""

    id: str
    tool: str
    purpose: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ActionPlan:
    """Explicit set of user-authorized tool operations for one request."""

    goals: tuple[str, ...]
    steps: tuple[PlanStep, ...]

    @property
    def allowed_tools(self) -> frozenset[str]:
        return frozenset(step.tool for step in self.steps)

    def ready_tools(self, completed_steps: frozenset[str]) -> frozenset[str]:
        """Return tools whose declared dependencies have completed."""
        return frozenset(
            step.tool
            for step in self.steps
            if step.id not in completed_steps
            and set(step.depends_on) <= completed_steps
        )

    def completed_after_calls(
        self,
        completed_steps: frozenset[str],
        called_tools: Sequence[str],
    ) -> frozenset[str]:
        """Mark one ready plan step complete for every executed tool call."""
        completed = set(completed_steps)
        for tool in called_tools:
            step = next(
                (
                    candidate
                    for candidate in self.steps
                    if candidate.id not in completed
                    and candidate.tool == tool
                    and set(candidate.depends_on) <= completed
                ),
                None,
            )
            if step is None:
                raise ActionPlanError(f"tool call violates plan order: {tool}")
            completed.add(step.id)
        return frozenset(completed)

    @classmethod
    def from_arguments(
        cls,
        arguments: Mapping[str, Any],
        *,
        available_tools: frozenset[str],
    ) -> ActionPlan:
        goals = _string_sequence(arguments.get("goals"), field="goals", maximum=8)
        raw_steps = arguments.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ActionPlanError("action plan must contain at least one step")
        if len(raw_steps) > 12:
            raise ActionPlanError("action plan contains too many steps")

        steps: list[PlanStep] = []
        known_ids: set[str] = set()
        for raw_step in raw_steps:
            if not isinstance(raw_step, Mapping):
                raise ActionPlanError("every action plan step must be an object")
            step_id = _string(raw_step.get("id"), field="step id", maximum=32)
            tool = _string(raw_step.get("tool"), field="step tool", maximum=100)
            purpose = _string(
                raw_step.get("purpose"), field="step purpose", maximum=300
            )
            if step_id in known_ids:
                raise ActionPlanError(f"duplicate action plan step id: {step_id}")
            if tool not in available_tools:
                raise ActionPlanError(f"unknown tool in action plan: {tool}")
            dependencies = _string_sequence(
                raw_step.get("depends_on", []),
                field="depends_on",
                maximum=12,
                allow_empty=True,
            )
            unknown_dependencies = set(dependencies) - known_ids
            if unknown_dependencies:
                raise ActionPlanError(
                    "action plan dependencies must reference earlier steps"
                )
            known_ids.add(step_id)
            steps.append(PlanStep(step_id, tool, purpose, dependencies))
        return cls(goals=goals, steps=tuple(steps))


def action_plan_tool_schema(
    available_tools: Mapping[str, str],
) -> Mapping[str, Any]:
    """Return the internal model-facing schema for declaring requested actions."""
    names = sorted(available_tools)
    catalog = "\n".join(f"- {name}: {available_tools[name]}" for name in names)
    return {
        "type": "function",
        "function": {
            "name": PLAN_TOOL_NAME,
            "description": (
                "Составить план только для явно запрошенных пользователем действий. "
                "Вызывай перед любыми прикладными инструментами. Для обычного "
                "разговора не вызывай этот инструмент и ответь текстом. "
                f"Доступные прикладные инструменты:\n{catalog}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goals": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {"type": "string", "minLength": 1, "maxLength": 200},
                        "description": "Независимые цели из текущей просьбы.",
                    },
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 12,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 32,
                                },
                                "tool": {"type": "string", "enum": names},
                                "purpose": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 300,
                                },
                                "depends_on": {
                                    "type": "array",
                                    "maxItems": 12,
                                    "items": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": 32,
                                    },
                                },
                            },
                            "required": ["id", "tool", "purpose", "depends_on"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["goals", "steps"],
                "additionalProperties": False,
            },
        },
    }


def _string(value: Any, *, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ActionPlanError(f"{field} must be a non-empty string")
    result = value.strip()
    if len(result) > maximum:
        raise ActionPlanError(f"{field} is too long")
    return result


def _string_sequence(
    value: Any,
    *,
    field: str,
    maximum: int,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ActionPlanError(f"{field} must be a non-empty array")
    if len(value) > maximum:
        raise ActionPlanError(f"{field} contains too many values")
    return tuple(_string(item, field=field, maximum=300) for item in value)


__all__ = [
    "PLAN_TOOL_NAME",
    "ActionPlan",
    "ActionPlanError",
    "PlanStep",
    "action_plan_tool_schema",
]
