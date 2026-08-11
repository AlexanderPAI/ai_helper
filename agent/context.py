"""Trusted request metadata supplied by an interface adapter."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class AgentRuntimeContext:
    """Metadata that tools receive from the application, never from the model."""

    chat_id: int
    user_id: int | None
    user_display_name: str | None
    message_id: int | None
    current_time: datetime
    timezone: str


__all__ = ["AgentRuntimeContext"]
