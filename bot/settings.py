"""Telegram bot settings loaded from environment variables."""

from datetime import timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config import PROJECT_ROOT
from reminder_app import ReminderWorkerOptions


class TelegramSettings(BaseSettings):
    """Validated Telegram bot configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="TELEGRAM_",
        env_ignore_empty=True,
        extra="ignore",
    )

    bot_token: SecretStr
    allowed_chat_ids: str = Field(min_length=1)
    calendar_default_timezone: str = "Europe/Moscow"

    @field_validator("allowed_chat_ids")
    @classmethod
    def validate_allowed_chat_ids(cls, value: str) -> str:
        """Ensure the comma-separated value contains only integer chat IDs."""
        try:
            chat_ids = {int(item.strip()) for item in value.split(",") if item.strip()}
        except ValueError as error:
            raise ValueError("must be a comma-separated list of integers") from error
        if not chat_ids:
            raise ValueError("must contain at least one chat ID")
        return value

    @field_validator("calendar_default_timezone")
    @classmethod
    def validate_calendar_default_timezone(cls, value: str) -> str:
        """Require an IANA timezone usable for chats without saved settings."""
        try:
            return ZoneInfo(value.strip()).key
        except (AttributeError, ZoneInfoNotFoundError) as error:
            raise ValueError("must be a valid IANA timezone") from error

    @property
    def chat_ids(self) -> frozenset[int]:
        """Return allowed Telegram chat IDs."""
        return frozenset(
            int(item.strip())
            for item in self.allowed_chat_ids.split(",")
            if item.strip()
        )


class ReminderWorkerSettings(BaseSettings):
    """Validated runtime settings for the embedded reminder worker."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="REMINDER_",
        env_ignore_empty=True,
        extra="ignore",
    )

    poll_interval: float = Field(default=10.0, gt=0)
    batch_size: int = Field(default=20, ge=1, le=1000)
    lease_timeout_seconds: float = Field(default=300.0, gt=0)
    max_attempts: int = Field(default=5, ge=1)
    retry_base_delay_seconds: float = Field(default=30.0, gt=0)
    retry_max_delay_seconds: float = Field(default=3600.0, gt=0)
    retry_jitter_ratio: float = Field(default=0.1, ge=0, le=1)
    shutdown_timeout: float = Field(default=30.0, gt=0)

    @property
    def worker_options(self) -> ReminderWorkerOptions:
        """Map environment settings to application-layer worker options."""
        return ReminderWorkerOptions(
            poll_interval=self.poll_interval,
            batch_size=self.batch_size,
            lease_timeout=timedelta(seconds=self.lease_timeout_seconds),
            max_attempts=self.max_attempts,
            retry_base_delay=timedelta(seconds=self.retry_base_delay_seconds),
            retry_max_delay=timedelta(seconds=self.retry_max_delay_seconds),
            retry_jitter_ratio=self.retry_jitter_ratio,
        )


__all__ = ["ReminderWorkerSettings", "TelegramSettings"]
