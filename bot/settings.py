"""Telegram bot settings loaded from environment variables."""

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config import PROJECT_ROOT


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

    @property
    def chat_ids(self) -> frozenset[int]:
        """Return allowed Telegram chat IDs."""
        return frozenset(
            int(item.strip())
            for item in self.allowed_chat_ids.split(",")
            if item.strip()
        )
