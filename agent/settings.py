"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class OpenRouterSettings(BaseSettings):
    """Validated OpenRouter configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="OPENROUTER_",
        env_ignore_empty=True,
        extra="ignore",
    )

    api_key: SecretStr
    model: str = Field(min_length=1)
    base_url: AnyHttpUrl = AnyHttpUrl("https://openrouter.ai/api/v1")
    site_url: AnyHttpUrl | None = None
    app_name: str | None = "ai-helper"
    timeout: float = Field(default=60.0, gt=0)


class HumorAPISettings(BaseSettings):
    """Validated Humor API configuration for the meme tool."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="HUMOR_API_",
        env_ignore_empty=True,
        extra="ignore",
    )

    api_key: SecretStr
    random_url: AnyHttpUrl
    search_url: AnyHttpUrl
    timeout: float = Field(default=15.0, gt=0)
    user_agent: str = Field(min_length=1)
