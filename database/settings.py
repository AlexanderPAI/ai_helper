"""Database settings loaded from environment variables."""

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

from config import PROJECT_ROOT


class DatabaseSettings(BaseSettings):
    """Validated settings for the asynchronous PostgreSQL connection pool."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="DATABASE_",
        env_ignore_empty=True,
        extra="ignore",
    )

    url: PostgresDsn
    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=5, ge=0)
    pool_timeout: float = Field(default=30.0, gt=0)
