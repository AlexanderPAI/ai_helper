"""SQLAlchemy engine and session factories."""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .settings import DatabaseSettings


def create_database_engine(settings: DatabaseSettings) -> AsyncEngine:
    """Create an asynchronous SQLAlchemy engine from validated settings."""
    return create_async_engine(
        str(settings.url),
        pool_pre_ping=True,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout,
    )


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create sessions that keep loaded values after transaction commits."""
    return async_sessionmaker(engine, expire_on_commit=False)
