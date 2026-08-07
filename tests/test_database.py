from __future__ import annotations

import os
import unittest

from pydantic import PostgresDsn
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

from database.models import Base
from database.settings import DatabaseSettings


class DatabaseSettingsTest(unittest.TestCase):
    def test_accepts_asyncpg_url_and_pool_settings(self) -> None:
        settings = DatabaseSettings(
            url=PostgresDsn("postgresql+asyncpg://app:secret@db:5432/app"),
            pool_size=7,
            max_overflow=3,
            pool_timeout=12,
        )

        self.assertEqual(
            str(settings.url), "postgresql+asyncpg://app:secret@db:5432/app"
        )
        self.assertEqual(settings.pool_size, 7)
        self.assertEqual(settings.max_overflow, 3)
        self.assertEqual(settings.pool_timeout, 12)


class CalendarMetadataTest(unittest.TestCase):
    def test_contains_initial_calendar_schema(self) -> None:
        self.assertEqual(
            set(Base.metadata.tables),
            {"calendar_chats", "calendar_events", "event_reminders"},
        )

        reminder_columns = Base.metadata.tables["event_reminders"].columns
        self.assertIn("message_text", reminder_columns)
        self.assertFalse(reminder_columns["message_text"].nullable)


@unittest.skipUnless(
    os.environ.get("DATABASE_TEST_URL"),
    "DATABASE_TEST_URL is required for PostgreSQL integration tests",
)
class MigratedPostgreSQLSchemaTest(unittest.IsolatedAsyncioTestCase):
    async def test_migration_created_tables_constraints_and_indexes(self) -> None:
        engine = create_async_engine(os.environ["DATABASE_TEST_URL"])
        try:
            async with engine.connect() as connection:
                schema = await connection.run_sync(self._inspect_schema)
        finally:
            await engine.dispose()

        self.assertEqual(
            schema["tables"],
            {
                "alembic_version",
                "calendar_chats",
                "calendar_events",
                "event_reminders",
            },
        )
        self.assertIn(
            "ix_calendar_events_chat_status_starts_at", schema["event_indexes"]
        )
        self.assertIn(
            "ix_event_reminders_status_next_attempt_at",
            schema["reminder_indexes"],
        )
        self.assertIn(
            "ix_event_reminders_status_locked_at",
            schema["reminder_indexes"],
        )
        self.assertIn(
            "ux_event_reminders_event_remind_at_open",
            schema["reminder_indexes"],
        )
        self.assertIn(
            "ck_event_reminders_message_text_not_blank",
            schema["reminder_check_constraints"],
        )

    @staticmethod
    def _inspect_schema(connection: object) -> dict[str, set[str]]:
        inspector = inspect(connection)
        return {
            "tables": set(inspector.get_table_names()),
            "event_indexes": {
                item["name"] for item in inspector.get_indexes("calendar_events")
            },
            "reminder_indexes": {
                item["name"] for item in inspector.get_indexes("event_reminders")
            },
            "reminder_check_constraints": {
                item["name"]
                for item in inspector.get_check_constraints("event_reminders")
            },
        }
