"""Add the lease recovery index for the reminder worker.

Revision ID: 20260807_02
Revises: 20260807_01
Create Date: 2026-08-07
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260807_02"
down_revision: str | None = "20260807_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Index processing reminders by lease timestamp."""
    op.create_index(
        "ix_event_reminders_status_locked_at",
        "event_reminders",
        ["status", "locked_at"],
    )


def downgrade() -> None:
    """Remove the lease recovery index."""
    op.drop_index("ix_event_reminders_status_locked_at", table_name="event_reminders")
