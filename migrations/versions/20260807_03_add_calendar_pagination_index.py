"""Add the stable calendar pagination index.

Revision ID: 20260807_03
Revises: 20260807_02
Create Date: 2026-08-07
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260807_03"
down_revision: str | None = "20260807_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Replace the list index with one supporting the cursor tie-breaker."""
    op.drop_index(
        "ix_calendar_events_chat_status_starts_at",
        table_name="calendar_events",
    )
    op.create_index(
        "ix_calendar_events_chat_status_starts_at_id",
        "calendar_events",
        ["chat_id", "status", "starts_at", "id"],
    )


def downgrade() -> None:
    """Restore the pre-pagination list index."""
    op.drop_index(
        "ix_calendar_events_chat_status_starts_at_id",
        table_name="calendar_events",
    )
    op.create_index(
        "ix_calendar_events_chat_status_starts_at",
        "calendar_events",
        ["chat_id", "status", "starts_at"],
    )
