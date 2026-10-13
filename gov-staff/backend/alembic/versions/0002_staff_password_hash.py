"""Optional password hash on staff_users (dummy test logins for now).

The sign-in method is still to be decided (PLAN 12.5). The column is nullable,
and only a salted scrypt hash is ever stored.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-13
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("staff_users") as batch:
        batch.add_column(sa.Column("password_hash", sa.String(255)))


def downgrade() -> None:
    with op.batch_alter_table("staff_users") as batch:
        batch.drop_column("password_hash")
