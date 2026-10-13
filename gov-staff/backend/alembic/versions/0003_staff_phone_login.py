"""Phone number login: one account per mobile number, and the auth-lookup
("self") policies can also find a user by `app.auth_phone`.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-13
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

APP_ROLE = "staff_app_rw"
SELF_OLD = ("email = current_setting('app.auth_email', true) "
            "OR id = current_setting('app.user_id', true)")
SELF_NEW = SELF_OLD + " OR mobile = current_setting('app.auth_phone', true)"


def _self_policies(condition: str) -> None:
    op.execute("DROP POLICY IF EXISTS staff_users_self_read ON staff_users")
    op.execute("DROP POLICY IF EXISTS staff_users_self_update ON staff_users")
    op.execute(f"CREATE POLICY staff_users_self_read ON staff_users FOR SELECT TO {APP_ROLE} USING ({condition})")
    op.execute(f"CREATE POLICY staff_users_self_update ON staff_users FOR UPDATE TO {APP_ROLE} "
               f"USING ({condition}) WITH CHECK ({condition})")


def upgrade() -> None:
    op.create_index("uq_staff_users_mobile", "staff_users", ["mobile"], unique=True)
    if op.get_bind().dialect.name == "postgresql":
        _self_policies(SELF_NEW)


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        _self_policies(SELF_OLD)
    op.drop_index("uq_staff_users_mobile", table_name="staff_users")
