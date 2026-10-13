"""P0: departments, staff_users, otp_codes, verification_requests, audit_log.

PostgreSQL also gets the isolation and integrity protections:
- a NOLOGIN role `staff_app_rw` that the app's login role is a member of
  (create the login role yourself: see docs/ARCHITECTURE.md), with only the
  privileges it needs; nothing for PUBLIC;
- Row-Level Security keyed on department (`app.department_id`, set per request
  from the authenticated user; `app.is_super_admin` for the platform admin),
  plus one narrow "self" policy so the auth code can find exactly one user by
  `app.auth_email` (login/OTP) or `app.user_id` (loading the session) before
  the department is known;
- a trigger making every new account pending and below admin (unless the super
  admin creates it), and refusing a change of department (super admin only), a promotion to
  super_admin (super admin only), and any other role change or activation
  outside an admin context (a department scope or the super admin);
- audit_log append-only (UPDATE/DELETE refused by trigger, and not granted).
SQLite (dev/tests) gets the same append-only trigger.

Revision ID: 0001
Revises:
Create Date: 2026-10-13
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

APP_ROLE = "staff_app_rw"
TS = sa.DateTime(timezone=True)
JSONT = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
SCOPED = ("staff_users", "verification_requests", "audit_log")
SCOPE = ("current_setting('app.is_super_admin', true) = 'true' "
         "OR department_id::text = current_setting('app.department_id', true)")
SELF = ("email = current_setting('app.auth_email', true) "
        "OR id = current_setting('app.user_id', true)")


def upgrade() -> None:
    op.create_table(
        "departments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("code", sa.String(40), nullable=False, unique=True),
        sa.Column("name_en", sa.String(200), nullable=False),
        sa.Column("name_hi", sa.String(200), nullable=False),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", TS, nullable=False),
    )
    op.create_table(
        "staff_users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(254), nullable=False, unique=True),
        sa.Column("full_name", sa.String(200), nullable=False),
        sa.Column("mobile", sa.String(20)),
        sa.Column("designation", sa.String(200), nullable=False),
        sa.Column("employee_id", sa.String(60)),
        sa.Column("department_id", sa.String(36), sa.ForeignKey("departments.id")),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("approved_at", TS),
        sa.Column("approved_by_id", sa.String(36), sa.ForeignKey("staff_users.id")),
        sa.Column("last_login_at", TS),
        sa.CheckConstraint("role IN ('super_admin','dept_admin','officer','viewer')", name="ck_staff_users_role"),
        sa.CheckConstraint("status IN ('pending_email','pending_approval','active','rejected','suspended')",
                           name="ck_staff_users_status"),
        # every user except the platform super admin belongs to a department
        sa.CheckConstraint("role = 'super_admin' OR department_id IS NOT NULL", name="ck_staff_users_department"),
    )
    op.create_index("ix_staff_users_department_id", "staff_users", ["department_id"])
    op.create_table(
        "otp_codes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(254), nullable=False),
        sa.Column("purpose", sa.String(20), nullable=False),
        sa.Column("code_hash", sa.String(128), nullable=False),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("expires_at", TS, nullable=False),
        sa.Column("consumed_at", TS),
        sa.Column("created_at", TS, nullable=False),
    )
    op.create_index("ix_otp_codes_email", "otp_codes", ["email"])
    op.create_table(
        "verification_requests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("staff_users.id"), nullable=False),
        sa.Column("department_id", sa.String(36), sa.ForeignKey("departments.id"), nullable=False),
        sa.Column("reason", sa.Text),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("decision_note", sa.Text),
        sa.Column("decided_by_id", sa.String(36), sa.ForeignKey("staff_users.id")),
        sa.Column("decided_at", TS),
        sa.Column("created_at", TS, nullable=False),
        sa.CheckConstraint("status IN ('open','approved','rejected')", name="ck_verification_requests_status"),
    )
    op.create_index("ix_verification_requests_user_id", "verification_requests", ["user_id"])
    op.create_index("ix_verification_requests_department_id", "verification_requests", ["department_id"])
    op.create_table(
        "audit_log",
        # client-generated id: inserts need no RETURNING, so appending works
        # before a user is identified (RLS applies SELECT policies to RETURNING)
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("at", TS, nullable=False),
        sa.Column("actor_user_id", sa.String(36)),
        sa.Column("department_id", sa.String(36)),
        sa.Column("action", sa.String(80), nullable=False),
        sa.Column("entity_type", sa.String(40)),
        sa.Column("entity_id", sa.String(64)),
        sa.Column("ip", sa.String(64)),
        sa.Column("details", JSONT, nullable=False),
    )
    for col in ("at", "actor_user_id", "department_id", "action"):
        op.create_index(f"ix_audit_log_{col}", "audit_log", [col])

    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        _postgres_protections()
    elif dialect == "sqlite":
        for verb in ("UPDATE", "DELETE"):
            op.execute(f"CREATE TRIGGER audit_log_no_{verb.lower()} BEFORE {verb} ON audit_log "
                       "BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END")


def _postgres_protections() -> None:
    op.execute(f"""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                CREATE ROLE {APP_ROLE} NOLOGIN;
            END IF;
        END $$""")
    tables = ("departments", "staff_users", "otp_codes", "verification_requests", "audit_log", "alembic_version")
    for table in tables:
        op.execute(f"REVOKE ALL ON {table} FROM PUBLIC")
    op.execute(f"GRANT SELECT ON departments TO {APP_ROLE}")  # departments are managed by migrations/scripts
    op.execute(f"GRANT SELECT, INSERT, UPDATE ON staff_users, verification_requests TO {APP_ROLE}")
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON otp_codes TO {APP_ROLE}")
    op.execute(f"GRANT SELECT, INSERT ON audit_log TO {APP_ROLE}")

    for table in SCOPED:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    op.execute(f"CREATE POLICY staff_users_scope ON staff_users TO {APP_ROLE} USING ({SCOPE}) WITH CHECK ({SCOPE})")
    # Auth lookup may read and update its own row, never insert (registration
    # inserts under the chosen department's scope). Role/activation changes are
    # further limited by the staff_users_guard trigger.
    op.execute(f"CREATE POLICY staff_users_self_read ON staff_users FOR SELECT TO {APP_ROLE} USING ({SELF})")
    op.execute(f"CREATE POLICY staff_users_self_update ON staff_users FOR UPDATE TO {APP_ROLE} "
               f"USING ({SELF}) WITH CHECK ({SELF})")
    op.execute(f"CREATE POLICY verification_requests_scope ON verification_requests TO {APP_ROLE} "
               f"USING ({SCOPE}) WITH CHECK ({SCOPE})")
    # Anyone may append (failed logins have no department); reading is scoped.
    op.execute(f"CREATE POLICY audit_log_read ON audit_log FOR SELECT TO {APP_ROLE} USING ({SCOPE})")
    op.execute(f"CREATE POLICY audit_log_append ON audit_log FOR INSERT TO {APP_ROLE} WITH CHECK (true)")

    op.execute("""
        CREATE OR REPLACE FUNCTION staff_users_guard() RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE
            is_super boolean := coalesce(current_setting('app.is_super_admin', true), '') = 'true';
            in_admin_context boolean := is_super OR coalesce(current_setting('app.department_id', true), '') <> '';
        BEGIN
            IF TG_OP = 'INSERT' THEN
                -- new accounts (registration) always start pending and below admin;
                -- only the super admin creates admins or active users directly
                IF NOT is_super AND (NEW.status NOT IN ('pending_email', 'pending_approval')
                                     OR NEW.role IN ('super_admin', 'dept_admin')) THEN
                    RAISE EXCEPTION 'new accounts must be pending and below admin';
                END IF;
                RETURN NEW;
            END IF;
            IF NEW.department_id IS DISTINCT FROM OLD.department_id AND NOT is_super THEN
                RAISE EXCEPTION 'only the super admin can move a user to another department';
            END IF;
            IF NEW.role = 'super_admin' AND OLD.role IS DISTINCT FROM 'super_admin' AND NOT is_super THEN
                RAISE EXCEPTION 'only the super admin can grant super_admin';
            END IF;
            IF NEW.role IS DISTINCT FROM OLD.role AND NOT in_admin_context THEN
                RAISE EXCEPTION 'role changes need an admin context';
            END IF;
            IF NEW.status = 'active' AND OLD.status IS DISTINCT FROM 'active' AND NOT in_admin_context THEN
                RAISE EXCEPTION 'activation needs an admin context';
            END IF;
            RETURN NEW;
        END $$""")
    op.execute("CREATE TRIGGER staff_users_guard BEFORE INSERT OR UPDATE ON staff_users "
               "FOR EACH ROW EXECUTE FUNCTION staff_users_guard()")
    op.execute("""
        CREATE OR REPLACE FUNCTION audit_log_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'audit_log is append-only'; END $$""")
    op.execute("CREATE TRIGGER audit_log_no_change BEFORE UPDATE OR DELETE ON audit_log "
               "FOR EACH ROW EXECUTE FUNCTION audit_log_append_only()")
    op.execute("CREATE TRIGGER audit_log_no_truncate BEFORE TRUNCATE ON audit_log "
               "FOR EACH STATEMENT EXECUTE FUNCTION audit_log_append_only()")


def downgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        op.execute("DROP TRIGGER IF EXISTS audit_log_no_truncate ON audit_log")
        op.execute("DROP TRIGGER IF EXISTS audit_log_no_change ON audit_log")
        op.execute("DROP FUNCTION IF EXISTS audit_log_append_only()")
        op.execute("DROP TRIGGER IF EXISTS staff_users_guard ON staff_users")
        op.execute("DROP FUNCTION IF EXISTS staff_users_guard()")
    elif dialect == "sqlite":
        op.execute("DROP TRIGGER IF EXISTS audit_log_no_update")
        op.execute("DROP TRIGGER IF EXISTS audit_log_no_delete")
    op.drop_table("audit_log")
    op.drop_table("verification_requests")
    op.drop_table("otp_codes")
    op.drop_table("staff_users")
    op.drop_table("departments")
