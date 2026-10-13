"""Isolation layer 2 on real PostgreSQL: Row-Level Security by department,
the narrow auth-lookup policy, restricted app role, append-only audit log.

Skipped unless TEST_POSTGRES_URL points at an EMPTY throwaway database as a
**superuser** (or a BYPASSRLS + CREATEROLE role), e.g.
    TEST_POSTGRES_URL=postgresql+psycopg://postgres:<pw>@localhost:5432/staff_test
The test runs the migrations, creates a temporary login role in staff_app_rw,
checks the policies through that role, then downgrades and drops the role.
"""

import os
import secrets
import uuid
from datetime import datetime, timezone

import pytest
from alembic import command
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import sessionmaker

from app.core.audit import write_audit
from app.core.db import set_auth_lookup, set_department_scope
from app.models import AuditLog, StaffUser
from tests.conftest import alembic_config

PG_URL = os.environ.get("TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="set TEST_POSTGRES_URL to run the PostgreSQL isolation checks")


@pytest.fixture
def pg():
    owner = create_engine(PG_URL, isolation_level="AUTOCOMMIT")
    role, password = f"staff_test_{secrets.token_hex(4)}", secrets.token_urlsafe(16)
    cfg = alembic_config(PG_URL)
    app = None
    try:
        command.upgrade(cfg, "head")
        now = datetime.now(timezone.utc)
        a, b = str(uuid.uuid4()), str(uuid.uuid4())
        with owner.connect() as c:
            c.execute(text(f"CREATE ROLE {role} LOGIN PASSWORD '{password}' IN ROLE staff_app_rw"))
            c.execute(text("SELECT set_config('app.is_super_admin', 'true', false)"))  # seeding = super admin action
            for dept, code in ((a, "agri"), (b, "health")):
                c.execute(text("INSERT INTO departments (id, code, name_en, name_hi, active, created_at) "
                               "VALUES (:i, :c, :c, :c, true, :t)"), {"i": dept, "c": code, "t": now})
                c.execute(text("INSERT INTO staff_users (id, email, full_name, designation, department_id, role, status, created_at) "
                               "VALUES (:i, :e, 'U', 'Officer', :d, 'officer', 'active', :t)"),
                          {"i": str(uuid.uuid4()), "e": f"user@{code}.gov.in", "d": dept, "t": now})
        app = create_engine(make_url(PG_URL).set(username=role, password=password))
        yield {"app": app, "owner": owner, "a": a, "b": b, "Session": sessionmaker(bind=app, expire_on_commit=False)}
    finally:
        if app is not None:
            app.dispose()
        command.downgrade(cfg, "base")
        with owner.connect() as c:
            c.execute(text(f"DROP ROLE IF EXISTS {role}"))
        owner.dispose()


def _scoped(conn, dept, super_admin=False):
    conn.execute(text("SELECT set_config('app.department_id', :d, true), set_config('app.is_super_admin', :s, true)"),
                 {"d": dept or "", "s": "true" if super_admin else "false"})


def test_users_see_only_their_department(pg):
    with pg["app"].begin() as conn:
        _scoped(conn, pg["a"])
        assert conn.execute(text("SELECT email FROM staff_users")).scalars().all() == ["user@agri.gov.in"]
    with pg["app"].begin() as conn:  # no scope = nothing
        assert conn.execute(text("SELECT count(*) FROM staff_users")).scalar() == 0
    with pg["app"].begin() as conn:
        _scoped(conn, None, super_admin=True)
        assert conn.execute(text("SELECT count(*) FROM staff_users")).scalar() == 2


def test_cannot_write_into_or_move_to_another_department(pg):
    with pytest.raises(DBAPIError):
        with pg["app"].begin() as conn:
            _scoped(conn, pg["a"])
            conn.execute(text("INSERT INTO staff_users (id, email, full_name, designation, department_id, role, status, created_at) "
                              "VALUES (:i, 'x@health.gov.in', 'X', 'Clerk', :d, 'viewer', 'active', now())"),
                         {"i": str(uuid.uuid4()), "d": pg["b"]})
    with pytest.raises(DBAPIError, match="only the super admin"):
        with pg["app"].begin() as conn:
            _scoped(conn, pg["a"])
            conn.execute(text("UPDATE staff_users SET department_id = :b WHERE email = 'user@agri.gov.in'"), {"b": pg["b"]})


def test_orm_audit_write_works_without_a_scope(pg):
    """Failed logins / registrations are audited before any user is known."""
    with pg["Session"]() as db:
        write_audit(db, "auth.login_failed", details={"reason": "unknown email"})
        write_audit(db, "x.dept_event", department_id=pg["a"])
        db.commit()
    with pg["Session"]() as db:
        set_department_scope(db, pg["a"])
        assert [r.action for r in db.scalars(select(AuditLog))] == ["x.dept_event"]  # reading stays scoped


def test_auth_lookup_finds_exactly_one_user(pg):
    with pg["Session"]() as db:
        set_auth_lookup(db, email="USER@agri.gov.in")
        users = db.scalars(select(StaffUser)).all()
        assert [u.email for u in users] == ["user@agri.gov.in"]  # never the other department's users
        user = users[0]
        user.last_login_at = datetime.now(timezone.utc)  # login can update its own row
        db.commit()
        user_id, dept = user.id, user.department_id
    with pg["Session"]() as db:
        set_auth_lookup(db, user_id=user_id)  # loading a session by id
        assert db.scalar(select(StaffUser)).department_id == dept
    with pg["Session"]() as db:
        set_auth_lookup(db, email="nobody@agri.gov.in")
        assert db.scalars(select(StaffUser)).all() == []


def test_auth_lookup_cannot_insert_or_self_promote(pg):
    """Before the department is known, the self policy reads/updates one row only:
    no inserts, no role change, no self-activation (a P1 bug can't become a breach)."""
    with pg["Session"]() as db:
        set_auth_lookup(db, email="evil@health.gov.in")
        db.add(StaffUser(email="evil@health.gov.in", full_name="E", designation="X", department_id=pg["b"],
                         role="dept_admin", status="active"))
        with pytest.raises(DBAPIError):
            db.commit()
    for change, error in (({"role": "super_admin"}, "super_admin"), ({"role": "dept_admin"}, "admin context"),
                          ({"status": "active"}, "admin context")):
        with pg["owner"].connect() as c:  # start from a pending user
            c.execute(text("UPDATE staff_users SET status = 'pending_email' WHERE email = 'user@agri.gov.in'"))
        with pg["Session"]() as db:
            set_auth_lookup(db, email="user@agri.gov.in")
            user = db.scalar(select(StaffUser))
            for key, value in change.items():
                setattr(user, key, value)
            with pytest.raises(DBAPIError, match=error):
                db.commit()
    with pg["Session"]() as db:  # the OTP step itself is allowed
        set_auth_lookup(db, email="user@agri.gov.in")
        db.scalar(select(StaffUser)).status = "pending_approval"
        db.commit()
    with pg["Session"]() as db:  # the department admin approves inside the department scope
        set_department_scope(db, pg["a"])
        db.scalar(select(StaffUser)).status = "active"
        db.commit()


def test_registration_flow_and_phone_lookup(pg):
    """What POST /api/auth/register does: insert pending under the chosen department,
    activate in that scope; then sign-in finds exactly that user by phone."""
    with pg["Session"]() as db:
        set_department_scope(db, pg["b"])
        user = StaffUser(email="reg@health.gov.in", mobile="9876543210", full_name="R", designation="Clerk",
                         department_id=pg["b"], role="officer", status="pending_email")
        db.add(user)
        db.flush()
        user.status = "active"
        db.commit()
    with pg["Session"]() as db:
        set_auth_lookup(db, phone="9876543210")
        found = db.scalars(select(StaffUser)).all()
        assert [u.email for u in found] == ["reg@health.gov.in"]
    with pg["Session"]() as db:  # the same phone can't register twice, even in another department
        set_department_scope(db, pg["a"])
        db.add(StaffUser(email="dup@agri.gov.in", mobile="9876543210", full_name="D", designation="Clerk",
                         department_id=pg["a"], role="officer", status="pending_email"))
        with pytest.raises(DBAPIError):
            db.commit()


def test_scope_survives_commit(pg):
    with pg["Session"]() as db:
        set_department_scope(db, pg["a"])
        assert len(db.scalars(select(StaffUser)).all()) == 1
        db.commit()
        assert len(db.scalars(select(StaffUser)).all()) == 1  # re-applied on the new transaction


def test_registration_insert_scoped_to_chosen_department(pg):
    with pg["Session"]() as db:
        set_department_scope(db, pg["b"])  # registration: the one place a chosen department is used
        db.add(StaffUser(email="new@health.gov.in", full_name="New", designation="Clerk", department_id=pg["b"],
                         role="viewer", status="pending_email"))
        db.commit()
    with pg["Session"]() as db:  # a registration can never create an active admin
        set_department_scope(db, pg["b"])
        db.add(StaffUser(email="boss@health.gov.in", full_name="B", designation="Clerk", department_id=pg["b"],
                         role="dept_admin", status="active"))
        with pytest.raises(DBAPIError, match="pending and below admin"):
            db.commit()
    with pg["Session"]() as db:  # the super admin may create the first department admin
        set_department_scope(db, None, is_super_admin=True)
        db.add(StaffUser(email="admin@health.gov.in", full_name="A", designation="Director", department_id=pg["b"],
                         role="dept_admin", status="active"))
        db.commit()
    with pg["Session"]() as db:
        set_department_scope(db, pg["a"])
        db.add(StaffUser(email="sneak@health.gov.in", full_name="S", designation="Clerk", department_id=pg["b"],
                         role="viewer", status="pending_email"))
        with pytest.raises(DBAPIError):
            db.commit()


def test_audit_log_is_append_only_even_for_superuser(pg):
    with pg["app"].begin() as conn:
        conn.execute(text("INSERT INTO audit_log (id, at, action, details) VALUES (:i, now(), 'x', '{}')"),
                     {"i": str(uuid.uuid4())})
    for stmt in ("UPDATE audit_log SET action = 'z'", "DELETE FROM audit_log", "TRUNCATE audit_log"):
        with pytest.raises(DBAPIError, match="append-only"):
            with pg["owner"].connect() as conn:
                conn.execute(text(stmt))


def test_app_role_cannot_change_departments_or_schema(pg):
    for stmt in ("INSERT INTO departments (id, code, name_en, name_hi, active, created_at) VALUES ('z','z','z','z',true,now())",
                 "DROP TABLE audit_log", "ALTER TABLE staff_users NO FORCE ROW LEVEL SECURITY",
                 "UPDATE audit_log SET action = 'z'"):
        with pytest.raises(DBAPIError):
            with pg["app"].begin() as conn:
                conn.execute(text(stmt))
