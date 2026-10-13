"""P0 foundation: health, departments list, schema/migrations, audit append-only, settings guard."""

import pytest
from alembic import command
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import DatabaseError, IntegrityError

from app.core.audit import write_audit
from app.core.config import Settings
from app.models import Department, StaffUser
from tests.conftest import alembic_config


def test_health_reports_database_and_security_headers(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["database"] == "ok"
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Cache-Control"] == "no-store"


def test_departments_lists_only_active_sorted(client, db):
    db.add_all([Department(code="health", name_en="Medical & Health", name_hi="चिकित्सा एवं स्वास्थ्य"),
                Department(code="agriculture", name_en="Agriculture", name_hi="कृषि"),
                Department(code="closed", name_en="Old Dept", name_hi="पुराना", active=False)])
    db.commit()
    body = client.get("/api/departments").json()
    assert [d["code"] for d in body] == ["agriculture", "health"]
    assert set(body[0]) == {"id", "code", "name_en", "name_hi"}  # nothing about members or documents


def test_cors_allows_only_the_staff_frontend(client):
    ok = client.options("/api/health", headers={"Origin": "http://localhost:5180", "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5180"
    bad = client.options("/api/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in bad.headers  # the citizen app's origin is not allowed


def test_migration_creates_schema_and_downgrades_cleanly(tmp_path):
    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    cfg = alembic_config(url)
    command.upgrade(cfg, "head")
    tables = set(inspect(create_engine(url)).get_table_names())
    assert {"departments", "staff_users", "otp_codes", "verification_requests", "audit_log"} <= tables
    command.downgrade(cfg, "base")
    assert set(inspect(create_engine(url)).get_table_names()) <= {"alembic_version"}


def test_audit_log_is_append_only(db):
    write_audit(db, "test.event", details={"k": "v"})
    db.commit()
    with pytest.raises(DatabaseError, match="append-only"):
        db.execute(text("UPDATE audit_log SET action = 'changed'"))
    db.rollback()
    with pytest.raises(DatabaseError, match="append-only"):
        db.execute(text("DELETE FROM audit_log"))
    db.rollback()
    assert db.execute(text("SELECT action FROM audit_log")).scalar() == "test.event"


def test_non_super_admin_must_have_a_department(db):
    db.add(StaffUser(email="x@rajasthan.gov.in", full_name="X", designation="Officer", role="officer", status="active"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    db.add(StaffUser(email="root@rajasthan.gov.in", full_name="Root", designation="Admin", role="super_admin", status="active"))
    db.commit()


def test_invalid_role_or_status_is_refused(db):
    dept = Department(code="agriculture", name_en="Agriculture", name_hi="कृषि")
    db.add(dept)
    db.commit()
    db.add(StaffUser(email="y@rajasthan.gov.in", full_name="Y", designation="Clerk", role="god", status="active",
                     department_id=dept.id))
    with pytest.raises(IntegrityError):
        db.commit()


@pytest.mark.parametrize("overrides, problem", [
    ({"database_url": "sqlite:///x.db"}, "PostgreSQL"),
    ({"session_secret": "short"}, "SESSION_SECRET"),
    ({"cors_origins": ["*"]}, "never *"),
    ({"cors_origins": ["http://staff.example.gov.in"]}, "https"),
    ({"cors_origins": ["http://localhost.evil.example"]}, "https"),
    ({"cors_origins": ["http://localhost:5180"]}, "https"),
    ({"cors_origins": ["https://staff.example.gov.in/app"]}, "https"),
    ({"session_secret": "# 32+ random characters outside development"}, "SESSION_SECRET"),
    ({"session_secret": "a" * 40}, "SESSION_SECRET"),
    ({"allowed_email_domains": []}, "ALLOWED_EMAIL_DOMAINS"),
    ({"otp_test_code": "1234"}, "OTP_TEST_CODE"),
])
def test_production_refuses_unsafe_settings(overrides, problem):
    good = dict(environment="production", database_url="postgresql+psycopg://u:p@h/db", session_secret="Zq7-vR2kLp9xW4mN8bT1cY6hJ3dF0gS5aE",
                cors_origins=["https://staff.example.gov.in"], allowed_email_domains=["rajasthan.gov.in"], otp_test_code="")
    Settings(**good)  # the safe baseline is accepted
    with pytest.raises(ValueError, match=problem):
        Settings(**{**good, **overrides})


def test_email_domains_are_normalised():
    s = Settings(allowed_email_domains="@Rajasthan.gov.in, NIC.in")
    assert s.allowed_email_domains == ["rajasthan.gov.in", "nic.in"]
