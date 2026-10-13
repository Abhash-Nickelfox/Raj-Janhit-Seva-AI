"""Tests run on a throwaway SQLite file built by the real Alembic migrations,
so the schema under test is exactly what production gets (minus Postgres-only RLS)."""

import os
import shutil
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="gov-staff-tests-"))
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["MIGRATION_DATABASE_URL"] = ""
# pin everything a developer's backend/.env could change
os.environ["CORS_ORIGINS"] = "http://localhost:5180"
os.environ["SESSION_SECRET"] = ""
os.environ["ALLOWED_EMAIL_DOMAINS"] = "rajasthan.gov.in,nic.in"
os.environ["SMTP_HOST"] = ""
os.environ["OTP_TEST_CODE"] = "1234"

import pytest  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]


def alembic_config(url: str | None = None) -> Config:
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    cfg.attributes["url"] = url or os.environ["DATABASE_URL"]
    return cfg


@pytest.fixture
def db():
    from app.core.db import SessionLocal, engine

    engine.dispose()
    cfg = alembic_config()
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def client(db):
    from app.main import app

    with TestClient(app) as c:
        yield c


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)
