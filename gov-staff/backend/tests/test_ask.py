"""POST /api/ask: signed-in only, department from the session, audited, input validated."""

import pytest
from sqlalchemy import text

from app.auth.passwords import hash_password
from app.models import Department, StaffUser
from app.routers import auth as auth_router

H = {"X-Requested-With": "fetch"}


@pytest.fixture
def signed_in(client, db):
    auth_router._failures.clear()
    agri = Department(code="agriculture", name_en="Agriculture", name_hi="कृषि")
    db.add(agri)
    db.flush()
    db.add(StaffUser(email="ravi@buildyourtech.com", full_name="Ravi", designation="Officer", department_id=agri.id,
                     role="officer", status="active", password_hash=hash_password("admin321")))
    db.commit()
    assert client.post("/api/auth/login", json={"login": "ravi@buildyourtech.com", "password": "admin321"},
                       headers=H).status_code == 200
    return client


def test_ask_requires_sign_in(client, db):
    assert client.post("/api/ask", json={"message": "hi"}, headers=H).status_code == 401


def test_ask_answers_for_the_users_department_and_is_audited(signed_in, db):
    r = signed_in.post("/api/ask", json={"message": "What is the latest seed subsidy circular?", "session_id": "s1"}, headers=H)
    assert r.status_code == 200
    assert "Agriculture" in r.json()["reply"] and r.json()["references"] == []
    hi = signed_in.post("/api/ask", json={"message": "बीज सब्सिडी", "preferred_lang": "hindi"}, headers=H).json()["reply"]
    assert "कृषि" in hi
    rows = db.execute(text("SELECT action, details FROM audit_log WHERE action = 'qa.ask'")).all()
    assert len(rows) == 2 and "seed subsidy" in str(rows[0][1])


def test_department_cannot_be_chosen_by_the_request(signed_in):
    r = signed_in.post("/api/ask", json={"message": "hi", "department_id": "other"}, headers=H)
    assert r.status_code == 422


@pytest.mark.parametrize("body", [{"message": ""}, {"message": "x" * 2001}, {"message": "hi", "preferred_lang": "french"}])
def test_ask_input_is_validated(signed_in, body):
    assert signed_in.post("/api/ask", json=body, headers=H).status_code == 422


def test_ask_needs_fetch_header(signed_in):
    assert signed_in.post("/api/ask", json={"message": "hi"}).status_code == 403
