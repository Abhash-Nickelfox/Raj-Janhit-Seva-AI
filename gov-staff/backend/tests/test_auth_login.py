"""Email + password sign-in for the test accounts: session cookie, who-am-I, sign-out,
same error for wrong email/password, lockout, CSRF header, tampered cookies."""

import pytest
from sqlalchemy import select, text

from app.auth.passwords import hash_password
from app.auth.sessions import COOKIE
from app.models import Department, StaffUser
from app.routers import auth as auth_router

H = {"X-Requested-With": "fetch"}


@pytest.fixture
def users(db):
    auth_router._failures.clear()
    health = Department(code="healthcare", name_en="Medical & Health", name_hi="चिकित्सा एवं स्वास्थ्य")
    agri = Department(code="agriculture", name_en="Agriculture", name_hi="कृषि")
    db.add_all([health, agri])
    db.flush()
    db.add_all([
        StaffUser(email="abhi@buildyourtech.com", full_name="Abhi", designation="Officer", department_id=health.id,
                  role="officer", status="active", password_hash=hash_password("admin123")),
        StaffUser(email="ravi@buildyourtech.com", full_name="Ravi", designation="Officer", department_id=agri.id,
                  role="officer", status="active", password_hash=hash_password("admin321")),
        StaffUser(email="nopass@buildyourtech.com", full_name="No", designation="Officer", department_id=agri.id,
                  role="officer", status="active"),
    ])
    db.commit()
    yield
    auth_router._failures.clear()


def login(client, email, password, headers=H):
    return client.post("/api/auth/login", json={"login": email, "password": password}, headers=headers)


def test_each_user_signs_in_to_their_own_department(client, users):
    for email, pw, dept in (("abhi@buildyourtech.com", "admin123", "healthcare"),
                            ("ravi@buildyourtech.com", "admin321", "agriculture")):
        client.cookies.clear()
        resp = login(client, email.upper(), pw)  # email is case-insensitive
        assert resp.status_code == 200
        cookie = resp.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=lax" in cookie and "path=/api" in cookie
        me = client.get("/api/auth/me").json()
        assert me["email"] == email and me["department"]["code"] == dept
        assert "password_hash" not in me


def test_wrong_password_and_unknown_email_get_the_same_answer(client, users):
    a = login(client, "abhi@buildyourtech.com", "wrong")
    b = login(client, "nobody@buildyourtech.com", "admin123")
    c = login(client, "nopass@buildyourtech.com", "")  # empty password is rejected by validation
    assert a.status_code == b.status_code == 401
    assert a.json() == b.json()
    assert c.status_code == 422
    assert login(client, "nopass@buildyourtech.com", "anything").status_code == 401  # no password set = can't sign in


def test_me_requires_a_valid_session(client, users):
    assert client.get("/api/auth/me").status_code == 401
    client.cookies.set(COOKIE, "forged.token", path="/api")
    assert client.get("/api/auth/me").status_code == 401
    client.cookies.clear()
    login(client, "abhi@buildyourtech.com", "admin123")
    token = next(c.value for c in client.cookies.jar if c.name == COOKIE)
    payload, sig = token.split(".")
    client.cookies.clear()
    client.cookies.set(COOKIE, payload + "x." + sig, path="/api")  # tampered payload
    assert client.get("/api/auth/me").status_code == 401


def test_logout_ends_the_session(client, users):
    login(client, "ravi@buildyourtech.com", "admin321")
    assert client.get("/api/auth/me").status_code == 200
    assert client.post("/api/auth/logout", headers=H).status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_suspended_user_is_signed_out(client, users, db):
    login(client, "ravi@buildyourtech.com", "admin321")
    db.execute(text("UPDATE staff_users SET status = 'suspended' WHERE email = 'ravi@buildyourtech.com'"))
    db.commit()
    assert client.get("/api/auth/me").status_code == 401


def test_lockout_after_five_failures(client, users):
    for _ in range(5):
        assert login(client, "abhi@buildyourtech.com", "wrong").status_code == 401
    assert login(client, "abhi@buildyourtech.com", "admin123").status_code == 429  # even the right password


def test_state_changing_calls_need_the_fetch_header(client, users):
    assert login(client, "abhi@buildyourtech.com", "admin123", headers={}).status_code == 403
    assert client.post("/api/auth/logout").status_code == 403


def test_every_sign_in_attempt_is_audited(client, users, db):
    login(client, "abhi@buildyourtech.com", "wrong")
    login(client, "abhi@buildyourtech.com", "admin123")
    client.post("/api/auth/logout", headers=H)
    actions = db.execute(text("SELECT action FROM audit_log ORDER BY at")).scalars().all()
    assert actions == ["auth.login_failed", "auth.login", "auth.logout"]
    assert "admin123" not in str(db.execute(text("SELECT details FROM audit_log")).scalars().all())


def test_extra_fields_are_refused(client, users):
    resp = client.post("/api/auth/login", headers=H,
                       json={"login": "abhi@buildyourtech.com", "password": "admin123", "department_id": "x"})
    assert resp.status_code == 422
