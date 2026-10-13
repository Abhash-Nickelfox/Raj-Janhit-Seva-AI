"""Registration: field validation, duplicates, department from the list only,
no role/status smuggling, then sign-in with the same email or phone + password."""

import pytest
from sqlalchemy import select, text

from app.auth.validation import normalize_phone
from app.models import Department, StaffUser
from app.routers import auth as auth_router

H = {"X-Requested-With": "fetch"}
GOOD = {"full_name": "Sunita Sharma", "email": "Sunita.Sharma@Example.org", "phone": "+91 98765 43210",
        "department_code": "healthcare", "designation": "Block Health Officer",
        "password": "Health2026", "confirm_password": "Health2026"}


@pytest.fixture
def depts(db):
    auth_router._failures.clear()
    auth_router._registrations.clear()
    db.add_all([Department(code="healthcare", name_en="Medical & Health", name_hi="चिकित्सा एवं स्वास्थ्य"),
                Department(code="agriculture", name_en="Agriculture", name_hi="कृषि"),
                Department(code="closed", name_en="Old", name_hi="पुराना", active=False)])
    db.commit()
    yield
    auth_router._failures.clear()
    auth_router._registrations.clear()


def register(client, **changes):
    return client.post("/api/auth/register", json={**GOOD, **changes}, headers=H)


def verify(client, email="sunita.sharma@example.org", channel="email", code="1234"):
    sent = client.post("/api/auth/register/send-otp", json={"email": email, "channel": channel}, headers=H)
    assert sent.status_code == 200, sent.text
    return client.post("/api/auth/register/verify-otp", json={"email": email, "code": code}, headers=H)


def test_register_verify_by_otp_then_sign_in_with_email_or_phone(client, depts, db):
    resp = register(client)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["email"] == "sunita.sharma@example.org" and body["phone_masked"].endswith("3210")
    user = db.scalar(select(StaffUser).where(StaffUser.email == "sunita.sharma@example.org"))
    assert user.mobile == "9876543210" and user.status == "pending_email" and user.role == "officer"
    assert user.password_hash.startswith("scrypt$") and "Health2026" not in user.password_hash

    # before verification: right password -> "verify first", wrong password -> generic error
    r = client.post("/api/auth/login", json={"login": "9876543210", "password": "Health2026"}, headers=H)
    assert r.status_code == 403 and r.json()["detail"]["code"] == "not_verified"
    r = client.post("/api/auth/login", json={"login": "9876543210", "password": "Wrong2026"}, headers=H)
    assert r.status_code == 401

    assert verify(client, channel="phone").status_code == 200
    db.expire_all()
    assert db.scalar(select(StaffUser).where(StaffUser.email == "sunita.sharma@example.org")).status == "active"
    for login in ("sunita.sharma@example.org", "9876543210", "+91 98765-43210", "098765 43210"):
        client.cookies.clear()
        r = client.post("/api/auth/login", json={"login": login, "password": "Health2026"}, headers=H)
        assert r.status_code == 200, (login, r.text)
        me = client.get("/api/auth/me").json()
        assert me["department"]["code"] == "healthcare" and me["phone"] == "9876543210"


def test_send_otp_reports_where_it_went(client, depts):
    register(client)
    r = client.post("/api/auth/register/send-otp", json={"email": "sunita.sharma@example.org", "channel": "email"}, headers=H)
    assert r.json()["sent_to"].endswith("@example.org") and r.json()["test_mode"] is True
    assert "sunita.sharma@" not in r.json()["sent_to"]  # masked


def test_wrong_otp_counts_attempts_and_locks(client, depts, db):
    register(client)
    client.post("/api/auth/register/send-otp", json={"email": "sunita.sharma@example.org", "channel": "email"}, headers=H)
    for left in (4, 3, 2, 1):
        r = client.post("/api/auth/register/verify-otp", json={"email": "sunita.sharma@example.org", "code": "9999"}, headers=H)
        assert r.status_code == 400 and f"{left} attempt" in r.json()["detail"]["message"]
    client.post("/api/auth/register/verify-otp", json={"email": "sunita.sharma@example.org", "code": "9999"}, headers=H)
    r = client.post("/api/auth/register/verify-otp", json={"email": "sunita.sharma@example.org", "code": "1234"}, headers=H)
    assert r.status_code == 429  # locked even with the right code; a new code is needed
    assert db.scalar(select(StaffUser).where(StaffUser.email == "sunita.sharma@example.org")).status == "pending_email"


def test_otp_resend_cooldown_and_hourly_limit(client, depts, db):
    register(client)
    send = lambda: client.post("/api/auth/register/send-otp", json={"email": "sunita.sharma@example.org", "channel": "email"}, headers=H)
    assert send().status_code == 200
    r = send()
    assert r.status_code == 429 and r.json()["detail"]["code"] == "otp_wait"
    db.execute(text("UPDATE otp_codes SET created_at = datetime(created_at, '-2 minutes')"))
    db.commit()
    for _ in range(4):
        assert send().status_code == 200
        db.execute(text("UPDATE otp_codes SET created_at = datetime(created_at, '-2 minutes')"))
        db.commit()
    assert send().json()["detail"]["code"] == "otp_limit"


def test_otp_needs_a_pending_registration(client, depts):
    for email in ("nobody@example.org",):
        r = client.post("/api/auth/register/send-otp", json={"email": email, "channel": "email"}, headers=H)
        assert r.status_code == 400
    register(client)
    assert verify(client).status_code == 200
    r = client.post("/api/auth/register/send-otp", json={"email": "sunita.sharma@example.org", "channel": "email"}, headers=H)
    assert r.status_code == 400  # already verified


def test_otp_input_is_validated(client, depts):
    register(client)
    for code in ("12", "abcd", "1234567", ""):
        r = client.post("/api/auth/register/verify-otp", json={"email": "sunita.sharma@example.org", "code": code}, headers=H)
        assert r.status_code == 422
    r = client.post("/api/auth/register/send-otp", json={"email": "sunita.sharma@example.org", "channel": "fax"}, headers=H)
    assert r.status_code == 422


def test_otp_codes_are_stored_hashed(client, depts, db):
    register(client)
    client.post("/api/auth/register/send-otp", json={"email": "sunita.sharma@example.org", "channel": "email"}, headers=H)
    stored = db.execute(text("SELECT code_hash FROM otp_codes")).scalar()
    assert stored and "1234" not in stored


@pytest.mark.parametrize("field, value", [
    ("full_name", "A"), ("full_name", "Sunita123"), ("full_name", ""),
    ("email", "not-an-email"), ("email", "a@b"),
    ("phone", "12345"), ("phone", "5876543210"), ("phone", "98765432101"), ("phone", "abcdefghij"),
    ("designation", "x"), ("designation", ""),
    ("password", "short1"), ("password", "onlyletters"), ("password", "12345678"), ("password", " Spaced123"),
])
def test_each_field_is_validated(client, depts, field, value):
    changes = {field: value}
    if field == "password":
        changes["confirm_password"] = value
    resp = register(client, **changes)
    assert resp.status_code == 422
    assert field in resp.json()["detail"]["fields"]


def test_passwords_must_match(client, depts):
    resp = register(client, confirm_password="Health2027")
    assert resp.status_code == 422
    assert "do not match" in str(resp.json()["detail"]["fields"])


def test_department_must_come_from_the_active_list(client, depts):
    for code in ("nonexistent", "closed"):
        resp = register(client, department_code=code)
        assert resp.status_code == 422 and resp.json()["detail"]["code"] == "bad_department"


def test_duplicate_email_or_phone_is_refused(client, depts):
    assert register(client).status_code == 201
    assert register(client, phone="9123456780").status_code == 409  # same email
    assert register(client, email="other@example.org").status_code == 409  # same phone
    assert register(client, email="other@example.org", phone="9123456780").status_code == 201


def test_role_status_or_department_id_cannot_be_smuggled(client, depts):
    for extra in ({"role": "super_admin"}, {"status": "active"}, {"department_id": "x"}):
        assert register(client, **extra).status_code == 422


def test_registration_needs_fetch_header_and_is_rate_limited(client, depts):
    assert client.post("/api/auth/register", json=GOOD).status_code == 403
    for i in range(auth_router.REGISTER_LIMIT_PER_HOUR):
        register(client, email=f"user{i}@example.org", phone=f"98765{i:05d}")
    assert register(client, email="late@example.org", phone="9000000001").status_code == 429


def test_registration_is_audited_without_the_password(client, depts, db):
    register(client)
    rows = db.execute(text("SELECT action, details FROM audit_log")).all()
    assert [r[0] for r in rows] == ["user.register"] and "Health2026" not in str(rows)


@pytest.mark.parametrize("raw, expected", [
    ("9876543210", "9876543210"), ("+91 98765 43210", "9876543210"), ("919876543210", "9876543210"),
    ("09876543210", "9876543210"), ("98765-43210", "9876543210"), ("1234567890", None), ("98765", None),
])
def test_phone_normalisation(raw, expected):
    assert normalize_phone(raw) == expected
