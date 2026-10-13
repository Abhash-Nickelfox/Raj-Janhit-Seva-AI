"""Register, sign in (email or phone + password), who am I, sign out.

Registration: the account is created as pending, the user verifies it with a
one-time code sent to their email or phone (testing: always 1234, nothing is
sent), and only then can sign in. Checking that a person really belongs to the
department they pick is decided later (PLAN 12.5 - staff list / access code)."""

import threading
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.deps import StaffActor, current_actor, require_fetch_header
from app.auth.passwords import DUMMY_HASH, hash_password, verify_password
from app.auth import otp as otps
from app.auth import validation as rules
from app.auth.sessions import COOKIE, cookie_args, make_token, read_token
from app.core.audit import write_audit
from app.core.db import get_db, set_auth_lookup, set_department_scope
from app.models import Department, StaffUser

router = APIRouter(prefix="/api/auth", tags=["auth"])

MAX_FAILURES, WINDOW_SECONDS = 5, 15 * 60
_failures: dict[str, list[float]] = {}
_failures_lock = threading.Lock()


REGISTER_LIMIT_PER_HOUR = 10
_registrations: dict[str, list[float]] = {}


class LoginIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    login: str = Field(min_length=3, max_length=254)  # email or mobile number
    password: str = Field(min_length=1, max_length=200)


class SendOtpIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(max_length=254)
    channel: str = Field(pattern="^(email|phone)$")


class VerifyOtpIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(max_length=254)
    code: str = Field(pattern=r"^\s*\d{4,6}\s*$")


class RegisterIn(BaseModel):
    model_config = ConfigDict(extra="forbid")  # role/status/department_id can't be smuggled in
    full_name: str = Field(max_length=200)
    email: str = Field(max_length=254)
    phone: str = Field(max_length=20)
    department_code: str = Field(max_length=40)
    designation: str = Field(max_length=200)
    password: str = Field(max_length=200)
    confirm_password: str = Field(max_length=200)

    _name = field_validator("full_name")(lambda cls, v: rules.check_name(v))
    _email = field_validator("email")(lambda cls, v: rules.check_email(v))
    _phone = field_validator("phone")(lambda cls, v: rules.check_phone(v))
    _designation = field_validator("designation")(lambda cls, v: rules.check_designation(v))
    _password = field_validator("password")(lambda cls, v: rules.check_password(v))

    @model_validator(mode="after")
    def passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match.")
        return self


def _recent_failures(key: str) -> int:
    now = time.time()
    with _failures_lock:
        _failures[key] = [t for t in _failures.get(key, []) if now - t < WINDOW_SECONDS]
        return len(_failures[key])


def _record_failure(key: str) -> None:
    with _failures_lock:
        _failures.setdefault(key, []).append(time.time())


def _me(actor: StaffActor) -> dict:
    d = actor.department
    return {
        "id": actor.user.id, "email": actor.user.email, "full_name": actor.user.full_name,
        "phone": actor.user.mobile, "designation": actor.user.designation, "role": actor.user.role,
        "department": {"id": d.id, "code": d.code, "name_en": d.name_en, "name_hi": d.name_hi} if d else None,
    }


@router.post("/register", status_code=201, dependencies=[Depends(require_fetch_header)])
def register(data: RegisterIn, request: Request, db: Session = Depends(get_db)):
    ip = request.client.host if request.client else None
    now = time.time()
    with _failures_lock:
        recent = [t for t in _registrations.get(ip or "", []) if now - t < 3600]
        if len(recent) >= REGISTER_LIMIT_PER_HOUR:
            raise HTTPException(429, detail={"code": "too_many_registrations",
                                             "message": "Too many registrations from this network. Try again later."})
        _registrations[ip or ""] = recent + [now]

    dept = db.scalar(select(Department).where(Department.code == data.department_code, Department.active.is_(True)))
    if dept is None:
        raise HTTPException(422, detail={"code": "bad_department", "field": "department_code",
                                         "message": "Choose a department from the list."})
    # The new row is written under the chosen department's scope, starts pending
    # (the database refuses anything else on insert) and is then activated.
    set_department_scope(db, dept.id)
    user = StaffUser(email=data.email, mobile=data.phone, full_name=data.full_name, designation=data.designation,
                     department_id=dept.id, role="officer", status="pending_email",
                     password_hash=hash_password(data.password))
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, detail={"code": "already_registered",
                                         "message": "This email or phone number is already registered. Please sign in."})
    write_audit(db, "user.register", actor_id=user.id, department_id=dept.id, entity_type="staff_user",
                entity_id=user.id, ip=ip, details={"email": user.email, "department": dept.code})
    db.commit()
    return _pending_info(user)


def _pending_info(user: StaffUser) -> dict:
    return {"email": user.email, "email_masked": otps.mask_email(user.email),
            "phone_masked": otps.mask_phone(user.mobile or ""), "has_phone": bool(user.mobile)}


def _pending_user(db: Session, email: str) -> StaffUser:
    email = (email or "").strip().lower()
    set_auth_lookup(db, email=email)
    user = db.scalar(select(StaffUser).where(StaffUser.email == email))
    if user is None or user.status != "pending_email":
        raise HTTPException(400, detail={"code": "nothing_to_verify",
                                         "message": "No registration is waiting for verification for this email."})
    return user


@router.post("/register/send-otp", dependencies=[Depends(require_fetch_header)])
def send_otp(data: SendOtpIn, request: Request, db: Session = Depends(get_db)):
    user = _pending_user(db, data.email)
    if data.channel == "phone" and not user.mobile:
        raise HTTPException(400, detail={"code": "no_phone", "message": "No phone number on this registration."})
    try:
        otps.issue(db, user.email)
    except otps.OtpError as exc:
        db.rollback()
        raise HTTPException(exc.status, detail={"code": exc.code, "message": exc.message, "retry_after": exc.retry_after})
    write_audit(db, "user.otp_sent", actor_id=user.id, department_id=user.department_id, entity_type="staff_user",
                entity_id=user.id, ip=request.client.host if request.client else None, details={"channel": data.channel})
    db.commit()
    sent_to = otps.mask_email(user.email) if data.channel == "email" else otps.mask_phone(user.mobile)
    return {"ok": True, "channel": data.channel, "sent_to": sent_to, "test_mode": otps.test_mode(),
            "resend_after": int(otps.RESEND_AFTER.total_seconds())}


@router.post("/register/verify-otp", dependencies=[Depends(require_fetch_header)])
def verify_otp(data: VerifyOtpIn, request: Request, db: Session = Depends(get_db)):
    user = _pending_user(db, data.email)
    ip = request.client.host if request.client else None
    try:
        otps.verify(db, user.email, data.code)
    except otps.OtpError as exc:
        write_audit(db, "user.otp_failed", actor_id=user.id, department_id=user.department_id, ip=ip,
                    details={"reason": exc.code})
        db.commit()  # keep the attempt count
        raise HTTPException(exc.status, detail={"code": exc.code, "message": exc.message})
    # verified: activate inside the user's own department scope
    set_department_scope(db, user.department_id)
    user.status, user.approved_at = "active", datetime.now(timezone.utc)
    write_audit(db, "user.verified", actor_id=user.id, department_id=user.department_id, entity_type="staff_user",
                entity_id=user.id, ip=ip)
    db.commit()
    return {"ok": True, "email": user.email}


@router.post("/login", dependencies=[Depends(require_fetch_header)])
def login(data: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)):
    raw = data.login.strip()
    phone = rules.normalize_phone(raw) if "@" not in raw else None
    ident = phone or raw.lower()
    ip = request.client.host if request.client else None
    keys = (f"login:{ident}", f"ip:{ip}")
    if any(_recent_failures(k) >= MAX_FAILURES for k in keys):
        write_audit(db, "auth.login_blocked", ip=ip, details={"login": ident})
        db.commit()
        raise HTTPException(429, detail={"code": "too_many_attempts", "message": "Too many attempts. Try again in 15 minutes."})

    if phone:
        set_auth_lookup(db, phone=phone)
        user = db.scalar(select(StaffUser).where(StaffUser.mobile == phone))
    else:
        set_auth_lookup(db, email=ident)
        user = db.scalar(select(StaffUser).where(StaffUser.email == ident))
    ok = verify_password(data.password, user.password_hash if user else DUMMY_HASH) and user is not None
    if ok and user.status == "pending_email":
        # right password, account not verified yet: send them to the OTP step
        raise HTTPException(403, detail={"code": "not_verified", "message": "Please verify your account first.",
                                         **_pending_info(user)})
    if not ok or user.status != "active":
        for k in keys:
            _record_failure(k)
        write_audit(db, "auth.login_failed", ip=ip, department_id=user.department_id if user else None,
                    details={"login": ident})
        db.commit()
        # same answer whether the email/phone or the password was wrong
        raise HTTPException(401, detail={"code": "bad_credentials", "message": "Email/phone or password is incorrect."})

    user.last_login_at = datetime.now(timezone.utc)
    write_audit(db, "auth.login", actor_id=user.id, department_id=user.department_id, entity_type="staff_user",
                entity_id=user.id, ip=ip)
    db.commit()
    with _failures_lock:
        for k in keys:
            _failures.pop(k, None)
    response.set_cookie(value=make_token(user.id), **cookie_args())
    return {"ok": True}


@router.get("/me")
def me(actor: StaffActor = Depends(current_actor)):
    return _me(actor)


@router.post("/logout", dependencies=[Depends(require_fetch_header)])
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    user_id = read_token(request.cookies.get(COOKIE))
    if user_id:
        write_audit(db, "auth.logout", actor_id=user_id, ip=request.client.host if request.client else None)
        db.commit()
    response.delete_cookie(COOKIE, path="/api")
    return {"ok": True}
