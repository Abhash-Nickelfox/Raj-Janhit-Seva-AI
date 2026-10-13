"""One-time codes for verifying a new account by email or phone.

Testing mode (development/test only): the code is always OTP_TEST_CODE (1234)
and nothing is actually sent. Real email/SMS delivery comes later; until then,
without a test code the API answers "OTP delivery is not set up".
Only a keyed hash of the code is stored. Codes expire after 10 minutes and
allow 5 attempts; a new code can be requested after 60 s, at most 5 per hour.
"""

import hashlib
import hmac
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.auth.sessions import _key
from app.core.config import get_settings
from app.models import OtpCode

TTL = timedelta(minutes=10)
MAX_ATTEMPTS = 5
RESEND_AFTER = timedelta(seconds=60)
MAX_PER_HOUR = 5


class OtpError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, retry_after: int | None = None):
        super().__init__(message)
        self.code, self.message, self.status, self.retry_after = code, message, status, retry_after


def _hash(email: str, code: str) -> str:
    return hmac.new(_key(), f"{email}:{code}".encode(), hashlib.sha256).hexdigest()


def test_mode() -> bool:
    s = get_settings()
    return bool(s.otp_test_code) and not s.is_production


def mask_email(email: str) -> str:
    name, _, domain = email.partition("@")
    return (name[:2] + "•" * max(1, len(name) - 2)) + "@" + domain


def mask_phone(phone: str) -> str:
    return "•" * 6 + phone[-4:] if phone else ""


def issue(db: Session, email: str, purpose: str = "register") -> None:
    """Creates a new code for this email (replacing older ones) and 'sends' it."""
    now = datetime.now(timezone.utc)
    recent = db.scalars(select(OtpCode).where(OtpCode.email == email, OtpCode.purpose == purpose,
                                              OtpCode.created_at > now - timedelta(hours=1))
                        .order_by(OtpCode.created_at.desc())).all()
    if recent and now - recent[0].created_at < RESEND_AFTER:
        wait = int((RESEND_AFTER - (now - recent[0].created_at)).total_seconds()) + 1
        raise OtpError("otp_wait", f"Please wait {wait} seconds before requesting a new code.", 429, wait)
    if len(recent) >= MAX_PER_HOUR:
        raise OtpError("otp_limit", "Too many codes requested. Please try again in an hour.", 429)
    if not test_mode():
        # Real delivery (email/SMS gateway) is not built yet; it will send a
        # random code such as f"{secrets.randbelow(10**6):06d}".
        raise OtpError("otp_unavailable", "OTP delivery is not set up yet.", 503)
    code = get_settings().otp_test_code
    # retire older codes (kept, so the per-hour limit still counts them)
    db.execute(update(OtpCode).where(OtpCode.email == email, OtpCode.purpose == purpose, OtpCode.consumed_at.is_(None))
               .values(consumed_at=now))
    db.add(OtpCode(email=email, purpose=purpose, code_hash=_hash(email, code), expires_at=now + TTL))


def verify(db: Session, email: str, code: str, purpose: str = "register") -> None:
    now = datetime.now(timezone.utc)
    otp = db.scalar(select(OtpCode).where(OtpCode.email == email, OtpCode.purpose == purpose,
                                          OtpCode.consumed_at.is_(None)).order_by(OtpCode.created_at.desc()))
    if otp is None or otp.expires_at < now:
        raise OtpError("otp_expired", "This code has expired. Please request a new one.")
    if otp.attempts >= MAX_ATTEMPTS:
        raise OtpError("otp_locked", "Too many wrong attempts. Please request a new code.", 429)
    if not hmac.compare_digest(otp.code_hash, _hash(email, code.strip())):
        otp.attempts += 1
        left = MAX_ATTEMPTS - otp.attempts
        raise OtpError("otp_wrong", f"Incorrect code. {left} attempt(s) left." if left else
                       "Too many wrong attempts. Please request a new code.")
    otp.consumed_at = now
