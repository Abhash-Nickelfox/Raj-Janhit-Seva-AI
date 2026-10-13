"""Signed session cookie: httpOnly (JavaScript can't read it), SameSite=Lax, Secure
outside development. Holds only the user id and expiry; department and role are
always re-read from the database on every request."""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time

from app.core.config import get_settings

COOKIE = "staff_session"
LIFETIME_SECONDS = 8 * 3600
log = logging.getLogger("staff.sessions")

_dev_secret = secrets.token_bytes(32)  # development without SESSION_SECRET: sessions end on restart


def _key() -> bytes:
    secret = get_settings().session_secret
    if secret:
        return secret.encode()
    if get_settings().is_production:
        raise RuntimeError("SESSION_SECRET is required")
    return _dev_secret


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_token(user_id: str) -> str:
    payload = _b64(json.dumps({"uid": user_id, "exp": int(time.time()) + LIFETIME_SECONDS}).encode())
    sig = _b64(hmac.new(_key(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def read_token(token: str | None) -> str | None:
    """The user id from a valid, unexpired token; otherwise None."""
    if not token or token.count(".") != 1:
        return None
    payload, sig = token.split(".")
    expected = _b64(hmac.new(_key(), payload.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        data = json.loads(_unb64(payload))
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("exp", 0) < time.time() or not isinstance(data.get("uid"), str):
        return None
    return data["uid"]


def cookie_args() -> dict:
    return {"key": COOKIE, "httponly": True, "samesite": "lax", "secure": get_settings().is_production,
            "max_age": LIFETIME_SECONDS, "path": "/api"}
