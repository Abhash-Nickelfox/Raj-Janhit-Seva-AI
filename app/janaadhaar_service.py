"""Jan Aadhaar 'Fetch Details' integration - the consumer-side business
logic sitting on top of a swappable transport (app/janaadhaar_client.py)
that actually speaks the wire protocol over real HTTP. Everything specific
to being a MOCK lives one layer down (janaadhaar_client.MockJanAadhaarTransport
calling janaadhaar_mock_routes.py's local endpoints, backed by
janaadhaar_mock_data.py's synthetic families) - this module only knows
about the transport's fetch_member_list/generate_otp/validate_otp
interface, so swapping JANAADHAAR_MODE=real in .env changes nothing here.

Still mirrors two architectural patterns from the real government stack
(RISL's Jan Aadhaar Yojana RFP, S4.2.2.4 Role-Based Redaction and S4.2.7
Aadhaar Data Vault) at THIS layer, since they're consumer-side
responsibilities, not part of the wire protocol itself:

  - Vault + Reference Key: the raw Jan Aadhaar ID is stored in exactly one
    place (_VAULT below) and everywhere else - including the caller in
    app/chat.py - only ever sees an opaque token, never the ID itself past
    the single fetch_member_list() call.
  - Audit trail: every fetch is logged (token/member_id + result code only,
    never the identifier/OTP/field values themselves) to a separate logger
    from the main app's conversation logging.

Only NAME_EN/NAME_LL/DOB/GENDER are confirmed fields from the real API
(Jan Aadhaar Integration Document v1.8) - everything else validate_otp
returns is a stand-in for "additional fields selected during onboarding,"
whose real names/format are unconfirmed until actual onboarding.
"""

import logging
import re
import secrets
import threading
from datetime import date

from app.config import JANAADHAAR_MODE, JANAADHAAR_MOCK_BASE_URL
from app.janaadhaar_client import MockJanAadhaarTransport, RealJanAadhaarTransport

_audit_logger = logging.getLogger("janaadhaar_audit")

_PROFILE_FIELD_KEYS = {
    "gender", "caste", "marital_status", "age",
    "family_income", "residence", "education_level", "disability",
}

_CONFIRMED_KEYS = ("NAME_EN", "NAME_LL", "DOB", "GENDER")


# ---------------------------------------------------------------------------
# Aadhaar Data Vault (mock): raw Jan Aadhaar ID <-> opaque Reference Key.
# Real vault is "a secure system inside the agency's infrastructure
# accessible only on need-to-know basis" (RFP S4.2.7) - an in-process dict is
# the mock equivalent of that isolation boundary; nothing outside this
# module ever reads _VAULT directly.
# ---------------------------------------------------------------------------
_VAULT: dict[str, str] = {}  # token -> raw jan aadhaar id
_VAULT_MAX = 5000
_vault_lock = threading.Lock()

_RAW_ID_RE = re.compile(r"^\d{10}$")


def tokenize_jan_aadhaar_id(raw_id: str) -> str:
    """Store the raw ID in the vault and return its opaque Reference Key.
    Callers (app/chat.py) should call this as soon as pii_utils extracts an
    ID, keep ONLY the token in conversation state, and pass the token to
    fetch_member_list() - so the raw ID never lives outside this module
    past the moment of extraction. Idempotent for the same ID."""
    with _vault_lock:
        for token, stored in _VAULT.items():
            if stored == raw_id:
                return token
        if len(_VAULT) >= _VAULT_MAX:
            # Oldest-first drop; a stale token just means the citizen
            # re-enters their ID.
            _VAULT.pop(next(iter(_VAULT)))
        token = secrets.token_hex(8)
        _VAULT[token] = raw_id
        return token


def is_vault_token(value: str) -> bool:
    return bool(value) and not _RAW_ID_RE.match(value) and value in _VAULT


def _resolve(id_or_token: str) -> tuple[str, str | None]:
    """(token, raw_id). Accepts either a vault token or a raw 10-digit ID
    (the latter kept for backward compatibility with callers that have
    not yet switched to tokenize_jan_aadhaar_id). raw_id is None for an
    unknown token."""
    if id_or_token and _RAW_ID_RE.match(id_or_token):
        return tokenize_jan_aadhaar_id(id_or_token), id_or_token
    with _vault_lock:
        return id_or_token, _VAULT.get(id_or_token)


def _age_from_dob(dob_ddmmyyyy: str) -> int:
    d, m, y = (int(part) for part in dob_ddmmyyyy.split("-"))
    today = date.today()
    return today.year - y - ((today.month, today.day) < (m, d))


class JanAadhaarService:
    def __init__(self, transport):
        self.transport = transport

    def fetch_member_list(self, jan_aadhaar_id: str) -> dict:
        """`jan_aadhaar_id` may be a vault token (preferred - see
        tokenize_jan_aadhaar_id) or a raw 10-digit ID (still accepted).
        The raw ID is resolved inside this module only. Returns
        {"status", "members", "token"} - the token is what callers should
        keep if they need to refer back to this ID."""
        token, raw_id = _resolve(jan_aadhaar_id)
        if raw_id is None:
            _audit_logger.info("member_list_fetch token=%s code=UNKNOWN_TOKEN", token)
            return {"status": "JAN_404", "members": [], "token": token}
        result = self.transport.fetch_member_list(raw_id)
        _audit_logger.info("member_list_fetch token=%s code=%s", token, result.get("responseCode"))
        if result.get("responseCode") != "JAN_200" or not result.get("data"):
            return {"status": result.get("responseCode", "JAN_404"), "members": [], "token": token}
        members = [
            {"member_id": m["MEMBERID"], "masked_name": m["NAME_EN"]}
            for m in result["data"]
        ]
        return {"status": "JAN_200", "members": members, "token": token}

    def generate_otp(self, member_id: str) -> dict:
        """Always returns {"status", "tid", "mock_otp"}; tid/mock_otp are
        None on failure. mock_otp is only ever set in mock mode with
        JANAADHAAR_DEMO_OTP_PREVIEW=1 (see janaadhaar_mock_data)."""
        result = self.transport.generate_otp(member_id)
        _audit_logger.info("generate_otp member_id=%s code=%s", member_id, result.get("responseCode"))
        if result.get("responseCode") != "JAN_200" or not result.get("tid"):
            return {"status": result.get("responseCode", "JAN_404"), "tid": None, "mock_otp": None}
        return {"status": "JAN_200", "tid": result["tid"], "mock_otp": result.get("_mock_otp_for_demo")}

    def validate_otp_and_fetch(self, member_id: str, tid: str, otp: str) -> dict:
        result = self.transport.validate_otp(member_id, tid, otp)
        _audit_logger.info("validate_otp member_id=%s code=%s", member_id, result.get("responseCode"))
        if result.get("responseCode") != "JAN_200":
            return {"status": result.get("responseCode", "JAN_401")}

        wire_data = result["data"]
        translated = {
            "name": wire_data["NAME_EN"],
            "gender": {"M": "male", "F": "female"}.get(wire_data.get("GENDER")),
            "age": _age_from_dob(wire_data["DOB"]) if wire_data.get("DOB") else None,
        }
        # Everything beyond the four confirmed fields is a stand-in for
        # "additional fields selected during onboarding" - passed through
        # as-is, since the mock already stores these in the shape our own
        # profile schema/citizen_status domain expects.
        for key, value in wire_data.items():
            if key not in _CONFIRMED_KEYS:
                translated[key] = value

        _audit_logger.info("validate_otp member_id=%s result=ok fields=%s", member_id, sorted(translated.keys()))
        return {"status": "JAN_200", "data": translated}

    def get_entitlement_details(self, member_id: str) -> dict:
        """UNVERIFIED SCHEMA end-to-end - see janaadhaar_mock_data.
        handle_entitlement_details for the full caveat. Returns every
        entitlement for this member (not scoped to one scheme); the caller
        matches/filters locally against whichever scheme the citizen asked
        about, or lists all of them for a general "what am I enrolled in"
        question."""
        result = self.transport.get_entitlement_details(member_id)
        _audit_logger.info("entitlement_details member_id=%s code=%s", member_id, result.get("responseCode"))
        if result.get("responseCode") != "JAN_200":
            return {"status": result.get("responseCode", "JAN_404"), "entitlements": []}

        entitlements = [
            {
                "scheme_code": e.get("schemeCode"),
                "scheme_name": e.get("schemeName"),
                "status": e.get("status"),
                "last_payment_amount": e.get("lastPaymentAmount"),
                "last_payment_date": e.get("lastPaymentDate"),
            }
            for e in (result.get("data") or [])
        ]
        return {"status": "JAN_200", "entitlements": entitlements}


_service = None


def get_janaadhaar_service() -> JanAadhaarService:
    global _service
    if _service is None:
        if JANAADHAAR_MODE == "real":
            transport = RealJanAadhaarTransport()  # raises NotImplementedError
        else:
            transport = MockJanAadhaarTransport(base_url=JANAADHAAR_MOCK_BASE_URL)
        _service = JanAadhaarService(transport)
    return _service
