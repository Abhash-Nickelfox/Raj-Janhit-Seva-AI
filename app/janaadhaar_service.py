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
import secrets
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


def _tokenize(raw_id: str) -> str:
    for token, stored in _VAULT.items():
        if stored == raw_id:
            return token
    token = secrets.token_hex(8)
    _VAULT[token] = raw_id
    return token


def _age_from_dob(dob_ddmmyyyy: str) -> int:
    d, m, y = (int(part) for part in dob_ddmmyyyy.split("-"))
    today = date.today()
    return today.year - y - ((today.month, today.day) < (m, d))


class JanAadhaarService:
    def __init__(self, transport):
        self.transport = transport

    def fetch_member_list(self, jan_aadhaar_id: str) -> dict:
        token = _tokenize(jan_aadhaar_id)
        result = self.transport.fetch_member_list(jan_aadhaar_id)
        _audit_logger.info("member_list_fetch token=%s code=%s", token, result.get("responseCode"))
        if result.get("responseCode") != "JAN_200" or not result.get("data"):
            return {"status": result.get("responseCode", "JAN_404"), "members": []}
        members = [
            {"member_id": m["MEMBERID"], "masked_name": m["NAME_EN"]}
            for m in result["data"]
        ]
        return {"status": "JAN_200", "members": members}

    def generate_otp(self, member_id: str) -> dict:
        result = self.transport.generate_otp(member_id)
        _audit_logger.info("generate_otp member_id=%s code=%s", member_id, result.get("responseCode"))
        if result.get("responseCode") != "JAN_200":
            return {"status": result.get("responseCode", "JAN_404")}
        # mock_otp is None in real mode - RealJanAadhaarTransport's response
        # never carries "_mock_otp_for_demo" at all, so this simply stays
        # absent with no special-casing needed here.
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
