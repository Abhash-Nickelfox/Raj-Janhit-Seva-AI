"""Consumer-side transport for the Jan Aadhaar Fetch Details flow - builds
the exact wire envelope the real API expects and parses the response back.
This is the ONE seam that changes between mock and real:
MockJanAadhaarTransport and RealJanAadhaarTransport share the same
interface (fetch_member_list/generate_otp/validate_otp), so
app/janaadhaar_service.py never needs to know or care which one it's
talking to - swapping JANAADHAAR_MODE in .env is the whole change.
"""

import secrets

import requests

from app import janaadhaar_crypto as crypto
from app.config import JANAADHAAR_APP_CODE, JANAADHAAR_CLIENT_ID, JANAADHAAR_SCHEME_CODE

_PREFIX = "/app/live/apiservice/janAadhaar/v1"


def _new_transaction_id() -> str:
    # Real transactionIds are generated fresh per call by the consumer, not
    # reused - the server rejects a repeat with JAN_501 (see
    # janaadhaar_mock_data.is_duplicate_transaction).
    return "TXN" + secrets.token_hex(6)


class MockJanAadhaarTransport:
    """Talks to our OWN local mock routes (app/janaadhaar_mock_routes.py)
    over real HTTP, using two DISTINCT keypairs
    (janaadhaar_crypto.CONSUMER_KEYPAIR / JANAADHAAR_KEYPAIR) so the round
    trip genuinely simulates two separate parties, not one party signing
    its own messages with a single key."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _call(self, path: str, fields: dict) -> dict:
        inner = {
            "appCode": JANAADHAAR_APP_CODE,
            "schemShortCode": JANAADHAAR_SCHEME_CODE,
            "transactionId": _new_transaction_id(),
            **fields,
        }
        sealed = crypto.seal(inner, "data", crypto.CONSUMER_KEYPAIR.private_key, crypto.JANAADHAAR_KEYPAIR.public_key)
        resp = requests.post(
            f"{self.base_url}{path}",
            params={"client_id": JANAADHAAR_CLIENT_ID},
            json={"data": sealed},
            timeout=10,
        )
        resp.raise_for_status()
        return crypto.open_envelope(
            resp.json()["data"], "response", crypto.CONSUMER_KEYPAIR.private_key, crypto.JANAADHAAR_KEYPAIR.public_key
        )

    def fetch_member_list(self, jan_id: str) -> dict:
        return self._call(f"{_PREFIX}/member-list", {"janId": jan_id})

    def generate_otp(self, member_id: str) -> dict:
        return self._call(f"{_PREFIX}/generate-otp", {"memberId": member_id})

    def validate_otp(self, member_id: str, tid: str, otp: str) -> dict:
        return self._call(f"{_PREFIX}/validate-otp", {"memberId": member_id, "tid": tid, "otp": otp})

    def get_entitlement_details(self, member_id: str) -> dict:
        # UNVERIFIED - see janaadhaar_mock_data.handle_entitlement_details
        # for the full caveat on this endpoint's real path/schema.
        return self._call(f"{_PREFIX}/entitlement-details", {"janaadhaarMemId": member_id})


class RealJanAadhaarTransport:
    """Placeholder - real access requires departmental onboarding (digital
    signature, RSA keypair exchange with Jan Aadhaar's actual public key,
    IP whitelisting, an approved Consumer Subscription via Sewadwaar). Once
    granted, this class would POST to https://apitest.sewadwaar.rajasthan.
    gov.in (or the production host) with the SAME paths/payload shape
    MockJanAadhaarTransport already uses - only base_url and the keypairs
    passed to crypto.seal/open_envelope would differ."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Real Jan Aadhaar integration requires departmental onboarding - not available yet. "
            "Set JANAADHAAR_MODE=mock in .env."
        )
