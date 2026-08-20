"""Local mock Sewadwaar/Jan Aadhaar HTTP endpoints - mirrors the real
apitest.sewadwaar.rajasthan.gov.in URL PATH structure exactly (everything
after the host), so MockJanAadhaarTransport and a future
RealJanAadhaarTransport differ ONLY in base_url, never in path or payload
shape. Mounted onto the same FastAPI app as the chatbot API itself
(app/main.py) purely for local-dev convenience - nothing here is meant to
be internet-facing, and client_id is accepted but not validated since
there's no real Consumer Subscription behind this mock.
"""

from fastapi import APIRouter, HTTPException, Request

from app import janaadhaar_crypto as crypto
from app import janaadhaar_mock_data as mockdata

router = APIRouter()

_PREFIX = "/app/live/apiservice/janAadhaar/v1"


async def _read_request(request: Request) -> dict:
    """Decrypts+verifies the inbound envelope using the SERVER's (Jan
    Aadhaar's) keys - decrypt with its own private key, verify against the
    consumer's public key - the mirror image of what the client does."""
    try:
        body = await request.json()
        inner = crypto.open_envelope(
            body["data"], "data", crypto.JANAADHAAR_KEYPAIR.private_key, crypto.CONSUMER_KEYPAIR.public_key
        )
        if not isinstance(inner, dict) or "transactionId" not in inner:
            raise ValueError("missing transactionId")
        return inner
    except Exception as e:  # malformed JSON, bad envelope, bad signature, missing keys
        raise HTTPException(status_code=400, detail=f"Malformed request envelope ({e.__class__.__name__})")


def _field(inner: dict, key: str) -> str:
    value = inner.get(key)
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=400, detail=f"Missing or invalid field: {key}")
    return value


def _seal_response(payload: dict) -> dict:
    """Signs+encrypts the outbound envelope using the SERVER's keys -
    sign with its own private key, encrypt for the consumer's public key."""
    sealed = crypto.seal(payload, "response", crypto.JANAADHAAR_KEYPAIR.private_key, crypto.CONSUMER_KEYPAIR.public_key)
    return {"data": sealed}


def _duplicate_response(transaction_id: str) -> dict:
    return {
        "status": False,
        "message": "Duplicate Transaction ID.",
        "responseCode": "JAN_501",
        "transactionId": transaction_id,
        "data": None,
    }


@router.post(f"{_PREFIX}/member-list")
async def member_list(request: Request, client_id: str = ""):
    inner = await _read_request(request)
    if mockdata.is_duplicate_transaction(inner["transactionId"]):
        result = _duplicate_response(inner["transactionId"])
    else:
        result = mockdata.handle_member_list(_field(inner, "janId"))
        result = {"transactionId": inner["transactionId"], "schemeCode": inner.get("schemShortCode"), "appCode": inner.get("appCode"), **result}
    return _seal_response(result)


@router.post(f"{_PREFIX}/generate-otp")
async def generate_otp(request: Request, client_id: str = ""):
    inner = await _read_request(request)
    if mockdata.is_duplicate_transaction(inner["transactionId"]):
        result = _duplicate_response(inner["transactionId"])
    else:
        result = mockdata.handle_generate_otp(_field(inner, "memberId"))
        result = {"transactionId": inner["transactionId"], "schemeCode": inner.get("schemShortCode"), "appCode": inner.get("appCode"), **result}
    return _seal_response(result)


@router.post(f"{_PREFIX}/validate-otp")
async def validate_otp(request: Request, client_id: str = ""):
    inner = await _read_request(request)
    if mockdata.is_duplicate_transaction(inner["transactionId"]):
        result = _duplicate_response(inner["transactionId"])
    else:
        result = mockdata.handle_validate_otp(_field(inner, "memberId"), _field(inner, "tid"), _field(inner, "otp"))
        result = {"transactionId": inner["transactionId"], "schemeCode": inner.get("schemShortCode"), "appCode": inner.get("appCode"), **result}
    return _seal_response(result)


@router.post(f"{_PREFIX}/entitlement-details")
async def entitlement_details(request: Request, client_id: str = ""):
    """UNVERIFIED SCHEMA - path/shape is a plausible guess (kebab-case,
    same /v1/ prefix as the other three confirmed endpoints), not a
    documented URL. See janaadhaar_mock_data.handle_entitlement_details for
    the same caveat on the response fields."""
    inner = await _read_request(request)
    if mockdata.is_duplicate_transaction(inner["transactionId"]):
        result = _duplicate_response(inner["transactionId"])
    else:
        result = mockdata.handle_entitlement_details(_field(inner, "janaadhaarMemId"))
        result = {"transactionId": inner["transactionId"], "appCode": inner.get("appCode"), **result}
    return _seal_response(result)
