"""What a real Jan Aadhaar server would do internally on receiving a
decrypted request - pure data/business logic, no crypto and no HTTP (those
live in janaadhaar_crypto.py and janaadhaar_mock_routes.py respectively).
Read this module as "the fake government database," in isolation from how
requests physically reach it.

Field names/values below follow the Jan Aadhaar Integration Document v1.8
exactly where the doc actually specifies them (NAME_EN/NAME_LL/DOB/GENDER,
DD-MM-YYYY date format, responseCode values from its Section 6 table). One
documented gap, called out inline: the doc has no specific "wrong OTP"
response code - JAN_001 ("Error processing request, please try again!") is
the closest generic, retryable code in its table; confirm the real one
during onboarding.
"""

import secrets
from datetime import date

# --- mock family data -------------------------------------------------------
# One single-member family (exercises the auto-select path in
# janaadhaar_handle_node) and one multi-member family (exercises the
# member-choice path). "profile"/"status" keys beyond NAME/DOB/GENDER are
# NOT part of the confirmed base schema - they stand in for whatever
# "additional fields selected during onboarding" the real API would return,
# which we have no public documentation for.
MOCK_FAMILIES = {
    "1234567890": {
        "members": [
            {
                "member_id": "MEM100001",
                "name": "Rekha Sharma",
                "name_ll": "रेखा शर्मा",
                "mobile": "9876543210",
                "dob": "1985-03-12",
                "gender_code": "F",
                "extra_fields": {
                    "caste": "obc", "marital_status": "married",
                    "education_level": "higher_education", "disability": "no",
                    "family_income": 210000, "residence": "rural",
                },
                "entitlements": [
                    {
                        "schemeCode": "SCM-SSP-001", "schemeName": "SOCIAL SECURITY PENSION",
                        "status": "Approved - active", "lastPaymentAmount": "1250.00", "lastPaymentDate": "01-07-2026",
                    },
                    {
                        "schemeCode": "SCM-PLY-002", "schemeName": "PALANHAR YOJANA",
                        "status": "Application under verification at Gram Panchayat level",
                        "lastPaymentAmount": None, "lastPaymentDate": None,
                    },
                ],
            },
        ],
    },
    "9988776655": {
        "members": [
            {
                "member_id": "MEM200001",
                "name": "Suresh Kumar Meena",
                "name_ll": "सुरेश कुमार मीणा",
                "mobile": "9911223344",
                "dob": "1962-11-02",
                "gender_code": "M",
                "extra_fields": {
                    "caste": "st", "marital_status": "married",
                    "education_level": "school", "disability": "no",
                    "family_income": 95000, "residence": "rural",
                },
                "entitlements": [
                    {
                        "schemeCode": "SCM-SSP-001", "schemeName": "SOCIAL SECURITY PENSION",
                        "status": "Approved - active", "lastPaymentAmount": "1000.00", "lastPaymentDate": "05-07-2026",
                    },
                ],
            },
            {
                "member_id": "MEM200002",
                "name": "Kavita Meena",
                "name_ll": "कविता मीणा",
                "mobile": "9911223355",
                "dob": "1990-07-19",
                "gender_code": "F",
                "extra_fields": {
                    "caste": "st", "marital_status": "married",
                    "education_level": "higher_education", "disability": "no",
                    "family_income": 95000, "residence": "rural",
                },
                "entitlements": [
                    {
                        "schemeCode": "SCM-SCH-003", "schemeName": "SCHOLARSHIP",
                        "status": "Application under review since 20-07-2026",
                        "lastPaymentAmount": None, "lastPaymentDate": None,
                    },
                ],
            },
        ],
    },
}

# tid -> {"member_id", "otp", "attempts"}
PENDING_OTPS: dict[str, dict] = {}

# Real transactionIds are generated fresh per call by the CONSUMER - the
# server-side is expected to reject a reuse (JAN_501), so we track what
# we've seen the same way a real gateway would.
SEEN_TRANSACTION_IDS: set[str] = set()


def find_member(member_id: str):
    for family in MOCK_FAMILIES.values():
        for member in family["members"]:
            if member["member_id"] == member_id:
                return member
    return None


def _to_ddmmyyyy(iso_date: str) -> str:
    y, m, d = iso_date.split("-")
    return f"{d}-{m}-{y}"


def is_duplicate_transaction(transaction_id: str) -> bool:
    if transaction_id in SEEN_TRANSACTION_IDS:
        return True
    SEEN_TRANSACTION_IDS.add(transaction_id)
    return False


def handle_member_list(jan_id: str) -> dict:
    family = MOCK_FAMILIES.get(jan_id)
    if not family:
        return {
            "status": False,
            "message": "No data found for the given Enrolment ID or Jan Aadhaar ID.",
            "responseCode": "JAN_404",
            "JanId": jan_id,
            "data": None,
        }
    members = [
        {"NAME_EN": _mask_name(m["name"]), "MEMBERID": m["member_id"]}
        for m in family["members"]
    ]
    return {
        "status": True,
        "message": "Members fetched successfully",
        "responseCode": "JAN_200",
        "JanId": jan_id,
        "data": members,
    }


def handle_generate_otp(member_id: str) -> dict:
    member = find_member(member_id)
    if not member:
        return {
            "status": False,
            "message": "No data found for the given Enrolment ID or Jan Aadhaar ID.",
            "responseCode": "JAN_404",
            "tid": None,
            "JanId": None,
            "data": None,
        }
    otp = f"{secrets.randbelow(10 ** 6):06d}"
    tid = "TID" + secrets.token_hex(6)
    PENDING_OTPS[tid] = {"member_id": member_id, "otp": otp, "attempts": 0}
    # A real integration never sees this OTP - it goes straight from Jan
    # Aadhaar to the member's own registered mobile via SMS. Printed here
    # only because there's no real SMS gateway in the mock.
    print(f"[MOCK SMS to {member['mobile']}] Your Jan Aadhaar OTP is {otp}")
    return {
        "status": True,
        "message": f"OTP sent to xxxxxx{member['mobile'][-4:]} successfully",
        "responseCode": "JAN_200",
        "tid": tid,
        "JanId": None,
        "data": None,
        # Mock-only convenience (2026-08-17): the REAL API never returns the
        # OTP itself in this response - it only ever reaches the citizen via
        # actual SMS. This field exists purely so the demo UI can show it as
        # a simulated "SMS arrived" popup instead of requiring a look at
        # server logs. RealJanAadhaarTransport's response will never carry
        # this key, so the popup naturally disappears once
        # JANAADHAAR_MODE=real - no special-casing needed anywhere else.
        "_mock_otp_for_demo": otp,
    }


def handle_validate_otp(member_id: str, tid: str, otp: str) -> dict:
    pending = PENDING_OTPS.get(tid)
    if not pending or pending["member_id"] != member_id:
        return {
            "status": False,
            "message": "No data found for the given Enrolment ID or Jan Aadhaar ID.",
            "responseCode": "JAN_404",
            "tid": tid,
            "JanId": None,
            "data": None,
        }
    pending["attempts"] += 1
    if pending["attempts"] > 3 or pending["otp"] != otp:
        return {
            "status": False,
            "message": "Invalid OTP, please try again.",
            "responseCode": "JAN_001",
            "tid": tid,
            "JanId": None,
            "data": None,
        }
    del PENDING_OTPS[tid]
    member = find_member(member_id)
    data = {
        "NAME_EN": member["name"],
        "NAME_LL": member["name_ll"],
        "DOB": _to_ddmmyyyy(member["dob"]),
        "GENDER": member["gender_code"],
        **member["extra_fields"],
    }
    return {
        "status": True,
        "message": "OTP validated successfully",
        "responseCode": "JAN_200",
        "tid": tid,
        "JanId": None,
        "data": data,
    }


def handle_entitlement_details(member_id: str) -> dict:
    """UNVERIFIED SCHEMA - "Jan Aadhaar Entitlement Details" is confirmed to
    exist as a published API name on Sewadwaar, but its real request/
    response fields are not documented anywhere publicly (confirmed by
    direct search, 2026-08-16). Everything below is an educated inference
    from the one schema we DO have - the DBT Cash/Non-Cash Transaction PUSH
    payload (schemeCode, entitlementId/entitlementMemId, paymentAmount,
    paymentDate) - on the reasoning that a generic "read my entitlements"
    endpoint would plausibly mirror what gets pushed IN via that same
    pipeline. Scheme NAMES are drawn from the real, confirmed "Jan Aadhaar
    integrated Schemes" dashboard (janaadhaar.rajasthan.gov.in) for realism;
    scheme CODES, status wording, and the response shape itself are
    invented. Returns EVERY entitlement for the member (not scoped to one
    scheme) - the chatbot filters/matches locally against whichever scheme
    the citizen actually asked about."""
    member = find_member(member_id)
    if not member:
        return {
            "status": False,
            "message": "No data found for the given Enrolment ID or Jan Aadhaar ID.",
            "responseCode": "JAN_404",
            "data": None,
        }
    return {
        "status": True,
        "message": "Entitlements fetched successfully",
        "responseCode": "JAN_200",
        "data": member["entitlements"],
    }


def _mask_name(name: str) -> str:
    def mask_word(word: str) -> str:
        if len(word) <= 2:
            return word
        return word[0] + "*" * (len(word) - 2) + word[-1]

    return " ".join(mask_word(w) for w in name.split())
