import pytest

from app import pii_utils as p


# --- OTP ---------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("123456", "123456"),
    ("OTP 123456 aaya, 2 minute pehle", "123456"),
    ("otp hai 12 34 56", "123456"),
    ("123-456", "123456"),
    ("१२३४५६", "123456"),
    ("mera otp १२३४५६ hai", "123456"),
    ("umar 65, income 2 lakh, 2024", None),          # scattered digits != OTP
    ("1234567", None),                                 # 7 contiguous
    ("12345", None),
    ("9876543210", None),                              # phone number is not an OTP
    ("", None),
])
def test_extract_otp(text, expected):
    assert p.extract_otp(text) == expected


# --- Jan Aadhaar ID ------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("1234567890", "1234567890"),
    ("mera id 1234 5678 90 hai", "1234567890"),
    ("1234-5678-90", "1234567890"),
    ("१२३४५६७८९०", "1234567890"),
    ("call me on 98765 and id 12345", None),          # 5 + 5 scattered
    ("12345678901", None),                             # 11 contiguous
    ("id 1234567890 ya 9876543210", "1234567890"),     # first group wins
])
def test_extract_jan_aadhaar_id(text, expected):
    assert p.extract_jan_aadhaar_id(text) == expected


# --- Consent -------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "haan", "Haan ji", "ji", "hanji", "bilkul", "zaroor", "kar do", "karo", "sure", "yeah", "yep",
    "theek hai", "ok", "okay", "yes", "हाँ", "हां", "जी", "बिल्कुल", "ठीक है", "करो",
    "haan, koi problem nahi", "ji haan kar do", "yes please", "हाँ करो",
])
def test_consent_yes(text):
    assert p.extract_consent(text) is True


@pytest.mark.parametrize("text", [
    "nahi", "nahin", "no", "nope", "mat karo", "rehne do", "chhodo", "cancel", "नहीं", "नही", "मत",
    "रहने दो", "ना", "nahi karna", "abhi nahi karna, haan baad mein",
])
def test_consent_no(text):
    assert p.extract_consent(text) is False


@pytest.mark.parametrize("text", [
    "pata nahi", "mujhe pata nahi", "पता नहीं", "confuse hoon",
    "mera naam Hanuman hai",          # "han" inside Hanuman, "na" inside naam
    "dhanyavad",
    "kya karna hai",                  # "ना" / "na" inside karna
    "अपना नाम बताओ",
    "",
    "haan nahi",                      # both, no structure
])
def test_consent_unclear(text):
    assert p.extract_consent(text) is None


# --- Cancel ----------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("cancel", True), ("skip karo", True), ("rehne do", True), ("chhodo yaar", True),
    ("baad mein karenge", True), ("nahi karna", True), ("band karo", True), ("wait", True),
    ("ruko", True), ("रहने दो", True), ("छोड़ो", True), ("बाद में", True),
    ("1234567890", False), ("haan", False), ("mera naam Ruko nahi hai", True), ("ok", False),
])
def test_extract_cancel(text, expected):
    assert p.extract_cancel(text) is expected


# --- Redaction -------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("otp 123456 hai", "otp XXXXXX hai"),
    ("id 1234 5678 90", "id XXXXXX"),
    ("phone 9876543210", "phone XXXXXX"),
    ("umar 65 hai", "umar 65 hai"),
    ("१२३४५६", "XXXXXX"),
    ("", ""),
])
def test_redact_digits(text, expected):
    assert p.redact_digits(text) == expected
