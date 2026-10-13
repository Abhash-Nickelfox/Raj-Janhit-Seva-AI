"""Field rules for registration and sign-in. The frontend checks the same rules
for instant feedback, but these server-side checks are the ones that count."""

import re

from email_validator import EmailNotValidError, validate_email

NAME_RE = re.compile(r"^[A-Za-zऀ-ॿ][A-Za-zऀ-ॿ .'-]{1,99}$")
DESIGNATION_RE = re.compile(r"^[A-Za-z0-9ऀ-ॿ][A-Za-z0-9ऀ-ॿ .,&()/'-]{1,99}$")
PHONE_RE = re.compile(r"^[6-9]\d{9}$")
PASSWORD_MIN, PASSWORD_MAX = 8, 64


def clean_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def check_name(value: str) -> str:
    value = clean_spaces(value)
    if not NAME_RE.match(value):
        raise ValueError("Enter your full name (2-100 letters; spaces, dot, hyphen and apostrophe allowed).")
    return value


def check_designation(value: str) -> str:
    value = clean_spaces(value)
    if not DESIGNATION_RE.match(value):
        raise ValueError("Enter your role/designation (2-100 characters).")
    return value


def check_email(value: str) -> str:
    try:
        return validate_email(clean_spaces(value), check_deliverability=False).normalized.lower()
    except EmailNotValidError as exc:
        raise ValueError("Enter a valid email address.") from exc


def normalize_phone(value: str) -> str | None:
    """Indian mobile number as 10 digits. Accepts spaces, dashes, +91, 91 or 0 in front."""
    digits = re.sub(r"[\s\-()]", "", value or "")
    if digits.startswith("+91"):
        digits = digits[3:]
    elif len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return digits if PHONE_RE.match(digits) else None


def check_phone(value: str) -> str:
    phone = normalize_phone(value)
    if not phone:
        raise ValueError("Enter a valid 10-digit Indian mobile number (starting with 6, 7, 8 or 9).")
    return phone


def check_password(value: str) -> str:
    if not PASSWORD_MIN <= len(value) <= PASSWORD_MAX:
        raise ValueError(f"Password must be {PASSWORD_MIN}-{PASSWORD_MAX} characters.")
    if value != value.strip():
        raise ValueError("Password can't start or end with a space.")
    if not (re.search(r"[A-Za-z]", value) and re.search(r"\d", value)):
        raise ValueError("Password must contain at least one letter and one number.")
    return value
