"""Loads .env (if present) once, so every module can just read os.environ."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")



def require_openrouter_key() -> str:
    """Raise (lazily, at first LLM use - not at import) if the key is unset.
    Importing app.config/app.main no longer crashes without a key, so
    /health can report the problem and tests can import the app."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and fill in your key."
        )
    return OPENROUTER_API_KEY


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# Comma-separated list of allowed CORS origins. "*" (default) is fine for
# local dev; app/main.py logs a warning when it's in effect.
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()
] or ["*"]

# Idle sessions (no /chat request for this long) are evicted from the
# in-memory checkpointer by app/main.py's background sweeper.
SESSION_TTL_MINUTES = int(os.environ.get("SESSION_TTL_MINUTES", "120"))

# Voice input (speech-to-text) - optional, unlike OPENROUTER_API_KEY above:
# the app is fully usable via text without this, so a missing key doesn't
# crash startup, only the /transcribe endpoint itself.
#
# Sarvam's Saaras (2026-08-18) replaces Groq's hosted Whisper as the actual
# transcription backend - purpose-built for Indian languages and code-mixed
# Hindi/English speech specifically, which is exactly where Whisper was
# shakiest (see stt_service.py's docstring for the translate-instead-of-
# transcribe bug this was already working around). GROQ_API_KEY/
# GROQ_STT_MODEL are kept, unused for now, in case of a future rollback.
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
SARVAM_STT_MODEL = os.environ.get("SARVAM_STT_MODEL", "saaras:v3")
# Language hint sent to Sarvam. "unknown" (auto-detect) made saaras:v3 emit
# Gurmukhi for Hindi speech, so hi-IN is the default; stt_service.py falls
# back to en-IN once if the transcript comes back in a non-Devanagari script.
SARVAM_STT_LANGUAGE = os.environ.get("SARVAM_STT_LANGUAGE", "hi-IN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")

# "mock" (default) calls our own local mock Sewadwaar routes
# (app/janaadhaar_mock_routes.py) over real HTTP. "real" is a placeholder
# that raises until actual departmental access is granted (digital
# signature + RSA keypair exchange + IP whitelisting via Sewadwaar's
# Consumer Subscription flow - see the sewadwaar-janaadhaar-api research
# memory). Swapping this one flag is the whole point of
# get_janaadhaar_service() being a factory, not a constructor called
# directly from app/chat.py.
JANAADHAAR_MODE = os.environ.get("JANAADHAAR_MODE", "mock")

# Base URL the client (app/janaadhaar_client.py) POSTs to - our own dev
# server by default, since the mock routes are mounted on this same FastAPI
# app. Swapping to the real host (apitest.sewadwaar.rajasthan.gov.in, then
# the production host) is the only change RealJanAadhaarTransport needs
# beyond the keypairs themselves.
JANAADHAAR_MOCK_BASE_URL = os.environ.get("JANAADHAAR_MOCK_BASE_URL", "http://localhost:8000")

# Placeholders standing in for values a real Consumer Subscription would
# assign us - appCode/schemShortCode identify OUR registered application
# and the specific onboarded scheme to Jan Aadhaar; client_id is Sewadwaar's
# own per-App identifier, passed as a query param on every call per its
# gateway convention.
JANAADHAAR_APP_CODE = os.environ.get("JANAADHAAR_APP_CODE", "MOCKAPP01")
JANAADHAAR_SCHEME_CODE = os.environ.get("JANAADHAAR_SCHEME_CODE", "MOCKSCM01")
JANAADHAAR_CLIENT_ID = os.environ.get("JANAADHAAR_CLIENT_ID", "mock-client-id")

# Mock-only: when "1", the mock's generated OTP is returned to the browser
# as a simulated "SMS arrived" popup. Default off - the OTP is then only
# written to the janaadhaar_audit log, so testers must set this to 1 (or
# read the server log) to complete a mock verification.
JANAADHAAR_DEMO_OTP_PREVIEW = _env_flag("JANAADHAAR_DEMO_OTP_PREVIEW", "0")
