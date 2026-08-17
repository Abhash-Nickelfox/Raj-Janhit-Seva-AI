"""Loads .env (if present) once, so every module can just read os.environ."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY not set. Copy .env.example to .env and fill in your key."
    )

# Voice input (speech-to-text) - optional, unlike OPENROUTER_API_KEY above:
# the app is fully usable via text without this, so a missing key doesn't
# crash startup, only the /transcribe endpoint itself.
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
