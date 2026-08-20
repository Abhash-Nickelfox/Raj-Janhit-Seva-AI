"""Speech-to-text for voice input, via Sarvam's Saaras model (2026-08-18,
replacing Groq's hosted Whisper - see app/config.py's SARVAM_API_KEY comment
for why).

Deliberately NOT routed through the LLM/conversation layer at all - this is
a pure audio-in, text-out utility. Whatever script this produces (Devanagari,
Roman, English) just becomes the citizen's message text and flows through
the EXISTING three-way language detection in extract_need_node - no separate
language handling needed here.

mode="transcribe" (Saaras's default): native-script output only (Devanagari
for Hindi) - deliberately NOT "codemix", because extract_need_node's
language classifier assumes one script per message.

Language hint (2026-08-20): language_code="unknown" (Sarvam auto-detect)
made saaras:v3 emit Gurmukhi/Bengali script for plainly Hindi speech, which
the downstream classifier can't handle at all. We now send
SARVAM_STT_LANGUAGE (default hi-IN). If the transcript still comes back in
a non-Devanagari Indic script, or Sarvam reports a language other than
hi-IN/en-IN, we retry ONCE with en-IN (covers citizens who actually spoke
English). If that is still neither Devanagari nor Latin, we give up with a
"please repeat" error rather than hand garbage to the chat graph.
"""

import logging
import re

import requests

from app.config import SARVAM_API_KEY, SARVAM_STT_LANGUAGE, SARVAM_STT_MODEL

logger = logging.getLogger("jan_seva_stt")

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
MAX_AUDIO_BYTES = 8 * 1024 * 1024  # 8 MB
REQUEST_TIMEOUT_S = 30

_ACCEPTED_LANGS = {"hi-IN", "en-IN"}

# Devanagari block, plus Latin letters. Anything in another Indic block
# (Gurmukhi U+0A00-0A7F, Bengali U+0980-09FF, Gujarati, Tamil, Telugu,
# Kannada, Malayalam, Odia, Urdu/Arabic) counts as "wrong script".
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_OTHER_INDIC_RE = re.compile(
    r"[ঀ-৿਀-੿઀-૿଀-୿஀-௿"
    r"ఀ-౿ಀ-೿ഀ-ൿ؀-ۿ]"
)

_COULD_NOT_UNDERSTAND = "Could not understand the audio - please repeat, or type your message."
_TOO_LARGE = "Audio is too large (max 8 MB) - please record a shorter message."
_UPSTREAM_FAILED = "Speech-to-text service failed - please try again, or type your message."


class TranscriptionError(Exception):
    pass


def has_foreign_script(text: str) -> bool:
    return bool(_OTHER_INDIC_RE.search(text or ""))


def is_usable_script(text: str) -> bool:
    """Devanagari and/or Latin only (digits/punctuation ignored)."""
    if has_foreign_script(text):
        return False
    return bool(_DEVANAGARI_RE.search(text) or _LATIN_RE.search(text))


def _call_sarvam(audio_bytes: bytes, filename: str, mime: str, language_code: str) -> tuple[str, str | None]:
    data = {
        "model": SARVAM_STT_MODEL,
        "mode": "transcribe",
        "language_code": language_code,
    }
    try:
        response = requests.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": (filename, audio_bytes, mime)},
            data=data,
            timeout=REQUEST_TIMEOUT_S,
        )
    except requests.Timeout:
        logger.error("sarvam stt timeout after %ss", REQUEST_TIMEOUT_S)
        raise TranscriptionError(_UPSTREAM_FAILED)
    except requests.RequestException as e:
        logger.error("sarvam stt request error: %s", e.__class__.__name__)
        raise TranscriptionError(_UPSTREAM_FAILED)

    if not response.ok:
        # Raw upstream body goes to the log only, never into the exception
        # message (which reaches the browser via /transcribe's 502 detail).
        logger.error("sarvam stt HTTP %s: %s", response.status_code, response.text[:500])
        if response.status_code == 413:
            raise TranscriptionError(_TOO_LARGE)
        if response.status_code in (400, 415, 422):
            raise TranscriptionError(_COULD_NOT_UNDERSTAND)
        raise TranscriptionError(_UPSTREAM_FAILED)

    try:
        payload = response.json()
    except ValueError:
        logger.error("sarvam stt returned non-JSON body")
        raise TranscriptionError(_UPSTREAM_FAILED)
    transcript = (payload.get("transcript") or "").strip()
    return transcript, payload.get("language_code")


def transcribe(audio_bytes: bytes, filename: str = "audio.webm", content_type: str = None) -> str:
    if not SARVAM_API_KEY:
        raise TranscriptionError(
            "SARVAM_API_KEY not set - copy .env.example's SARVAM_API_KEY into .env to enable voice input."
        )
    if not audio_bytes:
        raise TranscriptionError(_COULD_NOT_UNDERSTAND)
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise TranscriptionError(_TOO_LARGE)

    # Sarvam validates the part's Content-Type strictly; strip any
    # ";codecs=..." parameter and default to audio/webm.
    mime = (content_type or "audio/webm").split(";")[0].strip() or "audio/webm"

    transcript, lang = _call_sarvam(audio_bytes, filename, mime, SARVAM_STT_LANGUAGE)
    if transcript and not has_foreign_script(transcript) and (lang is None or lang in _ACCEPTED_LANGS):
        return transcript

    logger.info(
        "sarvam stt retry with en-IN (first pass lang=%s foreign_script=%s empty=%s)",
        lang, has_foreign_script(transcript), not transcript,
    )
    transcript, lang = _call_sarvam(audio_bytes, filename, mime, "en-IN")
    if transcript and is_usable_script(transcript):
        return transcript
    raise TranscriptionError(_COULD_NOT_UNDERSTAND)
