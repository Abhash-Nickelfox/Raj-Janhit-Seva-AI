"""Speech-to-text for voice input, via Groq's Whisper endpoint.

Deliberately NOT routed through the LLM/conversation layer at all - this is
a pure audio-in, text-out utility. Whatever script Whisper produces
(Devanagari, Roman, English) just becomes the citizen's message text and
flows through the EXISTING three-way language detection in
extract_need_node - no separate language handling needed here.

Language hint (2026-08-12): confirmed live - speaking Hindi with no language
hint got TRANSLATED to English text instead of transcribed faithfully (a
known Whisper quirk, especially on the turbo/distilled model variants used
here - without an explicit hint it sometimes decides to translate-to-English
rather than transcribe-in-original-language). Once the citizen has picked a
language via the frontend's language picker, we now know what they're
speaking and pass it through as an explicit hint, which stops this drift.
Before the picker existed there was no such signal available at all, which
is why this used to be deliberately left unset.
"""

import requests

from app.config import GROQ_API_KEY, GROQ_STT_MODEL

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Our reply_lang values ("hindi"/"hinglish"/"english") aren't ISO-639-1 codes.
# Hinglish is still spoken Hindi (just a script preference for the BOT's own
# replies) - the citizen's spoken audio is the same either way, so it maps
# to the same "hi" hint as "hindi".
_LANGUAGE_HINTS = {"hindi": "hi", "hinglish": "hi", "english": "en"}


class TranscriptionError(Exception):
    pass


def transcribe(audio_bytes: bytes, filename: str = "audio.webm", reply_lang: str = None) -> str:
    if not GROQ_API_KEY:
        raise TranscriptionError(
            "GROQ_API_KEY not set - copy .env.example's GROQ_API_KEY into .env to enable voice input."
        )

    data = {"model": GROQ_STT_MODEL, "response_format": "text"}
    hint = _LANGUAGE_HINTS.get(reply_lang)
    if hint:
        data["language"] = hint

    response = requests.post(
        GROQ_TRANSCRIPTION_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": (filename, audio_bytes)},
        data=data,
        timeout=30,
    )
    if not response.ok:
        raise TranscriptionError(f"Groq transcription failed ({response.status_code}): {response.text}")

    return response.text.strip()
