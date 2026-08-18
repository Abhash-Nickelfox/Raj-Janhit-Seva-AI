"""Speech-to-text for voice input, via Sarvam's Saaras model (2026-08-18,
replacing Groq's hosted Whisper - see app/config.py's SARVAM_API_KEY comment
for why).

Deliberately NOT routed through the LLM/conversation layer at all - this is
a pure audio-in, text-out utility. Whatever script this produces (Devanagari,
Roman, English) just becomes the citizen's message text and flows through
the EXISTING three-way language detection in extract_need_node - no separate
language handling needed here.

mode="transcribe" (Saaras's default): native-script output only (Devanagari
for Hindi, matching Whisper's old behavior) - deliberately NOT "codemix"
(Devanagari for Hindi words, Latin for English words in the SAME sentence),
even though Sarvam's own docs recommend codemix for naturally code-mixed
speech and it's likely more accurate content-wise. codemix's mixed-script
output doesn't map cleanly onto this app's existing hindi(Devanagari-only)/
hinglish(Roman-only)/english three-way split - extract_need_node's
detected_language classifier and normalize_node's DEVANAGARI_RE placeholder
guess both assume one script per message. "transcribe" keeps the exact same
downstream classification working unchanged; revisit codemix only alongside
updating that classifier to handle mixed-script text.

Language hint (2026-08-12, still needed): confirmed live - speaking Hindi
with no language hint got TRANSLATED to English text instead of transcribed
faithfully (a known Whisper quirk; not yet re-verified against Saaras, but
passing a hint costs nothing and Sarvam's own "unknown" value degrades
gracefully to auto-detect if this turns out to be unnecessary here too).
reply_lang was previously sourced from the frontend's language picker -
since that's been removed, main.py now threads through the LAST turn's own
reply_lang instead (see ChatResponse.reply_lang), so returning citizens still
get a hint after their first exchange; a citizen's very first-ever message
being voice still has no signal to hint from, same as before the picker
ever existed.
"""

import requests

from app.config import SARVAM_API_KEY, SARVAM_STT_MODEL

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

# Our reply_lang values ("hindi"/"hinglish"/"english") aren't BCP-47 codes.
# Hinglish is still spoken Hindi (just a script preference for the BOT's own
# replies) - the citizen's spoken audio is the same either way, so it maps
# to the same "hi-IN" hint as "hindi". "unknown" (Sarvam's own auto-detect
# value) is used when we have no signal at all, rather than omitting the
# field - Sarvam's own recommended way to say "figure it out", vs Groq's
# API where omitting the field entirely was the only option.
_LANGUAGE_HINTS = {"hindi": "hi-IN", "hinglish": "hi-IN", "english": "en-IN"}


class TranscriptionError(Exception):
    pass


def transcribe(audio_bytes: bytes, filename: str = "audio.webm", reply_lang: str = None,
                content_type: str = None) -> str:
    if not SARVAM_API_KEY:
        raise TranscriptionError(
            "SARVAM_API_KEY not set - copy .env.example's SARVAM_API_KEY into .env to enable voice input."
        )

    # Found live (2026-08-18): the 2-tuple form (filename, bytes) leaves
    # `requests` to send no explicit Content-Type at all for the file part -
    # Groq's endpoint tolerated that, but Sarvam's validates strictly against
    # its own allowed-MIME list and rejects "None" outright. Strip any
    # ";codecs=..." parameter (MediaRecorder's actual mimeType is typically
    # "audio/webm;codecs=opus", not the bare "audio/webm" Sarvam's list has)
    # and fall back to "audio/webm" if the browser didn't report one at all.
    mime = (content_type or "audio/webm").split(";")[0].strip()

    data = {
        "model": SARVAM_STT_MODEL,
        "mode": "transcribe",
        "language_code": _LANGUAGE_HINTS.get(reply_lang, "unknown"),
    }

    response = requests.post(
        SARVAM_STT_URL,
        headers={"api-subscription-key": SARVAM_API_KEY},
        files={"file": (filename, audio_bytes, mime)},
        data=data,
        timeout=30,
    )
    if not response.ok:
        raise TranscriptionError(f"Sarvam transcription failed ({response.status_code}): {response.text}")

    transcript = response.json().get("transcript", "")
    return transcript.strip()
