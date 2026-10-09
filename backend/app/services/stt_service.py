"""Speech-to-text for voice input, via Sarvam's Saaras model (2026-08-18,
replacing Groq's hosted Whisper - see app/core/config.py's SARVAM_API_KEY comment
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

No language hint sent (2026-08-19, reverting the 2026-08-18 attempt at one):
tried sourcing a hint from the PREVIOUS turn's own reply_lang, on the theory
that a citizen's spoken language stays constant across a conversation -
found live that this is false and actively harmful, not just unnecessary:
a citizen who spoke English, then switched to Hindi for their next voice
message, got that Hindi speech transcribed as English text, because the
stale hint from turn 1's English reply told Sarvam "language_code: en-IN"
for turn 2's actual Hindi speech, overriding what was really said. The
original 2026-08-12 concern (no-hint Whisper sometimes translating instead
of transcribing) doesn't even apply here - mode="transcribe" is already
forced explicitly below regardless of language, unlike Whisper's single
ambiguous endpoint. Always passing language_code="unknown" (Sarvam's own
auto-detect) lets it read the ACTUAL audio fresh every turn instead of
trusting a signal that can silently go stale the moment someone switches
languages mid-conversation - which voice users can do far more easily than
text users ever could.
"""

import requests

from app.core.config import SARVAM_API_KEY, SARVAM_STT_MODEL

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class TranscriptionError(Exception):
    pass


def transcribe(audio_bytes: bytes, filename: str = "audio.webm", content_type: str = None) -> str:
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
        "language_code": "unknown",
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
