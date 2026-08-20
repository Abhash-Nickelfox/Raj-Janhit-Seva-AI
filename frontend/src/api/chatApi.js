// The backend (app/main.py) returns {session_id, reply, sms_preview?,
// retryable?, error_code?} - no profile/shortlist/intent of any kind; that
// is written to the server log instead.
//
// Every fetch carries a 60 s AbortSignal timeout so a hung backend turns
// into a visible, retry-able error bubble instead of a spinner forever.
const REQUEST_TIMEOUT_MS = 60000;

function timeoutSignal() {
  if (typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function") {
    return AbortSignal.timeout(REQUEST_TIMEOUT_MS);
  }
  const controller = new AbortController();
  setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  return controller.signal;
}

// preferredLang ("hindi" | "hinglish" | "english" | null) locks reply_lang
// for the session once set.
export async function sendMessage(sessionId, message, preferredLang) {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message, preferred_lang: preferredLang || null }),
    signal: timeoutSignal(),
  });
  if (!res.ok) {
    throw new Error(`Network error (${res.status})`);
  }
  return res.json();
}

const MIME_TO_EXT = {
  "audio/webm": "webm",
  "audio/ogg": "ogg",
  "audio/mp4": "m4a",
  "audio/mpeg": "mp3",
  "audio/wav": "wav",
  "audio/x-wav": "wav",
  "audio/aac": "aac",
  "audio/flac": "flac",
};

export function filenameForBlob(blob) {
  const mime = (blob?.type || "audio/webm").split(";")[0].trim().toLowerCase();
  return `recording.${MIME_TO_EXT[mime] || "webm"}`;
}

// Voice input - stateless, no session_id involved (see app/main.py's
// /transcribe). The returned text is handed to sendMessage() like any typed
// message. No language hint is sent from here; app/stt_service.py decides.
export async function transcribeAudio(blob) {
  const formData = new FormData();
  formData.append("audio", blob, filenameForBlob(blob));

  const res = await fetch("/transcribe", { method: "POST", body: formData, signal: timeoutSignal() });
  if (!res.ok) {
    throw new Error(`Transcription failed (${res.status})`);
  }
  const { text } = await res.json();
  return text;
}
