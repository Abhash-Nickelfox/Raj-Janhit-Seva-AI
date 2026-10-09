// The backend (app/main.py) deliberately returns ONLY {session_id, reply} -
// no profile/shortlist/intent/traceability of any kind. That data still
// exists, it's just written to the server log instead, never sent here.
//
// preferredLang ("hindi" | "hinglish" | "english" | null) is sent on every
// request once the citizen picks one via the language picker - the backend
// locks reply_lang to it for the rest of the session, so voice input's
// transcription (which can come back as either Devanagari or Romanized
// Hindi for the same spoken sentence) can't cause it to flip mid-conversation.
export async function sendMessage(sessionId, message, preferredLang) {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message, preferred_lang: preferredLang || null }),
  });
  if (!res.ok) {
    throw new Error("Network error");
  }
  return res.json(); // { session_id, reply }
}

// Voice input - stateless, no session_id involved (see app/main.py's
// /transcribe). The returned text is just handed to sendMessage() above
// like any typed message.
//
// No language hint sent (2026-08-19, reverting a 2026-08-18 attempt at one
// sourced from the previous reply's language) - see app/stt_service.py's
// docstring for why that was actively wrong the moment a citizen switched
// languages between voice messages, not just unnecessary.
export async function transcribeAudio(blob) {
  const formData = new FormData();
  formData.append("audio", blob, "recording.webm");

  const res = await fetch("/transcribe", { method: "POST", body: formData });
  if (!res.ok) {
    throw new Error("Transcription failed");
  }
  const { text } = await res.json();
  return text;
}
