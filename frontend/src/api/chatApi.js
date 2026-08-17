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
// preferredLang is passed through as a language hint to Whisper - confirmed
// live (2026-08-12) that speaking Hindi with no hint got the transcription
// TRANSLATED into English text instead of transcribed faithfully.
export async function transcribeAudio(blob, preferredLang) {
  const formData = new FormData();
  formData.append("audio", blob, "recording.webm");
  if (preferredLang) {
    formData.append("preferred_lang", preferredLang);
  }

  const res = await fetch("/transcribe", { method: "POST", body: formData });
  if (!res.ok) {
    throw new Error("Transcription failed");
  }
  const { text } = await res.json();
  return text;
}
