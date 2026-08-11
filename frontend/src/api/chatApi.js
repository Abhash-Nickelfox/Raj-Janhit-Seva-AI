// The backend (app/main.py) deliberately returns ONLY {session_id, reply} -
// no profile/shortlist/intent/traceability of any kind. That data still
// exists, it's just written to the server log instead, never sent here.
export async function sendMessage(sessionId, message) {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.ok) {
    throw new Error("Network error");
  }
  return res.json(); // { session_id, reply }
}
