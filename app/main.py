"""FastAPI entry point (PLAN.md's suggested layout) - the HTTP layer over
app/chat.py's conversation graph. Deliberately thin: no business logic
lives here, just request/response plumbing plus the traceability split
this endpoint exists to enforce -

The API response NEVER includes profile/shortlist/intent/turn_type/etc -
only {session_id, reply}. All of that is logged server-side instead (see
_log_turn below), so a frontend can render a clean chat UI while the full
decision trail stays inspectable in the server logs, not exposed to the
client.

Run: uvicorn app.main:app --reload --port 8000
"""

import logging
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.chat import build_graph, get_llm, get_rag_service, get_job_service
from app.stt_service import TranscriptionError, transcribe
from app.janaadhaar_mock_routes import router as janaadhaar_mock_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("jan_seva_chat")

app = FastAPI(title="Jan Seva AI Chatbot API")

# Permissive CORS for local development - the frontend's Vite dev server
# proxies /chat and /health to this backend (see frontend/vite.config.js),
# so this is a safety net for anyone hitting the API cross-origin directly
# rather than the primary access path.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local mock Sewadwaar/Jan Aadhaar endpoints (app/janaadhaar_client.py's
# MockJanAadhaarTransport calls these over real HTTP) - mounted on this same
# process purely for local-dev convenience, namespaced under
# /app/live/apiservice/... so there's no collision with /chat, /health, etc.
app.include_router(janaadhaar_mock_router)

# Built once at process startup, not per-request - the graph object itself
# is stateless aside from the MemorySaver checkpointer, which keys state by
# thread_id, so concurrent sessions don't collide.
_graph = build_graph()

# get_rag_service()/get_llm() are lazy singletons in app/chat.py (only
# constructed the first time a node actually calls them) - building the
# graph above does NOT trigger that, since it only wires up nodes/edges,
# never runs a node's body. Left lazy, the embedding model (a real download/
# load, several seconds) and Chroma collections would only initialize on
# the first /chat request, making that first citizen's request slow while
# every later one is fast. Forcing it here instead means the delay happens
# once, at startup, logged clearly, before the server accepts any traffic.
logger.info("Warming up embedding model and LLM client...")
get_rag_service()
get_llm()
get_job_service()
logger.info("Warm-up complete - ready to serve requests.")


class ChatRequest(BaseModel):
    session_id: str
    message: str
    # Set once the citizen picks a language in the frontend's language
    # picker (shown alongside the welcome message) - "hindi" | "hinglish" |
    # "english". Sent on every request once chosen, not just the first, so
    # it keeps getting merged into the session's persistent state even
    # though only the first turn is what actually sets it. None/omitted
    # before a pick is made, which leaves reply_lang on normal per-turn
    # detection, unchanged from before this existed.
    preferred_lang: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    # Mock-only (2026-08-17): set only on the turn the Jan Aadhaar mock
    # generates an OTP, so the frontend can show it as a simulated "SMS
    # arrived" popup instead of requiring a look at server logs. None on
    # every other turn, and naturally None forever once JANAADHAAR_MODE=real,
    # since the real transport's response never carries the OTP value.
    sms_preview: Optional[str] = None


class TranscribeResponse(BaseModel):
    text: str


def _log_turn(session_id, message, result, error=None):
    """The one place full traceability gets recorded - never returned to
    the client, only written to the server log.

    `message` is expected to already be redacted (see the /chat handler
    below) whenever this turn was a Jan Aadhaar ID/OTP attempt - a plaintext
    server log is a real, persistent exposure channel distinct from the LLM,
    so it gets the same substitution conversation_history does."""
    if error is not None:
        logger.error("session=%s message=%r FAILED: %s", session_id, message, error, exc_info=True)
        return
    shortlist_names = [s["scheme_name"] for s in (result.get("shortlist") or [])[:5]]
    job_titles = [j["title"] for j in (result.get("job_shortlist") or [])[:5]]
    logger.info(
        "session=%s message=%r intent=%s turn_type=%s reply_lang=%s "
        "need=%r category=%s profile=%s shown_scheme=%s shortlist=%s job_shortlist=%s",
        session_id,
        message,
        result.get("intent"),
        result.get("turn_type"),
        result.get("reply_lang"),
        result.get("need_text"),
        result.get("need_category"),
        result.get("profile"),
        result.get("shown_scheme_name"),
        shortlist_names,
        job_titles,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(audio: UploadFile = File(...), preferred_lang: Optional[str] = Form(None)):
    """Voice input - stateless, independent of the conversation graph.
    Whatever text comes back is just handed to POST /chat like any typed
    message, so no session_id is needed here at all.

    preferred_lang ("hindi"/"hinglish"/"english"), if the citizen already
    picked one via the language picker, is passed through to Whisper as a
    language hint - confirmed live (2026-08-12) that speaking Hindi with no
    hint got translated to English text instead of transcribed faithfully."""
    try:
        audio_bytes = await audio.read()
        text = transcribe(audio_bytes, filename=audio.filename or "audio.webm", reply_lang=preferred_lang)
    except TranscriptionError as e:
        logger.error("transcription failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))
    return TranscribeResponse(text=text)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.session_id}}
    graph_input = {"raw_message": req.message}
    if req.preferred_lang:
        # Only included when actually set, so a request from BEFORE the
        # citizen picks a language doesn't overwrite an already-locked
        # value with None on some later turn - the checkpointer only
        # merges keys that are actually present in this dict.
        graph_input["locked_reply_lang"] = req.preferred_lang
    try:
        result = _graph.invoke(graph_input, config=config)
    except Exception as e:
        # Full exception (with traceback) goes to the log only - the client
        # gets a plain, apologetic fallback reply, never a stack trace.
        _log_turn(req.session_id, req.message, None, error=e)
        return ChatResponse(
            session_id=req.session_id,
            reply="Maaf kijiye, kuch takniki dikkat aa gayi. Kripya thodi der baad phir try karein.",
        )

    logged_message = result.get("redacted_raw_message") or req.message
    _log_turn(req.session_id, logged_message, result)
    return ChatResponse(
        session_id=req.session_id,
        reply=result.get("last_reply", ""),
        sms_preview=result.get("mock_sms_preview"),
    )
