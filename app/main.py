"""FastAPI entry point - the HTTP layer over app/chat.py's conversation
graph. Deliberately thin: request/response plumbing plus the traceability
split this endpoint exists to enforce -

The API response NEVER includes profile/shortlist/intent/turn_type/etc -
only {session_id, reply, ...}. All of that is logged server-side instead
(see _log_turn below), with every message redacted of 6-10 digit groups
(OTPs, Jan Aadhaar IDs, phone numbers) before it reaches the log.

Heavy imports (app.chat -> langchain/chroma/sentence-transformers) are
deferred into get_graph()/the lifespan handler so this module imports
cleanly in a slim test environment and tests can stub the graph.

Run: uvicorn app.main:app --reload --port 8000
"""

import logging
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import pii_utils
from app.config import (
    ALLOWED_ORIGINS,
    JANAADHAAR_MODE,
    OPENROUTER_API_KEY,
    SARVAM_API_KEY,
    SESSION_TTL_MINUTES,
)
from app.stt_service import MAX_AUDIO_BYTES, TranscriptionError, transcribe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("jan_seva_chat")

SESSION_ID_PATTERN = r"^[A-Za-z0-9_\-]{1,64}$"
MAX_MESSAGE_CHARS = 2000
MAX_TRACKED_SESSIONS = 10_000
SWEEP_INTERVAL_S = 10 * 60
FRIENDLY_ERROR_TEXT = "Maaf kijiye, kuch takniki dikkat aa gayi. Kripya thodi der baad phir try karein."

# ---------------------------------------------------------------------------
# Graph (lazy) + per-session locks + idle-session eviction
# ---------------------------------------------------------------------------
_graph = None
_graph_lock = threading.Lock()

# session_id -> threading.Lock, bounded LRU so a flood of one-off sessions
# can't grow this forever. Serialises graph.invoke per session: two
# concurrent requests on one thread_id would otherwise race on the
# checkpointer's read-modify-write.
_session_locks: "OrderedDict[str, threading.Lock]" = OrderedDict()
_session_locks_guard = threading.Lock()

# session_id -> last-seen monotonic time, for evict_idle_sessions().
_session_last_seen: dict[str, float] = {}
_last_seen_guard = threading.Lock()


def get_graph():
    """Build app.chat's compiled graph once (heavy import deferred here)."""
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                from app.chat import build_graph
                _graph = build_graph()
    return _graph


def get_session_lock(session_id: str) -> threading.Lock:
    with _session_locks_guard:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _session_locks[session_id] = lock
            while len(_session_locks) > MAX_TRACKED_SESSIONS:
                _session_locks.popitem(last=False)
        else:
            _session_locks.move_to_end(session_id)
        return lock


def touch_session(session_id: str) -> None:
    with _last_seen_guard:
        _session_last_seen[session_id] = time.monotonic()


def _delete_thread(checkpointer, thread_id: str) -> None:
    if checkpointer is None:
        return
    if hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(thread_id)
        return
    # Older MemorySaver: storage[thread_id] / writes keyed by (thread_id, ns, checkpoint_id)
    storage = getattr(checkpointer, "storage", None)
    if isinstance(storage, dict):
        storage.pop(thread_id, None)
    writes = getattr(checkpointer, "writes", None)
    if isinstance(writes, dict):
        for key in [k for k in writes if isinstance(k, tuple) and k and k[0] == thread_id]:
            writes.pop(key, None)


def evict_idle_sessions(ttl_seconds: Optional[float] = None, now: Optional[float] = None) -> int:
    """Drop checkpointer state + lock for sessions idle longer than the TTL.
    Returns the number evicted. Safe to call when the graph was never built."""
    ttl = SESSION_TTL_MINUTES * 60 if ttl_seconds is None else ttl_seconds
    now = time.monotonic() if now is None else now
    with _last_seen_guard:
        stale = [sid for sid, seen in _session_last_seen.items() if now - seen > ttl]
        for sid in stale:
            _session_last_seen.pop(sid, None)
    if not stale:
        return 0
    checkpointer = getattr(_graph, "checkpointer", None) if _graph is not None else None
    for sid in stale:
        try:
            _delete_thread(checkpointer, sid)
        except Exception as e:  # never let a sweep crash the server
            logger.warning("failed to evict session=%s: %s", sid, e.__class__.__name__)
        with _session_locks_guard:
            _session_locks.pop(sid, None)
    logger.info("evicted %d idle session(s)", len(stale))
    return len(stale)


def _sweeper_loop(stop_event: threading.Event) -> None:
    while not stop_event.wait(SWEEP_INTERVAL_S):
        try:
            evict_idle_sessions()
        except Exception as e:
            logger.warning("session sweeper error: %s", e.__class__.__name__)


# ---------------------------------------------------------------------------
# App + lifespan (warm-up moved here from import time)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    stop_event = threading.Event()
    sweeper = threading.Thread(target=_sweeper_loop, args=(stop_event,), daemon=True, name="session-sweeper")
    sweeper.start()
    try:
        # get_rag_service()/get_llm()/get_job_service() are lazy singletons in
        # app/chat.py; forcing them here means the embedding model load
        # happens once at startup, not on the first citizen's request.
        logger.info("Warming up graph, embedding model and LLM client...")
        from app.chat import get_job_service, get_llm, get_rag_service
        get_graph()
        get_rag_service()
        if OPENROUTER_API_KEY:
            get_llm()
        else:
            logger.warning("OPENROUTER_API_KEY not set - /chat will fail until it is; /health reports 503.")
        get_job_service()
        logger.info("Warm-up complete - ready to serve requests.")
    except Exception as e:
        # Don't take the process down: /health will report the problem.
        logger.error("Warm-up failed (%s): %s", e.__class__.__name__, e)
    yield
    stop_event.set()


app = FastAPI(title="Jan Seva AI Chatbot API", lifespan=lifespan)

if "*" in ALLOWED_ORIGINS:
    logger.warning("CORS allow_origins='*' - set ALLOWED_ORIGINS (comma-separated) for anything beyond local dev.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local mock Sewadwaar/Jan Aadhaar endpoints - only mounted in mock mode.
if JANAADHAAR_MODE == "mock":
    from app.janaadhaar_mock_routes import router as janaadhaar_mock_router
    app.include_router(janaadhaar_mock_router)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64, pattern=SESSION_ID_PATTERN)
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)
    # "hindi" | "hinglish" | "english" once the citizen picks one; None
    # before that, leaving reply_lang on per-turn detection.
    preferred_lang: Optional[str] = Field(None, max_length=16)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    # Mock-only: the simulated "SMS arrived" text on the turn the Jan
    # Aadhaar mock generates an OTP and JANAADHAAR_DEMO_OTP_PREVIEW=1.
    sms_preview: Optional[str] = None
    # Set on the friendly-error path (HTTP stays 200 for the frontend):
    # retryable=True tells the UI to offer a retry; error_code is a short
    # machine-readable tag (currently "graph_error").
    retryable: Optional[bool] = None
    error_code: Optional[str] = None


class TranscribeResponse(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Logging (everything redacted)
# ---------------------------------------------------------------------------
def _log_turn(session_id, message, result, error=None):
    """The one place full traceability gets recorded - never returned to
    the client. `message` is redacted of any 6-10 digit group before it
    gets here; the profile dict is logged as its KEYS only."""
    if error is not None:
        logger.error(
            "session=%s message=%r FAILED error=%s", session_id, message, error.__class__.__name__, exc_info=True,
        )
        return
    shortlist_names = [s.get("scheme_name") for s in (result.get("shortlist") or [])[:5]]
    job_titles = [j.get("title") for j in (result.get("job_shortlist") or [])[:5]]
    profile = result.get("profile") or {}
    profile_keys = sorted(k for k, v in profile.items() if v is not None) if isinstance(profile, dict) else []
    logger.info(
        "session=%s message=%r intent=%s turn_type=%s reply_lang=%s "
        "need=%r category=%s profile_keys=%s shown_scheme=%s shortlist=%s job_shortlist=%s",
        session_id,
        message,
        result.get("intent"),
        result.get("turn_type"),
        result.get("reply_lang"),
        pii_utils.redact_digits(result.get("need_text") or "") or None,
        result.get("need_category"),
        profile_keys,
        result.get("shown_scheme_name"),
        shortlist_names,
        job_titles,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    rag_ready = True
    try:
        from app.chat import get_rag_service
        svc = get_rag_service()
        if hasattr(svc, "is_ready"):
            rag_ready = bool(svc.is_ready())
    except Exception as e:
        logger.warning("health: rag check failed: %s", e.__class__.__name__)
        rag_ready = False
    with _last_seen_guard:
        jobs = len(_session_last_seen)
    body = {
        "status": "ok",
        "llm": bool(OPENROUTER_API_KEY),
        "stt": bool(SARVAM_API_KEY),
        "rag_ready": rag_ready,
        "jobs": jobs,
    }
    if not body["llm"] or not rag_ready:
        body["status"] = "degraded"
        return JSONResponse(status_code=503, content=body)
    return body


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe_audio(audio: UploadFile = File(...)):
    """Voice input - stateless, independent of the conversation graph.
    Plain `def` (not async) because transcribe() makes a blocking HTTP
    call; FastAPI runs it in the threadpool."""
    audio_bytes = audio.file.read(MAX_AUDIO_BYTES + 1)
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio is too large (max 8 MB) - please record a shorter message.")
    try:
        text = transcribe(
            audio_bytes,
            filename=audio.filename or "audio.webm",
            content_type=audio.content_type,
        )
    except TranscriptionError as e:
        logger.error("transcription failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))
    return TranscribeResponse(text=text)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.session_id}}
    graph_input = {"raw_message": req.message}
    if req.preferred_lang:
        graph_input["locked_reply_lang"] = req.preferred_lang
    redacted_request = pii_utils.redact_digits(req.message)
    touch_session(req.session_id)
    try:
        with get_session_lock(req.session_id):
            result = get_graph().invoke(graph_input, config=config)
    except Exception as e:
        _log_turn(req.session_id, redacted_request, None, error=e)
        return ChatResponse(
            session_id=req.session_id,
            reply=FRIENDLY_ERROR_TEXT,
            retryable=True,
            error_code="graph_error",
        )

    logged_message = pii_utils.redact_digits(result.get("redacted_raw_message") or req.message)
    _log_turn(req.session_id, logged_message, result)
    return ChatResponse(
        session_id=req.session_id,
        reply=result.get("last_reply", ""),
        sms_preview=result.get("mock_sms_preview"),
    )
