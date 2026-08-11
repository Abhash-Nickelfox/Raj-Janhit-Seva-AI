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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.chat import build_graph, get_llm, get_rag_service, get_job_service

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


class ChatResponse(BaseModel):
    session_id: str
    reply: str


def _log_turn(session_id, message, result, error=None):
    """The one place full traceability gets recorded - never returned to
    the client, only written to the server log."""
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


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.session_id}}
    try:
        result = _graph.invoke({"raw_message": req.message}, config=config)
    except Exception as e:
        # Full exception (with traceback) goes to the log only - the client
        # gets a plain, apologetic fallback reply, never a stack trace.
        _log_turn(req.session_id, req.message, None, error=e)
        return ChatResponse(
            session_id=req.session_id,
            reply="Maaf kijiye, kuch takniki dikkat aa gayi. Kripya thodi der baad phir try karein.",
        )

    _log_turn(req.session_id, req.message, result)
    return ChatResponse(session_id=req.session_id, reply=result.get("last_reply", ""))
