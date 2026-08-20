"""app.main with the conversation graph stubbed - no LLM/Chroma needed."""
import logging
import sys
import threading
import types

import pytest
from fastapi.testclient import TestClient


class _StubCheckpointer:
    def __init__(self):
        self.storage = {}
        self.deleted = []

    def delete_thread(self, thread_id):
        self.deleted.append(thread_id)
        self.storage.pop(thread_id, None)


class _StubGraph:
    def __init__(self, fail=False):
        self.fail = fail
        self.checkpointer = _StubCheckpointer()
        self.calls = []

    def invoke(self, graph_input, config=None):
        self.calls.append((graph_input, config))
        if self.fail:
            raise RuntimeError("boom")
        self.checkpointer.storage[config["configurable"]["thread_id"]] = True
        return {
            "last_reply": "namaste",
            "intent": "scheme",
            "profile": {"age": 65, "gender": None},
            "need_text": "pension 9876543210",
        }


class _StubRag:
    ready = True

    def is_ready(self):
        return self.ready


@pytest.fixture
def stub_chat(monkeypatch):
    """Fake app.chat so app.main never imports langchain/chroma."""
    mod = types.ModuleType("app.chat")
    rag = _StubRag()
    mod.build_graph = lambda checkpointer=None: _StubGraph()
    mod.get_rag_service = lambda: rag
    mod.get_llm = lambda: object()
    mod.get_job_service = lambda: object()
    monkeypatch.setitem(sys.modules, "app.chat", mod)
    return mod


@pytest.fixture
def main(stub_chat, monkeypatch):
    import app.main as m
    monkeypatch.setattr(m, "_graph", _StubGraph())
    monkeypatch.setattr(m, "OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(m, "SARVAM_API_KEY", "")
    m._session_last_seen.clear()
    m._session_locks.clear()
    return m


@pytest.fixture
def client(main):
    return TestClient(main.app)


def test_chat_happy_path(client):
    r = client.post("/chat", json={"session_id": "abc-123", "message": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "namaste"
    assert body["retryable"] is None
    assert body["error_code"] is None


def test_message_max_length(client):
    r = client.post("/chat", json={"session_id": "s1", "message": "x" * 2001})
    assert r.status_code == 422
    r = client.post("/chat", json={"session_id": "s1", "message": "x" * 2000})
    assert r.status_code == 200


def test_session_id_validation(client):
    assert client.post("/chat", json={"session_id": "a" * 65, "message": "hi"}).status_code == 422
    assert client.post("/chat", json={"session_id": "bad id!", "message": "hi"}).status_code == 422
    assert client.post("/chat", json={"session_id": "", "message": "hi"}).status_code == 422


def test_per_session_lock(main, client):
    client.post("/chat", json={"session_id": "lock-me", "message": "hi"})
    lock = main.get_session_lock("lock-me")
    assert isinstance(lock, type(threading.Lock()))
    assert main.get_session_lock("lock-me") is lock
    assert main.get_session_lock("other") is not lock


def test_redaction_in_logs(client, caplog):
    caplog.set_level(logging.INFO, logger="jan_seva_chat")
    client.post("/chat", json={"session_id": "s2", "message": "mera otp 123456 aur id 1234 5678 90 hai"})
    text = caplog.text
    assert "123456" not in text
    assert "1234 5678 90" not in text and "1234567890" not in text
    assert "XXXXXX" in text
    # need_text carried a phone number - redacted too; profile logged as keys only
    assert "9876543210" not in text
    assert "profile_keys=['age']" in text


def test_error_path_returns_retryable_and_redacts(main, client, caplog, monkeypatch):
    monkeypatch.setattr(main, "_graph", _StubGraph(fail=True))
    caplog.set_level(logging.ERROR, logger="jan_seva_chat")
    r = client.post("/chat", json={"session_id": "s3", "message": "otp 654321"})
    assert r.status_code == 200
    body = r.json()
    assert body["retryable"] is True
    assert body["error_code"] == "graph_error"
    assert "takniki dikkat" in body["reply"]
    assert "654321" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_health_shape_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "llm": True, "stt": False, "rag_ready": True, "jobs": 0}


def test_health_503_without_llm(main, client, monkeypatch):
    monkeypatch.setattr(main, "OPENROUTER_API_KEY", None)
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["llm"] is False
    assert r.json()["status"] == "degraded"


def test_health_503_when_rag_not_ready(main, client, stub_chat):
    stub_chat.get_rag_service().ready = False
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["rag_ready"] is False


def test_evict_idle_sessions(main, client):
    client.post("/chat", json={"session_id": "old", "message": "hi"})
    client.post("/chat", json={"session_id": "fresh", "message": "hi"})
    main._session_last_seen["old"] -= 10_000
    evicted = main.evict_idle_sessions(ttl_seconds=3600)
    assert evicted == 1
    assert main._graph.checkpointer.deleted == ["old"]
    assert "old" not in main._session_last_seen and "fresh" in main._session_last_seen
    assert "old" not in main._session_locks


def test_transcribe_rejects_oversized(client):
    big = b"0" * (8 * 1024 * 1024 + 1)
    r = client.post("/transcribe", files={"audio": ("a.webm", big, "audio/webm")})
    assert r.status_code == 413


def test_mock_route_bad_envelope_returns_400(client):
    r = client.post("/app/live/apiservice/janAadhaar/v1/member-list", json={"data": "not-an-envelope"})
    assert r.status_code == 400
    r = client.post("/app/live/apiservice/janAadhaar/v1/member-list", content=b"garbage")
    assert r.status_code == 400


def test_stt_retries_on_foreign_script(monkeypatch):
    from app import stt_service as stt

    monkeypatch.setattr(stt, "SARVAM_API_KEY", "k")
    calls = []

    def fake_call(audio, filename, mime, lang):
        calls.append(lang)
        if lang == "hi-IN":
            return "ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "pa-IN"   # Gurmukhi for Hindi speech
        return "widow pension chahiye", "en-IN"

    monkeypatch.setattr(stt, "_call_sarvam", fake_call)
    assert stt.transcribe(b"abc") == "widow pension chahiye"
    assert calls == ["hi-IN", "en-IN"]


def test_stt_gives_up_after_second_foreign_script(monkeypatch):
    from app import stt_service as stt

    monkeypatch.setattr(stt, "SARVAM_API_KEY", "k")
    monkeypatch.setattr(stt, "_call_sarvam", lambda *a: ("ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "pa-IN"))
    with pytest.raises(stt.TranscriptionError, match="please repeat"):
        stt.transcribe(b"abc")


def test_stt_size_guard(monkeypatch):
    from app import stt_service as stt

    monkeypatch.setattr(stt, "SARVAM_API_KEY", "k")
    with pytest.raises(stt.TranscriptionError, match="too large"):
        stt.transcribe(b"0" * (stt.MAX_AUDIO_BYTES + 1))
