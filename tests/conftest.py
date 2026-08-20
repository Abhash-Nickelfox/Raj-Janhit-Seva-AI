"""Shared fixtures: make `app.chat` importable in a slim environment (no
chromadb / sentence-transformers / OpenRouter key) by stubbing the heavy
vector-store modules, and provide scripted fakes for the LLM, RagService
and JanAadhaarService so whole-graph scenarios run deterministically."""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _Any:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return _Any()

    def __call__(self, *a, **k):
        return _Any()


def _stub(name, attrs=()):
    if name in sys.modules:
        return
    m = types.ModuleType(name)
    for a in attrs:
        setattr(m, a, _Any)
    sys.modules[name] = m


for _name, _attrs in (
    ("chromadb", ()),
    ("sentence_transformers", ("SentenceTransformer",)),
    ("langchain_chroma", ("Chroma",)),
    ("langchain_huggingface", ("HuggingFaceEmbeddings",)),
    ("langchain_community", ()),
    ("langchain_community.vectorstores", ("Chroma",)),
    ("langchain_community.embeddings", ("HuggingFaceEmbeddings",)),
):
    _stub(_name, _attrs)


# ---------------------------------------------------------------------------
# Scripted LLM
# ---------------------------------------------------------------------------
class _Msg:
    def __init__(self, content):
        self.content = content


class ScriptedStructured:
    """Returned by FakeLLM.with_structured_output: pops the next scripted
    result for that model class, or raises if the script says so."""

    def __init__(self, parent, model_cls):
        self.parent = parent
        self.model_cls = model_cls

    def invoke(self, messages):
        self.parent.calls.append((self.model_cls.__name__, messages))
        queue = self.parent.structured.get(self.model_cls.__name__) or []
        if not queue:
            raise AssertionError(f"no scripted result for {self.model_cls.__name__}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item) and not hasattr(item, "model_dump"):
            item = item(messages)
        return item


class FakeLLM:
    def __init__(self):
        self.structured = {}  # model name -> list of results / exceptions
        self.text = []  # plain invoke replies, in order
        self.calls = []
        self.text_calls = []

    def with_structured_output(self, model_cls, method=None):
        return ScriptedStructured(self, model_cls)

    def invoke(self, messages):
        self.text_calls.append(messages)
        if self.text:
            item = self.text.pop(0)
            if callable(item):
                item = item(messages)
            return _Msg(item)
        return _Msg("Theek hai.")


# ---------------------------------------------------------------------------
# Fake RagService
# ---------------------------------------------------------------------------
class FakeScheme:
    def __init__(self, scheme_id, scheme_name, metadata=None, lang="en", score=1.0):
        self.scheme_id = scheme_id
        self.scheme_name = scheme_name
        self.lang = lang
        self.score = score
        base = {"scheme_id": scheme_id, "scheme_name": scheme_name, "lang": lang, "caste": "all", "gender": "all",
                "marital_status": "all", "residence": "all", "education_level": "", "occupation_target_group": "",
                "field_type": "eligibility"}
        base.update(metadata or {})
        self.metadata = base


class FakeRag:
    def __init__(self, schemes):
        self.schemes = schemes
        self._meta_by_scheme_id = {s.scheme_id: s.metadata for s in schemes}
        self.shortlist_calls = []

    def shortlist(self, query, citizen, alt_citizen=None, top_n=10, trace=None):
        self.shortlist_calls.append(query)
        if trace is not None:
            trace["survivors"] = len(self.schemes)
        return list(self.schemes)[:top_n]

    def scheme_detail(self, scheme_id, lang, field_types=None):
        meta = dict(self._meta_by_scheme_id[scheme_id])
        meta["field_type"] = (field_types or ["eligibility"])[0]
        meta["lang"] = lang
        return [(f"{meta['scheme_name']} eligibility text", meta)]

    def get_discovery_card(self, scheme_id, lang):
        return f"card for {scheme_id}"

    def names_for(self, scheme_id):
        m = self._meta_by_scheme_id.get(scheme_id, {})
        return {m.get("scheme_name", ""), m.get("scheme_name_hi", "")} - {""}

    def is_ready(self):
        return True


@pytest.fixture
def fake_llm(monkeypatch):
    import app.chat as chat
    llm = FakeLLM()
    monkeypatch.setattr(chat, "get_llm", lambda: llm)
    monkeypatch.setattr(chat, "get_reply_llm", lambda: llm)
    return llm


@pytest.fixture
def fake_rag(monkeypatch):
    import app.chat as chat
    rag = FakeRag([
        FakeScheme("s1", "Mukhyamantri Vriddhjan Samman Pension Yojana", {"scheme_name_hi": "मुख्यमंत्री वृद्धजन सम्मान पेंशन योजना"}),
        FakeScheme("s2", "Indira Gandhi National Old Age Pension Scheme"),
        FakeScheme("s3", "Chief Minister Small and Marginal Farmers Older Pension Scheme", {"occupation_target_group": "farmer"}),
        FakeScheme("s4", "Palanhar Yojana"),
        FakeScheme("s5", "Kantedar Tarbandi Yojana"),
    ])
    monkeypatch.setattr(chat, "get_rag_service", lambda: rag)
    return rag


@pytest.fixture
def real_jobs(monkeypatch):
    import app.chat as chat
    from app.job_service import JobService
    svc = JobService()
    monkeypatch.setattr(chat, "get_job_service", lambda: svc)
    return svc


@pytest.fixture
def graph(fake_llm, fake_rag, real_jobs):
    import app.chat as chat
    return chat.build_graph()


def run_turn(graph, session, message, **extra):
    config = {"configurable": {"thread_id": session}}
    payload = {"raw_message": message}
    payload.update(extra)
    return graph.invoke(payload, config=config)
