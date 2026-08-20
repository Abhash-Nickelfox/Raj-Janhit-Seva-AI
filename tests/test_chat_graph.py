"""Whole-graph scenario tests replaying the failures from the 2026-08-20
tester reports, with the LLM scripted and RagService faked. These are the
regressions that kept coming back: each one is a conversation the bot must
now survive without the canned 'takniki dikkat' error."""

import pytest

import app.chat as chat
from app.chat import NeedExtraction, NeedItem, ProfileExtraction, SchemeAnswer
from tests.conftest import run_turn


def need(text, domain="scheme", intent="find_scheme", turn_type="new_need", lang="hinglish", **kw):
    item_kw = {k: v for k, v in kw.items() if k in NeedItem.model_fields}
    extra = {k: v for k, v in kw.items() if k not in NeedItem.model_fields}
    return NeedExtraction(
        needs=[NeedItem(need_text=text, domain=domain, about="self", **item_kw)],
        need_is_specific="yes", intent=intent, turn_type=turn_type, detected_language=lang, **extra,
    )


def script(llm, *, profile=None, needs=(), answers=(), texts=()):
    llm.structured.setdefault("ProfileExtraction", []).extend(profile or [ProfileExtraction()] * max(len(needs), 1))
    llm.structured.setdefault("NeedExtraction", []).extend(needs)
    llm.structured.setdefault("SchemeAnswer", []).extend(answers)
    llm.text.extend(texts)


# ---------------------------------------------------------------- 1. the crash loop
def test_off_enum_job_qualification_does_not_kill_the_turn(graph, fake_llm):
    """Tester: 'I am BCA related holder' -> every following turn errored."""
    script(fake_llm, needs=[need("BCA jobs", domain="job", job_qualification="BCA", job_min_salary="30,000")],
           texts=["LDC (Jaipur) - BCA ke liye fit hai."])
    out = run_turn(graph, "s1", "I am BCA related holder")
    assert out["last_reply"]
    assert out["needs"][0]["job_qualification"] in ("BCA", "Graduate")
    assert out["needs"][0]["job_min_salary"] == 30000
    assert out["job_title_named"] is False  # now actually stored in state


def test_validation_error_from_model_degrades_gracefully(graph, fake_llm):
    """Both NeedExtraction attempts blow up -> fallback need, still a reply."""
    from pydantic import ValidationError
    err = ValueError("ValidationError: 1 validation error for NeedExtraction")
    fake_llm.structured["ProfileExtraction"] = [ProfileExtraction()]
    fake_llm.structured["NeedExtraction"] = [err, err]
    fake_llm.structured["SchemeAnswer"] = [SchemeAnswer(reply="Aap kis cheez ke liye madad chahte hain?", asked_clarifying_question=True)]
    out = run_turn(graph, "s2", "In rajasthan")
    assert out["last_reply"]
    assert out["needs"][0]["need_text"] == "In rajasthan"


# ---------------------------------------------------------------- 2. stale-reply replay
def test_chitchat_without_reply_text_does_not_replay_previous_turn(graph, fake_llm):
    script(fake_llm,
           needs=[need("pension", lang="english"), need("thanks", intent="chitchat", lang="english", chitchat_reply=None)],
           answers=[SchemeAnswer(reply="Mukhyamantri Vriddhjan Samman Pension Yojana gives Rs 1000/month.", asked_clarifying_question=False)],
           texts=["You're welcome! Ask me anything about schemes or jobs."])
    first = run_turn(graph, "s3", "pension for me")
    second = run_turn(graph, "s3", "ok thanks")
    assert second["last_reply"] != first["last_reply"]
    assert "welcome" in second["last_reply"]


# ---------------------------------------------------------------- 3. repeated schemes
def test_show_more_never_repeats_already_explained_schemes(graph, fake_llm, fake_rag):
    script(fake_llm,
           needs=[need("pension", lang="english"), need("more", turn_type="wants_alternatives", lang="english")],
           answers=[SchemeAnswer(reply="Here are three pensions.", asked_clarifying_question=False),
                    SchemeAnswer(reply="Two more options.", asked_clarifying_question=False)])
    first = run_turn(graph, "s4", "pension schemes")
    shown_first = {s["scheme_id"] for s in first["shown_schemes"]}
    assert len(shown_first) == 3
    assert set(first["already_shown_scheme_ids"]) == shown_first  # all three recorded, not just #1
    second = run_turn(graph, "s4", "any other schemes?")
    shown_second = {s["scheme_id"] for s in second["shown_schemes"]}
    assert shown_second and not (shown_second & shown_first)


# ---------------------------------------------------------------- 4. domain switch hygiene
def test_switching_scheme_to_job_clears_scheme_focus(graph, fake_llm):
    script(fake_llm,
           needs=[need("pension"), need("sarkari naukri", domain="job")],
           answers=[SchemeAnswer(reply="Pension details.", asked_clarifying_question=False)],
           texts=["Kaunsi post mein interest hai?", "Yeh postings hain."])
    first = run_turn(graph, "s5", "pension chahiye")
    assert first["shown_scheme_id"]
    second = run_turn(graph, "s5", "koi sarkari naukri batao")
    assert second["last_domain"] == "job"
    assert second.get("shown_scheme_id") is None
    assert second.get("shortlist") == []


def test_followup_after_job_stays_in_job_domain(graph, fake_llm):
    """'iske documents?' after a job answer must not go to a scheme shown earlier."""
    script(fake_llm,
           needs=[need("pension"), need("patwari", domain="job", job_location="Jaipur"),
                  need("documents", domain=None, intent="documents_question", turn_type="followup_same_scheme")],  # model left domain unset
           answers=[SchemeAnswer(reply="Pension details.", asked_clarifying_question=False)],
           texts=["Patwari Jaipur details.", "Patwari ke liye documents: Aadhaar..."])
    run_turn(graph, "s6", "pension chahiye")
    second = run_turn(graph, "s6", "patwari job jaipur")
    assert second["shown_job_id"]
    third = run_turn(graph, "s6", "iske documents kya chahiye?")
    assert third["last_domain"] == "job"
    assert "Patwari" in third["last_reply"]


# ---------------------------------------------------------------- 5. language
def test_english_message_with_wrong_llm_guess_gets_english_reply(graph, fake_llm):
    script(fake_llm,
           needs=[need("schemes for graduates", lang="hinglish")],  # model guessed wrong
           answers=[SchemeAnswer(reply="Dekho, graduation ke baad jo schemes hote hain wo students ke liye hote hain, tum fit nahi hoge.", asked_clarifying_question=False)],
           texts=["After graduation, most schemes target current students, so you would not fit those."])
    out = run_turn(graph, "s7", "in Graudation Btech give me schemes related that")
    assert out["reply_lang"] == "english"
    assert not chat.DEVANAGARI_RE.search(out["last_reply"])
    assert "Dekho" not in out["last_reply"]  # rewritten by the language guard


def test_short_message_keeps_previous_language(graph, fake_llm):
    script(fake_llm,
           needs=[need("irrigation", lang="english"), need("irrigation", lang="hinglish", turn_type="followup_same_scheme")],
           answers=[SchemeAnswer(reply="Do you have 5 acres in Jaipur or Jodhpur?", asked_clarifying_question=True),
                    SchemeAnswer(reply="For Jodhpur the same state-level schemes apply.", asked_clarifying_question=False)])
    run_turn(graph, "s8", "I want irrigation schemes for my land")
    out = run_turn(graph, "s8", "In rajasthan")
    assert out["reply_lang"] == "english"


# ---------------------------------------------------------------- 6. jobs
def test_patwari_all_closed_is_reported_not_replaced(graph, fake_llm, real_jobs, monkeypatch):
    # Force every Patwari row closed via a service whose 'today' is far in the future.
    from app.job_service import JobService
    import datetime
    svc = JobService(today=datetime.date(2027, 6, 1))
    monkeypatch.setattr(chat, "get_job_service", lambda: svc)
    script(fake_llm, needs=[need("patwari", domain="job")], texts=["Patwari ki bharti abhi band hai."])
    out = run_turn(graph, "s9", "patwari bharti")
    assert out["job_shortlist"] == []
    assert out["job_title_named"] is True
    assert "Patwari" in out["job_search_info"]["closed_titles"]
    assert out["last_reply"]


def test_bare_job_query_returns_distinct_titles(graph, fake_llm):
    script(fake_llm, needs=[need("job", domain="job")], texts=["Kaunsi post?"])
    out = run_turn(graph, "s10", "mujhe job chahiye")
    titles = [j["title"] for j in out["job_shortlist"]]
    assert len(titles) == len(set(titles)) >= 3


# ---------------------------------------------------------------- 7. jan aadhaar
class FakeJan:
    def __init__(self):
        self.members = [{"member_id": "m1", "masked_name": "R**** S****"}]

    def fetch_member_list(self, jan_id):
        if jan_id == "1234567890":
            return {"status": "JAN_200", "members": self.members}
        return {"status": "JAN_404", "members": []}

    def generate_otp(self, member_id):
        return {"status": "JAN_200", "tid": "t1", "mock_otp": "111111"}

    def validate_otp_and_fetch(self, member_id, tid, otp):
        return {"status": "JAN_200" if otp == "111111" else "JAN_401", "data": {}}

    def get_entitlement_details(self, member_id):
        return {"entitlements": [{"scheme": "Old Age Pension", "status": "active"}]}


@pytest.fixture
def jan(monkeypatch):
    j = FakeJan()
    monkeypatch.setattr(chat, "get_janaadhaar_service", lambda: j)
    return j


def test_janaadhaar_consent_then_id_then_otp(graph, fake_llm, jan):
    script(fake_llm, needs=[need("kya meri pension aayi", domain="citizen_status")], texts=["Aapki Old Age Pension active hai."])
    t1 = run_turn(graph, "j1", "kya meri pension aayi?")
    assert t1["pending_janaadhaar_step"] == "awaiting_consent" and "sahmat" in t1["last_reply"]
    t2 = run_turn(graph, "j1", "haan, koi problem nahi")  # used to parse as NO
    assert t2["pending_janaadhaar_step"] == "awaiting_id"
    t3 = run_turn(graph, "j1", "mera id 9999999999 hai")  # unknown -> retry, not abort
    assert t3["pending_janaadhaar_step"] == "awaiting_id" and "record" in t3["last_reply"].lower() or "रिकॉर्ड" in t3["last_reply"]
    t4 = run_turn(graph, "j1", "1234567890")
    assert t4["pending_janaadhaar_step"] == "awaiting_otp" and t4["mock_sms_preview"]
    t5 = run_turn(graph, "j1", "OTP 111111 aaya, 2 minute pehle")  # used to concat all digits
    assert t5["pending_janaadhaar_step"] is None
    assert "Pension" in t5["last_reply"]
    # credentials never reach history
    assert all("1234567890" not in h["user"] and "111111" not in h["user"] for h in t5["conversation_history"])


def test_janaadhaar_cancel_mid_flow(graph, fake_llm, jan):
    script(fake_llm, needs=[need("pension status", domain="citizen_status")])
    run_turn(graph, "j2", "meri pension kab aayegi")
    run_turn(graph, "j2", "yes")
    out = run_turn(graph, "j2", "rehne do")
    assert out["pending_janaadhaar_step"] is None
    assert out["last_reply"]


def test_janaadhaar_unrelated_question_exits_and_is_answered(graph, fake_llm, jan):
    script(fake_llm,
           needs=[need("pension status", domain="citizen_status"), need("scholarship for daughter", lang="hinglish")],
           answers=[SchemeAnswer(reply="Beti ke liye yeh scholarship hai.", asked_clarifying_question=False)])
    run_turn(graph, "j3", "meri pension kab aayegi")
    run_turn(graph, "j3", "yes")
    out = run_turn(graph, "j3", "chhodo, beti ke liye scholarship batao pehle")
    assert out["pending_janaadhaar_step"] is None
    assert "scholarship" in out["last_reply"].lower()


def test_explicit_scheme_request_after_job_is_not_hijacked(graph, fake_llm):
    """Review 2026-08-20: 'koi aur yojana batao' after a job answer must search schemes."""
    script(fake_llm,
           needs=[need("patwari", domain="job"), need("yojana", domain="scheme", turn_type="wants_alternatives", need_benefit_type="subsidy")],
           answers=[SchemeAnswer(reply="Yeh yojanayein hain.", asked_clarifying_question=False)],
           texts=["Patwari details."])
    run_turn(graph, "r1", "patwari job")
    out = run_turn(graph, "r1", "koi aur yojana batao")
    assert out["last_domain"] == "scheme" and out["shortlist"]


def test_question_budget_resets_for_a_new_need(graph, fake_llm):
    script(fake_llm,
           needs=[need("job", domain="job"), need("pension", lang="english")],
           answers=[SchemeAnswer(reply="Which district are you in?", asked_clarifying_question=True)],
           texts=["Kaunsi post?"])
    first = run_turn(graph, "r2", "mujhe job chahiye")
    assert first["question_rounds"] == 1
    second = run_turn(graph, "r2", "pension for me")
    assert "HARD STOP" not in str(fake_llm.calls[-1])
    assert second["question_rounds"] == 1  # fresh need started from 0 and asked once


def test_otp_not_received_stays_in_flow(graph, fake_llm, jan):
    script(fake_llm, needs=[need("pension status", domain="citizen_status")])
    run_turn(graph, "r3", "meri pension kab aayegi")
    run_turn(graph, "r3", "yes")
    run_turn(graph, "r3", "1234567890")
    out = run_turn(graph, "r3", "abhi tak otp nahi aaya hai")
    assert out["pending_janaadhaar_step"] == "awaiting_otp"
    assert out["janaadhaar_tid"] == "t1"


def test_scheduled_tribe_is_st(graph, fake_llm):
    from app.chat import _normalize_caste
    assert _normalize_caste("Scheduled Tribe") == "st"
    assert _normalize_caste("SC") == "sc"
    assert _normalize_caste("OBC category") == "obc"
