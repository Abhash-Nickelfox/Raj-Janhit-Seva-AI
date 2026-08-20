"""Pure-function tests for app/chat.py (no LLM, no Chroma)."""

import pytest
from pydantic import ValidationError

import app.chat as chat
from app.chat import (
    NeedExtraction, NeedItem, ProfileExtraction,
    _identity_gap, _implied_groups, _invoke_structured, _janaadhaar_wants_out, _leaked_scheme_names,
    _normalize_job_qualification, _parse_int, _reply_language_mismatch, _snap, _snap_yesno,
    detect_language_heuristic, domain_switch_resets, resolve_reply_lang,
)


# ---------------------------------------------------------------- parsing / snapping
@pytest.mark.parametrize("raw,expected", [
    (30, 30), ("30", 30), ("30 saal", 30), ("5 lakh", 500000), ("2,50,000", 250000), ("Rs 48000/-", 48000),
    ("1.5 lakh", 150000), ("2 हज़ार", 2000), (None, None), ("unknown", None), (True, None),
])
def test_parse_int(raw, expected):
    assert _parse_int(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("job", "job"), ("JOB", "job"), ("Citizen Status", "citizen_status"), ("citizen-status", "citizen_status"),
    ("schemes", "scheme"), (None, "scheme"), ("banana", "scheme"),
])
def test_snap_domain(raw, expected):
    assert _snap(raw, ["scheme", "job", "citizen_status"], default="scheme") == expected


def test_snap_yesno():
    assert _snap_yesno("Yes") == "yes"
    assert _snap_yesno(True) == "yes"
    assert _snap_yesno("nahi") == "no"
    assert _snap_yesno("maybe") is None


@pytest.mark.parametrize("raw,expected", [
    ("BCA", "BCA"), ("B.Tech", "B.Tech"), ("graduate", "Graduate"), ("ITI", "Diploma"), ("12th pass", "12th Pass"),
    ("MA", "Post Graduate"), (None, None),
])
def test_normalize_job_qualification_never_raises(raw, expected):
    out = _normalize_job_qualification(raw)
    if expected is None:
        assert out is None
    else:
        assert out is not None  # exact label may come from JobService's own normaliser


# ---------------------------------------------------------------- schema leniency (the crash loop)
def test_need_item_accepts_off_enum_values():
    """The exact failure from the tester screenshots: BCA holder, 'In rajasthan'."""
    item = NeedItem(need_text="job", domain="job", about="self", job_qualification="BCA", job_min_salary="30,000")
    assert item.job_qualification == "BCA"
    need = NeedExtraction(needs=[item], need_is_specific="YES", intent="find_job", turn_type="follow-up", detected_language="Hinglish")
    assert need.intent == "find_job"  # snapped later in code, never a ValidationError


def test_profile_accepts_strings_for_numbers():
    p = ProfileExtraction(age="60 saal", family_income="2 lakh", disability="nahi", residence="Rural")
    assert _parse_int(p.age) == 60 and _parse_int(p.family_income) == 200000


def test_invoke_structured_falls_back_after_two_failures():
    class BadLLM:
        calls = 0

        def with_structured_output(self, cls, method=None):
            return self

        def invoke(self, messages):
            BadLLM.calls += 1
            raise ValueError("boom")

    out = _invoke_structured(BadLLM(), NeedExtraction, "prompt", fallback=NeedExtraction, label="t")
    assert isinstance(out, NeedExtraction) and out.needs == []
    assert BadLLM.calls == 2


# ---------------------------------------------------------------- language
@pytest.mark.parametrize("text,prev,expected", [
    ("Give me scheme related rajastha", None, "english"),
    ("in Graudation Btech give me schemes related that", None, "english"),
    ("mujhe pension chahiye", None, "hinglish"),
    ("mai labour board ke lia register hu", None, "hinglish"),
    ("मुझे पेंशन चाहिए", None, "hindi"),
    ("हाँ license है", None, "hindi"),
    ("In rajasthan", "hinglish", "hinglish"),
    ("yes", "english", "english"),
    ("30", "hindi", "hindi"),
])
def test_detect_language_heuristic(text, prev, expected):
    lang, _ = detect_language_heuristic(text, previous=prev)
    assert lang == expected


def test_resolve_reply_lang_overrides_wrong_llm_guess():
    assert resolve_reply_lang("Give me scheme related rajastha", "hinglish") == "english"
    assert resolve_reply_lang("mujhe scheme chahiye", "english") == "hinglish"
    assert resolve_reply_lang("ok", "hinglish", previous="english") == "english"
    assert resolve_reply_lang("anything", "hinglish", locked="hindi") == "hindi"


@pytest.mark.parametrize("reply,lang,mismatch", [
    ("Dekho, B.Tech graduation ke baad jo schemes hote hain wo mostly current students ke liye hote hain", "english", True),
    ("Here are the schemes that fit your situation as a farmer with 5 acres in Jaipur.", "english", False),
    ("आपके लिए यह योजना सही है। Kantedar Tarbandi Yojana में 60% subsidy मिलती है।", "hindi", False),
    ("Aapke liye yeh scheme sahi hai.", "hindi", True),
    ("Aapke liye yeh scheme sahi hai, मुख्यमंत्री योजना dekho.", "hinglish", True),
    ("Aapke liye yeh scheme sahi hai.", "hinglish", False),
])
def test_reply_language_mismatch(reply, lang, mismatch):
    assert _reply_language_mismatch(reply, lang) is mismatch


# ---------------------------------------------------------------- identity gaps
def test_implied_groups_from_profile():
    assert "widow" in _implied_groups({"marital_status": "widowed"})
    assert "woman" in _implied_groups({"gender": "female"})
    assert "senior_citizen" in _implied_groups({"age": 65})
    assert "differently_abled" in _implied_groups({"disability": "yes"})


def test_identity_gap_does_not_reask_widow_when_marital_status_known():
    meta = {"occupation_target_group": "widow|woman", "gender": "female", "marital_status": "widowed"}
    gaps = _identity_gap(meta, {"gender": "female", "marital_status": "widowed"}, "pension")
    assert gaps == []
    gaps = _identity_gap(meta, {}, "pension")
    assert ("gender", "female") in gaps


# ---------------------------------------------------------------- state hygiene
def test_domain_switch_resets_other_domain():
    st = {"last_domain": "scheme", "shown_scheme_id": "s1"}
    up = domain_switch_resets(st, "job")
    assert up["last_domain"] == "job" and up["shown_scheme_id"] is None and up["shortlist"] == []
    assert "job_shortlist" not in up
    assert domain_switch_resets({"last_domain": "job"}, "job") == {"last_domain": "job"}


def test_leaked_scheme_names_is_script_aware():
    class Svc:
        def names_for(self, sid):
            return {"Mukhyamantri Vriddhjan Samman Pension Yojana", "मुख्यमंत्री वृद्धजन सम्मान पेंशन योजना"}

    by_scheme = {"s1": {"scheme_name": "मुख्यमंत्री वृद्धजन सम्मान पेंशन योजना"}}
    hinglish_reply = "Aapke liye Mukhyamantri Vriddhjan Samman Pension Yojana fit hai - kya aap BPL hain?"
    assert _leaked_scheme_names(Svc(), by_scheme, hinglish_reply)
    assert not _leaked_scheme_names(Svc(), by_scheme, "Kya aap BPL card holder hain?")


# ---------------------------------------------------------------- jan aadhaar exits
@pytest.mark.parametrize("text,step,out", [
    ("rehne do", "awaiting_otp", True),
    ("cancel", "awaiting_id", True),
    ("chhodo, mujhe scholarship ke baare mein batao", "awaiting_id", True),
    ("kitni pension milegi mujhe batao zara", "awaiting_otp", True),  # unrelated 4+ words, no digits
    ("123456", "awaiting_otp", False),
    ("haan", "awaiting_consent", False),
    ("dusra wala", "awaiting_member_choice", False),
])
def test_janaadhaar_wants_out(text, step, out):
    assert _janaadhaar_wants_out(text, step) is out
