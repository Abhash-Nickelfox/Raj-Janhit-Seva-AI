"""Phase 3 (PLAN.md SS4/SS11): the conversation graph.

Three layers of state, per SS4:
  a. profile            - persistent citizen facts, merged (never overwritten wholesale)
  b. need_text/category  - resets every turn, drives this turn's retrieval only
  c. shown_scheme_*      - short-lived, only for pronoun follow-ups

Life of one query (matches PLAN.md SS11.c):
  normalize -> extract -> [branch: followup? chitchat? fresh need?]
  -> (stage_a ->) [branch: no match? too vague? proceed?]
  -> (clarify | stage_b | stage_b_followup) -> generate -> END

Deliberately NOT built here (later phases): rural-query normalization
beyond a basic strip (Phase 4), a persistent (non-in-memory) session
store, and any web/WhatsApp transport.
"""

import re
from datetime import date
from typing import Literal, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import OPENROUTER_API_KEY, OPENROUTER_MODEL
from app.state import ConvState
from app.rag_service import RagService, CitizenContext, OCCUPATION_KEYWORDS, eligibility_verdict
from app.job_service import JobService
from app import pii_utils
from app.janaadhaar_service import get_janaadhaar_service, _PROFILE_FIELD_KEYS

KNOWN_CATEGORIES = [
    "Education & Learning",
    "Social welfare & Empowerment",
    "Agriculture,Rural & Environment",
    "Women and Child",
    "Skills & Employment",
    "Health & Wellness",
    "Business & Entrepreneurship",
    "Housing & Shelter",
    "Banking,Financial Services and Insurance",
    "Sports & Culture",
    "Travel & Tourism",
]
KNOWN_BENEFIT_TYPES = [
    "scholarship", "pension", "loan", "subsidy", "equipment",
    "training", "insurance", "housing", "healthcare",
]

FIELD_TYPES_BY_INTENT = {
    "how_to_apply": ["application", "documents"],
    "documents_question": ["documents"],
    "ask_detail": ["eligibility", "benefits"],
    "find_scheme": ["eligibility", "benefits"],
}

# If the need is unspecific (no category/benefit implied) AND this many or
# more schemes survive the eligibility filter, asking one narrowing
# question beats guessing among that many plausible options.
CLARIFY_THRESHOLD = 6

# Every scalar identity/eligibility dimension eligibility_verdict already
# checks - not a hand-picked subset added one bug at a time. Deliberately
# NOT grown reactively (gender -> caste -> marital_status -> ...): every
# scheme, present or future, gets mapped into this SAME fixed schema by
# scripts/extract_eligibility.py, so this list is already complete by
# construction rather than needing an edit each time a new restriction type
# happens to cause a bug (found live 2026-08-10, after a disability-only-
# restricted scheme slipped through because "disability" wasn't on an
# earlier, hand-picked version of this list - see occupation_target_group's
# handling in _identity_gap for how that specific case is actually covered).
#
# Fail-open correctly keeps a narrowly-restricted scheme (e.g.
# gender=transgender) in the candidate pool when nothing rules it out - but
# that's a different question from whether it should be CONFIDENTLY
# PRESENTED as the answer when nothing confirms it either. Found live
# (2026-08-05): "I want to build my own startup" (no gender stated)
# surfaced "Transgender Utthan Kosh - Self Employment Programme" as the top
# pick - not wrong per the filter, but presenting a narrowly-targeted
# scheme as the confident answer to an unconfirmed identity is the kind of
# ambiguity worth resolving by asking, not guessing.
IDENTITY_FIELDS = ("gender", "caste", "marital_status", "residence", "education_level")

# How many of the top-ranked schemes get checked for an identity gap, not
# just the #1 - found live (2026-08-10): a #1 scheme unrestricted on gender
# let a #2-ranked, gender-restricted scheme (Transgender Utthan Kosh) slip
# through with no check at all, since only shortlist[0] was ever examined.
# Raised from 3 to 5 (2026-08-12): "start a business" still opened with a
# disability question even after the gap-free-alternative check existed,
# because THREE different narrow-population schemes (differently_abled,
# woman, transgender) happened to rank #1-#3, pushing a genuinely open
# scheme to #4 - just outside the old window. _prioritize_gap_free (which
# decides which schemes to actually SHOW once no question is asked) already
# scans the full shortlist with no such limit; this only widens how deep
# the "is there a gap-free alternative at all" check looks before giving up.
IDENTITY_GAP_CHECK_TOP_N = 5

# Bounds on the two rolling context mechanisms below - kept small and fixed
# deliberately: this is a safety margin for context the CURRENT turn's own
# retrieval might have missed (e.g. a narrow documents-only fetch), not a
# replacement for doing retrieval well. Unbounded growth would just make
# every future prompt larger and slower for no benefit once it exceeds a
# handful of recent exchanges.
MAX_CACHED_CHUNKS = 15
MAX_HISTORY_TURNS = 4

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def _chunk_key(meta):
    return (meta.get("scheme_id"), meta.get("field_type"), meta.get("lang"))


def _update_chunk_cache(existing_cache, new_chunks, max_size=MAX_CACHED_CHUNKS):
    """Append new_chunks to existing_cache, deduped by (scheme_id, field_type,
    lang) with the newest occurrence winning position (so a re-fetched chunk
    moves to "most recently touched" rather than duplicating), then keep only
    the most recent max_size entries."""
    by_key = {}
    for doc, meta in existing_cache or []:
        by_key[_chunk_key(meta)] = (doc, meta)
    for doc, meta in new_chunks or []:
        by_key[_chunk_key(meta)] = (doc, meta)  # overwrite/reinsert -> moves to end
    return list(by_key.values())[-max_size:]


def _identity_gap(scheme_meta, facts, need_text=""):
    """Which restrictions does this scheme have that current known facts
    haven't confirmed? Returns [(field, required_value), ...].

    Walks the full fixed IDENTITY_FIELDS schema, not a hand-picked subset.
    occupation_target_group is handled separately from the scalar fields
    above: it's usually already implied by what the need itself asks for
    (farm-equipment -> farmer, scholarship -> student), so it's only a real
    gap when the need text doesn't already confirm it - checked via the
    SAME OCCUPATION_KEYWORDS mapping extraction uses to derive this facet in
    the first place, not a new list. Without this exception, every farmer/
    student/entrepreneur-restricted scheme (the majority of the corpus)
    would trigger a redundant clarifying question the need already answered.
    """
    gaps = []
    for field_name in IDENTITY_FIELDS:
        value = scheme_meta.get(field_name, "")
        if value and value != "all" and not facts.get(field_name):
            gaps.append((field_name, value))

    occ_groups = [g for g in (scheme_meta.get("occupation_target_group") or "").split("|") if g]
    if occ_groups:
        # Check each tag independently, not "is ANY of them confirmed" - a
        # scheme tagged both 'differently_abled' and 'entrepreneur' must not
        # have the sensitive tag waved through just because a mundane one
        # ("business" -> entrepreneur) happens to match the need text too
        # (found live 2026-08-10 testing this exact fix: "start a business"
        # wrongly cleared a disability-restricted scheme's whole gap check).
        need_lower = (need_text or "").lower()
        unconfirmed_groups = [
            group for group in occ_groups
            if not any(kw in need_lower for kw in OCCUPATION_KEYWORDS.get(group, []))
        ]
        if unconfirmed_groups:
            gaps.append(("occupation_target_group", "|".join(unconfirmed_groups)))
    return gaps


def _first_identity_gap(shortlist, facts, need_text, top_n=IDENTITY_GAP_CHECK_TOP_N):
    """Check the top N shown schemes, not just the top 1, for an unconfirmed
    identity gap - returns (scheme_entry, gaps) for the first one found, or
    (None, []) if none of them have one.

    Only asks when EVERY candidate in the top N has a gap - if even one is
    already usable with no clarification needed, that one wins silently
    rather than blocking on a DIFFERENT candidate's narrower restriction.
    Found live (2026-08-11): "apna business start karna hai" kept opening
    with "are you disabled?" because a disability-gated scheme (occupation_
    target_group='differently_abled') happened to rank #1, even though
    #2/#3/#5 in the same shortlist were ordinary, ungated business schemes
    ready to present with no question at all. Asking about the rank-1
    candidate's gap regardless of what else is available meant a citizen
    never even reached the schemes that already fit them. Rank still
    matters when NOTHING is gap-free (the top-ranked gap is asked about,
    same as before) - this only changes behavior when a clean alternative
    genuinely exists.
    """
    gapped = []
    for entry in shortlist[:top_n]:
        gaps = _identity_gap(entry["metadata"], facts, need_text)
        if gaps:
            gapped.append((entry, gaps))
        else:
            return None, []  # a gap-free candidate exists - nothing to ask about
    if gapped:
        return gapped[0]
    return None, []


def _prioritize_gap_free(shortlist, facts, need_text, limit):
    """Same principle as _first_identity_gap, applied to WHICH schemes get
    explained once no clarifying question is asked - a narrow/rare-
    population restriction (disability, widow, ex-servicemen...) shouldn't
    dominate the citizen's first answer just because it happened to rank
    highest. Gap-free schemes fill the explained slots first; a gapped one
    only fills a remaining slot if there aren't enough gap-free candidates.
    Found live (2026-08-11): even after _first_identity_gap correctly
    stopped blocking on "are you disabled?", stage_b_node was still
    fetching the SAME disability-gated scheme (still ranked #1) as one of
    its top 3 to explain, so generate_node's own prose still centered the
    whole reply around disability instead of the broadly-applicable
    schemes sitting right alongside it."""
    gap_free, gapped = [], []
    for entry in shortlist:
        (gap_free if not _identity_gap(entry["metadata"], facts, need_text) else gapped).append(entry)
    return (gap_free + gapped)[:limit]


def _subject_facts_for_need(need, profile):
    """Whose facts should filter/gate THIS need: the citizen's own persistent
    profile, or the family member's facts stated for this specific need?
    Added 2026-08-10 - a need explicitly asked on behalf of someone else
    (e.g. a son's PhD scholarship) must be judged against THAT person's
    stated facts, never the citizen's own profile, and never left with no
    facts at all just because we correctly stopped attributing them to the
    citizen (see ProfileExtraction's third-party guardrail)."""
    if need and need.get("about") == "family_member":
        return need.get("subject_facts") or {}
    return profile or {}


_llm = None
_rag_service = None
_job_service = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )
    return _llm


def get_rag_service():
    global _rag_service
    if _rag_service is None:
        _rag_service = RagService()
    return _rag_service


def get_job_service():
    global _job_service
    if _job_service is None:
        _job_service = JobService()
    return _job_service


Intent = Literal["find_scheme", "ask_detail", "how_to_apply", "documents_question", "chitchat"]
YesNo = Literal["yes", "no"]
RuralUrban = Literal["rural", "urban"]
EducationLevel = Literal["school", "higher_education"]
# Three-way, not the old Devanagari-detected hi/en binary (app/chat.py's old
# normalize_node) - that binary silently bucketed Hinglish (Romanized Hindi)
# in with English, so a citizen writing "mujhe scheme chahiye" got replied
# to in prose English instead of matching their own Hinglish register.
# 'hindi' = Devanagari script. 'hinglish' = Hindi/Rajasthani words in Roman
# script (even mixed with occasional English words) - the common rural
# register. 'english' = genuine English vocabulary/grammar, not Romanized
# Hindi.
ReplyLanguage = Literal["hindi", "hinglish", "english"]
# Single mutually-exclusive classification, replacing what used to be two
# independent booleans (is_followup, wants_alternatives) that could - and
# did, live - contradict each other (is_followup=no + wants_alternatives=yes
# for a message that was neither a followup nor wanted alternatives, it was
# a plain new need). One field can't structurally disagree with itself the
# way two independently-generated fields can.
TurnType = Literal["new_need", "followup_same_scheme", "wants_alternatives"]


def _normalize_to_known(value, known_values):
    """Snap free-text LLM output to the closest known canonical string
    (case/whitespace-insensitive substring match), rather than forcing the
    LLM to emit a byte-perfect enum value - Groq's strict tool-call
    validation rejects harmless phrasing variance (e.g. an added space,
    "Agriculture, Rural" vs the stored "Agriculture,Rural") as a hard
    error, so normalization belongs in our own code, not the model's output.
    """
    if not value:
        return None
    squashed = re.sub(r"\s+", "", value).lower()
    for known in known_values:
        if re.sub(r"\s+", "", known).lower() == squashed:
            return known
    for known in known_values:
        if squashed in re.sub(r"\s+", "", known).lower():
            return known
    return None


# Strict, per-language reply instructions (2026-08-11) - replaces the old
# binary "Hindi (Devanagari)"/"English" instruction, which had no Hinglish
# option at all, so a citizen writing in Hinglish got a reply in prose
# English instead of matching their own register. Explicitly calls out
# "no mixing" in both directions, since the failure mode observed wasn't
# just "wrong language" but partial mixing (a Devanagari word inside an
# English sentence, or vice versa). Official scheme names are the one
# deliberate exception in all three - they're proper nouns, not translated.
_LANG_INSTRUCTIONS = {
    "hindi": (
        "Reply ENTIRELY in Hindi, in Devanagari script. Do not include any English words or "
        "Romanized Hindi (Hinglish) anywhere in the reply, except official scheme names, which "
        "are proper nouns and stay as-is."
    ),
    "hinglish": (
        "Reply ENTIRELY in Hinglish - Hindi words and grammar written in Roman/Latin script, the "
        "way a Hindi speaker naturally types (e.g. 'Aapko yeh scheme mil sakti hai', 'iske liye "
        "aapko yeh documents chahiye'). Do NOT use Devanagari script anywhere, and do NOT write "
        "in formal/prose English either - match the citizen's own Hinglish register exactly. "
        "Official scheme names are proper nouns and stay as-is."
    ),
    "english": (
        "Reply ENTIRELY in English. Do not include any Hindi words, in Devanagari or Romanized "
        "form, anywhere in the reply, except official scheme names, which are proper nouns and "
        "stay as-is."
    ),
}


def _lang_instruction(reply_lang):
    return _LANG_INSTRUCTIONS.get(reply_lang, _LANG_INSTRUCTIONS["english"])


def _source_lang(reply_lang):
    """Which of the corpus's two scraped language versions (en/hi) to pull
    scheme TEXT from - distinct from the reply language instruction above.
    Hinglish reuses the Hindi source text (genuine Hindi phrasing to work
    from) even though the REPLY itself must come out in Roman script, not
    Devanagari - conflating these two would mean fetching English source
    text and asking the model to invent Hindi-flavored phrasing from
    scratch instead of grounding it in the scheme's actual Hindi wording."""
    return "hi" if reply_lang in ("hindi", "hinglish") else "en"


class NeedItem(BaseModel):
    """One distinct need/topic within the message. A message almost always
    produces exactly one of these - it only becomes a list because a single
    message occasionally states two unrelated needs at once (a farming
    subsidy for the citizen AND a scholarship for their son), and each needs
    its own search + its own eligibility facts rather than being merged or
    silently dropped."""

    need_text: str = Field(
        ...,
        description=(
            "A SHORT (3-6 word) phrase capturing ONLY the core purpose/need - e.g. 'start a "
            "business', 'scholarship for SC student'. Do NOT include place names, institution "
            "names, degree specifics, or other biographical detail (that's captured separately) - "
            "this phrase is used directly as a search query, and extra detail hurts semantic "
            "search relevance rather than helping it (verified 2026-08-05: a query padded with "
            "'IIT Kharagpur'/'B.Tech computer science'/'Jodhpur' surfaced irrelevant agriculture "
            "schemes ahead of real business schemes; the same need stated as just 'start a "
            "company' retrieved correctly).\n"
            "If the citizen wrote in Hindi OR Hinglish (romanized Hindi), write this phrase in "
            "DEVANAGARI SCRIPT (transliterate/translate as needed), even if their own message "
            "used Roman letters. Verified 2026-08-07: 'beti ki shaadi' (Hinglish) retrieved "
            "irrelevant construction-worker schemes, while the same need as 'बेटी की शादी' "
            "(Devanagari) correctly retrieved actual marriage-assistance schemes - the embedding "
            "search matches natural Hindi phrasing far better in Devanagari than in Roman script. "
            "If they wrote in English, keep this phrase in English."
        ),
    )
    need_category: Optional[str] = Field(
        None,
        description=(
            f"one of {KNOWN_CATEGORIES} ONLY if the message itself names or clearly implies a "
            "subject matter. If the message is a pure procedural/reference follow-up with no "
            "subject matter of its own (e.g. 'what's the applying process of this', 'iske "
            "documents batao', 'just give me the process for this') - referring back to "
            "something already discussed rather than describing a new topic - leave this null. "
            "Do NOT guess a category from words like 'apply'/'process'/'documents' alone; those "
            "describe an ACTION, not a subject."
        ),
    )
    need_benefit_type: Optional[str] = Field(
        None,
        description=(
            f"one of {KNOWN_BENEFIT_TYPES} ONLY if the message itself names or clearly implies "
            "one, using the same rule as need_category - null for a content-free procedural "
            "follow-up, do not guess."
        ),
    )
    domain: Literal["scheme", "job", "citizen_status"] = Field(
        ...,
        description=(
            "'job' if this need is about a GOVERNMENT JOB VACANCY/RECRUITMENT - looking for a "
            "sarkari naukri, bharti, vacancy, a specific post (Patwari, LDC, Police Constable, "
            "Junior Assistant, etc.), or asking what jobs/posts they can apply for.\n"
            "'citizen_status' ONLY if the citizen is asking about THEIR OWN existing "
            "record/application/benefit status with the government - e.g. 'kya meri pension "
            "aayegi', 'mera application status kya hai', 'meri scholarship kab aayegi' - a "
            "question that can only be answered by looking up THEM specifically, not a general "
            "eligibility/discovery question. Do NOT use this for 'am I eligible for X' or 'what "
            "schemes can I get' - those are 'scheme', even if phrased personally, since they're "
            "about discovering options, not checking an existing record.\n"
            "'scheme' for everything else - a welfare benefit, subsidy, pension, scholarship, "
            "loan, or other government ASSISTANCE scheme (including business/employment-support "
            "schemes, which are benefits, not a job posting). When genuinely unclear, default to "
            "'scheme'."
        ),
    )
    job_location: Optional[str] = Field(
        None,
        description=(
            "ONLY if domain='job' and the citizen named a specific city for the job search (e.g. "
            "'Jaipur', 'Jodhpur'). null otherwise."
        ),
    )
    job_qualification: Optional[Literal["12th Pass", "Diploma", "Graduate", "Post Graduate"]] = Field(
        None,
        description=(
            "ONLY if domain='job' and the citizen explicitly stated THEIR OWN educational/technical "
            "qualification in THIS message. Normalize free text to exactly one of these 4 levels: "
            "'12th Pass' (10th/12th pass, no further study); 'Diploma' (polytechnic, ITI, any "
            "technical diploma); 'Graduate' (any bachelor's degree - B.A./B.Sc./B.Com/B.Tech/BCA/"
            "etc.); 'Post Graduate' (any master's degree or higher). null if no qualification was "
            "stated - do not guess this from their phrasing or tone."
        ),
    )
    job_min_salary: Optional[int] = Field(
        None,
        description=(
            "ONLY if domain='job' and the citizen explicitly stated a minimum/expected salary "
            "figure in THIS message (e.g. '30000 se zyada chahiye', 'at least 25k salary'). The "
            "number in rupees (convert 'k'/'lakh' as needed, e.g. '25k' -> 25000, '2 lakh' -> "
            "200000). null if no figure was stated."
        ),
    )
    about: Literal["self", "family_member"] = Field(
        ...,
        description=(
            "'family_member' ONLY if this specific need is explicitly stated as being for someone "
            "else the citizen is asking on behalf of (son/daughter/spouse/parent/sibling) - e.g. "
            "'scheme for my beti', 'mera beta PhD karna chahta hai uske liye kuch batao'. 'self' "
            "for everything else, including when the message doesn't specify - default to self."
        ),
    )
    subject_relation: Optional[str] = Field(
        None,
        description="ONLY if about='family_member': a short English word for who (son, daughter, spouse, mother, father, sibling). null if about='self'.",
    )
    subject_caste: Optional[str] = Field(
        None,
        description="ONLY if about='family_member' AND their caste was explicitly stated for THIS need (e.g. sc/st/obc/general/pvtg). null otherwise, and null if about='self' (use the citizen's own profile instead).",
    )
    subject_gender: Optional[str] = Field(
        None,
        description="ONLY if about='family_member' AND their gender was explicitly stated for THIS need. null otherwise.",
    )
    subject_education_level: Optional[EducationLevel] = Field(
        None,
        description=(
            "ONLY if about='family_member': 'school' if they're CURRENTLY STUDYING IN (not yet passed) "
            "school, up to Class 12. 'higher_education' if pursuing/holding a degree, diploma, or PhD - "
            "OR if they've PASSED/COMPLETED Class 12 and are now seeking to continue into further/higher "
            "studies (e.g. '12th pass, wants to study further') - that is 'higher_education' even before "
            "a degree has started, because that is the stage they are now entering. "
            "null if not stated or about='self'."
        ),
    )
    subject_disability: Optional[YesNo] = Field(
        None,
        description=(
            "ONLY if about='family_member': 'yes' if THEY were explicitly stated to have a "
            "disability, 'no' if explicitly stated to NOT have one (denying one specific thing "
            "does not confirm anything else - same rule as the citizen's own disability field "
            "above). null if not stated or about='self'."
        ),
    )
    subject_age: Optional[int] = Field(
        None,
        description="ONLY if about='family_member' AND their age was explicitly stated for THIS need. null otherwise.",
    )
    subject_marital_status: Optional[str] = Field(
        None,
        description=(
            "ONLY if about='family_member': married, never-married, widowed, divorced, separated - "
            "only if explicitly stated about THEM specifically. null if not stated or about='self'."
        ),
    )
    subject_family_income: Optional[int] = Field(
        None,
        description=(
            "ONLY if about='family_member' AND an income figure specific to them (not the general "
            "family income, which belongs on the citizen's own profile) was explicitly stated. "
            "null otherwise - most of the time this stays null even for family-member needs, since "
            "family income is usually a household-level fact already captured on the citizen."
        ),
    )
    subject_residence_evidence: Optional[str] = Field(
        None,
        description=(
            "ONLY if about='family_member': the EXACT words, copied verbatim, that specifically say "
            "WHAT TYPE of area THEY live in (village/rural vs. city/urban) - same rule as the "
            "citizen's own residence_evidence field above: a place name alone does not qualify, "
            "and this is usually null even for family-member needs since residence is normally a "
            "shared household fact already on the citizen's own profile."
        ),
    )
    subject_residence: Optional[RuralUrban] = Field(
        None,
        description="ONLY if about='family_member' and subject_residence_evidence above is also filled in with the literal supporting words. null otherwise.",
    )


class ProfileExtraction(BaseModel):
    """Just the mechanical, low-ambiguity part: did the citizen state a fact
    about THEMSELVES (the person sending this message) this message? Split
    out from need/intent classification (below) after finding that one call
    trying to fill ~10 fields at once was flaky - even outright dropping
    required fields on simple messages like "main mahila hoon". Fewer
    fields per call = fewer chances to fail.

    RE-CONFIRMED (2026-08-11): a later attempt to re-merge this with
    NeedExtraction into one call was tried again and reverted a second
    time - verified directly that it reintroduces a real regression (the
    residence_evidence guardrail below started confidently misclassifying
    'main Rajasthan ka rehne wala hu' as rural evidence, 11/11 across two
    rounds including after strengthening the field description, despite
    the standalone version below getting it right 5/5). Do not re-attempt
    merging this with need classification without re-verifying against
    that exact case first.

    Every field here is about the citizen who is typing, never about a
    family member (spouse/child/parent/sibling) they mention. A message
    like "mujhe beti ke liye scholarship chahiye" (I need a scholarship for
    my daughter) states a fact about the daughter, not the citizen - it
    does NOT mean the citizen is male or married; having a daughter implies
    nothing about the citizen's own gender or marital status. Similarly
    "mera beta 12th mein hai" describes the SON's education, not the
    citizen's own education_level. If a fact in the message clearly belongs
    to someone other than the citizen, leave that field null rather than
    attributing it to the citizen - only fill a field when the citizen is
    describing themselves.
    """

    caste: Optional[str] = Field(None, description="e.g. sc, st, obc, general, pvtg - only if the citizen just stated it about THEMSELVES this message")
    gender_evidence: Optional[str] = Field(
        None,
        description=(
            "The EXACT words, copied verbatim, that indicate the citizen's OWN gender - actively "
            "search for BOTH kinds of evidence before answering: (1) an explicit word - "
            "'mahila'/'aurat'/'female' for female, 'purush'/'aadmi'/'male' for male; OR (2) a "
            "Hindi grammatical marker - a masculine verb form or 'wala'/'wale' suffix in a "
            "first-person statement ('rehta hoon', 'rehne wala hoon', 'karta hoon') is JUST AS "
            "VALID evidence of male as a feminine verb form or 'wali' suffix ('rehti hoon', 'rehne "
            "wali hoon', 'karti hoon') is of female. The masculine grammatical form is NOT an "
            "unmarked default to skip past - it is direct, explicit evidence, exactly as much as "
            "the feminine form is. null if no such words are present anywhere in the message."
        ),
    )
    gender: Optional[str] = Field(
        None,
        description=(
            "male, female, transgender - set this whenever gender_evidence above identifies a "
            "real marker (explicit word or grammatical form, either masculine or feminine). Only "
            "if the citizen just stated their OWN gender, never inferred from a family member "
            "they mention."
        ),
    )
    marital_status: Optional[str] = Field(None, description="married, never-married, widowed, divorced, separated - only if the citizen just stated their OWN marital status, never inferred from mentioning a spouse or child")
    age: Optional[int] = Field(None, description="the citizen's own age - only if just stated")
    family_income: Optional[int] = Field(None, description="annual family income in rupees - only if just stated")
    residence_evidence: Optional[str] = Field(
        None,
        description=(
            "The EXACT words, copied verbatim from the citizen's message, that specifically say "
            "WHAT TYPE of area they live in - a village/countryside, or a city/town. This must go "
            "beyond merely naming a place or stating that they are a resident of somewhere: 'main "
            "Rajasthan ka rehne wala hu' / 'I live in Jaipur' / 'I'm from Delhi' name a PLACE but do "
            "NOT say what type of area it is, so none of those qualify - leave this null for them. "
            "Only a phrase that itself signals village/rural vs. city/urban character qualifies - "
            "e.g. 'gaon mein rehta hoon', 'dehat se hoon', 'main shehar mein rehti hoon', 'I live in "
            "a city'. If you cannot find such a phrase, leave this null even if a place name, "
            "occupation, or other detail is present elsewhere in the message."
        ),
    )
    residence: Optional[RuralUrban] = Field(
        None,
        description=(
            "MUST be exactly 'rural' or 'urban' if they described where THEY live in those terms - "
            "e.g. 'village', 'gaon' -> rural; 'city', 'shehar' -> urban. "
            "If they only named a specific place (a city/district/state name like 'Jaipur'), leave this "
            "null - a place name is NOT a value for this field, we don't track specific locations. "
            "Only set this if residence_evidence above is also filled in with the literal supporting words."
        ),
    )
    education_level: Optional[EducationLevel] = Field(
        None,
        description=(
            "The CITIZEN's own education level - 'school' if they said THEY are CURRENTLY STUDYING IN "
            "(not yet passed) school, up to Class 12 / 10+2. 'higher_education' if they said THEY are a "
            "graduate, post-graduate, currently pursuing/have completed a degree/diploma/professional "
            "course (e.g. 'I'm a BCA graduate', 'MCA graduate', 'pursuing engineering') - OR if they've "
            "PASSED/COMPLETED Class 12 and are now seeking to continue into further/higher studies (e.g. "
            "'12th pass, want to study further') - this is 'higher_education' even though they haven't "
            "started a degree yet, because that is the stage they are now entering; 'school' means still "
            "enrolled in and yet to finish school, not already past it. "
            "If the education level mentioned belongs to their child/sibling/spouse instead of "
            "themselves, leave this null - it is not the citizen's own education_level. "
            "Only if just stated this message."
        ),
    )
    disability: Optional[YesNo] = Field(
        None,
        description=(
            "'yes' ONLY if the citizen explicitly stated THEY THEMSELVES have a disability "
            "(e.g. 'I have 40% disability', 'mujhe viklangta hai'). 'no' ONLY if they explicitly "
            "stated THEY do NOT have a disability (e.g. 'no i dont have any disability'). null if "
            "disability wasn't mentioned about the citizen themselves at all - do not guess from "
            "occupation, need, or anything else."
        ),
    )


class NeedExtraction(BaseModel):
    """The judgment-heavy part: what do they want, and is it clear enough
    to act on. need_is_specific is the model's OWN direct clarity call -
    never inferred from whether need_category/need_benefit_type happened
    to populate (those two are unreliable and exist only for the rank
    boost, not for judging vagueness).

    Includes chitchat_reply (2026-08-11) - folding the chitchat reply
    itself into this same call (verified safe/reliable, unlike merging in
    ProfileExtraction's fields - see that class's docstring) skips a
    separate chitchat_node call entirely on chitchat turns."""

    needs: list[NeedItem] = Field(
        ...,
        description=(
            "One entry per DISTINCT need/topic in the message. Almost always exactly ONE entry - "
            "only produce more than one when the citizen clearly states two or more SEPARATE, "
            "unrelated purposes in the same message (e.g. a farming subsidy for themselves AND a "
            "scholarship for their son - two different needs, two different subjects). Do not split "
            "a single need into multiple entries just because it has several details attached to it."
        ),
    )
    need_is_specific: YesNo = Field(
        ...,
        description=(
            "'yes' if the need is specific enough to search for directly - e.g. names a "
            "scheme, a benefit type (pension/scholarship/loan/...), or a clear purpose - "
            "EVEN IF you can't map it to one of the exact category/benefit_type labels above. "
            "'no' ONLY if it's genuinely too broad/vague to know what kind of help they want "
            "(e.g. 'I need money', 'help chahiye' with no purpose stated)."
        ),
    )
    intent: Intent = Field(
        ...,
        description=(
            "What KIND of question this is, which decides which scheme fields get fetched:\n"
            "'how_to_apply' - asks about the APPLICATION PROCESS, even indirectly - not just "
            "'how do I apply', but also 'is there an online portal', 'where do I submit this', "
            "'online ya offline', 'kahan jama karu', asking for a form, a website, an office to "
            "visit. Fetches application+documents data.\n"
            "'documents_question' - asks specifically what documents/papers/certificates are "
            "needed. Fetches documents data only.\n"
            "'ask_detail' - asks about eligibility, benefits, amounts, who qualifies, or wants "
            "more general information about an already-identified scheme - anything that isn't "
            "specifically about the application process or documents.\n"
            "'find_scheme' - describes a need/purpose and wants a scheme found for it.\n"
            "'chitchat' - small talk, thanks, greetings - not a scheme question at all."
        ),
    )
    turn_type: TurnType = Field(
        ...,
        description=(
            "EXACTLY one, mutually exclusive - do not reason about these as separate yes/no "
            "questions, pick the single best-fitting category:\n"
            "'new_need' - the message describes a DIFFERENT need/topic than what's currently "
            "shown (even if phrased casually, e.g. 'and if i want to start a business' after a "
            "scholarship was shown).\n"
            "'followup_same_scheme' - asks for more detail about the SAME scheme already shown "
            "(documents, how to apply, eligibility, a pronoun/implicit reference like 'this'/"
            "'iske') - including terse procedural questions with no topic of their own.\n"
            "'wants_alternatives' - explicitly asks for OTHER/additional/more schemes for the "
            "SAME underlying need already established (e.g. 'what other schemes am I eligible "
            "for', 'any other options') - NOT a message that itself describes a new need, which "
            "is 'new_need' regardless of how it's phrased."
        ),
    )
    detected_language: ReplyLanguage = Field(
        ...,
        description=(
            "Which register the CITIZEN'S CURRENT message is actually written in - the reply "
            "must match this exactly, so classify carefully:\n"
            "'hindi' - written in Devanagari script (देवनागरी).\n"
            "'hinglish' - Hindi/Rajasthani words spelled out in Roman/Latin letters (e.g. 'mujhe "
            "scheme chahiye', 'mera beta 12th mein hai') - this is the common register for rural "
            "Hindi speakers typing in Roman script. Classify as hinglish even if a few English "
            "words are mixed in, as long as the sentence is fundamentally Hindi grammar/vocabulary "
            "in Roman letters.\n"
            "'english' - genuine English vocabulary and grammar, not Romanized Hindi.\n"
            "Base this ONLY on the citizen's message in THIS turn, not on what language previous "
            "turns used."
        ),
    )
    followup_target: Optional[str] = Field(
        None,
        description=(
            "ONLY relevant when several schemes OR jobs were listed above as shown last turn AND "
            "turn_type='followup_same_scheme': copy the EXACT scheme_name (or job title, if jobs "
            "were shown) from that list that this message is asking about - by position ('pehli'/"
            "'first' -> the 1st one listed, 'dusri'/'second' -> the 2nd), by topic ('education "
            "wali' -> whichever is about education), or by name. null if only one item was shown "
            "(nothing to disambiguate), or if the message clearly refers to ALL of them together."
        ),
    )
    chitchat_reply: Optional[str] = Field(
        None,
        description=(
            "ONLY if intent='chitchat': write the actual reply to send back, right here - not a "
            "description of what to say, the literal reply text itself. Keep it brief and warm, "
            "in the same language/register as detected_language above, and mention you can help "
            "them find Rajasthan government schemes and jobs. null for every other intent - do not "
            "write anything here unless intent is genuinely 'chitchat'."
        ),
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def normalize_node(state: ConvState):
    raw = state["raw_message"].strip()
    result = {
        "normalized_message": raw,
        # Cleared at the START of every turn, not just after use - if a
        # PREVIOUS turn set this (janaadhaar_handle_node redacting an ID/OTP
        # attempt), it must not linger and wrongly redact a later, unrelated
        # turn's conversation_history entry. janaadhaar_handle_node re-sets
        # this later in THIS SAME turn's run if it actually processes one.
        "redacted_raw_message": None,
    }
    if not state.get("pending_janaadhaar_step"):
        # Cheap placeholder only - extract_need_node's detected_language
        # always overrides this with a real 3-way classification (hindi/
        # hinglish/english) before any reply-generating node reads
        # reply_lang. This regex can only tell Devanagari from
        # not-Devanagari, so it can't tell Hinglish from English - kept only
        # as a fallback value, in the unlikely case something reads
        # reply_lang before extract_need_node completes.
        # Skipped entirely while a Jan Aadhaar verification turn is pending
        # (route_after_normalize is about to send this straight to
        # janaadhaar_handle_node, bypassing extract_need_node) - found live
        # (2026-08-14): without this guard, a Hinglish-typed Jan Aadhaar ID
        # ("mera jan aadhaar id ... hai", no Devanagari characters) got this
        # crude heuristic's "english" guess instead, silently downgrading a
        # conversation already locked to Hinglish since this heuristic's
        # value was never subsequently corrected by extract_need_node on
        # these bypassed turns.
        result["reply_lang"] = "hindi" if DEVANAGARI_RE.search(raw) else "english"
    return result


def route_after_normalize(state: ConvState):
    """Intercepts every turn where a Jan Aadhaar verification sub-flow is
    already in progress (an ID, member choice, or OTP is expected next) and
    sends it straight to janaadhaar_handle_node, BYPASSING extract_profile/
    extract_need entirely - both are LLM calls, and this turn's raw_message
    IS itself a credential the citizen just typed. Every other turn proceeds
    through the normal extraction pipeline unchanged."""
    if state.get("pending_janaadhaar_step"):
        return "janaadhaar_handle"
    return "extract_profile"


def extract_profile_node(state: ConvState):
    profile = state.get("profile", {})
    # Recent turns, not just the current message in isolation - found live
    # (2026-08-11): "this is her first marriage" (mid-conversation about the
    # citizen's DAUGHTER's wedding) got wrongly attributed as the citizen's
    # OWN marital_status, purely because this node had no way to know the
    # conversation was currently about someone else. Verified directly: the
    # model correctly identifies whose fact a message states once it can
    # actually see what's being discussed - it wasn't a reasoning gap, it
    # was missing the same context extract_need_node already gets.
    history_block = _format_conversation_history(state.get("conversation_history"))
    prompt = (
        history_block
        + (f"Known facts about this citizen so far: {profile}\n\n" if profile else "")
        + f'Citizen\'s new message: "{state["normalized_message"]}"\n\n'
        + "Extract only NEW facts about the citizen THEMSELVES (the person sending this message) "
        "stated in THIS message. Use the recent conversation above to tell whether this message is "
        "about the citizen or about someone else (e.g. a family member) already being discussed - a "
        "message continuing a conversation about a family member's situation is still about THAT "
        "person, not the citizen, even if it doesn't re-name them.\n\n"
        "For EVERY field: fill it in ONLY if the citizen explicitly and literally stated that "
        "fact about themselves. Never fill in a field because of an association, stereotype, or "
        "plausible-sounding guess connected to something else in the message - not from a place "
        "name, not from an occupation, not from their age, not from a fact about a family member, "
        "not from the general tone or completeness of their self-introduction. A detail is only "
        "evidence for a field if it is THAT field, stated directly - anything else, no matter how "
        "commonly correlated in the real world, is not evidence and must not be used. "
        "This holds even when the citizen states several other facts in the same message - stating "
        "more facts about themselves is not itself a reason to fill in additional, unstated fields. "
        "For example: naming a family member's need or trait (a scholarship 'for my beti/beta') is "
        "not a fact about the citizen's own gender, marital status, or education; naming a place "
        "they live in (a state or city name) is not, by itself, evidence of rural or urban - only "
        "an explicit description of where they live is; naming their occupation is not evidence of "
        "their education level or income - only an explicit statement about schooling, a degree, or "
        "an income figure is. Generalize this same standard to every field, not just these examples. "
        "When genuinely uncertain whether something counts as an explicit statement about the "
        "citizen themselves, leave the field null rather than guess. "
        "DENYING one specific value is NOT the same as confirming any value, including a different "
        "one: if the citizen says they are NOT something (e.g. 'I am not transgender', 'main "
        "transgender nahi hoon'), that rules out ONLY the denied value - it does not mean the field "
        "should be filled with a different guessed value (e.g. do not fill gender with 'female' "
        "just because 'transgender' was denied), and it absolutely does NOT mean the denied value "
        "itself should be filled in either (never output 'transgender' here because the citizen "
        "said they are NOT transgender - read the negation carefully, especially across 'nahi ... "
        "nahi hoon'-style Hindi negation). A denial with no separate positive statement leaves the "
        "field null. "
        "Do not restate facts already known above unless this message corrects them. "
        "Most messages state no new profile facts at all - that's fine, leave those null."
    )
    result = get_llm().with_structured_output(ProfileExtraction).invoke(prompt)

    # Evidence-grounding backstop for `residence` (2026-08-10): two rounds of
    # prompt-only guardrails failed to stop the model inferring rural/urban
    # from a place NAME's stereotype (e.g. "Rajasthan" -> rural) when another
    # fact was also stated - confirmed live, 2/2 reproducing, on both prompt
    # attempts. Rather than enumerate the words that count as real evidence
    # (a keyword list that can never keep pace with real phrasing diversity),
    # require the model to cite its own justification and verify that
    # citation is real: if there's no evidence, or the "evidence" isn't
    # actually present in the message, the classification is unfounded.
    evidence = (result.residence_evidence or "").strip().lower()
    if result.residence is not None and (not evidence or evidence not in state["normalized_message"].lower()):
        result.residence = None

    # Same evidence-grounding check for gender (2026-08-11): "rehne wala hu"
    # (masculine) was inconsistently NOT recognized as evidence while "rehne
    # wali hu" (feminine) reliably was, even after a prompt-only fix. Forcing
    # a cited quote doesn't stop hallucination here (the earlier failure was
    # under-recognition, not over-confident guessing) but requires the model
    # to actively locate real evidence before committing to a value, and a
    # claim with no real backing text is dropped either way.
    gender_evidence = (result.gender_evidence or "").strip().lower()
    if result.gender is not None and (not gender_evidence or gender_evidence not in state["normalized_message"].lower()):
        result.gender = None

    merged_profile = dict(profile)
    for field in ("caste", "gender", "marital_status", "age", "family_income", "residence", "education_level", "disability"):
        value = getattr(result, field)
        if value is not None:
            merged_profile[field] = value

    return {"profile": merged_profile}


def extract_need_node(state: ConvState):
    shown_scheme_name = state.get("shown_scheme_name")
    shown_schemes = state.get("shown_schemes") or []
    shown_jobs = state.get("shown_jobs") or []
    last_reply = state.get("last_reply")
    pending_need_text = state.get("pending_need_text")

    context_lines = []
    history = state.get("conversation_history")
    if history:
        # Full recent turns, not just the last reply - lets the model resolve
        # WHO a terse pronoun-only followup ("no she is not disable") refers
        # to by actually reading who was being discussed, the same way a
        # human would, instead of us pre-computing and threading a separate
        # "current subject" field through state. Found live (2026-08-10): a
        # followup about a family member (a citizen's wife) silently reverted
        # to being about the citizen themselves once the message no longer
        # named "wife" explicitly, because this node previously had no
        # access to the turn where "wife" was first established.
        context_lines.append(_format_conversation_history(history).rstrip())
    if len(shown_schemes) > 1:
        # Multiple schemes shown at once (a multi-need answer, or several
        # options) - give the model the actual list so it can resolve which
        # one a followup means, instead of us guessing positionally in code.
        # Found live (2026-08-10): "pehli scheme"/"dusri scheme" (first/
        # second scheme) followups had nothing to resolve against and got
        # answered against a stale single shown_scheme_id from an unrelated
        # earlier need.
        options = "; ".join(
            f'{i+1}. "{s["scheme_name"]}" (about: {s.get("need_text", "")})'
            for i, s in enumerate(shown_schemes)
        )
        context_lines.append(f"Multiple schemes were shown last turn, in this order: {options}")
    elif shown_scheme_name:
        context_lines.append(f"Scheme currently in focus (most recently shown): {shown_scheme_name}")
    if len(shown_jobs) > 1:
        job_options = "; ".join(f'{i+1}. "{j["title"]}" ({j.get("department", "")})' for i, j in enumerate(shown_jobs))
        context_lines.append(f"Multiple job postings were shown last turn, in this order: {job_options}")
    elif shown_jobs:
        context_lines.append(f"Job posting currently in focus (most recently shown): {shown_jobs[0]['title']}")
    if last_reply:
        context_lines.append(f"Your last reply to them was: {last_reply}")
    if pending_need_text:
        context_lines.append(
            f'They were previously asking about: "{pending_need_text}" - if this message is just a '
            "short answer to your last reply (yes/no/a fact about themselves) rather than a full "
            "restatement, treat need_text as still being about that."
        )
    locked_lang = state.get("locked_reply_lang")
    if locked_lang:
        context_lines.append(
            f"The citizen has picked '{locked_lang}' as their language for this whole conversation - "
            f"if you fill in chitchat_reply below, write it in '{locked_lang}' specifically, "
            "regardless of what detected_language would otherwise classify this message as."
        )
    context = "\n".join(context_lines)

    prompt = (
        f"{context}\n\n"
        f'Citizen\'s message: "{state["normalized_message"]}"\n\n'
        "Classify what they want and how clear it is."
    )

    result = get_llm().with_structured_output(NeedExtraction).invoke(prompt)

    needs = []
    for item in result.needs:
        needs.append({
            "need_text": item.need_text,
            "need_category": _normalize_to_known(item.need_category, KNOWN_CATEGORIES),
            "need_benefit_type": _normalize_to_known(item.need_benefit_type, KNOWN_BENEFIT_TYPES),
            "domain": item.domain,
            "job_location": item.job_location,
            "job_qualification": item.job_qualification,
            "job_min_salary": item.job_min_salary,
            "about": item.about,
            "subject_relation": item.subject_relation,
            "subject_facts": {
                k: v for k, v in {
                    "caste": item.subject_caste,
                    "gender": item.subject_gender,
                    "education_level": item.subject_education_level,
                    "disability": item.subject_disability,
                    "age": item.subject_age,
                    "marital_status": item.subject_marital_status,
                    "family_income": item.subject_family_income,
                    # Same evidence-grounding rule as the citizen's own residence
                    # (2026-08-10 fix): only trust it if the cited evidence is a
                    # real substring of the message, not just a place name.
                    "residence": (
                        item.subject_residence
                        if item.subject_residence is not None
                        and (item.subject_residence_evidence or "").strip().lower() in state["normalized_message"].lower()
                        else None
                    ),
                }.items() if v is not None
            },
        })
    if not needs:  # defensive - the model is expected to always return >=1
        # Found live (2026-08-14): a fully context-free, very generic message
        # ("what is my application status", first turn, nothing else stated)
        # can come back with an empty needs list from the model. domain
        # defaults to "scheme" here - matching NeedItem.domain's own "when
        # genuinely unclear, default to scheme" rule - so route_after_extract's
        # domain checks (job/citizen_status) have a defined value to read
        # instead of silently falling through on a missing key.
        needs = [{"need_text": state["normalized_message"], "need_category": None, "need_benefit_type": None,
                  "domain": "scheme", "about": "self", "subject_relation": None, "subject_facts": {}}]
    primary = needs[0]

    need_is_specific = result.need_is_specific == "yes"

    intent = result.intent
    if intent == "chitchat" and (primary["need_category"] or primary["need_benefit_type"] or need_is_specific):
        # the model contradicted itself - it extracted a real need but also
        # called this chitchat. A stated goal ("I want to start a business")
        # isn't idle chat just because it lacks a question mark; trust the
        # extracted need over the intent label when they conflict.
        intent = "find_scheme"

    result_dict = {
        "needs": needs,
        "need_text": primary["need_text"],
        "need_category": primary["need_category"],
        "need_benefit_type": primary["need_benefit_type"],
        "need_is_specific": need_is_specific,
        "intent": intent,
        "turn_type": result.turn_type,
        # A citizen-picked language (via the frontend's language picker)
        # always wins over this turn's own detection - see ConvState.
        # locked_reply_lang's docstring for why (voice transcription
        # flip-flopping between Devanagari and Romanized Hindi).
        "reply_lang": state.get("locked_reply_lang") or result.detected_language,
        "followup_target": result.followup_target,
        "pending_need_text": None,  # consumed - don't let it linger past the turn it was needed for
        # was the PREVIOUS turn a clarifying question (clarify/clarify_identity)?
        # If so, this message is almost certainly answering it, not referring
        # back to some possibly much-older shown_scheme - route_after_extract
        # uses this to force a fresh Stage A run regardless of what turn_type
        # says. Found live (2026-08-05): "education"/"undergraduate"/"she is
        # 12th pass" - answers to the bot's own clarifying questions - kept
        # getting misrouted to stage_b_followup against a stale shown_scheme
        # from many turns earlier (a different topic entirely), producing
        # answers about the wrong scheme and a dead-end loop the citizen
        # couldn't escape even by explicitly asking for a different scheme.
        "was_answering_clarification": bool(pending_need_text),
    }
    if not pending_need_text:
        # No clarification was pending coming into this turn, so this is a
        # fresh need with no unresolved clarify streak to cap - reset the
        # counter. Left untouched (not returned) while a clarification IS
        # pending, so the streak survives across the turns route_after_stage_a
        # actually needs to count it over.
        result_dict["clarify_attempts"] = 0
        result_dict["generate_clarify_rounds"] = 0

    # The reply itself was already composed as part of this SAME call
    # (chitchat_reply) - route_after_extract sends this straight to
    # finalize instead of a separate chitchat_node call when this is set,
    # cutting chitchat turns from 3 LLM calls down to 2 (extract_profile +
    # this call). Falls back to the old separate chitchat_node call (still
    # wired into the graph) if the model left chitchat_reply empty despite
    # intent='chitchat' - never silently produce an empty reply.
    if intent == "chitchat" and result.chitchat_reply:
        result_dict["last_reply"] = result.chitchat_reply
    return result_dict


def route_after_extract(state: ConvState):
    if state.get("was_answering_clarification"):
        return "stage_a"
    turn_type = state.get("turn_type")
    primary_need = (state.get("needs") or [None])[0]
    # A citizen-status question ("kya meri pension aayegi") can only be
    # answered by looking up THIS citizen's own record - kicks off the Jan
    # Aadhaar OTP-consent sub-flow instead of a scheme/job search. This turn's
    # own message (the status question itself) has no identifier in it, so
    # it's safe that it already went through extract_profile/extract_need
    # above - only the FOLLOWING turns (ID/OTP) get intercepted earlier, by
    # route_after_normalize.
    if primary_need and primary_need.get("domain") == "citizen_status":
        return "janaadhaar_handle"
    # Job vacancies are a separate domain from welfare schemes (no identity
    # eligibility gating, structured qualification/location/deadline data
    # instead) - routed to their own parallel subgraph rather than through
    # Stage A's scheme-shortlist/clarify machinery, which doesn't apply here.
    if primary_need and primary_need.get("domain") == "job":
        if turn_type == "followup_same_scheme" and state.get("shown_job_id"):
            return "stage_b_followup_jobs"
        return "stage_a_jobs"  # covers both a fresh job search and "show other jobs" in v1
    if turn_type == "wants_alternatives":
        shortlist = state.get("shortlist") or []
        already_shown = set(state.get("already_shown_scheme_ids") or [])
        remaining = [s for s in shortlist if s["scheme_id"] not in already_shown]
        if remaining:
            return "show_more"
        return "stage_a"  # nothing left to show from - fresh search instead of erroring
    if turn_type == "followup_same_scheme" and state.get("shown_scheme_id"):
        return "stage_b_followup"
    if state.get("intent") == "chitchat":
        # extract_need_node already composed the reply in the same call
        # (chitchat_reply) - skip the separate chitchat_node call entirely
        # and go straight to finalize. Only falls through to the old
        # chitchat_node path if that field came back empty.
        return "finalize" if state.get("last_reply") else "chitchat"
    return "stage_a"


def stage_a_node(state: ConvState):
    profile = state.get("profile", {})
    needs = state.get("needs") or [{
        "need_text": state["need_text"], "need_category": state.get("need_category"),
        "need_benefit_type": state.get("need_benefit_type"), "about": "self", "subject_facts": {},
    }]
    svc = get_rag_service()

    shortlists = []
    for need in needs:
        facts = _subject_facts_for_need(need, profile)
        citizen = CitizenContext(
            caste=facts.get("caste"),
            gender=facts.get("gender"),
            marital_status=facts.get("marital_status"),
            age=facts.get("age"),
            family_income=facts.get("family_income"),
            residence=facts.get("residence"),
            education_level=facts.get("education_level"),
            disability=facts.get("disability"),
            need_category=need.get("need_category"),
            need_benefit_type=need.get("need_benefit_type"),
        )
        # For a family-member need, also check the citizen's OWN profile as an
        # alternate candidate identity - some schemes "for" a family member
        # actually gate eligibility on the APPLICANT (e.g. a widow's-
        # daughter's-marriage scheme requires the widow/citizen's own marital
        # status, not the daughter's). A scheme should only be excluded if
        # NEITHER identity fits (see RagService.shortlist's alt_citizen doc).
        alt_citizen = None
        if need.get("about") == "family_member":
            alt_citizen = CitizenContext(
                caste=profile.get("caste"),
                gender=profile.get("gender"),
                marital_status=profile.get("marital_status"),
                age=profile.get("age"),
                family_income=profile.get("family_income"),
                residence=profile.get("residence"),
                education_level=profile.get("education_level"),
                disability=profile.get("disability"),
                need_category=need.get("need_category"),
                need_benefit_type=need.get("need_benefit_type"),
            )
        trace = {}
        results = svc.shortlist(need["need_text"], citizen, alt_citizen=alt_citizen, top_n=10, trace=trace)
        # plain dicts, not ScoredScheme dataclass instances - LangGraph's
        # checkpointer only reliably (de)serializes plain/registered types.
        shortlist = [
            {"scheme_id": s.scheme_id, "scheme_name": s.scheme_name, "lang": s.lang, "score": s.score, "metadata": s.metadata}
            for s in results
        ]
        shortlists.append({"need": need, "shortlist": shortlist, "candidate_count": trace.get("survivors", len(results))})

    primary = shortlists[0]
    return {
        "shortlists": shortlists,
        "shortlist": primary["shortlist"],
        "candidate_count": primary["candidate_count"],
    }


def route_after_stage_a(state: ConvState):
    shortlist = state.get("shortlist") or []
    if not shortlist:
        return "no_match"
    # Never clarify twice in a row for the same unresolved need - found live
    # (2026-08-10): a citizen who repeated the same vague answer twice got
    # the identical clarifying question back, verbatim, with no way out.
    # After one unanswered attempt, fall back to presenting the best
    # available candidates rather than asking again.
    if state.get("clarify_attempts", 0) < 1:
        # need_is_specific is the model's own direct judgment of clarity - NOT
        # inferred from whether need_category/need_benefit_type happened to
        # populate (those exist only for the rank boost and are unreliable
        # proxies for "was the need actually clear").
        if not state.get("need_is_specific", True) and state.get("candidate_count", 0) >= CLARIFY_THRESHOLD:
            return "clarify"
        primary_need = (state.get("needs") or [None])[0]
        facts = _subject_facts_for_need(primary_need, state.get("profile", {}))
        need_text = primary_need["need_text"] if primary_need else state.get("need_text", "")
        _, gaps = _first_identity_gap(shortlist, facts, need_text)
        if gaps:
            return "clarify_identity"
    return "stage_b"


def clarify_node(state: ConvState):
    shortlist = state.get("shortlist") or []
    options_desc = "; ".join(
        f"{s['scheme_name']} ({s['metadata'].get('category', '')})" for s in shortlist[:8]
    )
    lang_instruction = _lang_instruction(state.get("reply_lang"))
    prompt = (
        f'The citizen asked: "{state["need_text"]}"\n'
        f"This is too broad to answer directly - {state.get('candidate_count')} schemes could match, "
        f"spanning different purposes, e.g.: {options_desc}\n\n"
        f"Ask ONE short, simple clarifying question {lang_instruction} that would narrow this down "
        "(e.g. what specific kind of help they need). Do not list scheme names. One sentence only."
    )
    resp = get_llm().invoke(prompt)
    return {
        "last_reply": resp.content,
        "pending_need_text": state.get("need_text"),
        "clarify_attempts": state.get("clarify_attempts", 0) + 1,
    }


def clarify_identity_node(state: ConvState):
    shortlist = state.get("shortlist") or []
    primary_need = (state.get("needs") or [None])[0]
    facts = _subject_facts_for_need(primary_need, state.get("profile", {}))
    need_text = primary_need["need_text"] if primary_need else state.get("need_text", "")
    top, gaps = _first_identity_gap(shortlist, facts, need_text)
    if top is None:  # defensive - routing already confirmed a gap exists somewhere in the top N
        top = shortlist[0]
    # Capped at 2, same limit as generate_node's QUESTION LIMIT - a single
    # scheme restricted on 3+ unconfirmed fields at once (e.g. gender AND
    # caste AND disability) would otherwise get jammed into one compound
    # gap_desc and asked as a single overloaded question. Any gap beyond the
    # first 2 simply isn't asked about THIS turn - once these are answered
    # and the shortlist is re-evaluated with the new facts, a remaining gap
    # (on this scheme or another) surfaces on a later turn same as normal.
    gaps = gaps[:2]
    gap_desc = ", ".join(f"{field.replace('_', ' ')}: {value.replace('|', ' or ')}" for field, value in gaps)
    lang_instruction = _lang_instruction(state.get("reply_lang"))

    # Fetch the scheme's OWN eligibility text, not just the coarse
    # structured tag - found live (2026-08-11): "Assistance for Daughter's
    # Marriage" requires the APPLICANT (a widowed parent) to be widowed, not
    # the daughter, but this node had no way to know that from the bare tag
    # alone and defaulted to assuming every gap on a family-member need
    # belongs to that family member. Verified directly: given the real
    # eligibility text, the model correctly identifies who a restriction
    # actually applies to - it's an information gap, not a reasoning one.
    lang = _source_lang(state.get("reply_lang"))
    svc = get_rag_service()
    elig_chunks = svc.scheme_detail(top["scheme_id"], lang, field_types=["eligibility"])
    if not elig_chunks and lang != "en":
        elig_chunks = svc.scheme_detail(top["scheme_id"], "en", field_types=["eligibility"])
    elig_text = elig_chunks[0][0] if elig_chunks else ""

    # The "could this actually be about the applicant, not the family member"
    # framing is ONLY relevant when the need genuinely IS about a family
    # member - found live (2026-08-11): "mai kheti karta hu... mere liye"
    # (unambiguously self, no third party mentioned at all) still got asked
    # "are you a farmer, or asking on behalf of someone else?" because this
    # clause used to be included unconditionally, mentioning the family-
    # member possibility even when about='self' and there was never any
    # ambiguity to begin with.
    if primary_need and primary_need.get("about") == "family_member":
        subject_desc = primary_need.get("subject_relation") or "the family member you mentioned"
        # Give the model what's already confirmed about the CITIZEN too, not
        # just the family member's facts - found live (2026-08-11): a citizen
        # already known (this same turn, from "...karta hu") to be male still
        # got asked "are you a widow of an ex-serviceman?" for a scheme
        # requiring a widowed applicant, because this node only ever looked
        # at the family member's facts and had no way to know the citizen's
        # own confirmed gender already ruled that out. It's a context gap,
        # not a reasoning one - the model already reasons correctly about
        # WHO an eligibility restriction targets (see the paragraph below);
        # it just never got told what's already known about the citizen.
        profile = state.get("profile", {})
        known_bits = [
            f"{f.replace('_', ' ')}: {profile[f]}"
            for f in ("gender", "caste", "marital_status", "age", "family_income", "residence", "education_level", "disability")
            if profile.get(f)
        ]
        known_desc = ", ".join(known_bits) if known_bits else "nothing yet"
        who_clause = (
            f"This need is for the citizen's {subject_desc}, not the citizen themselves. What's "
            f"already confirmed about the CITIZEN THEMSELVES (not {subject_desc}) so far: {known_desc}.\n"
            f"The eligibility text is the ONLY thing that decides who this specific restriction ({gap_desc}) "
            f"is actually about - do not guess or default, read it literally: if it describes a "
            f"requirement of the APPLICANT/PARENT (e.g. 'the applicant should be...'), that requirement "
            f"is about the CITIZEN, even though the overall need is for {subject_desc} - e.g. a scheme "
            "assisting a widowed parent with their child's marriage requires the PARENT to be widowed, "
            f"so that specific question must be addressed to the citizen, not the child. Only if the "
            f"eligibility text itself describes a requirement of the beneficiary/child/dependent does "
            f"this restriction mean {subject_desc}.\n"
            f"If you conclude the restriction is about the citizen, but what's already confirmed about "
            f"the citizen above directly contradicts it (e.g. it requires a woman/widow but the citizen "
            f"is already known to be male), then this scheme does not actually apply to this household - "
            f"say so briefly instead of asking a question whose answer is already known."
        )
    else:
        who_clause = (
            "Read the eligibility text to determine what it actually requires of the citizen "
            "(there is no family member involved in this need)."
        )
    prompt = (
        f'The citizen asked about: "{state["need_text"]}"\n'
        f'The best-matching scheme is specifically restricted on: {gap_desc}. Its actual eligibility '
        f'text: "{elig_text}"\n\n'
        f"{who_clause}\n\n"
        f"Normally: ask ONE short, natural, non-intrusive question {lang_instruction} SPECIFICALLY "
        f"about the {gap_desc} criterion identified above - not a different, narrower requirement "
        "mentioned elsewhere in the eligibility text (e.g. a specific field of study/institution/"
        "sub-condition), even if that seems more concretely checkable. The eligibility text is only "
        "there to help phrase the question accurately and to determine who it's about (per the "
        "paragraph above) - it does not change WHICH criterion the question is about. Found live "
        "(2026-08-11): a generic 'give me a scholarship' request with an unconfirmed education_level "
        "gap got asked 'is your PhD in sociology/law/economics/...' instead - a wildly narrower, "
        "far less likely assumption than the actual gap (are they even in higher education at all), "
        "purely because that scheme's eligibility text happened to be dominated by that specific PhD "
        "requirement. "
        "If the identified gap is caste or marital status - fields with SEVERAL possible values "
        "(general/OBC/SC/ST/PVTG for caste; married/never-married/widowed/divorced/separated for "
        "marital status) - phrase the question OPEN-ENDED, asking what category/status the RELEVANT "
        "PERSON belongs to, NOT a presumptuous yes/no naming the one specific value THIS scheme "
        "happens to require (avoid 'are you SC?'/'are you widowed?' framings - that presumes an "
        "answer and reads as leading). This open-ended phrasing follows the SAME who-it's-addressed-"
        "to rule as everything else here - if the restriction is about the citizen, ask 'aap kis "
        "caste category se hain?'; if it's about a family member (e.g. their son), ask about THAT "
        "person by name/relation, e.g. 'aapke bete ki caste category kya hai?', never a bare 'aap/"
        "aapki' that would wrongly imply it's asking about the citizen themselves. Learning their "
        "actual category also resolves eligibility for every OTHER caste- or marital-status-"
        "restricted scheme in the shortlist from this one answer, not just this scheme. For every "
        "other gap field (residence, education level, disability, gender), a direct yes/no about "
        "the specific requirement is fine, since those are simple binary confirmations, not "
        "multi-way categories. "
        "Address the question to whichever person you determined above the restriction "
        "is actually about, and mention that general options exist either way. Do NOT name the "
        "specific scheme. "
        "But if the paragraph above says this scheme doesn't actually apply, say that briefly instead "
        f"{lang_instruction} and mention general options exist - do not ask a question in that case. "
        "One or two sentences either way."
    )
    resp = get_llm().invoke(prompt)
    return {
        "last_reply": resp.content,
        "pending_need_text": state.get("need_text"),
        "clarify_attempts": state.get("clarify_attempts", 0) + 1,
    }


def stage_b_node(state: ConvState):
    shortlists = state.get("shortlists") or [{"need": None, "shortlist": state.get("shortlist") or []}]
    lang = _source_lang(state.get("reply_lang"))
    field_types = FIELD_TYPES_BY_INTENT.get(state.get("intent"), ["eligibility", "benefits"])

    svc = get_rag_service()
    chunks = []
    already_shown = list(state.get("already_shown_scheme_ids") or [])
    shown_schemes = []
    # Fetch detail for every need's top schemes, not just the first - a
    # message stating two unrelated needs (2026-08-10) used to only ever see
    # the first need's schemes in the final answer, since this only looped
    # over the single `shortlist`. shown_schemes records one entry per need
    # so a later followup ("pehli scheme"/"dusri scheme") has something real
    # to resolve against, instead of a single stale shown_scheme_id that only
    # ever reflected the first need.
    profile = state.get("profile", {})
    for entry in shortlists:
        need = entry.get("need") or {}
        facts = _subject_facts_for_need(need, profile)
        top = _prioritize_gap_free(entry["shortlist"] or [], facts, need.get("need_text", ""), 3)
        # Record EVERY scheme actually presented for this need, not just the
        # top one - found live (2026-08-11): a single need surfacing 3
        # candidate schemes together (the normal case, not just the multi-
        # need case) left shown_schemes with only 1 entry even though all 3
        # were fetched and shown in the answer below, so "dusri scheme"
        # (second scheme) had nothing real to resolve against.
        for s in top:
            shown_schemes.append({
                "scheme_id": s["scheme_id"],
                "scheme_name": s["scheme_name"],
                "need_text": need.get("need_text", ""),
            })
        for s in top:
            detail = svc.scheme_detail(s["scheme_id"], lang, field_types=field_types)
            if not detail and lang != "en":
                detail = svc.scheme_detail(s["scheme_id"], "en", field_types=field_types)
            chunks.extend(detail)
        if top and top[0]["scheme_id"] not in already_shown:
            already_shown.append(top[0]["scheme_id"])

    return {
        "detail_chunks": chunks,
        "chunk_cache": _update_chunk_cache(state.get("chunk_cache"), chunks),
        "shown_schemes": shown_schemes,
        "shown_scheme_id": shown_schemes[0]["scheme_id"] if shown_schemes else state.get("shown_scheme_id"),
        "shown_scheme_name": shown_schemes[0]["scheme_name"] if shown_schemes else state.get("shown_scheme_name"),
        "shown_scheme_lang": lang,
        "already_shown_scheme_ids": already_shown,
        "clarify_attempts": 0,  # a real answer is being delivered - the unresolved streak ends here
        "focused_scheme_ids": None,  # a genuinely fresh, possibly multi-scheme answer - no narrowing to carry over
    }


def show_more_node(state: ConvState):
    """'What other schemes am I eligible for' - a distinct request from a
    single-scheme followup (turn_type='followup_same_scheme'), answered from
    the shortlist already computed by the last Stage A run rather than a
    fresh search, since that data already exists (see ConvState.shortlist's
    docstring). Excludes every scheme already shown this conversation, not
    just the current one, so asking "what else" repeatedly doesn't repeat
    an earlier answer."""
    already_shown = set(state.get("already_shown_scheme_ids") or [])
    remaining = [s for s in (state.get("shortlist") or []) if s["scheme_id"] not in already_shown][:3]
    lang = _source_lang(state.get("reply_lang"))
    field_types = FIELD_TYPES_BY_INTENT.get(state.get("intent"), ["eligibility", "benefits"])

    svc = get_rag_service()
    chunks = []
    for s in remaining:
        detail = svc.scheme_detail(s["scheme_id"], lang, field_types=field_types)
        if not detail and lang != "en":
            detail = svc.scheme_detail(s["scheme_id"], "en", field_types=field_types)
        chunks.extend(detail)

    new_already_shown = list(already_shown)
    if remaining and remaining[0]["scheme_id"] not in new_already_shown:
        new_already_shown.append(remaining[0]["scheme_id"])

    shown_schemes = [
        {"scheme_id": s["scheme_id"], "scheme_name": s["scheme_name"], "need_text": state.get("need_text", "")}
        for s in remaining
    ]

    return {
        "detail_chunks": chunks,
        "chunk_cache": _update_chunk_cache(state.get("chunk_cache"), chunks),
        "shown_schemes": shown_schemes or state.get("shown_schemes"),
        "shown_scheme_id": remaining[0]["scheme_id"] if remaining else state.get("shown_scheme_id"),
        "shown_scheme_name": remaining[0]["scheme_name"] if remaining else state.get("shown_scheme_name"),
        "shown_scheme_lang": lang,
        "already_shown_scheme_ids": new_already_shown,
        "focused_scheme_ids": None,  # showing a fresh batch of alternatives - no narrowing to carry over
    }


def stage_b_followup_node(state: ConvState):
    lang = state.get("shown_scheme_lang", "en")
    field_types = FIELD_TYPES_BY_INTENT.get(state.get("intent"), ["eligibility", "benefits"])

    # Resolve which shown scheme(s) this followup actually means, rather than
    # always defaulting to a single shown_scheme_id - found live (2026-08-10):
    # when several schemes were shown at once, "pehli scheme"/"education wali"
    # style followups had nothing to resolve against and silently answered
    # about the wrong (stale, unrelated) scheme. followup_target is resolved
    # by extract_need_node from the actual message text against the real
    # list of shown schemes, not guessed positionally here.
    shown_schemes = state.get("shown_schemes") or []
    target_name = state.get("followup_target")
    targets = [s for s in shown_schemes if s["scheme_name"] == target_name] if target_name else []
    if not targets:
        targets = shown_schemes or [{"scheme_id": state["shown_scheme_id"], "scheme_name": state.get("shown_scheme_name")}]

    svc = get_rag_service()

    # Re-verify eligibility against whatever's been learned SINCE this
    # scheme was first shown, before presenting it again - a followup never
    # re-runs Stage A, so `targets` can be stale. Found live (2026-08-10): a
    # citizen who explicitly said "no disability" kept being shown a
    # disability-restricted scheme turn after turn, because nothing ever
    # re-checked it against that new fact.
    primary_need = (state.get("needs") or [None])[0]
    profile = state.get("profile", {})
    facts = _subject_facts_for_need(primary_need, profile)
    citizen = CitizenContext(
        caste=facts.get("caste"), gender=facts.get("gender"), marital_status=facts.get("marital_status"),
        age=facts.get("age"), family_income=facts.get("family_income"), residence=facts.get("residence"),
        education_level=facts.get("education_level"), disability=facts.get("disability"),
    )
    # For a family-member need, also check the citizen's OWN profile as an
    # alternate identity - mirrors stage_a_node's alt_citizen (same fail-open
    # "exclude only if NEITHER identity fits" reasoning). Found live
    # (2026-08-11): this re-check only ever looked at the family member's
    # facts, so a scheme whose applicant-only requirement was already
    # contradicted by the citizen's own known profile (e.g. married, not
    # widowed) still came back "still eligible" on a followup, even though a
    # fresh clarify_identity_node evaluation would correctly reject it.
    alt_citizen = None
    if primary_need and primary_need.get("about") == "family_member":
        alt_citizen = CitizenContext(
            caste=profile.get("caste"), gender=profile.get("gender"), marital_status=profile.get("marital_status"),
            age=profile.get("age"), family_income=profile.get("family_income"), residence=profile.get("residence"),
            education_level=profile.get("education_level"), disability=profile.get("disability"),
        )
    still_eligible, now_excluded = [], []
    for t in targets:
        meta = svc._meta_by_scheme_id.get(t["scheme_id"])
        fits = meta is None or eligibility_verdict(meta, citizen)[0] or (
            alt_citizen is not None and eligibility_verdict(meta, alt_citizen)[0]
        )
        if fits:
            still_eligible.append(t)
        else:
            now_excluded.append(t["scheme_id"])

    # If the citizen explicitly asked about ONE specific scheme (via
    # followup_target) and it turns out excluded, still fetch and present
    # ITS OWN detail so the answer can explain why - found live (2026-08-11):
    # "3rd scheme ki eligibility batao" asked about a scheme that's excluded
    # once the citizen's gender contradicts it, still_eligible came back
    # empty, and the answer silently talked about two OTHER schemes instead
    # of ever addressing the one actually asked about. Only falls back to
    # "explain the excluded one" when a SPECIFIC target was named - a plain
    # followup with no named target should still just quietly drop anything
    # excluded, not resurface it.
    to_fetch = still_eligible if (still_eligible or not target_name) else targets

    chunks = []
    for t in to_fetch:
        detail = svc.scheme_detail(t["scheme_id"], lang, field_types=field_types)
        if not detail and lang != "en":
            detail = svc.scheme_detail(t["scheme_id"], "en", field_types=field_types)
        chunks.extend(detail)

    # Purge any now-disconfirmed scheme from the PERSISTENT rolling cache -
    # it can still be fetched fresh above to explain THIS turn's answer, but
    # shouldn't linger in the cache afterward as if it were still a live
    # candidate.
    cache = [(doc, meta) for doc, meta in (state.get("chunk_cache") or []) if meta.get("scheme_id") not in now_excluded]
    cacheable_ids = {t["scheme_id"] for t in still_eligible}
    cacheable_chunks = [c for c in chunks if c[1].get("scheme_id") in cacheable_ids]

    # When the citizen explicitly narrowed to ONE specific scheme
    # (followup_target matched something), tell generate_node to restrict
    # itself to just that scheme - found live (2026-08-11): "tell me about
    # the last scheme" correctly narrowed THIS turn's own fetch, but
    # chunk_cache still held the other 2 previously-shown schemes from the
    # prior turn, and generate_node's "fill in recent context" merge
    # silently reintroduced them, defeating the narrowing entirely. Always
    # explicitly set (never left to the shallow-merge default) so a
    # narrowing from an earlier turn can't linger onto a later turn that
    # didn't narrow anything.
    focused_scheme_ids = [t["scheme_id"] for t in to_fetch] if target_name else None
    return {
        "detail_chunks": chunks,
        "chunk_cache": _update_chunk_cache(cache, cacheable_chunks),
        "focused_scheme_ids": focused_scheme_ids,
    }


def _structured_facts_summary(meta):
    """The free-text field chunks (description/benefits/eligibility prose)
    don't always mention every structured eligibility fact myScheme's API
    provides - e.g. a scheme's caste facet can include OBC/SC/ST while its
    free-text eligibility description only talks about income and
    institution type, never mentioning caste at all (verified 2026-08-05
    on Mukhya Mantri Sarvajan Uchcha Shiksha Chhatravriti Yojana). Stage A's
    filter already used these structured facts to decide this scheme is
    relevant - generation needs to see them too, or it answers blind to
    exactly the signal that made the scheme worth showing.

    Identity dimensions (caste/gender/marital_status) are ALWAYS stated
    explicitly, including when unrestricted ("open to all") - found live
    (2026-08-05): omitting caste entirely when it's 'all' left the LLM
    seeing silence on both the structured data AND the free text (which
    also doesn't mention caste), and it concluded "doesn't mention OBC"
    (hedging) instead of "no caste restriction, OBC qualifies" - silence
    is not the same as "no restriction" unless stated as such.
    """
    parts = []
    for key in ("caste", "gender", "marital_status"):
        val = meta.get(key) or "all"
        parts.append(
            f"{key.replace('_', ' ')}: open to all"
            if val == "all"
            else f"{key.replace('_', ' ')}: {val.replace('|', ', ')}"
        )
    for key in ("occupation_target_group", "benefit_type_fine"):
        val = meta.get(key)
        if val:
            parts.append(f"{key.replace('_', ' ')}: {val.replace('|', ', ')}")
    if meta.get("family_income_ceiling"):
        parts.append(f"family income ceiling: Rs {meta['family_income_ceiling']}")
    if meta.get("age_min") is not None or meta.get("age_max") is not None:
        parts.append(f"age range: {meta.get('age_min')}-{meta.get('age_max')}")
    return "; ".join(parts)


def _format_conversation_history(history):
    if not history:
        return ""
    lines = [f'Citizen: "{turn["user"]}"\nYou: "{turn["bot"]}"' for turn in history]
    return "Recent conversation (for context only - answer the CURRENT question, don't repeat old answers):\n" + "\n\n".join(lines) + "\n\n"


def _format_profile(profile):
    if not profile:
        return ""
    facts = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in profile.items())
    return f"What we know about this citizen: {facts}\n\n"


class SchemeAnswer(BaseModel):
    """Structured wrapper around generate_node's reply purely to know
    whether it asked a clarifying question - added 2026-08-16 so
    pending_need_text can be set the same way clarify_node/
    clarify_identity_node already do, giving a terse follow-up reply next
    turn something structured to anchor to (see generate_node's return)."""

    reply: str = Field(..., description="The actual reply to send to the citizen - identical in content to what a plain text reply would be.")
    asked_clarifying_question: bool = Field(
        ...,
        description=(
            "True if `reply` asks the citizen to confirm/provide ANY information needed to "
            "determine eligibility (per QUESTION LIMIT above) - i.e. it is NOT yet a complete, "
            "final answer. False if `reply` is a complete answer with no outstanding question "
            "(eligibility confirmed, or plainly stating nothing matches)."
        ),
    )


def generate_node(state: ConvState):
    # this turn's own retrieval is authoritative and always included; the
    # rolling cache only fills in recently-discussed schemes/fields THIS
    # turn's retrieval didn't happen to fetch (e.g. this turn only pulled
    # "documents" but the citizen's question also implicitly touches
    # "eligibility" discussed two turns ago) - current-turn chunks win on
    # any (scheme_id, field_type, lang) key collision.
    current = state.get("detail_chunks") or []
    current_keys = {_chunk_key(meta) for _, meta in current}
    cached_extra = [(doc, meta) for doc, meta in (state.get("chunk_cache") or []) if _chunk_key(meta) not in current_keys]

    # When the citizen just explicitly narrowed to ONE specific scheme (see
    # stage_b_followup_node's focused_scheme_ids), restrict the cache
    # fill-in to that same scheme - found live (2026-08-11): "tell me about
    # the last scheme" correctly narrowed this turn's own fetch, but the
    # cache still held the other previously-shown schemes and got merged
    # back in here, silently undoing the narrowing the citizen just asked
    # for. Every other path leaves this unset, so the normal "fill in
    # recent context across multiple schemes" behavior is unaffected.
    focused_scheme_ids = state.get("focused_scheme_ids")
    if focused_scheme_ids:
        cached_extra = [(doc, meta) for doc, meta in cached_extra if meta.get("scheme_id") in focused_scheme_ids]

    chunks = current + cached_extra

    structured_by_scheme = {}
    for _, meta in chunks:
        structured_by_scheme.setdefault(meta["scheme_name"], meta)
    structured_text = "\n".join(
        f"[{name}] structured eligibility data: {_structured_facts_summary(meta)}"
        for name, meta in structured_by_scheme.items()
    )

    chunks_text = "\n\n".join(
        f"[{meta['scheme_name']} / {meta['field_type']}]\n{doc}" for doc, meta in chunks
    )

    # Pre-synthesized short cards (name + who it's for + one-line
    # description), reused from Stage A's own search index rather than
    # having the model improvise a summary from the full eligibility/
    # benefit text below - see "RESPONSE ORDER"/clarifying-question
    # instructions further down for when this gets used INSTEAD of the
    # full text, not alongside it.
    svc = get_rag_service()
    discovery_cards_text = "\n".join(
        f"[{name}] {svc.get_discovery_card(meta['scheme_id'], meta['lang'])}"
        for name, meta in structured_by_scheme.items()
    )

    lang_instruction = _lang_instruction(state.get("reply_lang"))

    needs = state.get("needs") or [{"need_text": state.get("need_text"), "about": "self", "subject_relation": None}]
    if len(needs) > 1:
        needs_lines = []
        for i, n in enumerate(needs, 1):
            who = f" (for the citizen's {n.get('subject_relation') or 'family member'}, not the citizen themselves)" if n.get("about") == "family_member" else ""
            needs_lines.append(f'{i}. "{n["need_text"]}"{who}')
        question_block = (
            "Citizen's CURRENT questions (there is more than one - answer EACH separately and "
            "clearly under its own heading, do not merge them into one answer):\n" + "\n".join(needs_lines)
        )
    else:
        who = f" (for the citizen's {needs[0].get('subject_relation') or 'family member'}, not the citizen themselves)" if needs[0].get("about") == "family_member" else ""
        question_block = f'Citizen\'s CURRENT question: "{needs[0]["need_text"]}"{who}'

    # need_text is a deliberately distilled search phrase (see NeedItem's own
    # field description) - it can lose specific facts the citizen just
    # confirmed, especially ones about a referenced third party supporting
    # THEIR OWN need (e.g. "but my parents are construction worker" for a
    # scheme requiring the applicant's PARENT to be one) - there's no
    # structured field for that (it isn't the need's own subject, so
    # subject_facts doesn't apply, and it isn't the citizen's own fact
    # either). Found live (2026-08-11): need_text collapsed that exact
    # message to 'higher studies scholarship', and the answer confidently
    # denied eligibility, having never seen the confirmation at all. Passing
    # the literal raw message lets the model use whatever specific facts
    # were actually stated, however they relate to the scheme's real
    # eligibility text, without needing a new field for every possible kind
    # of referenced third party.
    question_block += f'\nCitizen\'s exact words this turn: "{state.get("raw_message", "")}"'

    generate_clarify_rounds = state.get("generate_clarify_rounds", 0)
    hard_stop_block = (
        f"HARD STOP: this need has already gone through {generate_clarify_rounds} round(s) of your "
        "own clarifying questions without landing on an answer. You MUST give your FINAL best-effort "
        "answer this turn - set asked_clarifying_question to false, recommend the most likely-fitting "
        "scheme(s) from what is already known, and clearly caveat anything still unconfirmed. Do NOT "
        "ask another question, even a reasonable one. Found live (2026-08-17): a business-loan need "
        "was asked about across SEVEN consecutive turns and never once reached an actual "
        "recommendation - one of those questions even repeated something already answered several "
        "turns earlier, since only a few recent turns of conversation history are ever kept.\n\n"
        if generate_clarify_rounds >= 2 else ""
    )

    prompt = (
        "Answer the citizen's question using ONLY the information below (both the "
        "structured eligibility data AND the scheme text). Do not invent eligibility "
        "criteria, amounts, or steps not present in either. If it doesn't fully answer, "
        "say what's missing rather than guessing.\n\n"
        "The structured eligibility data is a COARSE category tag (e.g. a single caste value), "
        "not a substitute for the scheme text - the scheme text is the actual, complete, "
        "government-written eligibility description, and is authoritative when the two disagree. "
        "Confirmed live (2026-08-11): a real scheme's structured caste tag said 'SC', but its own "
        "written eligibility text named a specific, different list of OBC sub-castes with no "
        "mention of SC at all - a genuine contradiction in the source government data itself, not "
        "something invented here. When the tag and the written text disagree like this, do NOT "
        "call the scheme a 'match' or 'best match' based on the tag - say plainly that the tag and "
        "the actual written criteria disagree, and that eligibility depends on the WRITTEN "
        "criteria, not the tag. If the citizen has already stated a fact the written text clearly "
        "excludes (e.g. they named a caste that isn't in the text's specific list), say so directly "
        "and drop the scheme from consideration immediately - do not keep re-asking a question "
        "about the same requirement the citizen has already effectively answered by stating an "
        "incompatible fact.\n\n"
        "Never assume or state the citizen's OWN caste, disability status, gender, or any other "
        "identity/category fact unless it is explicitly listed in 'What we know about this "
        "citizen' below - if a scheme's eligibility depends on something not confirmed there, say "
        "plainly that it isn't confirmed yet (e.g. 'this requires knowing your caste/disability "
        "status, which hasn't been mentioned') rather than labeling the citizen into a category "
        "('as a general-category applicant', 'since you have no disability') that was never stated. "
        "If NONE of the schemes below are confirmed to match what's actually known about the "
        "citizen, say so directly and plainly rather than repeatedly re-listing the same "
        "conditional options turn after turn.\n\n"
        "If the citizen's exact words this turn are a short answer (yes/no/haan/nahi) to your own "
        "PREVIOUS reply, and that previous reply asked about TWO OR MORE separate conditions at "
        "once, a bare 'nahi'/'no' is AMBIGUOUS about which one(s) it's denying - do not silently "
        "assume it means all of them, and do not just repeat the same conditional information "
        "again unchanged. Instead ask ONE short follow-up naming the specific conditions and "
        "asking which apply, so the next answer is unambiguous.\n\n"
        "RESPONSE ORDER: if this answer needs a clarifying question (per QUESTION LIMIT below), lead "
        "with a SHORT list of the 2-3 relevant scheme NAMES (one line each, from 'Scheme summaries' "
        "below - not the full eligibility/benefit text), then put the clarifying question(s) AFTER "
        "that, at the end. This reads as a natural conversation - here's what's out there, and here's "
        "what I need to know to narrow it down - rather than opening with a bare question with no "
        "context. Reversed 2026-08-17 from an earlier version of this instruction that put the "
        "question first - that was itself a fix for a different problem (a long wall of full scheme "
        "text/conditions burying the question at the very end), which QUESTION LIMIT and SCHEME "
        "DETAIL LEVEL below already solve on their own by capping questions at 2 and keeping scheme "
        "mentions short - so leading with a few short names no longer buries anything.\n\n"
        "QUESTION LIMIT: ask AT MOST 2 clarifying questions in this single reply, even if more "
        "facts remain unconfirmed across the schemes below - pick the 2 most important/narrowing "
        "ones. Found live (2026-08-14): a vague scholarship need surfaced FIVE separate questions "
        "(residence, age, class/degree, disability, income) in one reply, all before any scheme "
        "info - overwhelming, and answering all 5 at once isn't how a real conversation works. Do "
        "NOT try to compensate by cramming multiple conditions into one combined question either - "
        "each of the (at most 2) questions should be about ONE thing. Whatever remains unconfirmed "
        "after these 2 are answered will naturally surface as further questions on a LATER turn, "
        "once this turn's answers are known - you do not need to mention or promise that here.\n\n"
        "QUESTION PRIORITY: when choosing WHICH (at most 2) unconfirmed facts to ask about, prefer "
        "broad, low-sensitivity ones - age, what class/degree they're studying in, family income, "
        "residence - over narrow, identity-defining, or sensitive ones - disability status, caste, "
        "marital status, gender. Ask about a sensitive/narrow fact FIRST only if literally no broad "
        "fact remains unconfirmed for any shortlisted scheme. Found live (2026-08-16): 'paisa chahiye "
        "beti ki padhai ke liye' (a plain request for help with a daughter's education) got asked "
        "'does she have a 40%+ disability certificate' as the OPENING question, purely because one of "
        "three shortlisted scholarships happened to be disability-gated - presumptuous and needlessly "
        "personal as a first question when her age or which class she's in would narrow things down "
        "just as well without asking about a sensitive medical/identity fact up front.\n\n"
        "SCHEME DETAIL LEVEL: if this reply includes a clarifying question (per the above), describe "
        "each scheme using ONLY its short description below (under 'Scheme summaries') - just what "
        "it's for, in a line or two - NOT the full eligibility criteria or benefit text ('Scheme "
        "text' below). Dumping full eligibility/benefit detail for schemes whose fit isn't even "
        "confirmed yet buries the actual question under paragraphs the citizen can't act on. Once "
        "no more clarifying questions are needed (eligibility is confirmed, or you're giving a final "
        "answer), use the full 'Scheme text' as before - this limit is ONLY for a reply that also "
        "asks something.\n\n"
        "ANSWER EARLY: prefer giving a caveated best-guess answer over asking yet another question "
        "whenever you already have enough to make a reasonable recommendation. Only ask about a fact "
        "if NOT knowing it would change WHICH scheme applies, or whether the citizen is eligible at "
        "all - not for minor refinements (e.g. whether they need special machinery vs a general "
        "setup loan doesn't usually change scheme eligibility, only which scheme is a marginally "
        "better fit - that's exactly the kind of thing to note as a caveat, not gate the whole answer "
        "behind). When you do answer without every detail confirmed, say plainly what you assumed or "
        "couldn't confirm, rather than presenting it as certain.\n\n"
        f"{hard_stop_block}"
        f"{_format_conversation_history(state.get('conversation_history'))}"
        f"{_format_profile(state.get('profile'))}"
        f"{question_block}\n\n"
        f"Structured eligibility data:\n{structured_text}\n\n"
        f"Scheme summaries (short - use these instead of 'Scheme text' when asking a clarifying "
        f"question):\n{discovery_cards_text}\n\n"
        f"Scheme text (full detail - use once no clarifying question is needed):\n{chunks_text}\n\n"
        f"{lang_instruction} Keep it concise and practical."
    )
    result = get_llm().with_structured_output(SchemeAnswer).invoke(prompt)
    # Mechanically enforced regardless of what the model decided - the
    # HARD STOP prompt instruction above asks nicely, but prompt-following
    # isn't guaranteed, and the whole point of this cap is a GUARANTEE the
    # loop terminates, not just a strong hint. Once generate_clarify_rounds
    # already hit the cap coming into this turn, this turn is forced to
    # count as a final answer no matter what asked_clarifying_question says.
    asked_question = result.asked_clarifying_question and generate_clarify_rounds < 2
    # Set the SAME way clarify_node/clarify_identity_node already do, so a
    # terse follow-up reply ("nahi"/"18 saal") next turn has something
    # structured to anchor to - found live (2026-08-16): generate_node's own
    # clarifying questions had NO such anchor (only clarify_node/
    # clarify_identity_node set this), so a citizen's direct "nahi" to a
    # disability question never made it into subject_facts at all, and the
    # same question got asked again two turns later despite already being
    # answered. Cleared (None) when this reply didn't ask anything, so a
    # later unrelated message doesn't get wrongly treated as answering a
    # question that was never actually asked.
    return {
        "last_reply": result.reply,
        "pending_need_text": state.get("need_text") if asked_question else None,
        "generate_clarify_rounds": (generate_clarify_rounds + 1) if asked_question else 0,
    }


def chitchat_node(state: ConvState):
    """Fallback only - the normal path composes the chitchat reply directly
    inside extract_need_node (NeedExtraction.chitchat_reply) and skips this
    node entirely via route_after_extract. This only runs if that field
    came back empty despite intent='chitchat'."""
    lang_instruction = _lang_instruction(state.get("reply_lang"))
    prompt = (
        f'The citizen said: "{state["normalized_message"]}"\n'
        f"This isn't a scheme/job question. Reply briefly and warmly. {lang_instruction} "
        "Mention you can help them find Rajasthan government schemes and jobs."
    )
    resp = get_llm().invoke(prompt)
    return {"last_reply": resp.content}


# ---------------------------------------------------------------------------
# Job-vacancy domain (parallel to the scheme Stage A/B/generate nodes above -
# see app/job_service.py's docstring for why this is a separate path rather
# than routed through RagService's eligibility-gating machinery)
# ---------------------------------------------------------------------------

def stage_a_jobs_node(state: ConvState):
    primary_need = (state.get("needs") or [None])[0]
    need_text = primary_need["need_text"] if primary_need else state.get("need_text", "")
    location = primary_need.get("job_location") if primary_need else None
    qualification = primary_need.get("job_qualification") if primary_need else None
    min_salary = primary_need.get("job_min_salary") if primary_need else None
    turn_type = state.get("turn_type")

    svc = get_job_service()
    already_shown = list(state.get("already_shown_job_ids") or [])
    # "show other jobs" excludes what's already been shown; a fresh/different
    # search (even a repeat of the same title) should not - a citizen asking
    # about "Patwari" again later should still see it.
    exclude_ids = already_shown if turn_type == "wants_alternatives" else None
    results = svc.search(
        need_text, location=location, exclude_ids=exclude_ids, top_n=5,
        qualification=qualification, min_salary=min_salary,
    )

    shown_jobs = [{"job_id": j["id"], "title": j["title"], "department": j["department"]} for j in results]
    # Record EVERY job shown this turn, not just the first - a "show more"
    # excludes already_shown_job_ids, so if only the top result were tracked,
    # postings #2-5 from an earlier batch could reappear in a later "show
    # more" call (found testing this directly: "Police Constable" shown as
    # the #2 result in one turn reappeared in the very next "aur konsi jobs
    # hai" call, since only the #1 result's id had been recorded as shown).
    for j in results:
        if j["id"] not in already_shown:
            already_shown.append(j["id"])

    return {
        "job_shortlist": results,
        "shown_jobs": shown_jobs,
        "shown_job_id": results[0]["id"] if results else None,
        "already_shown_job_ids": already_shown,
    }


def stage_b_followup_jobs_node(state: ConvState):
    shown_jobs = state.get("shown_jobs") or []
    target_title = state.get("followup_target")
    targets = [j for j in shown_jobs if j["title"] == target_title] if target_title else []
    if not targets:
        targets = shown_jobs or ([{"job_id": state["shown_job_id"]}] if state.get("shown_job_id") else [])

    svc = get_job_service()
    jobs = [svc.get(t["job_id"]) for t in targets]
    jobs = [j for j in jobs if j]
    return {"job_shortlist": jobs}


_JOB_FIELD_FOCUS_BY_INTENT = {
    "how_to_apply": "focus on the official application portal link and what documents are needed to apply.",
    "documents_question": "focus ONLY on exactly what documents are needed.",
    "ask_detail": "focus on qualification required, salary, vacancies, and the application deadline.",
}


def generate_jobs_node(state: ConvState):
    lang_instruction = _lang_instruction(state.get("reply_lang"))
    jobs = state.get("job_shortlist") or []

    if not jobs:
        prompt = (
            f'The citizen asked about a government job: "{state.get("need_text", "")}"\n'
            f"No matching postings were found. Say so briefly {lang_instruction} and suggest they "
            "check back later or ask about a different post/location. One or two sentences."
        )
        resp = get_llm().invoke(prompt)
        return {"last_reply": resp.content}

    jobs_desc = "\n".join(
        f"- {j['title']} ({j['department']}), location: {j['location']}, status: {j['status']}, "
        f"vacancies: {j['vacancies']}, qualification required: {j['qualification']}, "
        f"salary: {j['salary']}, applications open {j['startDate']} to {j['lastDate']}, "
        f"documents needed: {', '.join(j['documentsNeeded'])}, "
        f"application portal: {j['officialApplicationPortal']}"
        for j in jobs
    )
    focus = _JOB_FIELD_FOCUS_BY_INTENT.get(state.get("intent"), "give a clear overview of the matching postings.")
    prompt = (
        f"Today's date is {date.today().isoformat()}.\n"
        f'The citizen asked: "{state.get("need_text", "")}"\n'
        f"Matching government job posting(s):\n{jobs_desc}\n\n"
        f"Answer their question {lang_instruction} using ONLY this data - never invent a detail "
        f"not listed above. {focus} Use today's date above to reason correctly about whether the "
        "application window has already started, is upcoming, or has already closed - do not guess "
        "this from the dates alone without comparing them to today. The 'status' field is the "
        "authoritative signal for whether it's open ('active'), not yet open ('coming soon'), or "
        "over ('closed') - if your date reasoning and the status field seem to disagree, trust "
        "status and phrase the reply around it rather than asserting a contradiction. Format as a "
        "short list if there are multiple postings."
    )
    resp = get_llm().invoke(prompt)
    return {"last_reply": resp.content}


_NO_MATCH_TEXT = {
    "hindi": (
        "माफ़ कीजिए, मुझे आपके लिए कोई योजना नहीं मिली। कृपया थोड़ा और बताइए कि आपको किस चीज़ के लिए "
        "मदद चाहिए।"
    ),
    "hinglish": (
        "Maaf kijiye, mujhe aapke liye koi scheme nahi mili. Kripya thoda aur bataiye "
        "ki aapko kis cheez ke liye madad chahiye."
    ),
    "english": "Sorry, I couldn't find a matching scheme. Could you tell me a bit more about "
    "what kind of help you need?",
}


def no_match_node(state: ConvState):
    text = _NO_MATCH_TEXT.get(state.get("reply_lang"), _NO_MATCH_TEXT["english"])
    return {"last_reply": text}


# ---------------------------------------------------------------------------
# Jan Aadhaar citizen-status verification sub-flow (mock, 2026-08-14) - a
# scripted 3-step OTP-consent dialogue (member choice -> OTP -> fetch),
# mirroring the real Jan Aadhaar Integration Document v1.8 flow. Every
# message in it is parsed deterministically via app/pii_utils.py, never by
# an LLM - see route_after_normalize/route_after_extract above for how turns
# get routed here instead of through the normal extraction pipeline.
# ---------------------------------------------------------------------------

_ASK_ID_TEXT = {
    "hindi": "आपकी स्थिति जानने के लिए मुझे आपका Jan Aadhaar verify करना होगा। कृपया अपना 10 अंकों का Jan Aadhaar ID बताएं।",
    "hinglish": "Aapki status jaanne ke liye mujhe aapka Jan Aadhaar verify karna hoga. Kripya apna 10 anko ka Jan Aadhaar ID bataiye.",
    "english": "To check your status, I'll need to verify your Jan Aadhaar. Please share your 10-digit Jan Aadhaar ID.",
}
_ID_INVALID_TEXT = {
    "hindi": "माफ़ कीजिए, यह एक मान्य Jan Aadhaar ID नहीं लग रहा। कृपया सही 10 अंकों का ID दोबारा भेजें।",
    "hinglish": "Maaf kijiye, yeh valid Jan Aadhaar ID nahi lag raha. Kripya sahi 10 anko ka ID dobara bhejein.",
    "english": "Sorry, that doesn't look like a valid Jan Aadhaar ID. Please send the correct 10-digit ID again.",
}
_ID_NOT_FOUND_TEXT = {
    "hindi": "माफ़ कीजिए, इस Jan Aadhaar ID से कोई रिकॉर्ड नहीं मिला। कृपया ID दोबारा जांच कर बताएं, या कुछ और पूछें।",
    "hinglish": "Maaf kijiye, is Jan Aadhaar ID se koi record nahi mila. Kripya ID dobara check karke bataiye, ya kuch aur pooch sakte hain.",
    "english": "Sorry, no record was found for that Jan Aadhaar ID. Please double-check and try again, or ask me something else.",
}
_MEMBER_CHOICE_HEADER = {
    "hindi": "आपके Jan Aadhaar में ये सदस्य मिले हैं। कृपया बताएं यह जानकारी किसके लिए चाहिए (नंबर बताएं):",
    "hinglish": "Aapke Jan Aadhaar mein ye members mile hain. Kripya bataiye yeh jaankari kiske liye chahiye (number bataiye):",
    "english": "Your Jan Aadhaar has these family members - please tell me which one this is for (reply with the number):",
}
# Appended after the member list every time (2026-08-14) - a relative living
# in a DIFFERENT household has their own separate Jan Aadhaar family ID (per
# janaadhaar.rajasthan.gov.in's own FAQ: a split household gets "a new Jan
# Aadhaar family ID"), so they simply won't appear in THIS list at all - the
# member list itself carries no relation metadata (just masked names), so
# only the citizen can recognize whether the person they mean is even here.
# Offered proactively rather than only after a failed match, so a citizen
# asking about someone outside their own household doesn't have to guess
# their way through a wrong list first.
_MEMBER_CHOICE_FOOTER = {
    "hindi": "(अगर इनमें से कोई नहीं है, तो बताएं और उनका अपना Jan Aadhaar ID दें।)",
    "hinglish": "(Agar inme se koi nahi hai, to bataiye aur unka apna Jan Aadhaar ID dijiye.)",
    "english": "(If none of these are who you mean, let me know and share their own Jan Aadhaar ID instead.)",
}
_NOT_IN_LIST_TEXT = {
    "hindi": "ठीक है। अगर वह व्यक्ति आपके साथ एक ही Jan Aadhaar परिवार में नहीं हैं, तो कृपया उनका अपना 10 अंकों का Jan Aadhaar ID बताएं।",
    "hinglish": "Theek hai. Agar wo vyakti aapke saath ek hi Jan Aadhaar parivar mein nahi hain, to kripya unka apna 10 anko ka Jan Aadhaar ID bataiye.",
    "english": "Understood. If they're not part of your own Jan Aadhaar household, please share their own 10-digit Jan Aadhaar ID instead.",
}
_NONE_OF_THESE_PHRASES = (
    "inme se koi nahi", "in mein se koi nahi", "koi nahi hai", "koi bhi nahi",
    "इनमें से कोई नहीं", "कोई नहीं है", "none of these", "not in the list",
    "not on this list", "alag rehte", "अलग रहते",
)
_ASK_OTP_TEXT = {
    "hindi": "आपके रजिस्टर्ड मोबाइल नंबर पर एक OTP भेज दिया गया है। कृपया वह 6 अंकों का OTP यहां बताएं।",
    "hinglish": "Aapke registered mobile number par ek OTP bhej diya gaya hai. Kripya wo 6 anko ka OTP yahan bataiye.",
    "english": "An OTP has been sent to your registered mobile number. Please share that 6-digit OTP here.",
}
_OTP_INVALID_TEXT = {
    "hindi": "यह OTP सही नहीं लग रहा। कृपया दोबारा सही 6 अंकों का OTP भेजें।",
    "hinglish": "Yeh OTP sahi nahi lag raha. Kripya dobara sahi 6 anko ka OTP bhejein.",
    "english": "That OTP doesn't look correct. Please send the correct 6-digit OTP again.",
}
_FLOW_ABORTED_TEXT = {
    "hindi": "माफ़ कीजिए, वेरिफिकेशन पूरा नहीं हो पाया। कृपया थोड़ी देर बाद दोबारा कोशिश करें, या मुझसे कुछ और पूछें।",
    "hinglish": "Maaf kijiye, verification poora nahi ho paaya. Kripya thodi der baad dobara koshish karein, ya mujhse kuch aur poochein.",
    "english": "Sorry, verification couldn't be completed. Please try again in a little while, or ask me something else.",
}

_MEMBER_ORDINAL_WORDS = {
    "1": 0, "first": 0, "pehla": 0, "pehli": 0, "पहला": 0, "पहली": 0,
    "2": 1, "second": 1, "dusra": 1, "dusri": 1, "दूसरा": 1, "दूसरी": 1,
    "3": 2, "third": 2, "teesra": 2, "teesri": 2, "तीसरा": 2, "तीसरी": 2,
}

_MAX_JANAADHAAR_ATTEMPTS = 3


def _pick_text(reply_lang, texts):
    return texts.get(reply_lang, texts["english"])


def _format_member_choice_text(reply_lang, members):
    header = _pick_text(reply_lang, _MEMBER_CHOICE_HEADER)
    lines = "\n".join(f"{i + 1}. {m['masked_name']}" for i, m in enumerate(members))
    footer = _pick_text(reply_lang, _MEMBER_CHOICE_FOOTER)
    return f"{header}\n{lines}\n{footer}"


def _is_none_of_these(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _NONE_OF_THESE_PHRASES)


def _resolve_member_choice(text, members):
    # Tokenize rather than requiring the WHOLE trimmed message to match one
    # of the ordinal words exactly - found live (2026-08-14): "dusra wala"
    # ("the second one", a natural Hinglish phrasing with a trailing "wala")
    # failed an exact-match lookup even though "dusra" itself is a known
    # ordinal, because the full string "dusra wala" isn't itself a dict key.
    words = re.findall(r"[\wऀ-ॿ]+", text.lower())
    for word in words:
        if word in _MEMBER_ORDINAL_WORDS:
            idx = _MEMBER_ORDINAL_WORDS[word]
            if idx < len(members):
                return members[idx]
    digits = pii_utils.digits_only(text)
    if digits:
        idx = int(digits) - 1
        if 0 <= idx < len(members):
            return members[idx]
    return None


def _abort_janaadhaar_flow(reply_lang):
    return {
        "last_reply": _pick_text(reply_lang, _FLOW_ABORTED_TEXT),
        "pending_janaadhaar_step": None,
        "janaadhaar_member_id": None,
        "janaadhaar_tid": None,
        "janaadhaar_members": [],
        "janaadhaar_attempts": 0,
    }


def janaadhaar_handle_node(state: ConvState):
    step = state.get("pending_janaadhaar_step")
    reply_lang = state.get("reply_lang")
    attempts = state.get("janaadhaar_attempts", 0)
    svc = get_janaadhaar_service()

    if step is None:
        # Fresh trigger from a citizen_status need - THIS turn's own message
        # (the status question itself) already safely went through
        # extract_profile/extract_need, nothing to redact yet.
        return {
            "last_reply": _pick_text(reply_lang, _ASK_ID_TEXT),
            "pending_janaadhaar_step": "awaiting_id",
            "janaadhaar_attempts": 0,
        }

    # From here on, THIS turn's raw_message is itself a credential attempt
    # (an ID, a member choice, or an OTP) - never let it reach
    # conversation_history/logs unredacted, regardless of which branch below
    # actually runs.
    redacted = {"redacted_raw_message": "[Jan Aadhaar verification input redacted]"}

    if step == "awaiting_id":
        jan_id = pii_utils.extract_jan_aadhaar_id(state["raw_message"])
        if not jan_id:
            if attempts + 1 >= _MAX_JANAADHAAR_ATTEMPTS:
                return {**redacted, **_abort_janaadhaar_flow(reply_lang)}
            return {**redacted, "last_reply": _pick_text(reply_lang, _ID_INVALID_TEXT), "janaadhaar_attempts": attempts + 1}

        result = svc.fetch_member_list(jan_id)
        if result["status"] != "JAN_200" or not result["members"]:
            return {**redacted, **_abort_janaadhaar_flow(reply_lang)}

        members = result["members"]
        if len(members) == 1:
            otp_result = svc.generate_otp(members[0]["member_id"])
            return {
                **redacted,
                "last_reply": _pick_text(reply_lang, _ASK_OTP_TEXT),
                "pending_janaadhaar_step": "awaiting_otp",
                "janaadhaar_member_id": members[0]["member_id"],
                "janaadhaar_tid": otp_result["tid"],
                "janaadhaar_members": members,
                "janaadhaar_attempts": 0,
            }
        return {
            **redacted,
            "last_reply": _format_member_choice_text(reply_lang, members),
            "pending_janaadhaar_step": "awaiting_member_choice",
            "janaadhaar_members": members,
            "janaadhaar_attempts": 0,
        }

    if step == "awaiting_member_choice":
        members = state.get("janaadhaar_members") or []
        if _is_none_of_these(state["raw_message"]):
            # The person they mean isn't in THIS family's list at all - most
            # likely a relative in a different household (their own,
            # separate Jan Aadhaar family ID) rather than a resolution
            # failure - restart the flow asking for THAT ID instead of
            # re-showing a list that can never contain them.
            return {
                **redacted,
                "last_reply": _pick_text(reply_lang, _NOT_IN_LIST_TEXT),
                "pending_janaadhaar_step": "awaiting_id",
                "janaadhaar_members": [],
                "janaadhaar_attempts": 0,
            }
        chosen = _resolve_member_choice(state["raw_message"], members)
        if not chosen:
            if attempts + 1 >= _MAX_JANAADHAAR_ATTEMPTS:
                return {**redacted, **_abort_janaadhaar_flow(reply_lang)}
            return {
                **redacted,
                "last_reply": _format_member_choice_text(reply_lang, members),
                "janaadhaar_attempts": attempts + 1,
            }

        otp_result = svc.generate_otp(chosen["member_id"])
        return {
            **redacted,
            "last_reply": _pick_text(reply_lang, _ASK_OTP_TEXT),
            "pending_janaadhaar_step": "awaiting_otp",
            "janaadhaar_member_id": chosen["member_id"],
            "janaadhaar_tid": otp_result["tid"],
            "janaadhaar_attempts": 0,
        }

    if step == "awaiting_otp":
        otp = pii_utils.extract_otp(state["raw_message"])
        if not otp:
            if attempts + 1 >= _MAX_JANAADHAAR_ATTEMPTS:
                return {**redacted, **_abort_janaadhaar_flow(reply_lang)}
            return {**redacted, "last_reply": _pick_text(reply_lang, _OTP_INVALID_TEXT), "janaadhaar_attempts": attempts + 1}

        result = svc.validate_otp_and_fetch(state.get("janaadhaar_member_id"), state.get("janaadhaar_tid"), otp)
        if result["status"] != "JAN_200":
            if attempts + 1 >= _MAX_JANAADHAAR_ATTEMPTS:
                return {**redacted, **_abort_janaadhaar_flow(reply_lang)}
            return {**redacted, "last_reply": _pick_text(reply_lang, _OTP_INVALID_TEXT), "janaadhaar_attempts": attempts + 1}

        data = result["data"]
        profile_fields = {k: v for k, v in data.items() if k in _PROFILE_FIELD_KEYS and v is not None}
        merged_profile = dict(state.get("profile", {}))
        merged_profile.update(profile_fields)

        # Entitlement Details is a SEPARATE fetch from the identity data
        # above (unverified schema - see janaadhaar_mock_data.
        # handle_entitlement_details) - this is what actually carries
        # per-scheme status (enrolled/approved/paid, last payment amount/
        # date), not the identity fetch itself.
        entitlement_result = svc.get_entitlement_details(state.get("janaadhaar_member_id"))
        entitlements = entitlement_result.get("entitlements", [])

        # Only entitlements (never the ID/OTP, which never reached this
        # point as anything but a local variable) go into this prompt.
        lang_instruction = _lang_instruction(reply_lang)
        prompt = (
            f'The citizen asked about their own status: "{state.get("need_text", "")}"\n'
            f"Their identity has just been verified via Jan Aadhaar OTP. Their scheme entitlements:\n"
            f"{entitlements}\n\n"
            f"Answer their question directly using ONLY this data {lang_instruction} - match it to "
            "whichever scheme they actually asked about (by name, even if phrased differently, "
            "abbreviated, or in Hindi/Hinglish). If they asked generally what they're enrolled in "
            "rather than about one specific scheme, briefly list all of the entitlements above. If "
            "NONE of the entitlements match what they asked about, say plainly that no record was "
            "found for that specific scheme, rather than guessing from an unrelated one. Do not "
            "invent any detail not listed above. Keep it to one or two sentences unless listing "
            "multiple entitlements. Do not mention Jan Aadhaar, OTP, or the verification process in "
            "this reply - just answer their status question naturally, as if you already knew this "
            "about them."
        )
        resp = get_llm().invoke(prompt)
        return {
            **redacted,
            "last_reply": resp.content,
            "profile": merged_profile,
            "janaadhaar_status_data": entitlements,
            "pending_janaadhaar_step": None,
            "janaadhaar_member_id": None,
            "janaadhaar_tid": None,
            "janaadhaar_members": [],
            "janaadhaar_attempts": 0,
        }

    # Defensive fallback - shouldn't be reachable given the three states above.
    return {**redacted, **_abort_janaadhaar_flow(reply_lang)}


def finalize_node(state: ConvState):
    """Every path (generate/clarify/clarify_identity/chitchat/no_match) routes
    through here before END, so conversation_history updates consistently
    regardless of which one produced the reply - rather than duplicating
    this update in five different terminal nodes."""
    history = list(state.get("conversation_history") or [])
    # redacted_raw_message (set by janaadhaar_handle_node) stands in for the
    # real message whenever this turn was a Jan Aadhaar ID/OTP attempt - this
    # history gets replayed into FUTURE extract_profile/extract_need/generate
    # prompts, so persisting the raw value here would leak it to an LLM one
    # turn later even if this turn's own handling never did.
    persisted_message = state.get("redacted_raw_message") or state["raw_message"]
    history.append({"user": persisted_message, "bot": state.get("last_reply", "")})
    return {"conversation_history": history[-MAX_HISTORY_TURNS:]}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(ConvState)
    graph.add_node("normalize", normalize_node)
    graph.add_node("extract_profile", extract_profile_node)
    graph.add_node("extract_need", extract_need_node)
    graph.add_node("stage_a", stage_a_node)
    graph.add_node("stage_b", stage_b_node)
    graph.add_node("stage_b_followup", stage_b_followup_node)
    graph.add_node("show_more", show_more_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("clarify_identity", clarify_identity_node)
    graph.add_node("chitchat", chitchat_node)
    graph.add_node("no_match", no_match_node)
    graph.add_node("generate", generate_node)
    graph.add_node("stage_a_jobs", stage_a_jobs_node)
    graph.add_node("stage_b_followup_jobs", stage_b_followup_jobs_node)
    graph.add_node("generate_jobs", generate_jobs_node)
    graph.add_node("janaadhaar_handle", janaadhaar_handle_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("normalize")
    graph.add_conditional_edges(
        "normalize",
        route_after_normalize,
        {"janaadhaar_handle": "janaadhaar_handle", "extract_profile": "extract_profile"},
    )
    graph.add_edge("extract_profile", "extract_need")
    graph.add_conditional_edges(
        "extract_need",
        route_after_extract,
        {
            "stage_b_followup": "stage_b_followup",
            "show_more": "show_more",
            "chitchat": "chitchat",
            "finalize": "finalize",
            "stage_a": "stage_a",
            "stage_a_jobs": "stage_a_jobs",
            "stage_b_followup_jobs": "stage_b_followup_jobs",
            "janaadhaar_handle": "janaadhaar_handle",
        },
    )
    graph.add_conditional_edges(
        "stage_a",
        route_after_stage_a,
        {"no_match": "no_match", "clarify": "clarify", "clarify_identity": "clarify_identity", "stage_b": "stage_b"},
    )
    graph.add_edge("stage_b", "generate")
    graph.add_edge("stage_b_followup", "generate")
    graph.add_edge("show_more", "generate")
    graph.add_edge("generate", "finalize")
    graph.add_edge("stage_a_jobs", "generate_jobs")
    graph.add_edge("stage_b_followup_jobs", "generate_jobs")
    graph.add_edge("generate_jobs", "finalize")
    graph.add_edge("clarify", "finalize")
    graph.add_edge("clarify_identity", "finalize")
    graph.add_edge("chitchat", "finalize")
    graph.add_edge("no_match", "finalize")
    graph.add_edge("janaadhaar_handle", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=MemorySaver())
