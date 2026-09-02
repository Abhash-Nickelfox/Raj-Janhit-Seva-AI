"""Structured lookup over Rajasthan government job vacancies (jobs.json).

Unlike RagService's scheme corpus, this needs no embeddings/chunking: it's
120 small, fully structured records (title/department/qualification/
location/dates/salary/documents/status), so exact/fuzzy field matching is
both simpler and more precise here than semantic search would be. Jobs are
a genuinely different domain from welfare schemes - they gate on
qualification/location/deadline, not caste/gender/marital-status identity -
so this is a separate, purpose-built service rather than bolted onto
RagService's eligibility-gating machinery.
"""

import json
import re
from datetime import date
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

JOBS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "jobs.json"

# Verified empirically against this corpus's 12 job titles across a range of
# realistic queries: a genuine match always scores 90-100 (even "teacher
# grade 3 vacancy" vs "Teacher Grade III" -> 90.3), while the highest
# spurious cross-match ("police constable bharti" vs the unrelated "LDC")
# scores only 66.7 - very short titles (3-4 letters) are prone to scoring
# deceptively high against unrelated longer queries on pure chance character
# overlap. 80 sits cleanly in the gap between the two, same threshold
# RagService uses for scheme-name matching.
TITLE_MATCH_THRESHOLD = 80

# The dataset's qualification field is a small closed set (verified directly
# against jobs.json, 2026-08-14), not free text - '12th Pass' < 'Diploma'
# (covers polytechnic/ITI) < 'Graduate'/'BCA'/'B.Tech' (same tier - all
# bachelor's-level) < 'Post Graduate'. This tier number is what
# _rank_key uses to prioritize jobs the citizen is actually likely eligible
# for, WITHOUT ever excluding higher-tier ones (see search()'s docstring).
_QUALIFICATION_TIER = {
    # 'Below 12th' is a CITIZEN-side value only (NeedItem.job_qualification) -
    # no posting in jobs.json requires less than 12th Pass, which is exactly
    # why it needs its own tier below 0 rather than being folded into it
    # (2026-09-02, found live: a 10th-pass citizen was recorded as '12th Pass'
    # because neither the extraction Literal nor this table had anywhere else
    # to put them, and got shown 12th-Pass-only vacancies as if eligible).
    "Below 12th": -1,
    "12th Pass": 0,
    "Diploma": 1,
    "Graduate": 2, "BCA": 2, "B.Tech": 2,
    "Post Graduate": 3,
}

# The lowest requirement any posting in this dataset actually asks for.
# Computed from the table rather than hardcoded so it stays correct if the
# corpus ever gains a lower-requirement posting.
MIN_JOB_QUALIFICATION_TIER = _QUALIFICATION_TIER["12th Pass"]


def _qualification_tier(value: Optional[str]) -> int:
    return _QUALIFICATION_TIER.get(value, 2)  # unknown values default to the common bachelor's tier


def is_under_qualified_for_all(qualification: Optional[str]) -> bool:
    """Whether this citizen's stated qualification is below what EVERY posting
    in the corpus requires - i.e. nothing here is applicable to them.

    Separate from search()'s deliberate fail-open ranking (see its docstring):
    fail-open is right for OVER-qualification, where a Graduate can still
    apply to a 12th-Pass post, but it is not right for UNDER-qualification,
    where the citizen simply cannot apply at all. search() still returns the
    postings so the reply can explain the requirement; this lets the caller
    say so plainly instead of presenting them as options."""
    if not qualification:
        return False
    return _qualification_tier(qualification) < MIN_JOB_QUALIFICATION_TIER


def _min_salary(job) -> int:
    """"₹33843 - ₹97275" -> 33843. First number in the string is always the
    lower bound of the range in this dataset."""
    nums = re.findall(r"\d+", job["salary"])
    return int(nums[0]) if nums else 0


def _compute_status(job, today: date) -> str:
    """Computed fresh every call rather than trusted from the stored
    "status" field - found live (2026-08-14): the stored field went stale
    the moment "today" moved past a job's lastDate without the file being
    re-edited (9 of 120 records had status="active" with a lastDate already
    in the past). A static field can only ever be correct as of whenever it
    was last written; computing it from startDate/lastDate/today is correct
    forever, no matter how much time passes."""
    start = date.fromisoformat(job["startDate"])
    last = date.fromisoformat(job["lastDate"])
    if today < start:
        return "coming soon"
    if today > last:
        return "closed"
    return "active"


# Hand-verified Hindi/Hinglish/abbreviation aliases for this corpus's 12 job
# titles (2026-08-25, found live) - every title in jobs.json is an
# English-only string, so fuzz.partial_ratio against a Devanagari query
# scores near-zero regardless of a genuine match (confirmed: "पटवारी भर्ती"
# scored <=15 against every title, including the real Patwari postings) -
# NeedItem.need_text is REQUIRED to be Devanagari for any Hindi/Hinglish
# message (see its own field description), with no domain exception for
# jobs, so this isn't a rare edge case, it's every Hindi-script job query.
# Same class of gap as OCCUPATION_KEYWORDS (app/rag_service.py) - a small,
# literal table, not exhaustive; anything missed just falls back to the
# English fuzzy match, same as before this fix.
_TITLE_ALIASES = {
    "Account Assistant": ["लेखा सहायक", "accounts assistant", "lekha sahayak"],
    "Computer Instructor": ["कंप्यूटर अनुदेशक", "computer anudeshak"],
    "Forest Guard": ["वनरक्षक", "वन रक्षक", "vanrakshak", "van rakshak"],
    "Junior Assistant": ["कनिष्ठ सहायक", "kanishtha sahayak", "kanisht sahayak"],
    "Junior Engineer": ["कनिष्ठ अभियंता", "kanishtha abhiyanta", "जूनियर इंजीनियर"],
    "LDC": ["लोअर डिवीजन क्लर्क", "lower division clerk", "एलडीसी"],
    "Lab Assistant": ["प्रयोगशाला सहायक", "prayogshala sahayak", "लैब सहायक"],
    "Patwari": ["पटवारी"],
    "Police Constable": ["पुलिस कांस्टेबल", "पुलिस भर्ती", "sipahi", "सिपाही", "police bharti"],
    "RAS Officer": ["आरएएस अधिकारी", "राजस्थान प्रशासनिक सेवा", "rajasthan administrative service"],
    "Teacher Grade III": [
        "अध्यापक तृतीय श्रेणी", "तृतीय श्रेणी अध्यापक", "shikshak tritiya shreni",
        "teacher grade 3", "3rd grade teacher", "third grade teacher",
    ],
    "Village Development Officer": ["ग्राम विकास अधिकारी", "gram vikas adhikari", "वीडीओ"],
}


def _title_alias_score(query_lower, job):
    """100 if any known Hindi/Hinglish/abbreviation alias for this job's
    title appears in the query, else 0 - a deliberately binary signal
    (found-or-not), not a fuzzy one, since these are exact hand-verified
    phrases, not approximations to fuzzy-match against."""
    aliases = _TITLE_ALIASES.get(job["title"])
    if not aliases:
        return 0.0
    return 100.0 if any(alias in query_lower for alias in aliases) else 0.0


def _title_match_score(query, job):
    query_lower = query.lower()
    return max(
        fuzz.partial_ratio(query_lower, job["title"].lower()),
        fuzz.partial_ratio(query_lower, job["department"].lower()),
        _title_alias_score(query_lower, job),
    )


class JobService:
    def __init__(self):
        self.jobs = json.loads(JOBS_PATH.read_text(encoding="utf-8"))
        self._by_id = {j["id"]: j for j in self.jobs}

    def _with_live_status(self, job):
        return {**job, "status": _compute_status(job, date.today())}

    def search(self, query_text, location: Optional[str] = None, status: Optional[str] = "active",
               exclude_ids: Optional[list] = None, top_n: int = 5,
               qualification: Optional[str] = None, min_salary: Optional[int] = None):
        """Fuzzy-matches query_text against title/department, optionally
        filtered by status/location/min_salary. When a specific post is
        named, sorted by title-match score. Otherwise (a qualification-
        driven or bare "sarkari naukri" query), sorted by qualification fit
        and vacancy count - see below.

        Never EXCLUDES on qualification - fail-open, same principle as
        scheme eligibility: a Graduate asking about a "12th Pass" posting
        shouldn't have it hidden, since overqualification is normally fine
        for these posts. The requirement is surfaced in the reply instead,
        so the citizen can self-assess. What qualification DOES change is
        ranking: found live (2026-08-14) - "maine polytechnic ITI kiya hua
        hai, koi job batao" produced a need_text ("polytechnic ITI graduate
        job") that never fuzzy-matched any real job TITLE (titles are post
        names like "Patwari", not qualification categories), so it fell
        into the bare-query fallback and returned B.Tech-required postings
        sorted by pure deadline, ignoring the stated qualification entirely.
        Now: when no specific post is named but a qualification IS stated,
        jobs at or below the citizen's tier are ranked first (real fit),
        then by vacancy count (more open seats = better odds), then by
        deadline - higher-tier jobs still appear, just later, never dropped.

        If a location/min_salary filter drops every candidate, it's
        silently relaxed (dropped) rather than returning nothing - a
        citizen asking about a city/pay bar with no current openings still
        gets useful alternatives back.
        """
        today = date.today()
        candidates = self.jobs
        if status:
            candidates = [j for j in candidates if _compute_status(j, today) == status]
        if exclude_ids:
            candidates = [j for j in candidates if j["id"] not in exclude_ids]

        if location:
            narrowed = [j for j in candidates if location.lower() in j["location"].lower()]
            if narrowed:
                candidates = narrowed

        if min_salary:
            narrowed = [j for j in candidates if _min_salary(j) >= min_salary]
            if narrowed:
                candidates = narrowed

        scored = [(_title_match_score(query_text, j), j) for j in candidates]
        strong = [(score, j) for score, j in scored if score >= TITLE_MATCH_THRESHOLD]
        if strong:
            # A real title/department match exists - only return those, not
            # padded out with irrelevant postings just to reach top_n (found
            # testing this directly: "patwari job" was returning Patwari
            # correctly at #1, but also Teacher Grade III/Police Constable
            # in the remaining slots purely because nothing else scored
            # higher, not because they're actually relevant).
            strong.sort(key=lambda pair: (-pair[0], pair[1]["lastDate"]))
            return [self._with_live_status(j) for _, j in strong[:top_n]]

        if qualification:
            citizen_tier = _qualification_tier(qualification)
            ranked = sorted(
                candidates,
                key=lambda j: (_qualification_tier(j["qualification"]) > citizen_tier, -j["vacancies"], j["lastDate"]),
            )
        else:
            # No specific title AND no stated qualification (a bare "sarkari
            # naukri") - most vacancies first (better realistic odds), then
            # soonest deadline, instead of an arbitrary/pure-deadline order.
            ranked = sorted(candidates, key=lambda j: (-j["vacancies"], j["lastDate"]))
        return [self._with_live_status(j) for j in ranked[:top_n]]

    def has_title_match(self, query_text, status: Optional[str] = "active") -> bool:
        """Whether query_text names a specific post/department confidently
        enough that no clarifying question is needed about WHICH job -
        reuses the exact same scoring search() itself already computes
        internally to decide between its title-matched result set and the
        generic ranked-fallback path, so the two can never disagree. Added
        2026-08-18 for route_after_stage_a_jobs: a bare "mujhe job ke liye
        kuch batao" should trigger a narrowing question, but "Patwari job ke
        baare mein batao" (a specific post named, just no location/
        qualification/salary stated) should not - job_location/
        job_qualification/job_min_salary alone can't tell those two apart,
        since both leave all three null."""
        today = date.today()
        candidates = [j for j in self.jobs if not status or _compute_status(j, today) == status]
        return any(_title_match_score(query_text, j) >= TITLE_MATCH_THRESHOLD for j in candidates)

    def get(self, job_id):
        job = self._by_id.get(job_id)
        return self._with_live_status(job) if job else None
