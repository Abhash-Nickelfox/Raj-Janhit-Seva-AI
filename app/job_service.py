"""Structured lookup over Rajasthan government job vacancies (jobs.json).

Unlike RagService's scheme corpus, this needs no embeddings/chunking: it's
~100 small, fully structured records (title/department/qualification/
location/dates/salary/documents), so exact/fuzzy field matching is both
simpler and more precise here than semantic search would be. Jobs are a
genuinely different domain from welfare schemes - they gate on
qualification/location/deadline, not caste/gender/marital-status identity -
so this is a separate, purpose-built service rather than bolted onto
RagService's eligibility-gating machinery.

This module deliberately imports nothing from app.config so it can be
tested without API keys.

Public API
----------
    normalize_qualification(text) -> Optional[str]
    JobService(jobs=None, today=None)
        .search_with_info(query_text, location=None, qualification=None,
                          min_salary=None, exclude_ids=None, top_n=5,
                          status="active") -> (results, info)
        .search(query_text, location=None, status="active", exclude_ids=None,
                top_n=5, qualification=None, min_salary=None) -> results
        .has_title_match(query_text, status="active") -> bool
        .get(job_id) -> job | None

`info` keys: title_named, matched_titles, closed_matches, upcoming_matches,
relaxed, location_requested, qualification_normalized, below_min_qualification.
"""

import json
import re
from datetime import date
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

JOBS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "jobs.json"

# ---------------------------------------------------------------------------
# Title matching
# ---------------------------------------------------------------------------

# Fuzzy threshold on token_set_ratio between the cleaned query and a title
# (or one of its aliases). Matching is ONLY against the title - never the
# department - because department strings all contain "Rajasthan"/
# "Department", which made every "Rajasthan me job" query a 100% "match"
# against every row under the old partial_ratio scheme.
TITLE_MATCH_THRESHOLD = 80
MIN_NEEDLE_LEN = 3
# Per-word typo tolerance ("patvari" vs "patwari" = 86).
WORD_FUZZ_THRESHOLD = 85

# Canonical title -> aliases citizens actually type (Hinglish + Hindi).
# Multi-word aliases are matched as phrases; single tokens as whole words.
TITLE_ALIASES = {
    "LDC": ["ldc", "clerk", "lower division clerk", "लिपिक", "एलडीसी"],
    "RAS Officer": ["ras", "ras officer", "administrative service", "आरएएस"],
    "Patwari": ["patwari", "पटवारी"],
    "Police Constable": ["constable", "police", "sipahi", "police constable", "पुलिस", "कांस्टेबल", "सिपाही"],
    "Junior Engineer": ["je", "engineer", "junior engineer", "इंजीनियर"],
    "Teacher Grade III": ["teacher", "shikshak", "teacher grade 3", "teacher grade iii", "reet", "शिक्षक", "अध्यापक"],
    "Village Development Officer": ["vdo", "gram sevak", "gram vikas adhikari", "village development officer", "ग्राम सेवक", "ग्राम विकास अधिकारी"],
    "Lab Assistant": ["lab assistant", "prayogshala sahayak", "प्रयोगशाला सहायक"],
    "Computer Instructor": ["computer instructor", "computer teacher", "कंप्यूटर अनुदेशक"],
    "Account Assistant": ["account assistant", "accountant", "lekha sahayak", "लेखा सहायक"],
    "Forest Guard": ["forest guard", "vanrakshak", "van rakshak", "forester", "वनरक्षक"],
    "Junior Assistant": ["junior assistant", "kanishth sahayak", "कनिष्ठ सहायक"],
}

# Words that carry no information about WHICH post is wanted. Stripped
# before fuzzy matching so "rajasthan me sarkari naukri batao" reduces to
# an empty needle (=> no title named) instead of fuzz-matching something.
STOPWORDS = {
    "job", "jobs", "naukri", "nokri", "sarkari", "government", "govt", "rajasthan",
    "vacancy", "vacancies", "bharti", "recruitment", "post", "posts", "me", "mein",
    "main", "ke", "ki", "ka", "liye", "koi", "batao", "bataye", "bataiye", "chahiye",
    "chahiye", "hai", "hain", "kya", "kaun", "konsi", "kaunsi", "si", "sa", "se",
    "aur", "or", "and", "the", "a", "an", "for", "in", "of", "to", "about", "any",
    "mujhe", "muje", "mereko", "mere", "mera", "meri", "hu", "hun", "hoon", "want",
    "tell", "show", "find", "kuch", "bhi", "please", "plz", "sir", "madam", "ji",
    "options", "option", "opening", "openings", "list", "latest", "new", "current",
    "hoon", "ho", "raha", "rahi", "karna", "karni", "kar", "sakta", "sakti",
    "नौकरी", "सरकारी", "राजस्थान", "भर्ती", "में", "के", "लिए", "कोई", "बताओ",
    "बताइए", "चाहिए", "है", "हैं", "मुझे", "और", "जॉब", "कौनसी", "कौन", "सी",
}

_WORD_RE = re.compile(r"""[^\s,.!?;:()\[\]"'/\\|\-]+""")  # whitespace/punct split keeps Devanagari matras intact


def _tokens(text: str) -> list:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _clean_query(text: str) -> str:
    return " ".join(t for t in _tokens(text) if t not in STOPWORDS)


def _has_phrase(tokens: list, phrase: str) -> bool:
    p = phrase.split()
    n = len(p)
    return any(tokens[i:i + n] == p for i in range(len(tokens) - n + 1))


def match_titles(query_text: str) -> dict:
    """Returns {canonical_title: score} for titles the query names.
    Whole-word alias hits score 100; otherwise token_set_ratio of the
    stopword-stripped query against the title/aliases (>= threshold)."""
    tokens = [t for t in _tokens(query_text) if t not in STOPWORDS]
    needle = " ".join(tokens)
    if len(needle.replace(" ", "")) < MIN_NEEDLE_LEN:
        return {}
    scores = {}
    for title, aliases in TITLE_ALIASES.items():
        best = 0
        for alias in [title.lower()] + aliases:
            if _has_phrase(tokens, alias):
                best = 100
                break
            if len(alias) < 4:  # short aliases (je, ras) need an exact whole-word hit
                continue
            alias_words = alias.split()
            if len(alias_words) > 1:
                # Multi-word alias: EVERY alias word needs a (typo-tolerant)
                # counterpart in the query - plain token_set_ratio would let
                # "lab assistant" match "Account Assistant" on the shared word.
                if all(any(fuzz.ratio(w, t) >= WORD_FUZZ_THRESHOLD for t in tokens) for w in alias_words):
                    best = max(best, 95)
            else:
                best = max(best, max((fuzz.ratio(alias, t) for t in tokens), default=0))
        if best >= TITLE_MATCH_THRESHOLD:
            scores[title] = best
    return scores


# ---------------------------------------------------------------------------
# Qualification
# ---------------------------------------------------------------------------

# Canonical dataset values -> tier. '12th Pass' < 'Diploma' < bachelor's
# level ('Graduate'/'BCA'/'B.Tech') < 'Post Graduate'.
_QUALIFICATION_TIER = {
    "10th Pass": 0,  # not a dataset value; below every posting - flagged via info
    "12th Pass": 1,
    "12th Pass (Science)": 1,
    "Diploma": 2,
    "Graduate": 3, "BCA": 3, "B.Tech": 3,
    "Post Graduate": 4,
}

_QUAL_PATTERNS = [
    (r"\b(post\s*graduat\w*|pg|m\.?\s?a\b|m\.?\s?sc\b|m\.?\s?com\b|mca|m\.?\s?tech|mba|master'?s?|snatakottar|परास्नातक|एमए|एमएससी)\b", "Post Graduate"),
    (r"\b(b\.?\s?tech|b\.?\s?e\b|engineering\s+degree|btech)\b", "B.Tech"),
    (r"\b(bca)\b", "BCA"),
    (r"\b(graduat\w*|b\.?\s?a\b|b\.?\s?sc\b|b\.?\s?com\b|bachelor'?s?|degree|snatak|ग्रेजुएट|स्नातक|बीए|बीकॉम|बीएससी)\b", "Graduate"),
    (r"\b(diploma|polytechnic|iti|डिप्लोमा|आईटीआई)\b", "Diploma"),
    (r"\b(12th|12|xii|intermediate|inter|hsc|senior\s+secondary|barahvi|barvi|12vi|12vin|barhvi|बारहवीं|12वीं)\b", "12th Pass"),
    (r"\b(10th|10|matric\w*|sslc|high\s+school|dasvi|dasvin|10vi|10vin|दसवीं|10वीं)\b", "10th Pass"),
]


def normalize_qualification(text: Optional[str]) -> Optional[str]:
    """Maps citizen free text ("maine B.Com kiya hai", "12vi pass", "ITI")
    to a canonical tier label. Returns one of the dataset qualification
    values, "10th Pass" (below every posting), or None if unrecognised.
    Highest qualification mentioned wins ("10th 12th graduate" -> Graduate)."""
    if not text:
        return None
    t = text.lower().strip()
    if t in _QUALIFICATION_TIER:
        return t
    for canon in _QUALIFICATION_TIER:
        if t == canon.lower():
            return canon
    for pattern, canon in _QUAL_PATTERNS:
        if re.search(pattern, t):
            return canon
    return None


def _qualification_tier(value: Optional[str]) -> int:
    return _QUALIFICATION_TIER.get(value, 3)  # unknown values default to the bachelor's tier


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

LOCATION_ALIASES = {
    "Jaipur": ["jaipur", "जयपुर", "jaypur"],
    "Jodhpur": ["jodhpur", "जोधपुर"],
    "Udaipur": ["udaipur", "उदयपुर"],
    "Kota": ["kota", "कोटा"],
    "Ajmer": ["ajmer", "अजमेर"],
    "Bikaner": ["bikaner", "बीकानेर"],
    "Alwar": ["alwar", "अलवर"],
    "Sikar": ["sikar", "सीकर"],
}
_LOCATION_LOOKUP = {alias: city for city, aliases in LOCATION_ALIASES.items() for alias in aliases}


def normalize_location(text: Optional[str]) -> Optional[str]:
    """"जयपुर" / "jaipur city" / "Jaipur" -> "Jaipur". None if unknown."""
    if not text:
        return None
    t = text.strip().lower()
    if t in _LOCATION_LOOKUP:
        return _LOCATION_LOOKUP[t]
    for tok in _tokens(t):
        if tok in _LOCATION_LOOKUP:
            return _LOCATION_LOOKUP[tok]
    return None


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _min_salary(job) -> int:
    """"₹20800 - ₹65900 (Level-5)" -> 20800."""
    nums = re.findall(r"\d+", job["salary"])
    return int(nums[0]) if nums else 0


def _compute_status(job, today: date) -> str:
    """Computed from startDate/lastDate/today on every call; the file stores
    no status field because a stored one goes stale the day after it's
    written."""
    start = date.fromisoformat(job["startDate"])
    last = date.fromisoformat(job["lastDate"])
    if today < start:
        return "coming soon"
    if today > last:
        return "closed"
    return "active"


class JobService:
    def __init__(self, jobs: Optional[list] = None, today: Optional[date] = None):
        """`jobs`/`today` are injectable for tests; production uses jobs.json
        and the real date."""
        self.jobs = jobs if jobs is not None else json.loads(JOBS_PATH.read_text(encoding="utf-8"))
        self._by_id = {j["id"]: j for j in self.jobs}
        self._today = today

    @property
    def today(self) -> date:
        return self._today or date.today()

    def _with_live_status(self, job):
        return {**job, "status": _compute_status(job, self.today)}

    # -- core -------------------------------------------------------------

    def search_with_info(self, query_text, location: Optional[str] = None,
                         qualification: Optional[str] = None, min_salary: Optional[int] = None,
                         exclude_ids: Optional[list] = None, top_n: int = 5,
                         status: Optional[str] = "active"):
        """Returns (results, info).

        Order of operations: title match -> exclude_ids -> status filter ->
        location/min_salary filters (each relaxed if it empties the set,
        recorded in info["relaxed"]) -> ranking.

        Title matching runs BEFORE the status filter so a named post whose
        postings are all closed/upcoming is reported as such
        (info["closed_matches"] / info["upcoming_matches"]) instead of
        silently falling through to generic results.

        Ranking:
          * title named: by title score, then qualification fit, then soonest
            deadline. Only matching titles are returned - never padded.
          * no title named: diversity - at most one posting per title,
            ordered by qualification fit (postings at/below the citizen's
            tier first), then soonest deadline. Vacancy count is not used.

        Qualification is fail-open: never excludes, only ranks.
        """
        today = self.today
        exclude = set(exclude_ids or [])
        info = {
            "title_named": False,
            "matched_titles": [],
            "closed_matches": [],
            "upcoming_matches": [],
            "relaxed": [],
            "location_requested": location,
            "qualification_normalized": None,
            "below_min_qualification": False,
        }

        # 1. Title match over the WHOLE corpus (pre-status).
        title_scores = match_titles(query_text)
        if title_scores:
            info["title_named"] = True
            info["matched_titles"] = sorted(title_scores, key=lambda t: -title_scores[t])
            pool = [j for j in self.jobs if j["title"] in title_scores and j["id"] not in exclude]
        else:
            pool = [j for j in self.jobs if j["id"] not in exclude]

        # 2. Status filter, recording what was lost for named titles.
        if status:
            if title_scores:
                info["closed_matches"] = [self._with_live_status(j) for j in pool
                                          if _compute_status(j, today) == "closed"]
                info["upcoming_matches"] = [self._with_live_status(j) for j in pool
                                            if _compute_status(j, today) == "coming soon"]
            pool = [j for j in pool if _compute_status(j, today) == status]

        # 3. Location / salary, each relaxed if it empties the pool.
        city = normalize_location(location)
        if location:
            if city:
                narrowed = [j for j in pool if j["location"] == city]
            else:
                narrowed = [j for j in pool if location.lower() in j["location"].lower()]
            if narrowed:
                pool = narrowed
            else:
                info["relaxed"].append("location")

        if min_salary:
            narrowed = [j for j in pool if _min_salary(j) >= min_salary]
            if narrowed:
                pool = narrowed
            else:
                info["relaxed"].append("min_salary")

        # 4. Qualification tier (ranking only).
        citizen_q = normalize_qualification(qualification) if qualification else None
        info["qualification_normalized"] = citizen_q
        if citizen_q == "10th Pass":
            info["below_min_qualification"] = True
        citizen_tier = _qualification_tier(citizen_q) if citizen_q else None

        def fit(j):
            if citizen_tier is None:
                return 0
            # 0 = citizen meets the requirement; otherwise how many tiers short
            # they are, so a 10th-pass citizen sees 12th-pass posts before B.Tech.
            return max(0, _qualification_tier(j["qualification"]) - citizen_tier)

        # 5. Rank.
        if title_scores:
            pool.sort(key=lambda j: (-title_scores[j["title"]], fit(j), j["lastDate"]))
            results = pool[:top_n]
        else:
            pool.sort(key=lambda j: (fit(j), j["lastDate"]))
            seen, results = set(), []
            for j in pool:
                if j["title"] in seen:
                    continue
                seen.add(j["title"])
                results.append(j)
                if len(results) >= top_n:
                    break

        return [self._with_live_status(j) for j in results], info

    # -- compatibility wrappers ---------------------------------------------

    def search(self, query_text, location: Optional[str] = None, status: Optional[str] = "active",
               exclude_ids: Optional[list] = None, top_n: int = 5,
               qualification: Optional[str] = None, min_salary: Optional[int] = None):
        """Old signature; returns just the results list."""
        results, _ = self.search_with_info(
            query_text, location=location, qualification=qualification, min_salary=min_salary,
            exclude_ids=exclude_ids, top_n=top_n, status=status,
        )
        return results

    def has_title_match(self, query_text, status: Optional[str] = "active") -> bool:
        """Whether query_text names a specific post. `status` is accepted for
        backward compatibility but a named post counts even if every posting
        for it is closed - callers use search_with_info's closed_matches to
        explain that, rather than asking a clarifying question."""
        return bool(match_titles(query_text))

    def get(self, job_id):
        job = self._by_id.get(job_id)
        return self._with_live_status(job) if job else None
