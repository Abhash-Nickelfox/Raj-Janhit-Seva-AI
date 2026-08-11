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


def _title_match_score(query, job):
    query_lower = query.lower()
    return max(
        fuzz.partial_ratio(query_lower, job["title"].lower()),
        fuzz.partial_ratio(query_lower, job["department"].lower()),
    )


class JobService:
    def __init__(self):
        self.jobs = json.loads(JOBS_PATH.read_text(encoding="utf-8"))
        self._by_id = {j["id"]: j for j in self.jobs}

    def search(self, query_text, location: Optional[str] = None, status: Optional[str] = "active",
               exclude_ids: Optional[list] = None, top_n: int = 5):
        """Fuzzy-matches query_text against title/department, optionally
        filtered by status/location, sorted by title-match score (falling
        back to soonest deadline when nothing specifically matches, so a
        bare "sarkari naukri" query still returns the most time-sensitive
        postings rather than an arbitrary order).

        Never excludes on qualification - fail-open, same principle as
        scheme eligibility: a Graduate asking about a "12th Pass" posting
        shouldn't have it hidden, since overqualification is normally fine
        for these posts. The requirement is surfaced in the reply instead,
        so the citizen can self-assess.

        If a location filter drops every candidate, it's silently relaxed
        (dropped) rather than returning nothing - a citizen asking about a
        city with no current openings still gets useful alternatives back.
        """
        candidates = self.jobs
        if status:
            candidates = [j for j in candidates if j["status"] == status]
        if exclude_ids:
            candidates = [j for j in candidates if j["id"] not in exclude_ids]

        if location:
            narrowed = [j for j in candidates if location.lower() in j["location"].lower()]
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
            return [j for _, j in strong[:top_n]]
        # No specific title/department named (a bare "sarkari naukri") -
        # return the broadest, most time-sensitive set instead of noise-order.
        return sorted(candidates, key=lambda j: j["lastDate"])[:top_n]

    def get(self, job_id):
        return self._by_id.get(job_id)
