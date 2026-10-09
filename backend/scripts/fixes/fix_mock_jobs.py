"""One-time patch to data/processed/jobs.json (2026-08-26), two fixes:

1. Qualification consistency: every one of this dataset's 12 job titles had
   its qualification requirement vary WILDLY across postings (e.g. Patwari
   ranged from '12th Pass' to 'B.Tech' to 'Post Graduate' depending on
   which record you looked at, RAS Officer down to '12th Pass' - unrealistic
   for a real recruitment corpus, since the SAME post should require the
   SAME qualification everywhere it's posted, just with different
   department/location/vacancy counts). Reassigns qualification per title to
   a single, realistic value matching actual Rajasthan government
   recruitment norms, and regenerates documentsNeeded to match (its
   qualification-specific document pair was tied to the OLD per-record
   value, e.g. "Post Graduate Degree & Marksheet" - left unfixed, a job
   would keep asking for a Post Graduate certificate after being reassigned
   to '12th Pass').

2. Active-job scarcity: only 20 of 120 postings were active as of today,
   because every lastDate was fixed at scrape time and drifts into the past
   as real time passes (this mock corpus has no auto-refresh). Extends
   lastDate for ~50 currently-closed postings out to various dates in
   December 2026, spread across titles/departments rather than one bulk
   date, so more of the corpus is genuinely searchable without the whole
   thing looking artificially uniform.

Run: python scripts/fixes/fix_mock_jobs.py
"""

import json
from datetime import date
from pathlib import Path

JOBS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "jobs.json"

# One realistic qualification per title, per actual RSMSSB/RPSC recruitment
# norms for these posts (Patwari/RAS Officer/Teacher Grade III/VDO require a
# bachelor's degree; LDC/Junior Assistant/Forest Guard/Lab Assistant/Police
# Constable are 12th-pass-entry posts; Junior Engineer requires a diploma;
# Computer Instructor requires a computer-specific degree).
TITLE_QUALIFICATION = {
    "Account Assistant": "Graduate",
    "Computer Instructor": "BCA",
    "Forest Guard": "12th Pass",
    "Junior Assistant": "12th Pass",
    "Junior Engineer": "Diploma",
    "LDC": "12th Pass",
    "Lab Assistant": "12th Pass",
    "Patwari": "Graduate",
    "Police Constable": "12th Pass",
    "RAS Officer": "Graduate",
    "Teacher Grade III": "Graduate",
    "Village Development Officer": "Graduate",
}

_DOCS_PREFIX = [
    "Aadhaar Card", "Recent Passport-size Photograph", "Signature Specimen",
    "Valid Photo ID Proof", "Rajasthan Domicile/Residence Certificate",
]
_DOCS_SUFFIX = ["Category/Caste Certificate (if applicable)", "Experience Certificate (if applicable)"]
# Verified against this dataset's own existing records (one example of each
# qualification value) before writing this table - not invented.
_QUALIFICATION_DOCS_MIDDLE = {
    "12th Pass": ["10th Marksheet & Certificate", "12th Marksheet & Certificate"],
    "Diploma": ["10th & 12th Marksheet", "Diploma Certificate & Marksheet"],
    "Graduate": ["10th & 12th Marksheet", "Graduation Degree & Marksheet"],
    "BCA": ["10th & 12th Marksheet", "BCA Degree & Marksheet"],
    "B.Tech": ["10th & 12th Marksheet", "B.Tech Degree & Marksheet"],
    "Post Graduate": ["Graduation Certificate", "Post Graduate Degree & Marksheet"],
}


def _build_docs(qualification):
    return _DOCS_PREFIX + _QUALIFICATION_DOCS_MIDDLE[qualification] + _DOCS_SUFFIX


def _compute_status(job, today):
    start = date.fromisoformat(job["startDate"])
    last = date.fromisoformat(job["lastDate"])
    if today < start:
        return "coming soon"
    if today > last:
        return "closed"
    return "active"


# Spread across December rather than one bulk date - cycled deterministically
# per job, not random, so re-running this script is reproducible.
_DECEMBER_DATES = [f"2026-12-{d:02d}" for d in (5, 10, 15, 20, 25, 31)]


def main():
    with open(JOBS_PATH, encoding="utf-8") as f:
        jobs = json.load(f)

    today = date.today()

    # --- Fix 1: qualification consistency ---
    qual_changed = 0
    for job in jobs:
        correct_qual = TITLE_QUALIFICATION.get(job["title"])
        if correct_qual and job["qualification"] != correct_qual:
            job["qualification"] = correct_qual
            job["documentsNeeded"] = _build_docs(correct_qual)
            qual_changed += 1

    # --- Fix 2: extend ~50 currently-closed postings into December ---
    closed = [j for j in jobs if _compute_status(j, today) == "closed"]
    # Spread picks across the closed list (every ~len/50th one) for variety
    # across titles/departments, rather than the first 50 in file order.
    target_count = min(50, len(closed))
    step = max(1, len(closed) // target_count)
    to_extend = closed[::step][:target_count]
    for i, job in enumerate(to_extend):
        job["lastDate"] = _DECEMBER_DATES[i % len(_DECEMBER_DATES)]

    # --- Keep the stored status field consistent with the now-updated dates
    # (the app itself always recomputes this live and never trusts the
    # stored value - see app/services/job_service.py's _compute_status docstring -
    # but keeping the raw data internally consistent matters for anyone
    # inspecting the JSON directly). ---
    for job in jobs:
        job["status"] = _compute_status(job, today)

    with open(JOBS_PATH, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)

    print(f"Reassigned qualification (+ documentsNeeded) on {qual_changed} record(s).")
    print(f"Extended lastDate to December on {len(to_extend)} previously-closed record(s).")
    statuses = {}
    for job in jobs:
        statuses[job["status"]] = statuses.get(job["status"], 0) + 1
    print("New status breakdown:", statuses)


if __name__ == "__main__":
    main()
