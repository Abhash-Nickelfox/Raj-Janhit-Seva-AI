"""One-time, no-LLM patch (2026-08-24) for two real occupation_target_group
gaps found live, applied to the CURRENT data/processed/eligibility_structured.json
without re-running the full extraction pipeline (which would also
re-trigger classify_education_level()'s real LLM call for every
"student"-tagged scheme - ~157 schemes total, many qualify).

1. TRANSGENDER (safe, automated): OCCUPATION_KEYWORDS was missing a
   "transgender" entry entirely (now added in app/services/rag_service.py). Every
   "Transgender Utthan Kosh" scheme's own NAME literally contains
   "Transgender", so re-running the normal name/description/tags keyword
   match with the new entry catches all of them safely - verified directly
   against each one's real eligibility text (all 7 state an unconditional
   "the applicant should/must belong to the transgender community").

2. DIFFERENTLY_ABLED (hand-verified, NOT automated): several schemes use a
   euphemistic name ("Vishesh Yogyajan", "Specially Abled") that never
   matches OCCUPATION_KEYWORDS at all, so this gap can't be caught by
   scanning name/description/tags. Scanning the FULL eligibility text
   instead was tried and reverted (see extract_eligibility.py's own
   comment) - it hits real false positives from negation ("should NOT have
   any disability"), "preference will be given to X" (non-mandatory), and
   inclusive OR-lists that also list "General Category" (so the restriction
   doesn't actually gate anyone). Every entry below was individually
   confirmed against its real, published eligibility text before being
   added here - this is NOT a blind keyword match, it's a short, reasoned
   list in the same spirit as a hand-verified corrections file, just scoped
   to only what's been directly checked.

Run: python scripts/fixes/fix_occupation_tags.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.rag_service import OCCUPATION_KEYWORDS  # noqa: E402
from scripts.pipeline.extract_eligibility import latest_scrape_dir  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_PATH = DATA_DIR / "processed" / "eligibility_structured.json"

# Hand-verified against each scheme's real eligibility text (2026-08-24) -
# every entry states an unconditional "applicant should have X% disability"
# / "applicant should be a specially-abled individual" requirement, not a
# conditional sub-clause, a "preference" clause, or part of an inclusive
# OR-list. Schemes considered and explicitly REJECTED (kept as-is,
# NOT added here) because their real text turned out to be one of those
# unsafe patterns: "Chief Minister Higher Education Scholarship Scheme"
# (disability is a conditional sub-clause for a general scholarship, not a
# restriction on who may apply at all), "Mukhyamantri Work Form Home-Job
# Work Yojana" ("preference will be given to" disabled/widowed/etc, not a
# requirement), "Operation of Sports Hostels" (eligibility text says
# applicants should NOT have a disability - negated, backwards), and
# "Scholarship for Children of Ex-Servicemen for Vocational and Technical
# Education" ("not admissible in ... Disability ... cases" - excludes,
# doesn't include). "Indira Priyadarshini Award" also rejected: disability
# is one of several OR-qualifying categories that also includes "General
# Category", so the restriction doesn't actually gate anyone in practice.
DIFFERENTLY_ABLED_CONFIRMED_SLUGS = {
    "ssfsap",  # Vishesh Yogyajan Chhatravriti Yojana - "should have at least 40% disability"
    "cmsasps",  # Chief Minister Specially Abled Samman Pension Scheme - "Specially abled persons ... with 40% and above disability"
    "mky",  # Mukhyamantri Kanyadaan Yojana - "This scheme is specifically for Divyang and Women..."
    "sdvay",  # Vishesh Yogyajan Sukhad Dampatya Jeevan Yojana - "should possess a disability certificate with 40% or more disability"
}


def main():
    scrape_dir = latest_scrape_dir()
    with open(PROCESSED_PATH, encoding="utf-8") as f:
        records = json.load(f)

    changed = []
    seen_slugs = set()
    for record in records:
        seen_slugs.add(record["slug"])
        old_group = list(record.get("occupation_target_group") or [])
        new_group = set(old_group)

        # 1. Transgender - safe, automated (name/description/tags only).
        raw_path = scrape_dir / f"{record['slug']}.json"
        if raw_path.exists():
            with open(raw_path, encoding="utf-8") as f:
                raw = json.load(f)
            en = raw.get("en", {})
            basic = en.get("basicDetails", {})
            content = en.get("schemeContent", {})
            signal_text = " ".join(
                [
                    basic.get("schemeName", ""),
                    content.get("briefDescription", ""),
                    " ".join(basic.get("tags", []) or []),
                ]
            ).lower()
            if any(k in signal_text for k in OCCUPATION_KEYWORDS["transgender"]):
                new_group.add("transgender")

        # 2. Differently-abled - hand-verified list only.
        if record["slug"] in DIFFERENTLY_ABLED_CONFIRMED_SLUGS:
            new_group.add("differently_abled")

        new_group = sorted(new_group)
        if new_group != sorted(old_group):
            changed.append((record["scheme_name"], old_group, new_group))
            record["occupation_target_group"] = new_group

    unmatched = DIFFERENTLY_ABLED_CONFIRMED_SLUGS - seen_slugs
    if unmatched:
        print(f"WARNING: slug(s) not found in {PROCESSED_PATH}: {unmatched}")

    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Checked {len(records)} schemes, updated {len(changed)} occupation_target_group tag(s):")
    for name, old, new in changed:
        print(f"  {name}\n    before: {old}\n    after:  {new}")


if __name__ == "__main__":
    main()
