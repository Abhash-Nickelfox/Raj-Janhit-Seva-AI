"""Parse myScheme's native eligibility data into a flat, filterable schema.

Mostly no LLM involved - the source API already returns machine-coded
eligibility facets (see scrape_myscheme.py's module docstring for how that
was confirmed). This script normalizes that object's shape and derives two
cheap, mechanical facets the API doesn't give directly (occupation/target-
group, fine-grained benefit type) via small keyword tables - scheme name/
tags/category almost always name these directly, no LLM needed.

education_level is the ONE exception, and deliberately uses a real LLM call
(once per relevant scheme, cached to disk forever after: the previous
eligibility_structured.json is read back and its education_level reused by
slug, so re-running this script after a facet/keyword fix costs zero LLM
calls and needs no API key - see load_education_cache()). Found live (2026-08-05): most schemes with an
education-level restriction (e.g. "Rajya Pratibha Khoj Pariksha Chhatravriti
Yojana" is Class 10/12-only) don't name that level anywhere in the scheme
name or tags - it only exists in the free-text eligibility description,
phrased however that scheme's author happened to write it ("enrolled in
Class 10 or 12", "pursuing PhD", "enrolled in IIT/IIM/NIT/AIIMS"...). A
keyword table over scheme names would have caught only the few schemes
that literally say "Pre-Matric"/"Post Matric" and silently missed the rest -
this genuinely needs reading comprehension over free text, which an LLM
does and a keyword table can't. This is a ONE-TIME BATCH read (not a live
conversation-time call) - the result is cached in eligibility_structured.json
like everything else here, never touched again until the next re-scrape.

Finally data/processed/scheme_overrides.json (hand-verified per-slug facet
corrections + canonical_group) is layered on last - the same file
app/rag_service.py applies at load time, so the two can't disagree.

Input:  data/raw/{scrape-date}/{slug}.json  (from scrape_myscheme.py)
        data/processed/scheme_overrides.json (curated, optional)
Output: data/processed/eligibility_structured.json  (list of flat records)

Usage: python scripts/extract_eligibility.py [--no-llm]
  --no-llm  never call the classifier; schemes with no cached
            education_level just keep null (deterministic re-run).
"""

import json
import sys
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field  # noqa: E402

from app.rag_service import OCCUPATION_KEYWORDS, apply_overrides_to_record, load_overrides  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_PATH = PROCESSED_DIR / "eligibility_structured.json"

# "pension" in a scheme name is not proof of an old-age scheme (widow,
# disability, WWII and journalists' pensions all start at 18 or lower) -
# the senior_citizen tag additionally needs a real minimum age. 55 is the
# lowest old-age threshold in this corpus (women on the state pensions).
SENIOR_CITIZEN_MIN_AGE = 55

# myScheme encodes "no age restriction" as a 0-100 or 1-100 row
NO_AGE_RESTRICTION = {(0, 100), (1, 100)}

ALL_VALUES = {"all", "both", "none"}

BENEFIT_TYPE_KEYWORDS = {
    "scholarship": ["scholarship", "chatravriti"],
    "pension": ["pension"],
    "loan": ["loan", "rin"],
    "subsidy": ["subsidy", "anudan"],
    "equipment": ["yantra", "equipment", "machine", "tool"],
    "training": ["training", "prashikshan", "skill"],
    "insurance": ["insurance", "bima"],
    "housing": ["housing", "awas", "ghar"],
    "healthcare": ["health", "chikitsa", "medical", "hospital", "ayushman"],
}

# Only worth an LLM call for schemes that plausibly restrict by education
# level at all - most categories (Agriculture, Housing, Health...) never do.
# Bounds the one-time cost to roughly the Education & Learning schemes +
# any stragglers tagged "student" elsewhere, not all 157.
EDUCATION_LEVEL_RELEVANT_CATEGORY = "Education & Learning"

EducationLevel = Literal["school", "higher_education"]


class EducationLevelClassification(BaseModel):
    education_level: Optional[EducationLevel] = Field(
        None,
        description=(
            "'school' if this scheme's eligibility restricts to students up to and including "
            "Class 12 / 10+2 / pre-matric / a specific class range (e.g. 'Class 6 to 8', "
            "'Class 10 or 12'). 'higher_education' if it requires enrollment in or completion "
            "of a college/university degree, diploma, professional course, PhD, or a named "
            "higher-education institution (IIT/IIM/NIT/AIIMS etc). "
            "null if the eligibility text doesn't restrict by education level at all, or covers "
            "both school and higher-education students without distinction - do not guess."
        ),
    )


_classifier_llm = None


def _get_classifier():
    global _classifier_llm
    if _classifier_llm is None:
        # lazy: app.config requires an API key at import time, and a
        # cache-only re-run never needs either
        from langchain_openai import ChatOpenAI

        from app.config import OPENROUTER_API_KEY, OPENROUTER_MODEL

        _classifier_llm = ChatOpenAI(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        ).with_structured_output(EducationLevelClassification)
    return _classifier_llm


def classify_education_level(scheme_name, eligibility_text):
    if not eligibility_text or not eligibility_text.strip():
        return None
    prompt = (
        f"Scheme: {scheme_name}\n"
        f"Eligibility text: {eligibility_text}\n\n"
        "Classify the education level this scheme's eligibility restricts to, if any."
    )
    result = _get_classifier().invoke(prompt)
    return result.education_level


def _values(entries):
    """Extract lowercased .value codes from a myScheme facet list, e.g.
    [{"value": "sc", "label": "..."}] -> ["sc"]. Returns ["all"] if the
    facet is unrestricted (any entry's value is All/Both/None) or missing -
    an explicit "no restriction" marker, not an empty/ambiguous list.
    """
    if not entries:
        return ["all"]
    values = [str(e.get("value", "")).lower() for e in entries if e.get("value") is not None]
    if not values or any(v in ALL_VALUES for v in values):
        return ["all"]
    return values


def _age_range(age_entries):
    """(age_min, age_max, age_min_female, age_min_male) from myScheme's
    per-category age rows. Rows that say 0/1-100 are "no restriction" and
    are ignored (None/None if that's all there is). age_min stays the
    overall minimum for backwards compatibility; the per-gender minimums
    are only set when the female/male rows actually differ (e.g. 55/58
    on the Rajasthan old-age pensions)."""
    rows = []
    for a in age_entries or []:
        gte, lte = a.get("gte"), a.get("lte")
        if (gte, lte) in NO_AGE_RESTRICTION:
            continue
        rows.append((gte, lte, str(a.get("category") or "").strip().lower()))
    if not rows:
        return None, None, None, None
    gtes = [g for g, _, _ in rows if g is not None]
    ltes = [l for _, l, _ in rows if l is not None]
    by_gender = {}
    for g, _, cat in rows:
        if cat in ("female", "male") and g is not None:
            by_gender[cat] = min(g, by_gender.get(cat, g))
    female, male = by_gender.get("female"), by_gender.get("male")
    if female is None or male is None or female == male:
        female = male = None
    return (min(gtes) if gtes else None), (max(ltes) if ltes else None), female, male


def _keyword_match(text, table):
    text = text.lower()
    return [group for group, keywords in table.items() if any(k in text for k in keywords)]


def derive_occupation_target_group(signal_text, age_min):
    groups = _keyword_match(signal_text, OCCUPATION_KEYWORDS)
    if "senior_citizen" in groups and (age_min is None or age_min < SENIOR_CITIZEN_MIN_AGE):
        groups.remove("senior_citizen")
    return groups


def load_education_cache(path=OUTPUT_PATH):
    """slug -> education_level from the previous run's output, so the
    LLM is only ever asked about schemes it hasn't classified before."""
    if not path.exists():
        return {}
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {r["slug"]: r.get("education_level") for r in previous if r.get("slug")}


_MISSING = object()


def extract_one(record, education_cache=None, overrides=None, allow_llm=True):
    en = record.get("en", {})
    basic = en.get("basicDetails", {})
    content = en.get("schemeContent", {})
    elig = record.get("eligibility_structured", {})
    scheme_name = (basic.get("schemeName") or "").strip() or None

    signal_text = " ".join(
        [
            scheme_name or "",
            content.get("briefDescription", ""),
            " ".join(basic.get("tags", []) or []),
        ]
    )

    age_min, age_max, age_min_female, age_min_male = _age_range(elig.get("age"))
    category = [c.get("label") for c in basic.get("schemeCategory", []) or []]
    occupation_target_group = derive_occupation_target_group(signal_text, age_min)

    education_level = None
    if EDUCATION_LEVEL_RELEVANT_CATEGORY in category or "student" in occupation_target_group:
        cached = (education_cache or {}).get(record["slug"], _MISSING)
        if cached is not _MISSING:
            education_level = cached
        elif allow_llm:
            eligibility_text = (
                en.get("eligibilityCriteria", {}).get("eligibilityDescription_md", "") or ""
            )
            education_level = classify_education_level(scheme_name or "", eligibility_text)

    out = {
        "scheme_id": record["scheme_id"],
        "slug": record["slug"],
        "scheme_name": scheme_name,
        "level": (basic.get("level") or {}).get("label"),
        "category": category,
        "department": (basic.get("nodalDepartmentName") or {}).get("label"),
        "caste": _values(elig.get("caste")),
        "gender": _values(elig.get("gender")),
        "marital_status": _values(elig.get("maritalStatus")),
        "residence": _values(elig.get("residence")),
        "age_min": age_min,
        "age_max": age_max,
        "age_min_female": age_min_female,
        "age_min_male": age_min_male,
        "family_income_ceiling": (elig.get("familyIncomeAnnual") or {}).get("lte"),
        "individual_income_ceiling": (elig.get("individualIncomeAnnual") or {}).get("lte"),
        "parent_income_ceiling": (elig.get("parentIncomeAnnual") or {}).get("lte"),
        "benefit_type_coarse": (content.get("benefitTypes") or {}).get("value"),
        "benefit_type_fine": _keyword_match(signal_text, BENEFIT_TYPE_KEYWORDS),
        "occupation_target_group": occupation_target_group,
        "education_level": education_level,
        "canonical_group": None,
    }
    return apply_overrides_to_record(out, (overrides or {}).get(record["slug"]))


def latest_scrape_dir():
    raw_dir = DATA_DIR / "raw"
    dated_dirs = sorted((d for d in raw_dir.iterdir() if d.is_dir()), reverse=True)
    if not dated_dirs:
        sys.exit(f"No scrape snapshots found under {raw_dir}")
    return dated_dirs[0]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    allow_llm = "--no-llm" not in argv
    scrape_dir = latest_scrape_dir()
    print(f"Reading raw scrape from {scrape_dir}")

    education_cache = load_education_cache()
    overrides = load_overrides()
    print(f"education_level cache: {len(education_cache)} slugs; overrides: {len(overrides)} slugs; llm={'on' if allow_llm else 'off'}")

    paths = [p for p in sorted(scrape_dir.glob("*.json")) if p.name != "_listing.json"]
    records = []
    for i, path in enumerate(paths, 1):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        record = extract_one(raw, education_cache=education_cache, overrides=overrides, allow_llm=allow_llm)
        if record.get("education_level") is not None or i % 20 == 0:
            print(f"  [{i}/{len(paths)}] {record['slug']}: education_level={record.get('education_level')}")
        records.append(record)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} structured eligibility records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
