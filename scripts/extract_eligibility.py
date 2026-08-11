"""Parse myScheme's native eligibility data into a flat, filterable schema.

Mostly no LLM involved - the source API already returns machine-coded
eligibility facets (see scrape_myscheme.py's module docstring for how that
was confirmed). This script normalizes that object's shape and derives two
cheap, mechanical facets the API doesn't give directly (occupation/target-
group, fine-grained benefit type) via small keyword tables - scheme name/
tags/category almost always name these directly, no LLM needed.

education_level is the ONE exception, and deliberately uses a real LLM call
(once per relevant scheme, cached to disk forever after - see
classify_education_level()). Found live (2026-08-05): most schemes with an
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

Input:  data/raw/{scrape-date}/{slug}.json  (from scrape_myscheme.py)
Output: data/processed/eligibility_structured.json  (list of flat records)
"""

import json
import sys
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.config import OPENROUTER_API_KEY, OPENROUTER_MODEL  # noqa: E402
from app.rag_service import OCCUPATION_KEYWORDS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

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
    if not age_entries:
        return None, None
    gtes = [a["gte"] for a in age_entries if a.get("gte") is not None]
    ltes = [a["lte"] for a in age_entries if a.get("lte") is not None]
    return (min(gtes) if gtes else None), (max(ltes) if ltes else None)


def _keyword_match(text, table):
    text = text.lower()
    return [group for group, keywords in table.items() if any(k in text for k in keywords)]


def extract_one(record):
    en = record.get("en", {})
    basic = en.get("basicDetails", {})
    content = en.get("schemeContent", {})
    elig = record.get("eligibility_structured", {})

    signal_text = " ".join(
        [
            basic.get("schemeName", ""),
            content.get("briefDescription", ""),
            " ".join(basic.get("tags", []) or []),
        ]
    )

    age_min, age_max = _age_range(elig.get("age"))
    category = [c.get("label") for c in basic.get("schemeCategory", []) or []]
    occupation_target_group = _keyword_match(signal_text, OCCUPATION_KEYWORDS)

    education_level = None
    if EDUCATION_LEVEL_RELEVANT_CATEGORY in category or "student" in occupation_target_group:
        eligibility_text = (
            en.get("eligibilityCriteria", {}).get("eligibilityDescription_md", "") or ""
        )
        education_level = classify_education_level(basic.get("schemeName", ""), eligibility_text)

    return {
        "scheme_id": record["scheme_id"],
        "slug": record["slug"],
        "scheme_name": basic.get("schemeName"),
        "level": (basic.get("level") or {}).get("label"),
        "category": category,
        "department": (basic.get("nodalDepartmentName") or {}).get("label"),
        "caste": _values(elig.get("caste")),
        "gender": _values(elig.get("gender")),
        "marital_status": _values(elig.get("maritalStatus")),
        "residence": _values(elig.get("residence")),
        "age_min": age_min,
        "age_max": age_max,
        "family_income_ceiling": (elig.get("familyIncomeAnnual") or {}).get("lte"),
        "individual_income_ceiling": (elig.get("individualIncomeAnnual") or {}).get("lte"),
        "parent_income_ceiling": (elig.get("parentIncomeAnnual") or {}).get("lte"),
        "benefit_type_coarse": (content.get("benefitTypes") or {}).get("value"),
        "benefit_type_fine": _keyword_match(signal_text, BENEFIT_TYPE_KEYWORDS),
        "occupation_target_group": occupation_target_group,
        "education_level": education_level,
    }


def latest_scrape_dir():
    raw_dir = DATA_DIR / "raw"
    dated_dirs = sorted((d for d in raw_dir.iterdir() if d.is_dir()), reverse=True)
    if not dated_dirs:
        sys.exit(f"No scrape snapshots found under {raw_dir}")
    return dated_dirs[0]


def main():
    scrape_dir = latest_scrape_dir()
    print(f"Reading raw scrape from {scrape_dir}")

    paths = [p for p in sorted(scrape_dir.glob("*.json")) if p.name != "_listing.json"]
    records = []
    classified = 0
    for i, path in enumerate(paths, 1):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        record = extract_one(raw)
        if record.get("education_level") is not None or i % 20 == 0:
            print(f"  [{i}/{len(paths)}] {record['slug']}: education_level={record.get('education_level')}")
        records.append(record)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "eligibility_structured.json"
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} structured eligibility records to {out_path}")


if __name__ == "__main__":
    main()
