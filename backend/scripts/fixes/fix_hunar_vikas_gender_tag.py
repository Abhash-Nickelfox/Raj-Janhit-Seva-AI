"""One-time, no-LLM patch (2026-09-03) for a real occupation_target_group
tagging error found live, applied to the CURRENT
data/processed/eligibility_structured.json without re-running the full
extraction pipeline - same pattern as scripts/fixes/fix_occupation_tags.py.

Found live: a citizen who never stated their gender was told a scheme was
"un ladkiyon ke liye" (for those girls) and didn't fit them - a fact never
confirmed, that the model appears to have inferred from this scheme's own
occupation_target_group tag containing 'woman'. Traced to source: all three
"Mukhyamantri Hunar Vikas Yojana" (Higher Education, Professional/Technical
Training, Self-Employment/Business) variants are tagged 'woman' in
occupation_target_group, but:
  - their own structured `gender` field already says ['male', 'female'] -
    i.e. open to both, contradicting the occupation tag outright
  - their REAL eligibility text (data/raw/2026-08-05/mhvs-{hte,ptts,sebs}.json)
    never mentions gender/women anywhere - only Palanhar-beneficiary status,
    an age window (15-21 or 21+), and course/registration conditions

So 'woman' has zero basis in either the scheme's own gender field or its
authoritative real text for these three - hand-verified by reading each raw
eligibility description directly, not inferred from a keyword scan.

NOT touched: "Urdu Scholarship- Rajasthan" (slug 'usr'), which ALSO has this
same gender=['male','female'] + occupation_target_group=['woman'] shape but
is a genuinely different case - its real text states "at graduation level,
only female students are eligible; at post-graduation level, both male and
female are eligible". That's an actual, education-level-conditional gender
restriction the flat schema can't represent either way (removing 'woman'
would hide a real restriction for undergrad applicants; nothing here can
safely resolve which half of the record is right without a schema change
for a conditional-on-another-field restriction). Left as-is deliberately.

Run: python scripts/fixes/fix_hunar_vikas_gender_tag.py
"""

import json
from pathlib import Path

PROCESSED_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "eligibility_structured.json"

# Hand-verified against each scheme's real eligibility text (2026-09-03) -
# see module docstring. Every one of these three has gender=['male','female']
# already, and 'woman' appears nowhere in the real, government-written
# eligibility description.
CONFIRMED_SPURIOUS_WOMAN_TAG_SLUGS = {"mhvs-hte", "mhvs-ptts", "mhvs-sebs"}


def main():
    with open(PROCESSED_PATH, encoding="utf-8") as f:
        records = json.load(f)

    changed = []
    seen_slugs = set()
    for record in records:
        seen_slugs.add(record["slug"])
        if record["slug"] not in CONFIRMED_SPURIOUS_WOMAN_TAG_SLUGS:
            continue
        old_group = list(record.get("occupation_target_group") or [])
        new_group = sorted(g for g in old_group if g != "woman")
        if new_group != sorted(old_group):
            changed.append((record["scheme_name"], old_group, new_group))
            record["occupation_target_group"] = new_group

    unmatched = CONFIRMED_SPURIOUS_WOMAN_TAG_SLUGS - seen_slugs
    if unmatched:
        print(f"WARNING: slug(s) not found in {PROCESSED_PATH}: {unmatched}")

    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Checked {len(records)} schemes, updated {len(changed)} occupation_target_group tag(s):")
    for name, old, new in changed:
        print(f"  {name}\n    before: {old}\n    after:  {new}")


if __name__ == "__main__":
    main()
