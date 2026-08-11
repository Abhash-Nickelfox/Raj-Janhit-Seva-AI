"""Test Phase 2's actual Stage A/B retrieval (app/rag_service.py) - the
fail-open eligibility filter + category/benefit-type rank boost - as
opposed to query_kb.py, which only shows raw semantic search with no
filtering at all. Use this to compare "before" (query_kb.py) vs "after"
(this) on the same query.

Usage:
  python scripts/query_retrieval.py "pension for widow" --gender female --marital-status widowed --benefit pension
  python scripts/query_retrieval.py "SC scholarship" --caste sc
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from app.rag_service import RagService, CitizenContext  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--caste")
    parser.add_argument("--gender")
    parser.add_argument("--marital-status")
    parser.add_argument("--age", type=int)
    parser.add_argument("--income", type=int)
    parser.add_argument("--residence")
    parser.add_argument("--category", help="need_category - matched against scheme category for a rank boost")
    parser.add_argument("--benefit", help="need_benefit_type - matched against benefit_type_fine for a rank boost")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    citizen = CitizenContext(
        caste=args.caste,
        gender=args.gender,
        marital_status=args.marital_status,
        age=args.age,
        family_income=args.income,
        residence=args.residence,
        need_category=args.category,
        need_benefit_type=args.benefit,
    )

    print("Loading knowledge base...")
    svc = RagService()

    trace = {}
    results = svc.shortlist(args.query, citizen, top_n=args.top, trace=trace)

    print(f"\nQuery: {args.query!r}  Citizen: {citizen}")
    print(f"Candidates before filter: {trace['candidates_before_filter']}, dropped: {len(trace['dropped'])}, survivors: {trace['survivors']}")
    if trace["dropped"]:
        print("Dropped (eligibility contradiction):")
        for d in trace["dropped"][:10]:
            print(f"  - {d['scheme_name']}: {d['reason']}")

    print(f"\nTop {args.top} (Stage A shortlist, filtered + boosted):")
    for rank, r in enumerate(results, 1):
        print(f"{rank}. [{r.score:.4f}] {r.scheme_name} ({r.lang})")


if __name__ == "__main__":
    main()
