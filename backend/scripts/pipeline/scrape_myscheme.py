"""Scrape Rajasthan government scheme data from myScheme.gov.in's public API.

myscheme.gov.in is a Next.js SPA - the page itself carries no scheme data.
All data comes from api.myscheme.gov.in, which the site's own frontend calls
with a client-side API key embedded in its JS bundle (public, not a secret -
every browser visiting the site sends the same key). Endpoints and request
shapes below were confirmed live on 2026-08-05 by inspecting that bundle and
calling the API directly; myScheme has changed its API version before, so if
this starts failing, re-derive it the same way (view the bundle's requests
in a browser's Network tab, or re-grep the downloaded JS chunks for
"api.myscheme.gov.in").

Key discovery: the *structured* eligibility facets (age range per category,
income ceilings, gender/caste/marital-status value codes, etc.) are returned
natively by the API in schemes/v6/public/schemes?lang=hi - NOT lang=en -
under data["hi"]["eligibilityCriteria"]. The lang=en response for the same
scheme only has free-text eligibility description, no structured fields.
This was verified across multiple schemes and is exploited here rather than
building an LLM-based eligibility extractor: this data is already
machine-coded by the source, an LLM would only be re-deriving what the API
already gives us.
"""

import json
import time
from datetime import date
from pathlib import Path

import requests

API_BASE = "https://api.myscheme.gov.in"
API_KEY = "tYTy5eEhlu9rFjyxuCr7ra7ACp4dv1RH8gWuHTDc"
HEADERS = {
    "x-api-key": API_KEY,
    "Origin": "https://www.myscheme.gov.in",
    "Referer": "https://www.myscheme.gov.in/",
    "User-Agent": "Mozilla/5.0 (compatible; JanSevaSchemeBot/1.0)",
}

SEARCH_MAX_SIZE = 100  # API rejects size > 100 with a 412
REQUEST_DELAY_SECONDS = 0.4  # be a polite client - this is someone else's public API

RAJASTHAN_STATE_FILTERS = [
    {"identifier": "beneficiaryState", "value": "Rajasthan"},
    {"identifier": "level", "value": "State"},
]

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _get(url, params=None):
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def search_schemes(filters, keyword="", sort="", size=SEARCH_MAX_SIZE):
    """Page through /search/v6/schemes for the given facet filters.

    Yields each hit's `fields` dict (slug, schemeName, briefDescription,
    tags, schemeCategory, level, beneficiaryState, ...).
    """
    q = json.dumps(filters)
    from_ = 0
    while True:
        data = _get(
            f"{API_BASE}/search/v6/schemes",
            params={
                "lang": "en",
                "q": q,
                "keyword": keyword,
                "sort": sort,
                "from": from_,
                "size": size,
            },
        )
        hits = data["data"]["hits"]
        for item in hits["items"]:
            yield item["fields"]
        page = hits["page"]
        from_ += page["size"]
        time.sleep(REQUEST_DELAY_SECONDS)
        if from_ >= page["total"]:
            break


def fetch_scheme_detail(slug, lang):
    return _get(
        f"{API_BASE}/schemes/v6/public/schemes", params={"slug": slug, "lang": lang}
    )["data"]


def fetch_documents(scheme_id, lang):
    return _get(
        f"{API_BASE}/schemes/v6/public/schemes/{scheme_id}/documents",
        params={"lang": lang},
    )["data"]


def fetch_faqs(scheme_id, lang):
    return _get(
        f"{API_BASE}/schemes/v6/public/schemes/{scheme_id}/faqs", params={"lang": lang}
    )["data"]


def scrape_scheme(slug):
    """Fetch en + hi detail/documents/faqs for one scheme, merge into one record."""
    en = fetch_scheme_detail(slug, "en")
    time.sleep(REQUEST_DELAY_SECONDS)
    hi = fetch_scheme_detail(slug, "hi")
    time.sleep(REQUEST_DELAY_SECONDS)

    scheme_id = en["_id"]

    docs_en = fetch_documents(scheme_id, "en")
    time.sleep(REQUEST_DELAY_SECONDS)
    docs_hi = fetch_documents(scheme_id, "hi")
    time.sleep(REQUEST_DELAY_SECONDS)

    faqs_en = fetch_faqs(scheme_id, "en")
    time.sleep(REQUEST_DELAY_SECONDS)
    faqs_hi = fetch_faqs(scheme_id, "hi")
    time.sleep(REQUEST_DELAY_SECONDS)

    return {
        "scheme_id": scheme_id,
        "slug": slug,
        "en": en.get("en", {}),
        "hi": hi.get("hi", {}),
        "eligibility_structured": hi.get("hi", {}).get("eligibilityCriteria", {}),
        "documents": {"en": docs_en.get("en", {}), "hi": docs_hi.get("hi", {})},
        "faqs": {"en": faqs_en.get("en", {}), "hi": faqs_hi.get("hi", {})},
    }


def main():
    snapshot_dir = RAW_DIR / date.today().isoformat()
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching Rajasthan state scheme listing...")
    listing = list(search_schemes(RAJASTHAN_STATE_FILTERS))
    print(f"Found {len(listing)} Rajasthan state schemes")

    (snapshot_dir / "_listing.json").write_text(
        json.dumps(listing, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    failures = []
    for i, entry in enumerate(listing, 1):
        slug = entry["slug"]
        out_path = snapshot_dir / f"{slug}.json"
        if out_path.exists():
            continue  # resume-safe: re-running the script skips already-scraped schemes

        print(f"[{i}/{len(listing)}] {slug}")
        try:
            record = scrape_scheme(slug)
        except requests.HTTPError as e:
            print(f"  FAILED: {e}")
            failures.append(slug)
            continue

        out_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"Done. {len(listing) - len(failures)} scraped, {len(failures)} failed.")
    if failures:
        print("Failed slugs:", failures)


if __name__ == "__main__":
    main()
