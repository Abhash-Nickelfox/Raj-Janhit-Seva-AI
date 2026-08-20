"""Unit tests for app.job_service - no API key / network needed."""
import sys
from pathlib import Path
from datetime import date

import pytest

from app.job_service import JobService, normalize_qualification, normalize_location

TODAY = date(2026, 8, 20)


def test_importing_job_service_does_not_import_config():
    # Run in a fresh interpreter - other test modules legitimately import app.config.
    import subprocess
    code = "import sys, app.job_service; sys.exit(1 if 'app.config' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], cwd=str(Path(__file__).resolve().parents[1])).returncode == 0


@pytest.fixture(scope="module")
def svc():
    return JobService(today=TODAY)


# --- title matching ---------------------------------------------------------

def test_patwari_all_closed_reports_closed_matches():
    jobs = [
        {"id": "A", "title": "Patwari", "department": "Revenue Department", "vacancies": 10,
         "qualification": "Graduate", "location": "Jaipur", "startDate": "2026-05-01",
         "lastDate": "2026-06-01", "salary": "₹20800 - ₹65900", "documentsNeeded": [],
         "officialApplicationPortal": "x"},
        {"id": "B", "title": "LDC", "department": "RSMSSB", "vacancies": 10,
         "qualification": "12th Pass", "location": "Kota", "startDate": "2026-08-01",
         "lastDate": "2026-09-30", "salary": "₹20800 - ₹65900", "documentsNeeded": [],
         "officialApplicationPortal": "x"},
    ]
    s = JobService(jobs=jobs, today=TODAY)
    results, info = s.search_with_info("patwari ki bharti")
    assert info["title_named"] is True
    assert info["matched_titles"] == ["Patwari"]
    assert results == []  # never padded with unrelated LDC rows
    assert [j["id"] for j in info["closed_matches"]] == ["A"]
    assert info["closed_matches"][0]["status"] == "closed"
    assert s.has_title_match("patwari ki bharti") is True


def test_rajasthan_me_job_is_not_a_title(svc):
    results, info = svc.search_with_info("Rajasthan me job")
    assert info["title_named"] is False
    assert info["matched_titles"] == []
    assert results
    assert svc.has_title_match("Rajasthan me job") is False


@pytest.mark.parametrize("q", ["job", "naukri", "sarkari naukri chahiye", "mujhe job ke liye kuch batao"])
def test_bare_queries_return_diverse_titles(svc, q):
    results, info = svc.search_with_info(q)
    assert info["title_named"] is False
    titles = [j["title"] for j in results]
    assert len(titles) == 5
    assert len(set(titles)) == 5  # one posting per title, not the same high-vacancy trio


def test_bare_query_not_sorted_by_vacancies(svc):
    results = svc.search("job")
    vac = [j["vacancies"] for j in results]
    assert vac != sorted(vac, reverse=True) or len(set(vac)) == 1


def test_ras_does_not_match_rajasthan(svc):
    assert svc.has_title_match("Rajasthan me sarkari naukri") is False
    _, info = svc.search_with_info("RAS ki taiyari kar raha hu")
    assert info["matched_titles"] == ["RAS Officer"]


def test_department_words_do_not_match(svc):
    assert svc.has_title_match("department job") is False
    assert svc.has_title_match("rajasthan") is False


@pytest.mark.parametrize("q,title", [
    ("ldc job jaipur me", "LDC"),
    ("clerk", "LDC"),
    ("police constable bharti", "Police Constable"),
    ("sipahi", "Police Constable"),
    ("पुलिस भर्ती", "Police Constable"),
    ("पटवारी की नौकरी", "Patwari"),
    ("patvari", "Patwari"),  # typo tolerance
    ("engineer bharti", "Junior Engineer"),
    ("shikshak", "Teacher Grade III"),
    ("teacher grade 3 vacancy", "Teacher Grade III"),
    ("vdo", "Village Development Officer"),
    ("gram sevak", "Village Development Officer"),
    ("vanrakshak", "Forest Guard"),
    ("computer instructor", "Computer Instructor"),
])
def test_aliases(svc, q, title):
    _, info = svc.search_with_info(q)
    assert info["matched_titles"][0] == title


def test_multiword_titles_do_not_cross_match(svc):
    _, info = svc.search_with_info("lab assistant")
    assert info["matched_titles"] == ["Lab Assistant"]
    _, info = svc.search_with_info("teacher")
    assert info["matched_titles"] == ["Teacher Grade III"]


def test_title_results_only_contain_that_title(svc):
    results, _ = svc.search_with_info("junior engineer")
    assert results and all(j["title"] == "Junior Engineer" for j in results)
    assert all(j["status"] == "active" for j in results)


def test_short_needle_ignored(svc):
    assert svc.has_title_match("je") is False  # below MIN_NEEDLE_LEN
    assert svc.has_title_match("") is False


# --- filters ----------------------------------------------------------------

def test_exclude_ids_honoured_on_plain_search(svc):
    first = svc.search("job")
    ids = [j["id"] for j in first]
    second = svc.search("job", exclude_ids=ids)
    assert second
    assert not set(ids) & {j["id"] for j in second}


def test_location_exact_and_hindi(svc):
    results, info = svc.search_with_info("job", location="जयपुर")
    assert results and all(j["location"] == "Jaipur" for j in results)
    assert info["relaxed"] == []
    assert info["location_requested"] == "जयपुर"
    results, _ = svc.search_with_info("job", location="JAIPUR")
    assert all(j["location"] == "Jaipur" for j in results)


def test_location_relaxed_is_flagged():
    jobs = [{"id": "A", "title": "LDC", "department": "RSMSSB", "vacancies": 10,
             "qualification": "12th Pass", "location": "Kota", "startDate": "2026-08-01",
             "lastDate": "2026-09-30", "salary": "₹20800 - ₹65900", "documentsNeeded": [],
             "officialApplicationPortal": "x"}]
    s = JobService(jobs=jobs, today=TODAY)
    results, info = s.search_with_info("ldc", location="Jaipur")
    assert [j["id"] for j in results] == ["A"]
    assert info["relaxed"] == ["location"]


def test_min_salary_relaxed_is_flagged(svc):
    results, info = svc.search_with_info("job", min_salary=10_000_000)
    assert results
    assert "min_salary" in info["relaxed"]


def test_qualification_fail_open_ranks_fit_first(svc):
    results, info = svc.search_with_info("job", qualification="12vi pass")
    assert info["qualification_normalized"] == "12th Pass"
    quals = [j["qualification"] for j in results]
    assert quals[0].startswith("12th Pass")
    # higher-tier postings still present later, never dropped
    assert any(not q.startswith("12th") for q in quals)


def test_tenth_pass_flagged(svc):
    _, info = svc.search_with_info("job", qualification="10th")
    assert info["below_min_qualification"] is True


# --- normalisers ------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("BCA", "BCA"),
    ("B.Tech", "B.Tech"),
    ("btech", "B.Tech"),
    ("B.Com", "Graduate"),
    ("maine B.Com kiya hai", "Graduate"),
    ("ITI", "Diploma"),
    ("polytechnic diploma", "Diploma"),
    ("10th", "10th Pass"),
    ("12th", "12th Pass"),
    ("12th Pass", "12th Pass"),
    ("12th Pass (Science)", "12th Pass (Science)"),
    ("graduate", "Graduate"),
    ("Graduation", "Graduate"),
    ("post graduate", "Post Graduate"),
    ("MA", "Post Graduate"),
    ("M.Sc", "Post Graduate"),
    ("diploma", "Diploma"),
    ("स्नातक", "Graduate"),
    ("kuch nahi", None),
    ("", None),
    (None, None),
])
def test_normalize_qualification(text, expected):
    assert normalize_qualification(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("jaipur", "Jaipur"), ("जोधपुर", "Jodhpur"), ("Kota city", "Kota"), ("Delhi", None), (None, None),
])
def test_normalize_location(text, expected):
    assert normalize_location(text) == expected


# --- data sanity ------------------------------------------------------------

def test_dataset_is_sane(svc):
    jobs = svc.jobs
    assert len({(j["title"], j["location"]) for j in jobs}) == len(jobs)  # no dupes
    assert "status" not in jobs[0]
    statuses = [svc.get(j["id"])["status"] for j in jobs]
    assert statuses.count("active") >= 40
    assert statuses.count("coming soon") > 0 and statuses.count("closed") > 0
    expected = {"Patwari": "Graduate", "LDC": "12th Pass", "Police Constable": "12th Pass",
                "Junior Engineer": "B.Tech", "Computer Instructor": "BCA", "RAS Officer": "Graduate"}
    for j in jobs:
        if j["title"] in expected:
            assert j["qualification"] == expected[j["title"]]
        assert "Rajasthan Department" not in j["department"]


def test_get(svc):
    j = svc.get("RJ001")
    assert j and j["status"] in {"active", "coming soon", "closed"}
    assert svc.get("nope") is None
