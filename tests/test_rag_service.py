"""Unit tests for the pure (Chroma-free) parts of app/rag_service.py:
whole-word name matching, override application, canonical-group dedupe
and the eligibility verdict. RagService itself needs the embedding +
Chroma stack and is exercised via its _init_index() path with fake card
metadata instead of a real collection."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import rag_service as rs  # noqa: E402
from app.rag_service import (  # noqa: E402
    CitizenContext,
    RagService,
    ScoredScheme,
    apply_overrides_to_metadata,
    apply_overrides_to_record,
    dedupe_by_group,
    eligibility_verdict,
    income_demotion,
    is_name_needle,
    load_overrides,
    score_names,
)

OVERRIDES = load_overrides()
ELIG = {r["slug"]: r for r in json.loads((rs.DATA_DIR / "processed" / "eligibility_structured.json").read_text("utf-8"))}


def _names_by_slug():
    """slug -> names set built the same way RagService._init_index does,
    from the regenerated eligibility JSON + overrides (no Chroma)."""
    out = {}
    for slug, rec in ELIG.items():
        names = {rec["scheme_name"] or ""}
        names.update(OVERRIDES.get(slug, {}).get("aliases", []))
        names.discard("")
        out[slug] = names
    return out


NAMES = _names_by_slug()
GROUP = {slug: rec.get("canonical_group") for slug, rec in ELIG.items()}


def _top(query):
    scores, decisive = score_names(query, NAMES, GROUP.get)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return ranked, decisive


# --------------------------------------------------------------------- names


class TestNameMatch:
    def test_short_needle_never_matches(self):
        assert not is_name_needle("job")
        assert not is_name_needle("old")
        scores, decisive = score_names("job", NAMES, GROUP.get)
        assert scores == {} and not decisive

    def test_single_long_token_is_a_needle(self):
        assert is_name_needle("palanhar")
        assert is_name_needle("chiranjeevi")
        assert is_name_needle("पालनहार")
        assert not is_name_needle("काम")

    def test_stopwords_do_not_count_as_tokens(self):
        assert not is_name_needle("yojana ke baare mein batao")

    @pytest.mark.parametrize(
        "query, expected_slugs",
        [
            ("vridha pension", {"ignvpyr", "mvspy"}),
            ("old age pension", {"ignvpyr", "mvspy"}),
            ("वृद्धावस्था पेंशन", {"ignvpyr", "mvspy"}),
            ("बुढ़ापा पेंशन", {"ignvpyr", "mvspy"}),
            ("widow pension", {"ignvpyrr", "cmenspy"}),
            ("विधवा पेंशन", {"ignvpyrr", "cmenspy"}),
            ("kisan pension", {"cmsmfops", "slesvkspy"}),
            ("किसान पेंशन", {"cmsmfops", "slesvkspy"}),
            ("viklang pension", {"cmsasps", "mvyspy"}),
            ("दिव्यांग पेंशन", {"cmsasps", "mvyspy"}),
            ("palanhar", {"ps", "saps"}),
            ("पालनहार योजना", {"ps", "saps"}),
            ("chiranjeevi", {"mcsby"}),
            ("krishi yantra", {"ky"}),
            ("krishi yantra ke baare mein batao", {"ky"}),
            ("कृषि यंत्र", {"ky"}),
        ],
    )
    def test_aliases_resolve_to_the_right_programme(self, query, expected_slugs):
        ranked, decisive = _top(query)
        hits = {slug for slug, score in ranked if score >= rs.NAME_MATCH_THRESHOLD}
        assert hits == expected_slugs, ranked
        assert decisive, ranked

    def test_vridha_pension_does_not_hit_widow_pension(self):
        ranked, _ = _top("vridha pension")
        assert "ignvpyrr" not in dict(ranked)

    def test_widow_pension_does_not_hit_world_war_pension(self):
        ranked, _ = _top("widow pension")
        assert "wwiip" not in dict(ranked)

    def test_old_age_pension_does_not_hit_farmer_pension(self):
        ranked, _ = _top("old age pension")
        assert "cmsmfops" not in dict(ranked)
        assert "slesvkspy" not in dict(ranked)

    def test_job_does_not_hit_work_from_home(self):
        scores, _ = score_names("job", NAMES, GROUP.get)
        assert "mwfhjwy" not in scores

    def test_generic_phrase_is_not_decisive(self):
        scores, decisive = score_names("financial assistance", NAMES, GROUP.get)
        assert len(scores) > rs.NAME_MATCH_DISTINCT_LIMIT
        assert not decisive

    def test_tie_between_different_programmes_is_not_decisive(self):
        names = {"a": {"Samman Pension Yojana"}, "b": {"Kisan Samman Pension Nidhi"}}
        scores, decisive = score_names("samman pension", names, lambda sid: None)
        assert scores == {"a": 100.0, "b": 100.0}
        assert not decisive

    def test_tie_within_same_group_is_decisive(self):
        names = {"a": {"Palanhar Scheme"}, "b": {"Vishesh Yogyajan Palanhar Yojana"}}
        scores, decisive = score_names("palanhar", names, lambda sid: "palanhar")
        assert set(scores) == {"a", "b"}
        assert decisive

    def test_fuzzy_token_tolerates_spelling_but_not_vidhva_vs_vridha(self):
        assert rs._token_in("yojna", ["yojana"])
        assert rs._token_in("palanhaar", ["palanhar"])
        assert not rs._token_in("vidhva", ["vridha"])
        assert not rs._token_in("older", ["old"])


# ----------------------------------------------------------------- overrides


class TestOverrides:
    def test_record_override_replaces_only_facets(self):
        rec = {"slug": "x", "occupation_target_group": ["student"], "residence": ["rural"]}
        out = apply_overrides_to_record(rec, {"_reason": "r", "aliases": ["foo"], "occupation_target_group": ["differently_abled"]})
        assert out["occupation_target_group"] == ["differently_abled"]
        assert out["residence"] == ["rural"]
        assert "aliases" not in out and "_reason" not in out
        assert rec["occupation_target_group"] == ["student"]  # not mutated

    def test_metadata_override_joins_lists_and_drops_none(self):
        meta = {"slug": "x", "occupation_target_group": "student|woman", "age_min": 1}
        out = apply_overrides_to_metadata(meta, {"occupation_target_group": ["differently_abled"], "age_min": None, "canonical_group": "g"})
        assert out["occupation_target_group"] == "differently_abled"
        assert "age_min" not in out
        assert out["canonical_group"] == "g"

    def test_overrides_file_is_well_formed(self):
        assert OVERRIDES, "data/processed/scheme_overrides.json missing or empty"
        for slug, entry in OVERRIDES.items():
            assert slug in ELIG, f"override for unknown slug {slug}"
            assert entry.get("_reason"), f"{slug}: every override needs a _reason"

    @pytest.mark.parametrize(
        "slug, expected",
        [
            ("msyr", ["voluntary_organisation"]),
            ("saps", ["differently_abled"]),
            ("sdvay", ["differently_abled"]),
            ("ssfsap", ["student", "differently_abled"]),
            ("cmsasps", ["differently_abled"]),
            ("slesvkspy", ["farmer", "senior_citizen"]),
            ("msveay", []),
        ],
    )
    def test_regenerated_json_has_corrected_facets(self, slug, expected):
        assert ELIG[slug]["occupation_target_group"] == expected

    def test_regenerated_json_misc_fixes(self):
        assert ELIG["lpy"]["residence"] == ["all"]
        for slug in ("imspesy", "mjpmsky", "msveay", "nsjbsy"):
            assert ELIG[slug]["scheme_name"] == ELIG[slug]["scheme_name"].strip()
        for slug in ("cmenspy", "wwiip", "fasfj", "ignvpyrr", "cmdrsr", "nsjbsy", "mvyspy", "cmsasps"):
            assert "senior_citizen" not in ELIG[slug]["occupation_target_group"], slug
        for rec in ELIG.values():
            if "senior_citizen" in rec["occupation_target_group"]:
                assert rec["age_min"] is not None and rec["age_min"] >= 55, rec["slug"]
        assert ELIG["cmsasps"]["age_min"] is None  # 1-100 means unrestricted
        assert ELIG["mvspy"]["age_min_female"] == 55 and ELIG["mvspy"]["age_min_male"] == 58
        assert ELIG["cmsmfops"]["age_min_male"] == 58

    def test_canonical_groups(self):
        assert GROUP["cmsmfops"] == GROUP["slesvkspy"] == "farmer_old_age_pension"
        assert GROUP["cmsasps"] == GROUP["mvyspy"] == "disability_pension"
        assert GROUP["ignvpyr"] == GROUP["mvspy"] == "old_age_pension"
        assert GROUP["ps"] == GROUP["saps"] == "palanhar"
        assert GROUP["ky"] is None


# -------------------------------------------------------------------- dedupe


def _scored(sid, score, name=None):
    return ScoredScheme(scheme_id=sid, scheme_name=name or sid, lang="en", score=score, metadata={"scheme_id": sid})


class TestDedupe:
    def test_keeps_best_ranked_member_and_records_siblings(self):
        groups = {"a": "pension", "b": "pension", "c": None, "d": "pension"}
        ranked = [_scored("a", 1.0), _scored("c", 2.0), _scored("b", 3.0), _scored("d", 4.0)]
        kept = dedupe_by_group(ranked, groups.get)
        assert [s.scheme_id for s in kept] == ["a", "c"]
        assert kept[0].sibling_ids == ["b", "d"]
        assert kept[0].metadata["sibling_ids"] == "b|d"
        assert kept[1].sibling_ids == [] and "sibling_ids" not in kept[1].metadata

    def test_ungrouped_schemes_untouched(self):
        ranked = [_scored("x", 1.0), _scored("y", 2.0)]
        assert [s.scheme_id for s in dedupe_by_group(ranked, lambda sid: None)] == ["x", "y"]


# -------------------------------------------------------------- eligibility


class TestEligibilityVerdict:
    def test_unknown_facts_never_exclude(self):
        meta = {"gender": "female", "age_min": 60, "family_income_ceiling": 100}
        assert eligibility_verdict(meta, CitizenContext())[0]

    def test_gender_mismatch_excludes(self):
        ok, reason = eligibility_verdict({"gender": "female"}, CitizenContext(gender="male"))
        assert not ok and "gender" in reason

    def test_age_bounds(self):
        meta = {"age_min": 18, "age_max": 40}
        assert not eligibility_verdict(meta, CitizenContext(age=17))[0]
        assert eligibility_verdict(meta, CitizenContext(age=18))[0]
        assert not eligibility_verdict(meta, CitizenContext(age=41))[0]

    def test_gender_specific_age_min(self):
        meta = {"age_min": 55, "age_min_female": 55, "age_min_male": 58}
        assert eligibility_verdict(meta, CitizenContext(age=56, gender="female"))[0]
        assert not eligibility_verdict(meta, CitizenContext(age=56, gender="male"))[0]
        assert eligibility_verdict(meta, CitizenContext(age=56))[0]  # unknown gender -> fail-open minimum

    def test_income_never_excludes_but_demotes(self):
        meta = {"family_income_ceiling": 100000}
        citizen = CitizenContext(family_income=500000)
        assert eligibility_verdict(meta, citizen)[0]
        assert income_demotion(meta, citizen)
        assert income_demotion(meta, CitizenContext(family_income=50000)) is None

    def test_income_ceiling_type_must_match(self):
        meta = {"individual_income_ceiling": 48000}
        assert income_demotion(meta, CitizenContext(family_income=500000)) is None
        assert income_demotion(meta, CitizenContext(individual_income=60000))
        assert income_demotion({"parent_income_ceiling": 10}, CitizenContext(parent_income=11))

    def test_explicit_no_disability_excludes_disability_scheme(self):
        meta = {"occupation_target_group": "differently_abled"}
        assert not eligibility_verdict(meta, CitizenContext(disability="no"))[0]
        assert eligibility_verdict(meta, CitizenContext())[0]


# -------------------------------------------------- RagService index (fake)


def _fake_service():
    svc = RagService.__new__(RagService)
    svc._card_count = 4
    svc._overrides_by_slug = OVERRIDES
    cards = []
    for slug in ("ignvpyr", "mvspy", "ky", "cmsasps"):
        rec = ELIG[slug]
        for lang in ("en", "hi"):
            cards.append({
                "scheme_id": f"id-{slug}", "slug": slug, "scheme_name": rec["scheme_name"],
                "scheme_name_hi": "", "lang": lang, "occupation_target_group": "student|senior_citizen", "age_min": 1,
            })
    svc._init_index(cards)
    return svc


class TestRagServiceIndex:
    def test_is_ready_group_of_names_for(self):
        svc = _fake_service()
        assert svc.is_ready()
        assert svc.group_of("id-ignvpyr") == "old_age_pension"
        assert svc.siblings_of("id-ignvpyr") == ["id-mvspy"]
        assert svc.group_of("id-ky") is None
        names = svc.names_for("id-ignvpyr")
        assert "Indira Gandhi National Vrdhavastha Pension Yojana" in names
        assert "वृद्धावस्था पेंशन" in names and "old age pension" in names
        assert svc.names_for("nope") == set()

    def test_overrides_applied_to_loaded_metadata(self):
        svc = _fake_service()
        meta = svc.scheme_meta("id-cmsasps")
        assert meta["occupation_target_group"] == "differently_abled"
        assert meta["canonical_group"] == "disability_pension"
        assert svc._meta_by_scheme_id["id-cmsasps"] is meta  # chat.py reads this attr directly


# ------------------------------------------------------ shortlist (stubbed)


class _Doc:
    def __init__(self, metadata):
        self.metadata = metadata


class _FakeCards:
    """Stands in for the Chroma discovery_cards collection: returns a
    fixed best-first hit list regardless of the query."""

    def __init__(self, hits):
        self.hits = hits

    def similarity_search_with_score(self, query, k):
        return [(_Doc(meta), dist) for meta, dist in self.hits][:k]


def _card(slug, lang, **extra):
    rec = ELIG[slug]
    meta = {"scheme_id": f"id-{slug}", "slug": slug, "scheme_name": rec["scheme_name"], "scheme_name_hi": "", "lang": lang}
    meta.update(extra)
    return meta


class TestShortlistStubbed:
    def _svc(self, hits):
        svc = RagService.__new__(RagService)
        svc._card_count = len(hits)
        svc._overrides_by_slug = OVERRIDES
        svc.cards = _FakeCards(hits)
        svc._init_index([meta for meta, _ in hits])
        return svc

    def test_pool_keeps_best_distance_per_scheme(self):
        # hi card of ky ranks 1st, en card 3rd: the scheme must be scored on the better one
        hits = [(_card("ky", "hi"), 5.0), (_card("kkt", "en"), 6.0), (_card("ky", "en"), 9.0)]
        svc = self._svc(hits)
        out = svc.shortlist("tarbandi subsidy details", CitizenContext(), top_n=5)
        by_id = {s.scheme_id: s for s in out}
        assert by_id["id-ky"].score == pytest.approx(5.0)
        assert by_id["id-ky"].lang == "hi"

    def test_name_boost_and_group_dedupe_with_siblings(self):
        hits = [(_card("ky", "en"), 5.0), (_card("mvspy", "en"), 8.0), (_card("ignvpyr", "en"), 9.0)]
        svc = self._svc(hits)
        trace = {}
        out = svc.shortlist("budhapa pension", CitizenContext(), top_n=5, trace=trace)
        assert out[0].scheme_id == "id-mvspy"  # name boost lifts it over the closer semantic hit
        ids = [s.scheme_id for s in out]
        assert "id-ignvpyr" not in ids  # folded into its old_age_pension twin
        assert out[0].sibling_ids == ["id-ignvpyr"]
        assert out[0].metadata["sibling_ids"] == "id-ignvpyr"
        assert trace["name_boost_applied"] is True

    def test_income_demotes_instead_of_excluding(self):
        hits = [(_card("ky", "en", family_income_ceiling=100000), 5.0), (_card("kkt", "en"), 6.0)]
        svc = self._svc(hits)
        trace = {}
        out = svc.shortlist("fencing", CitizenContext(family_income=300000), top_n=5, trace=trace)
        ids = [s.scheme_id for s in out]
        assert ids == ["id-kkt", "id-ky"]  # ky still listed, now ranked below
        assert out[1].demotion_reason and out[1].metadata["income_over_ceiling"]
        assert trace["dropped"] == []

    def test_overridden_facet_is_what_the_verdict_sees(self):
        # cmsasps card still carries the stale 'student' tag from an old index build;
        # the override makes it differently_abled, so an explicit "no disability" excludes it
        hits = [(_card("cmsasps", "en", occupation_target_group="student|senior_citizen"), 5.0)]
        svc = self._svc(hits)
        trace = {}
        out = svc.shortlist("samman pension", CitizenContext(disability="no"), top_n=5, trace=trace)
        assert out == [] and trace["dropped"][0]["reason"].startswith("occupation_target_group")
