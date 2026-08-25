"""Phase 2 (PLAN.md SS3): two-stage retrieval on top of the Chroma knowledge
base built by build_knowledge_base.py.

Stage A (shortlist): semantic search over discovery_cards, narrowed by a
fail-open eligibility filter over what's known about the citizen, then
re-ranked with a category/benefit-type match boost.

Stage B (detail): pull only the field_chunks that matter for a specific
scheme + the fields the current question actually needs.

No conversation memory here (that's Phase 3) - this module takes whatever
citizen facts and need/category the caller already knows and answers a
single, self-contained query. It's deliberately testable standalone via
scripts/query_retrieval.py, independent of any chat/session logic.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chromadb"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Semantic distance is unbounded but in this corpus observed roughly in the
# 5-25 range (lower = more similar). Boosts are score REDUCTIONS applied
# before sorting (still lower = better) - sized to meaningfully re-rank
# within that observed range without swamping semantic relevance entirely.
CATEGORY_MATCH_BOOST = 3.0
BENEFIT_TYPE_MATCH_BOOST = 5.0

# Hybrid recall: semantic search alone is weak on short/bare scheme-name
# queries - verified "krishi yantra" ranks the actual "Krishi Yantra" scheme
# 9th of 76 eligible candidates via embeddings alone. A fuzzy match against
# the scheme's own name (English or Hindi) is a much bigger, decisive boost,
# and is also used to pull a strongly name-matched scheme into the candidate
# pool even when semantic search didn't retrieve it in the top pool at all.
NAME_MATCH_THRESHOLD = 80  # rapidfuzz partial_ratio, 0-100
NAME_MATCH_BOOST = 10.0
NAME_MATCH_FALLBACK_DISTANCE = 15.0  # baseline "distance" for a name-matched scheme absent from the semantic pool - roughly this corpus's median observed distance

# A genuine bare-name query ("krishi yantra") names exactly one scheme. A
# generic phrase that happens to be a substring of many scheme titles
# ("financial assistance" matches 6+ "Financial Assistance Scheme for X"
# titles at partial_ratio=100) is not a name lookup at all - verified this
# corpus: "krishi yantra" hits 1 scheme, "financial assistance" hits 6.
# Above this count, treat the phrase as generic and skip NAME_MATCH_BOOST
# entirely rather than letting every incidental title match win top rank.
NAME_MATCH_DISTINCT_LIMIT = 2

SHORTLIST_POOL_SIZE = 100  # how many raw semantic hits to pull before filtering/ranking

# Keyword table used to derive occupation_target_group during extraction
# (scripts/extract_eligibility.py) AND, at conversation time, to check
# whether a citizen's stated NEED already implies membership in a target
# group before treating that restriction as an unconfirmed identity gap
# worth asking about (see app/chat.py's _identity_gap) - defined once here
# since both consumers need the exact same mapping, not two copies that
# could drift apart. Deliberately small and literal - meant to catch the
# common, obvious cases; anything missed just fails open on both ends
# (extraction leaves the facet unpopulated; the gap-check asks rather than
# silently guessing).
# Devanagari entries alongside every Romanized one (2026-08-11) - need_text
# is REQUIRED to be written in Devanagari for any Hindi/Hinglish message
# (see NeedItem.need_text's own field description - verified this retrieves
# far better than Romanized text), so a Romanized-only keyword list here
# silently never matches for the majority of this app's actual users.
# Confirmed live: "बेटी की शादी" (daughter's marriage) never matched
# 'woman'/'beti' at all, so a scheme tagged occupation_target_group='woman'
# with NO OTHER restriction (gender/marital_status/caste all 'all') still
# showed up as having an unconfirmed gap, causing a much narrower, unlikely
# scheme (widow-only) to be explained/asked about first instead - the exact
# "narrow scheme wins the gap-check race" failure mode already fixed
# elsewhere, just via a different route (a script-mismatch false negative,
# not a ranking issue).
OCCUPATION_KEYWORDS = {
    "farmer": ["farmer", "kisan", "krishi", "किसान", "कृषि"],
    "student": ["student", "scholarship", "chatra", "vidyarthi", "छात्र", "छात्रा", "विद्यार्थी", "छात्रवृत्ति"],
    "widow": ["widow", "vidhwa", "ekal nari", "विधवा", "एकल नारी"],
    "differently_abled": ["disab", "divyang", "viklang", "दिव्यांग", "विकलांग"],
    "unemployed": ["unemploy", "berojgar", "बेरोजगार"],
    "senior_citizen": ["senior citizen", "old age", "vridha", "pension", "वृद्ध", "बुजुर्ग", "वरिष्ठ नागरिक", "पेंशन"],
    "woman": ["women", "mahila", "girl", "beti", "kanya", "महिला", "बेटी", "कन्या", "लड़की"],
    "artisan": ["artisan", "weaver", "handicraft", "shilpkar", "शिल्पकार", "बुनकर", "कारीगर"],
    "labourer": ["labour", "worker", "shramik", "mazdoor", "श्रमिक", "मजदूर"],
    "entrepreneur": ["entrepreneur", "business", "udyam", "startup", "व्यवसाय", "उद्यम", "व्यापार", "बिजनेस", "बिज़नेस"],
    "ex_servicemen": ["ex-servicemen", "ex servicemen", "sainik", "veteran", "war widow", "सैनिक", "पूर्व सैनिक"],
    # Added 2026-08-24 - was missing entirely, so every "Transgender Utthan
    # Kosh" scheme (7 variants) had NO occupation_target_group signal for
    # this restriction at all. Safe to match from scheme name/description/
    # tags alone (the usual, narrow signal_text extract_eligibility.py
    # scans) - every one of these schemes literally has "Transgender" in its
    # own name, unlike the differently_abled gap below, which needed reading
    # the full eligibility text since euphemistic names ("Vishesh Yogyajan")
    # don't self-describe the same way.
    "transgender": ["transgender", "trans ", "kinnar", "third gender", "किन्नर", "ट्रांसजेंडर", "तृतीय लिंग"],
}

# NOTE on differently_abled under-tagging (2026-08-24): several schemes
# whose name is a euphemism ("Vishesh Yogyajan", "Specially Abled") rather
# than a literal match for this group's keywords are NOT reliably catchable
# by keyword-matching AT ALL, even against the full eligibility text -
# tried and reverted: scanning eligibility text directly hit real false
# positives from negation ("should NOT have any disability"), "preference
# will be given to X" (non-mandatory), and inclusive OR-lists that also
# list "General Category" (so the restriction doesn't actually gate
# anyone). This needs a human reading each candidate's real text, the same
# reason classify_education_level() above uses an LLM instead of a keyword
# table for its own facet - see scripts/fix_occupation_tags.py for the
# small, hand-verified list of schemes actually confirmed this way.


@dataclass
class CitizenContext:
    """Whatever is currently known about the citizen. Any field left None
    is simply never checked by the eligibility filter (fail-open)."""

    caste: Optional[str] = None
    gender: Optional[str] = None
    marital_status: Optional[str] = None
    age: Optional[int] = None
    family_income: Optional[int] = None
    residence: Optional[str] = None
    education_level: Optional[str] = None  # "school" | "higher_education" - see extract_eligibility.py's classify_education_level
    disability: Optional[str] = None  # "yes" | "no" - only set when explicitly stated either way
    # what the CURRENT query is about - not persisted profile, just this call's need
    need_category: Optional[str] = None
    need_benefit_type: Optional[str] = None


@dataclass
class ScoredScheme:
    scheme_id: str
    scheme_name: str
    lang: str
    score: float
    metadata: dict
    excluded_reason: Optional[str] = None


def _split(value):
    return value.split("|") if value else []


def eligibility_verdict(meta, citizen: CitizenContext):
    """Fail-open check: exclude a scheme ONLY on an explicit contradiction
    between a known citizen fact and a restrictive scheme facet. A citizen
    fact that's unknown (None), or a scheme facet that's unrestricted
    ("all"/missing), never causes an exclusion.

    Returns (eligible: bool, reason: str | None).
    """
    checks = [
        ("caste", citizen.caste, meta.get("caste", "")),
        ("gender", citizen.gender, meta.get("gender", "")),
        ("marital_status", citizen.marital_status, meta.get("marital_status", "")),
        ("residence", citizen.residence, meta.get("residence", "")),
        ("education_level", citizen.education_level, meta.get("education_level", "")),
    ]
    for label, citizen_value, scheme_value in checks:
        if citizen_value is None:
            continue
        allowed = _split(scheme_value)
        if not allowed or "all" in allowed:
            continue
        if citizen_value.lower() not in [a.lower() for a in allowed]:
            return False, f"{label}: citizen={citizen_value!r} not in scheme's {allowed!r}"

    if citizen.age is not None:
        age_min, age_max = meta.get("age_min"), meta.get("age_max")
        if age_min is not None and citizen.age < age_min:
            return False, f"age: citizen={citizen.age} below scheme min {age_min}"
        if age_max is not None and citizen.age > age_max:
            return False, f"age: citizen={citizen.age} above scheme max {age_max}"

    if citizen.family_income is not None:
        ceiling = meta.get("family_income_ceiling")
        if ceiling is not None and citizen.family_income > ceiling:
            return False, f"income: citizen={citizen.family_income} exceeds ceiling {ceiling}"

    # A scheme requiring the 'differently_abled' occupation_target_group is
    # (per this corpus's actual data) FOR disabled applicants specifically,
    # not merely open to them alongside everyone else - so an explicit "no
    # disability" is a real contradiction, same fail-open principle as every
    # other check above: only excludes on an explicit stated fact, never on
    # missing information (found live 2026-08-10).
    if citizen.disability == "no" and "differently_abled" in _split(meta.get("occupation_target_group", "")):
        return False, "occupation_target_group: citizen explicitly stated no disability, scheme requires it"

    return True, None


def _name_match_score(query, meta):
    """0-100 best fuzzy match between the query and this scheme's name, in
    whichever language (English or Hindi) matches better. partial_ratio
    (not plain ratio) so a query with extra words around the name
    ("krishi yantra ke baare mein batao") still scores near-perfectly on
    the substring that IS the name."""
    query_lower = query.lower()
    best = 0.0
    name_en = meta.get("scheme_name") or ""
    if name_en:
        best = max(best, fuzz.partial_ratio(query_lower, name_en.lower()))
    name_hi = meta.get("scheme_name_hi") or ""
    if name_hi:
        best = max(best, fuzz.partial_ratio(query, name_hi))
    return best


def _identity_query_terms(citizen: CitizenContext) -> str:
    """Known identity facts as extra search terms. eligibility_verdict and
    _rank_boost only ever see these facts as filters/category-boosts - the
    embedding search itself never learns them, so a generic query like
    "financial support" from a citizen already known to be a widow can't
    semantically favor widow-specific schemes. Appending them to the query
    text is how that already-known context reaches the search itself."""
    terms = []
    for value in (citizen.marital_status, citizen.gender, citizen.caste, citizen.residence, citizen.education_level):
        if value and value.lower() != "all":
            terms.append(value)
    return " ".join(terms)


def _rank_boost(meta, citizen: CitizenContext, name_score: float = 0.0):
    boost = 0.0
    if citizen.need_category and citizen.need_category.lower() in (meta.get("category", "") or "").lower():
        boost += CATEGORY_MATCH_BOOST
    if citizen.need_benefit_type and citizen.need_benefit_type.lower() in _split(meta.get("benefit_type_fine", "")):
        boost += BENEFIT_TYPE_MATCH_BOOST
    if name_score >= NAME_MATCH_THRESHOLD:
        boost += NAME_MATCH_BOOST * (name_score / 100)
    return boost


class RagService:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.cards = Chroma(
            collection_name="discovery_cards",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        self.chunks = Chroma(
            collection_name="field_chunks",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        # one representative card's metadata per scheme_id, for the hybrid
        # name-match pass in shortlist() - built once, not per-query, since
        # it's a fixed, small (157-scheme) list.
        self._meta_by_scheme_id = {}
        for meta in self.cards.get(include=["metadatas"])["metadatas"]:
            self._meta_by_scheme_id.setdefault(meta["scheme_id"], meta)

    def shortlist(self, query: str, citizen: CitizenContext, alt_citizen: Optional[CitizenContext] = None,
                  top_n: int = 5, trace: Optional[dict] = None):
        """Stage A: filter + rank. Returns up to top_n ScoredScheme, best first.

        alt_citizen: an alternate candidate identity to also check when a
        need is asked on behalf of a family member. Not every eligibility
        facet on a matched scheme is necessarily about that family member -
        e.g. a scheme assisting a WIDOW with her daughter's marriage gates on
        the widow/applicant's own marital status, not the daughter's, even
        though the need itself is "for" the daughter. Rather than guess
        which facet belongs to which person, a scheme is only excluded if
        NEITHER candidate identity fits - the same fail-open principle
        ("never exclude on missing info") extended to also cover "never
        exclude on a fact that may have been checked against the wrong
        person" (found live 2026-08-10).
        """
        identity_terms = _identity_query_terms(citizen)
        search_query = f"{query} {identity_terms}".strip() if identity_terms else query
        raw_hits = self.cards.similarity_search_with_score(search_query, k=SHORTLIST_POOL_SIZE)
        pool = {doc.metadata["scheme_id"]: (doc.metadata, distance) for doc, distance in raw_hits}

        # hybrid recall: pull in any scheme whose NAME closely matches the
        # query even if semantic search didn't retrieve it in the top pool
        # at all - short/bare-name queries carry too little signal for
        # embeddings alone (verified: "krishi yantra" misses its own exact
        # scheme in the top 100 semantic hits for some phrasings).
        name_scores = {}
        for scheme_id, meta in self._meta_by_scheme_id.items():
            score = _name_match_score(query, meta)
            if score >= NAME_MATCH_THRESHOLD:
                name_scores[scheme_id] = score
                if scheme_id not in pool:
                    pool[scheme_id] = (meta, NAME_MATCH_FALLBACK_DISTANCE)

        is_specific_name_lookup = len(name_scores) <= NAME_MATCH_DISTINCT_LIMIT
        effective_name_scores = name_scores if is_specific_name_lookup else {}

        if trace is not None:
            trace["candidates_before_filter"] = len(pool)
            trace["dropped"] = []
            trace["name_matches"] = name_scores
            trace["name_boost_applied"] = is_specific_name_lookup
            trace["identity_terms"] = identity_terms

        best_by_scheme = {}
        for scheme_id, (meta, distance) in pool.items():
            eligible, reason = eligibility_verdict(meta, citizen)
            if not eligible and alt_citizen is not None:
                eligible_alt, _ = eligibility_verdict(meta, alt_citizen)
                eligible = eligible_alt
            if not eligible:
                if trace is not None:
                    trace["dropped"].append({"scheme_name": meta["scheme_name"], "reason": reason})
                continue

            adjusted_score = distance - _rank_boost(meta, citizen, effective_name_scores.get(scheme_id, 0.0))
            existing = best_by_scheme.get(scheme_id)
            if existing is None or adjusted_score < existing.score:
                best_by_scheme[scheme_id] = ScoredScheme(
                    scheme_id=scheme_id,
                    scheme_name=meta["scheme_name"],
                    lang=meta["lang"],
                    score=adjusted_score,
                    metadata=meta,
                )

        ranked = sorted(best_by_scheme.values(), key=lambda s: s.score)
        if trace is not None:
            trace["survivors"] = len(ranked)
            trace["shortlist"] = [(s.scheme_name, round(s.score, 4)) for s in ranked[:top_n]]

        return ranked[:top_n]

    def scheme_detail(self, scheme_id: str, lang: str, field_types: Optional[list] = None):
        """Stage B: pull only the field chunks that matter for this scheme
        (and, if given, only these field_type(s))."""
        clauses = [{"scheme_id": scheme_id}, {"lang": lang}]
        if field_types:
            clauses.append({"field_type": {"$in": field_types}})
        where = {"$and": clauses}
        results = self.chunks.get(where=where, include=["documents", "metadatas"])
        return list(zip(results["documents"], results["metadatas"]))

    def get_discovery_card(self, scheme_id: str, lang: str) -> str:
        """The pre-synthesized one-paragraph card already used for Stage A
        search (name + who it's for + benefit type + a short description) -
        reused here as a lightweight scheme summary distinct from the full
        field-chunk text, e.g. for generate_node's reply when a clarifying
        question is needed and a full eligibility/benefit dump would bury
        it. Falls back to the English card if the requested language's
        version isn't found, same pattern as scheme_detail's callers use."""
        where = {"$and": [{"scheme_id": scheme_id}, {"lang": lang}]}
        results = self.cards.get(where=where, include=["documents"])
        if results["documents"]:
            return results["documents"][0]
        if lang != "en":
            return self.get_discovery_card(scheme_id, "en")
        return ""
