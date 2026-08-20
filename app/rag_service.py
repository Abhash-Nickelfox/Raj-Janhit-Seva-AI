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

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from rapidfuzz import fuzz

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chromadb"
OVERRIDES_PATH = DATA_DIR / "processed" / "scheme_overrides.json"
# Single source of truth - scripts/build_knowledge_base.py imports this, so
# the index and the query side can never silently drift onto different
# models. ~128-token input window (the tail of anything longer is ignored).
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
KB_BUILD_COMMAND = "python scripts/build_knowledge_base.py"

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
NAME_MATCH_THRESHOLD = 80  # whole-word coverage score, 0-100 (see _name_match_score)
NAME_MATCH_BOOST = 10.0
# Two different schemes (different canonical groups) scoring within this
# many points of each other on the name match means the query doesn't
# decisively name either - no boost rather than a coin flip. Members of
# the SAME canonical group (pension twins) tying is fine: they're the same
# programme and get deduped anyway.
NAME_MATCH_TIE_MARGIN = 5
# A needle shorter than this (and made of a single content token) is too
# little to be a scheme-name lookup at all - "job" used to boost the "Work
# From Home-Job Work" scheme, "old" matched "Older Pension" etc.
NAME_MATCH_MIN_CHARS = 8
NAME_MATCH_MIN_CHARS_DEVANAGARI = 5  # matras share code points with consonants: "पालनहार" is 7, "पेंशन" 5
NAME_MATCH_MIN_TOKENS = 2
# Per-token fuzzy tolerance so "yojna"/"yojana", "kisaan"/"kisan",
# "palanhaar"/"palanhar" still count as the same word. Calibrated so that
# "vidhva"/"vridha" (widow vs old-age, ratio 83) does NOT.
TOKEN_FUZZY_THRESHOLD = 86
TOKEN_FUZZY_MIN_LEN = 5

# A known income above the matching ceiling is a DEMOTION, not an
# exclusion: income figures citizens state are rough, ceilings are
# routinely waived for BPL/Antyodaya etc., and the exact ceiling type
# (family/individual/parent) rarely maps cleanly onto what was said.
INCOME_OVER_CEILING_PENALTY = 4.0
NAME_MATCH_FALLBACK_DISTANCE = 15.0  # baseline "distance" for a name-matched scheme absent from the semantic pool - roughly this corpus's median observed distance

# A genuine bare-name query ("krishi yantra") names exactly one scheme. A
# generic phrase that happens to be a substring of many scheme titles
# ("financial assistance" matches 6+ "Financial Assistance Scheme for X"
# titles with full coverage) is not a name lookup at all - verified this
# corpus: "krishi yantra" hits 1 scheme, "financial assistance" hits 6.
# Above this many distinct canonical groups, treat the phrase as generic
# and skip NAME_MATCH_BOOST entirely rather than letting every incidental
# title match win top rank.
NAME_MATCH_DISTINCT_LIMIT = 2

SHORTLIST_POOL_SIZE = 100  # how many raw semantic hits to pull before filtering/ranking

# Filler words dropped before whole-word name matching - English, Hinglish
# and Devanagari - so "krishi yantra ke baare mein batao" has the content
# tokens {krishi, yantra} and nothing else. Generic scheme-title words
# ("yojana", "scheme", "rajasthan") are fillers too: they're in dozens of
# titles and carry no identifying signal.
NAME_STOPWORDS = {  # (compared after NFC normalisation + lowercasing)
    # English
    "a", "an", "the", "of", "for", "to", "in", "on", "about", "and", "or", "is", "are", "i", "me", "my",
    "want", "need", "get", "how", "what", "which", "tell", "apply", "information", "info", "details",
    "scheme", "schemes", "yojana", "yojna", "yojanaen", "programme", "program", "government", "govt",
    "rajasthan", "sarkari", "sarkar", "state",
    # Hinglish
    "ke", "ki", "ka", "ko", "se", "mein", "me", "main", "mai", "mujhe", "mujhko", "hai", "hain", "ho",
    "kya", "kaise", "kaun", "konsi", "kaunsi", "chahiye", "chahie", "liye", "lie", "baare", "bare",
    "batao", "bataiye", "bataye", "bataen", "jankari", "jaankari", "milegi", "milega", "mil", "sakta",
    "sakti", "hu", "hun", "mera", "meri", "mere", "aur", "ya", "par", "ek", "koi",
    # Devanagari
    "योजना", "योजनाएं", "स्कीम", "सरकारी", "सरकार", "राजस्थान", "के", "की", "का", "को", "से", "में",
    "मैं", "मुझे", "है", "हैं", "हूँ", "हूं", "क्या", "कैसे", "कौन", "कौनसी", "चाहिए", "लिए", "बारे",
    "बताओ", "बताइए", "बताएं", "जानकारी", "मिलेगी", "मिलेगा", "मिल", "सकता", "सकती", "मेरा", "मेरी",
    "मेरे", "और", "या", "पर", "एक", "कोई",
}
NAME_STOPWORDS = {unicodedata.normalize("NFC", w) for w in NAME_STOPWORDS}
_TOKEN_SPLIT_RE = re.compile(r"[\s\-–—/,.()|:;!?\"'’‘“”]+")

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
    "farmer": ["farmer", "kisan", "krishak", "krishi", "किसान", "कृषक", "कृषि"],
    "student": ["student", "scholarship", "chatra", "vidyarthi", "छात्र", "छात्रा", "विद्यार्थी", "छात्रवृत्ति"],
    "widow": ["widow", "vidhwa", "ekal nari", "विधवा", "एकल नारी"],
    "differently_abled": ["disab", "divyang", "viklang", "yogyajan", "specially abled", "specially-abled", "pwd", "दिव्यांग", "विकलांग", "विशेष योग्यजन"],
    "unemployed": ["unemploy", "berojgar", "बेरोजगार"],
    # NB: "pension" alone is NOT proof of old age (widow/disability/WWII
    # pensions) - extraction additionally requires age_min >= 55 before it
    # keeps this tag (see scripts/extract_eligibility.py SENIOR_CITIZEN_MIN_AGE).
    "senior_citizen": ["senior citizen", "old age", "vridha", "vriddh", "pension", "वृद्ध", "बुजुर्ग", "वरिष्ठ नागरिक", "पेंशन"],
    "woman": ["women", "mahila", "girl", "beti", "kanya", "महिला", "बेटी", "कन्या", "लड़की"],
    "artisan": ["artisan", "weaver", "handicraft", "shilpkar", "शिल्पकार", "बुनकर", "कारीगर"],
    "labourer": ["labour", "worker", "shramik", "mazdoor", "श्रमिक", "मजदूर"],
    "entrepreneur": ["entrepreneur", "business", "udyam", "startup", "व्यवसाय", "उद्यम", "व्यापार", "बिजनेस", "बिज़नेस"],
    "ex_servicemen": ["ex-servicemen", "ex servicemen", "sainik", "veteran", "war widow", "सैनिक", "पूर्व सैनिक"],
}


@dataclass
class CitizenContext:
    """Whatever is currently known about the citizen. Any field left None
    is simply never checked by the eligibility filter (fail-open)."""

    caste: Optional[str] = None
    gender: Optional[str] = None
    marital_status: Optional[str] = None
    age: Optional[int] = None
    family_income: Optional[int] = None
    individual_income: Optional[int] = None  # the applicant's own annual income, when stated separately
    parent_income: Optional[int] = None  # for student schemes gated on the parents' income
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
    # soft signals - the scheme stays in the list but ranks lower
    demotion_reason: Optional[str] = None
    # other scheme_ids that are the SAME programme (canonical_group twins)
    # and were folded into this entry by shortlist()'s dedupe
    sibling_ids: list = field(default_factory=list)


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
        age_min, age_max = _age_min_for(meta, citizen.gender), meta.get("age_max")
        if age_min is not None and citizen.age < age_min:
            return False, f"age: citizen={citizen.age} below scheme min {age_min}"
        if age_max is not None and citizen.age > age_max:
            return False, f"age: citizen={citizen.age} above scheme max {age_max}"

    # Income is deliberately NOT an exclusion - see income_demotion().

    # A scheme requiring the 'differently_abled' occupation_target_group is
    # (per this corpus's actual data) FOR disabled applicants specifically,
    # not merely open to them alongside everyone else - so an explicit "no
    # disability" is a real contradiction, same fail-open principle as every
    # other check above: only excludes on an explicit stated fact, never on
    # missing information (found live 2026-08-10).
    if citizen.disability == "no" and "differently_abled" in _split(meta.get("occupation_target_group", "")):
        return False, "occupation_target_group: citizen explicitly stated no disability, scheme requires it"

    return True, None


def _age_min_for(meta, gender):
    """Gender-specific minimum age when the scheme has one (e.g. 55 for
    women / 58 for men on the Rajasthan old-age pensions); the overall
    minimum otherwise (and for an unknown gender - fail-open: the lower of
    the two is the one that can't wrongly exclude anyone)."""
    if gender:
        g = gender.lower()
        if g == "female" and meta.get("age_min_female") is not None:
            return meta["age_min_female"]
        if g == "male" and meta.get("age_min_male") is not None:
            return meta["age_min_male"]
    return meta.get("age_min")


def income_demotion(meta, citizen: CitizenContext) -> Optional[str]:
    """Soft counterpart of eligibility_verdict for income: a reason string
    when a KNOWN citizen income exceeds the scheme ceiling of the SAME
    type (family vs individual vs parent), else None. Never compares
    across types - a stated family income says nothing definite about
    the applicant's individual income ceiling. Callers demote, never
    exclude, on this."""
    pairs = (
        ("family", citizen.family_income, meta.get("family_income_ceiling")),
        ("individual", citizen.individual_income, meta.get("individual_income_ceiling")),
        ("parent", citizen.parent_income, meta.get("parent_income_ceiling")),
    )
    for label, income, ceiling in pairs:
        if income is not None and ceiling is not None and income > ceiling:
            return f"income: citizen {label} income {income} exceeds ceiling {ceiling}"
    return None


_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def _tokens(text: str) -> list:
    """Lowercased, NFC-normalised whole-word content tokens with
    NAME_STOPWORDS removed. NFC matters for Devanagari: "ढ़" arrives
    either precomposed (U+095D) or as DDHA + NUKTA depending on the
    keyboard/STT, and they must compare equal."""
    text = unicodedata.normalize("NFC", text).lower()
    return [t for t in _TOKEN_SPLIT_RE.split(text) if t and t not in NAME_STOPWORDS]


def _min_chars_for(token: str) -> int:
    return NAME_MATCH_MIN_CHARS_DEVANAGARI if _DEVANAGARI_RE.search(token) else NAME_MATCH_MIN_CHARS


def _token_in(token: str, haystack: Iterable[str]) -> bool:
    for other in haystack:
        if token == other:
            return True
        if (
            len(token) >= TOKEN_FUZZY_MIN_LEN
            and len(other) >= TOKEN_FUZZY_MIN_LEN
            and fuzz.ratio(token, other) >= TOKEN_FUZZY_THRESHOLD
        ):
            return True
    return False


def is_name_needle(query: str) -> bool:
    """Whether the query carries enough to be a scheme-name lookup at all:
    at least NAME_MATCH_MIN_TOKENS content tokens, or a single one of at
    least NAME_MATCH_MIN_CHARS characters ("palanhar", "chiranjeevi")."""
    toks = _tokens(query)
    if len(toks) >= NAME_MATCH_MIN_TOKENS:
        return True
    return bool(toks) and len(toks[0]) >= _min_chars_for(toks[0])


def _name_match_score(query, names: Iterable[str]) -> float:
    """0-100: the share of the query's content tokens found (whole-word,
    lightly fuzzy) in the best-matching of this scheme's names/aliases.

    Whole-word coverage rather than character partial_ratio because the
    latter on a short needle matched the WRONG scheme with near-perfect
    scores: "vridha pension" -> "...VIDHVA Pension" (93), "widow pension"
    -> "World War II Pension", "old age pension" -> a farmers' pension.
    Query-side coverage (not name-side) so a partial title still scores
    100 ("krishi yantra" against "Krishi Yantra Anudan Yojana"), while
    sharing only a generic token ("pension") scores 50 or less.
    """
    if not is_name_needle(query):
        return 0.0
    q_tokens = _tokens(query)
    best = 0.0
    for name in names:
        if not name:
            continue
        n_tokens = _tokens(name)
        if not n_tokens:
            continue
        matched = sum(1 for t in q_tokens if _token_in(t, n_tokens))
        best = max(best, 100.0 * matched / len(q_tokens))
    return best


def score_names(query: str, names_by_scheme: dict, group_of=None) -> tuple:
    """Name-match every scheme and decide whether the result is decisive.

    names_by_scheme: scheme_id -> iterable of names/aliases.
    group_of: optional scheme_id -> canonical_group (None when ungrouped).
    Returns (scores_at_or_above_threshold, decisive). Not decisive when
    the phrase matches too many distinct schemes (a generic title
    fragment, see NAME_MATCH_DISTINCT_LIMIT) or when the top two are
    different programmes within NAME_MATCH_TIE_MARGIN of each other.
    """
    scores = {}
    for scheme_id, names in names_by_scheme.items():
        score = _name_match_score(query, names)
        if score >= NAME_MATCH_THRESHOLD:
            scores[scheme_id] = score
    if not scores:
        return scores, False

    def _grp(scheme_id):
        return (group_of(scheme_id) if group_of else None) or f"_solo:{scheme_id}"

    distinct_groups = {_grp(sid) for sid in scores}
    if len(distinct_groups) > NAME_MATCH_DISTINCT_LIMIT:
        return scores, False
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if len(ranked) >= 2:
        (top_id, top), (second_id, second) = ranked[0], ranked[1]
        if _grp(top_id) != _grp(second_id) and top - second < NAME_MATCH_TIE_MARGIN:
            return scores, False
    return scores, True


def load_overrides(path: Path = OVERRIDES_PATH) -> dict:
    """slug -> override entry from data/processed/scheme_overrides.json
    (curated, hand-verified corrections layered on top of the mechanical
    extraction: facet fixes, canonical_group, aliases). Missing file ->
    {} so nothing here is load-bearing for startup."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {slug: entry for slug, entry in data.items() if not slug.startswith("_")}


# keys in an override entry that are NOT eligibility facets
OVERRIDE_META_KEYS = {"_reason", "aliases"}


def apply_overrides_to_record(record: dict, override: Optional[dict]) -> dict:
    """Layer one override entry over an eligibility_structured.json-style
    record (list-valued facets). Returns a new dict; the record is not
    mutated. Only facet keys are applied (aliases/_reason are skipped)."""
    out = dict(record)
    if not override:
        return out
    for key, value in override.items():
        if key in OVERRIDE_META_KEYS:
            continue
        out[key] = value
    return out


def apply_overrides_to_metadata(meta: dict, override: Optional[dict]) -> dict:
    """Same, for a Chroma-style metadata dict (list facets joined with
    "|", None means "drop the key")."""
    out = dict(meta)
    if not override:
        return out
    for key, value in override.items():
        if key in OVERRIDE_META_KEYS:
            continue
        if value is None:
            out.pop(key, None)
        elif isinstance(value, list):
            out[key] = "|".join(str(v) for v in value)
        else:
            out[key] = value
    return out


def dedupe_by_group(ranked: list, group_of) -> list:
    """Collapse canonical_group twins (e.g. the two listings of the farmer
    old-age pension) down to the best-ranked member, recording the
    folded scheme_ids on the survivor's sibling_ids AND in its metadata
    ("sibling_ids", "|"-joined) so a plain-dict consumer sees them too.
    Ungrouped schemes pass through untouched. Input must already be
    sorted best-first."""
    kept = []
    leader_by_group = {}
    for scheme in ranked:
        group = group_of(scheme.scheme_id)
        if not group:
            kept.append(scheme)
            continue
        leader = leader_by_group.get(group)
        if leader is None:
            leader_by_group[group] = scheme
            kept.append(scheme)
            continue
        leader.sibling_ids.append(scheme.scheme_id)
        leader.metadata = dict(leader.metadata, sibling_ids="|".join(leader.sibling_ids))
    return kept


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
        # imported here, not at module top, so the pure functions above stay
        # importable/testable without the embedding + Chroma stack installed
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        if not CHROMA_DIR.exists():
            raise RuntimeError(
                f"Knowledge base not found at {CHROMA_DIR} - build it first with: {KB_BUILD_COMMAND}"
            )
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
        # Chroma silently creates an EMPTY collection when the directory
        # exists but holds nothing (or a stale/partial build) - which used
        # to surface only as "no_match" for every single need. Fail loudly.
        self._card_count = self.cards._collection.count()
        if self._card_count == 0:
            raise RuntimeError(
                f"Knowledge base at {CHROMA_DIR} has no discovery cards - (re)build it with: {KB_BUILD_COMMAND}"
            )

        self._overrides_by_slug = load_overrides()
        self._init_index(self.cards.get(include=["metadatas"])["metadatas"])

    def _init_index(self, card_metadatas):
        """Build the per-scheme lookup tables from the card metadata
        (separate from __init__ so tests can feed fake metadata)."""
        # one representative card's metadata per scheme_id, for the hybrid
        # name-match pass in shortlist() - built once, not per-query, since
        # it's a fixed, small (157-scheme) list. Overrides are layered on
        # here too, so a corrected facet/group takes effect without a
        # rebuild of the index.
        self._meta_by_scheme_id = {}
        self._override_by_scheme_id = {}
        for meta in card_metadatas:
            scheme_id = meta["scheme_id"]
            if scheme_id in self._meta_by_scheme_id:
                continue
            override = self._overrides_by_slug.get(meta.get("slug"))
            self._override_by_scheme_id[scheme_id] = override
            self._meta_by_scheme_id[scheme_id] = apply_overrides_to_metadata(meta, override)

        self._group_by_scheme_id = {
            sid: (meta.get("canonical_group") or None) for sid, meta in self._meta_by_scheme_id.items()
        }
        self._names_by_scheme_id = {}
        for sid, meta in self._meta_by_scheme_id.items():
            names = {(meta.get("scheme_name") or "").strip(), (meta.get("scheme_name_hi") or "").strip()}
            override = self._override_by_scheme_id.get(sid) or {}
            names.update(a.strip() for a in override.get("aliases", []) or [])
            names.discard("")
            self._names_by_scheme_id[sid] = names

    def is_ready(self) -> bool:
        """True once the discovery_cards collection is loaded and non-empty
        (the constructor already raises otherwise, so this is for health
        checks that hold a reference)."""
        return getattr(self, "_card_count", 0) > 0 and bool(self._meta_by_scheme_id)

    def group_of(self, scheme_id: str) -> Optional[str]:
        """canonical_group of a scheme (None when it has no known twins)."""
        return self._group_by_scheme_id.get(scheme_id)

    def siblings_of(self, scheme_id: str) -> list:
        """Other scheme_ids in the same canonical_group."""
        group = self.group_of(scheme_id)
        if not group:
            return []
        return [sid for sid, g in self._group_by_scheme_id.items() if g == group and sid != scheme_id]

    def names_for(self, scheme_id: str) -> set:
        """Every name this scheme goes by: English name, Hindi name and all
        curated aliases (both scripts) - so a "did the reply mention this
        scheme" check works whichever script the model answered in."""
        return set(self._names_by_scheme_id.get(scheme_id, set()))

    def scheme_meta(self, scheme_id: str) -> Optional[dict]:
        """Override-corrected representative metadata for a scheme."""
        return self._meta_by_scheme_id.get(scheme_id)

    def _apply_meta_overrides(self, meta: dict) -> dict:
        return apply_overrides_to_metadata(meta, self._overrides_by_slug.get(meta.get("slug")))

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
        # a scheme has an en AND a hi card; both can land in the pool -
        # keep the BEST distance per scheme (hits arrive best-first, so
        # a plain dict comprehension would have kept the worse one)
        pool = {}
        for doc, distance in raw_hits:
            scheme_id = doc.metadata["scheme_id"]
            existing = pool.get(scheme_id)
            if existing is None or distance < existing[1]:
                pool[scheme_id] = (self._apply_meta_overrides(doc.metadata), distance)

        # hybrid recall: pull in any scheme whose NAME closely matches the
        # query even if semantic search didn't retrieve it in the top pool
        # at all - short/bare-name queries carry too little signal for
        # embeddings alone (verified: "krishi yantra" misses its own exact
        # scheme in the top 100 semantic hits for some phrasings).
        name_scores, is_specific_name_lookup = score_names(query, self._names_by_scheme_id, self.group_of)
        for scheme_id in name_scores:
            if scheme_id not in pool:
                pool[scheme_id] = (self._meta_by_scheme_id[scheme_id], NAME_MATCH_FALLBACK_DISTANCE)
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
            demotion = income_demotion(meta, citizen)
            if demotion and alt_citizen is not None and income_demotion(meta, alt_citizen) is None:
                demotion = None
            if demotion:
                adjusted_score += INCOME_OVER_CEILING_PENALTY
                meta = dict(meta, income_over_ceiling=demotion)
            best_by_scheme[scheme_id] = ScoredScheme(
                scheme_id=scheme_id,
                scheme_name=meta["scheme_name"],
                lang=meta["lang"],
                score=adjusted_score,
                metadata=meta,
                demotion_reason=demotion,
            )

        ranked = sorted(best_by_scheme.values(), key=lambda s: s.score)
        ranked = dedupe_by_group(ranked, self.group_of)
        if trace is not None:
            trace["survivors"] = len(ranked)
            trace["demoted"] = [(s.scheme_name, s.demotion_reason) for s in ranked if s.demotion_reason]
            trace["deduped"] = {s.scheme_name: s.sibling_ids for s in ranked if s.sibling_ids}
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
