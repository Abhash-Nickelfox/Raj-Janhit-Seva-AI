"""Multi-turn conversation stress tests for Phase 3 (app/graph/chat.py).

Unlike eval/golden_set.py (mostly deterministic, no/minimal LLM calls),
this replays real multi-turn conversations through the live graph - real
Groq calls every turn. Run deliberately, not on every save.

Each scenario is a list of (message, check_fn) turns. check_fn receives
the graph state AFTER that turn (plus the state from the previous turn,
for before/after comparisons) and returns a list of (name, passed, detail)
tuples. Checks are behavioral (profile keys, shown_scheme identity, intent,
category, turn_type) rather than exact text matches, since LLM phrasing
varies turn to turn even at temperature=0 - that's inherent to an
LLM-based system, not a determinism bug to chase.

Run: python eval/conversation_tests.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from app.graph.chat import build_graph  # noqa: E402

results = []


def check(scenario, turn_idx, name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] turn {turn_idx}: {name}" + (f" -- {detail}" if detail else ""))
    results.append((scenario, turn_idx, name, passed, detail))


# ---------------------------------------------------------------------------
# Scenario 1: cross-topic profile persistence (the original motivating case)
# ---------------------------------------------------------------------------

def scenario_cross_topic_persistence():
    return [
        (
            "main SC hoon, mera annual income 150000 hai, scholarship chahiye",
            lambda s, prev: [
                ("caste extracted into profile", s["profile"].get("caste", "").lower() == "sc", f"got {s['profile']}"),
                ("income extracted into profile", s["profile"].get("family_income") == 150000, f"got {s['profile']}"),
                ("category = Education", s.get("need_category") == "Education & Learning", f"got {s.get('need_category')}"),
            ],
        ),
        (
            "krishi ke liye koi yojana hai kya",
            lambda s, prev: [
                ("caste STILL in profile after topic switch", s["profile"].get("caste", "").lower() == "sc", f"got {s['profile']}"),
                ("income STILL in profile after topic switch", s["profile"].get("family_income") == 150000, f"got {s['profile']}"),
                ("category switched to Agriculture", s.get("need_category") == "Agriculture,Rural & Environment", f"got {s.get('need_category')}"),
                ("shown_scheme changed from turn 1", s.get("shown_scheme_name") != prev.get("shown_scheme_name"), f"still {s.get('shown_scheme_name')}"),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Scenario 2: follow-up resolves to the MOST RECENT scheme, across a switch
# ---------------------------------------------------------------------------

def scenario_followup_after_switch():
    return [
        ("scholarship for SC students", lambda s, prev: [
            ("not a followup (fresh need)", s.get("turn_type") == "new_need", f"got {s.get('turn_type')}"),
        ]),
        ("krishi yantra ke baare mein batao", lambda s, prev: [
            ("shown_scheme switched to a farming scheme", "Krishi" in (s.get("shown_scheme_name") or ""), f"got {s.get('shown_scheme_name')}"),
        ]),
        ("iske liye kaise apply karu", lambda s, prev: [
            ("detected as followup", s.get("turn_type") == "followup_same_scheme", f"got {s.get('turn_type')}"),
            ("shown_scheme UNCHANGED (still the farming one, not reverted)", s.get("shown_scheme_name") == prev.get("shown_scheme_name"), f"was {prev.get('shown_scheme_name')}, now {s.get('shown_scheme_name')}"),
            ("intent = how_to_apply", s.get("intent") == "how_to_apply", f"got {s.get('intent')}"),
        ]),
    ]


# ---------------------------------------------------------------------------
# Scenario 3: multi-fact accumulation across separate turns (not just the
# last-stated fact overwriting everything else)
# ---------------------------------------------------------------------------

def scenario_fact_accumulation():
    return [
        ("main mahila hoon", lambda s, prev: [
            ("gender extracted", s["profile"].get("gender", "").lower() == "female", f"got {s['profile']}"),
        ]),
        ("meri age 45 saal hai", lambda s, prev: [
            ("gender STILL present", s["profile"].get("gender", "").lower() == "female", f"got {s['profile']}"),
            ("age newly added", s["profile"].get("age") == 45, f"got {s['profile']}"),
        ]),
        ("main vidhwa hoon", lambda s, prev: [
            ("gender still present", s["profile"].get("gender", "").lower() == "female", f"got {s['profile']}"),
            ("age still present", s["profile"].get("age") == 45, f"got {s['profile']}"),
            ("marital_status newly added", s["profile"].get("marital_status", "").lower() == "widowed", f"got {s['profile']}"),
        ]),
    ]


# ---------------------------------------------------------------------------
# Scenario 4: correcting a previously stated fact updates it, not appends
# ---------------------------------------------------------------------------

def scenario_fact_correction():
    return [
        ("meri annual income 500000 hai", lambda s, prev: [
            ("income = 500000", s["profile"].get("family_income") == 500000, f"got {s['profile']}"),
        ]),
        ("sorry galti se bola, meri income 100000 hai", lambda s, prev: [
            ("income CORRECTED to 100000, not stuck at 500000", s["profile"].get("family_income") == 100000, f"got {s['profile']}"),
        ]),
    ]


# ---------------------------------------------------------------------------
# Scenario 5: chitchat interleaved shouldn't wipe profile/focus
# ---------------------------------------------------------------------------

def scenario_chitchat_preserves_context():
    return [
        ("main OBC hoon, business loan ke liye scheme chahiye", lambda s, prev: [
            ("caste extracted", s["profile"].get("caste", "").lower() == "obc", f"got {s['profile']}"),
            ("shown_scheme set", bool(s.get("shown_scheme_name")), ""),
        ]),
        ("thank you so much", lambda s, prev: [
            ("classified as chitchat", s.get("intent") == "chitchat", f"got {s.get('intent')}"),
            ("profile SURVIVES the chitchat turn", s["profile"].get("caste", "").lower() == "obc", f"got {s['profile']}"),
            ("shown_scheme SURVIVES the chitchat turn", s.get("shown_scheme_name") == prev.get("shown_scheme_name"), f"was {prev.get('shown_scheme_name')}, now {s.get('shown_scheme_name')}"),
        ]),
        ("iske eligibility criteria kya hai", lambda s, prev: [
            ("followup still resolves correctly after the chitchat turn", s.get("turn_type") == "followup_same_scheme", f"got {s.get('turn_type')}"),
            ("shown_scheme still the same one from turn 1", s.get("shown_scheme_name") == prev.get("shown_scheme_name"), ""),
        ]),
    ]


# ---------------------------------------------------------------------------
# Scenario 6: vague need -> clarify -> specific answer
# ---------------------------------------------------------------------------

def scenario_vague_then_clarify():
    return [
        ("mujhe sarkar se paisa chahiye", lambda s, prev: [
            ("no category/benefit extracted (genuinely vague)", not s.get("need_category") and not s.get("need_benefit_type"), f"got category={s.get('need_category')} benefit={s.get('need_benefit_type')}"),
            ("a clarifying question was asked, not a scheme dump", "?" in (s.get("last_reply") or "") and len(s.get("last_reply") or "") < 400, f"reply len={len(s.get('last_reply') or '')}"),
        ]),
        ("mujhe apna khud ka business shuru karna hai", lambda s, prev: [
            ("category now resolved to Business", s.get("need_category") == "Business & Entrepreneurship", f"got {s.get('need_category')}"),
            ("a real scheme was shown this time", bool(s.get("shown_scheme_name")), ""),
        ]),
    ]


# ---------------------------------------------------------------------------
# Scenario 7: mixed-script follow-up (Hindi turn, then English follow-up)
# ---------------------------------------------------------------------------

def scenario_mixed_script_followup():
    return [
        ("मुझे विधवा पेंशन योजना चाहिए", lambda s, prev: [
            ("reply_lang detected as hi", s.get("reply_lang") == "hi", f"got {s.get('reply_lang')}"),
            ("shown_scheme set", bool(s.get("shown_scheme_name")), ""),
        ]),
        ("what documents are needed for this", lambda s, prev: [
            ("reply_lang switches to en for this message", s.get("reply_lang") == "en", f"got {s.get('reply_lang')}"),
            ("still correctly detected as a followup despite script switch", s.get("turn_type") == "followup_same_scheme", f"got {s.get('turn_type')}"),
            ("shown_scheme unchanged across the script switch", s.get("shown_scheme_name") == prev.get("shown_scheme_name"), ""),
        ]),
    ]


# ---------------------------------------------------------------------------
# Scenario 8: out-of-scope / adversarial message doesn't crash or hallucinate
# ---------------------------------------------------------------------------

def scenario_out_of_scope():
    return [
        ("aaj mausam kaisa hai", lambda s, prev: [
            ("handled as chitchat, not forced into a scheme search", s.get("intent") == "chitchat", f"got {s.get('intent')}"),
            ("no scheme was fabricated as 'shown'", not s.get("shown_scheme_name"), f"got {s.get('shown_scheme_name')}"),
        ]),
    ]


# ---------------------------------------------------------------------------
# Scenario 9: a fact stated about a family member must not leak onto the
# citizen's own profile - regression for the 2026-08-10 fix where mentioning
# "beti"/"ladka" (daughter/son) made the model hallucinate the CITIZEN's own
# gender/marital_status, and a mentioned child's education level got
# attributed to the citizen instead. Fixed by explicit "about the citizen
# themselves, not a family member" instructions in extract_node's (formerly
# extract_profile_node's) prompt text (field-level Pydantic descriptions
# alone did not work).
# ---------------------------------------------------------------------------

def scenario_profile_third_party_isolation():
    return [
        ("mujhe thoda sa scholarship chahiye beti ke liye", lambda s, prev: [
            ("citizen's own gender NOT hallucinated from mentioning a daughter", "gender" not in s.get("profile", {}), f"got {s['profile']}"),
            ("citizen's own marital_status NOT hallucinated from mentioning a daughter", "marital_status" not in s.get("profile", {}), f"got {s['profile']}"),
        ]),
        ("main khud OBC hoon, aur mera beta MCA graduate hai", lambda s, prev: [
            ("citizen's own caste correctly captured", s["profile"].get("caste", "").lower() == "obc", f"got {s['profile']}"),
            ("son's education_level NOT attributed to the citizen", s["profile"].get("education_level") is None, f"got {s['profile']}"),
        ]),
    ]


SCENARIOS = [
    ("cross_topic_persistence", scenario_cross_topic_persistence),
    ("followup_after_switch", scenario_followup_after_switch),
    ("fact_accumulation", scenario_fact_accumulation),
    ("fact_correction", scenario_fact_correction),
    ("chitchat_preserves_context", scenario_chitchat_preserves_context),
    ("vague_then_clarify", scenario_vague_then_clarify),
    ("mixed_script_followup", scenario_mixed_script_followup),
    ("out_of_scope", scenario_out_of_scope),
    ("profile_third_party_isolation", scenario_profile_third_party_isolation),
]


def main():
    print("Building conversation graph...")
    graph = build_graph()

    for name, scenario_fn in SCENARIOS:
        print(f"\n=== Scenario: {name} ===")
        config = {"configurable": {"thread_id": f"stress-{name}"}}
        prev_state = {}
        for turn_idx, (message, check_fn) in enumerate(scenario_fn(), 1):
            print(f" you> {message}")
            state = graph.invoke({"raw_message": message}, config=config)
            print(f" bot> {(state.get('last_reply') or '')[:200]}")
            for assertion_name, passed, detail in check_fn(state, prev_state):
                check(name, turn_idx, assertion_name, passed, detail)
            prev_state = state

    total = len(results)
    passed = sum(1 for *_, p, _ in results if p)
    print(f"\n{passed}/{total} checks passed.")
    failures = [(s, t, n, d) for s, t, n, p, d in results if not p]
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for s, t, n, d in failures:
            print(f"  - [{s}] turn {t}: {n}" + (f" -- {d}" if d else ""))
        sys.exit(1)


if __name__ == "__main__":
    main()
