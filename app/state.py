"""LangGraph conversation state (PLAN.md SS4). Persisted across turns via
the graph's checkpointer (keyed by thread_id/session_id) - a node's return
dict is shallow-merged into this, so a key a node doesn't return simply
keeps its previous value (that's how `profile` and `shown_scheme_*` persist
across turns without every node needing to re-state them).
"""

from typing import Optional, TypedDict


class ConvState(TypedDict, total=False):
    # --- per-turn input/derived, overwritten every turn ---
    raw_message: str
    normalized_message: str
    # "hindi" | "hinglish" | "english" - classified by extract_need_node's
    # detected_language every turn (normalize_node's Devanagari-regex guess
    # is only a placeholder until then). Three-way, not the old hi/en binary,
    # since that bucketed Hinglish (Romanized Hindi) in with English and
    # replies came out in prose English for Hinglish-writing citizens.
    reply_lang: str

    # need_text/category/benefit_type mirror needs[0] (the PRIMARY need) for
    # every existing single-need code path (clarify, identity-gap, followup,
    # show_more) that only ever reasoned about one need - see `needs` below
    # for the full per-need breakdown used by stage_a/stage_b/generate.
    need_text: str
    need_category: Optional[str]
    need_benefit_type: Optional[str]
    need_is_specific: bool  # the model's own direct clarity judgment - NOT inferred from category/benefit_type presence

    # One entry per DISTINCT need/topic stated in the message - almost always
    # length 1. Added 2026-08-10: a message stating two unrelated needs (a
    # farming subsidy for the citizen AND a scholarship for their son) used
    # to have the second one silently discarded, since need_text was a single
    # string. Each entry: {need_text, need_category, need_benefit_type,
    # about: "self"|"family_member", subject_relation, subject_facts}.
    # subject_facts holds whatever identity facts were stated about the
    # family member THIS need is for (caste/gender/education_level) - kept
    # separate from the citizen's own persistent `profile` so a fact stated
    # about a son/daughter never leaks into the citizen's own identity.
    needs: list
    # Parallel to `needs`: {"need": <needs[i]>, "shortlist": [...], "candidate_count": int}
    # per need, from the same Stage A run - shortlist/candidate_count above
    # mirror shortlists[0] for single-need code paths.
    shortlists: list
    intent: str  # find_scheme | ask_detail | how_to_apply | documents_question | chitchat
    # single mutually-exclusive classification (new_need | followup_same_scheme |
    # wants_alternatives) - replaced two independent booleans (is_followup,
    # wants_alternatives) that could structurally contradict each other; see
    # app/chat.py's TurnType for why.
    turn_type: str
    was_answering_clarification: bool  # was the previous turn a clarifying question? forces a fresh Stage A run, overriding turn_type

    # shortlist is set by stage_a_node but, since stage_b_followup/clarify/
    # show_more don't overwrite it, it PERSISTS from the last Stage A run
    # across however many single-scheme-followup turns come after - that's
    # what lets show_more_node answer "what else is there" from data already
    # computed, without a fresh search.
    shortlist: list  # list[dict] (scheme_id/scheme_name/lang/score/metadata), from the last Stage A run
    candidate_count: int  # pre-truncation survivor count, for the clarify-or-not decision
    detail_chunks: list  # list[(doc_text, metadata)], this turn's Stage B result

    last_reply: str  # what the bot said this turn - also serves as "last reply" context next turn

    # --- persistent across the whole conversation (SS4.a / SS4.c) ---
    profile: dict  # caste/gender/marital_status/age/family_income/residence - merged, never reset
    # One entry per scheme actually shown last turn - {scheme_id, scheme_name,
    # need_text}, in the order presented. shown_scheme_id/name/lang mirror
    # shown_schemes[0] for old single-scheme code paths. Added 2026-08-10:
    # with multi-need answers, "the" shown scheme is ambiguous - a followup
    # like "pehli scheme ki eligibility batao" or one asking about the SECOND
    # need's schemes needs the full list to resolve against, not a single
    # stale slot that only ever reflected the primary need.
    shown_schemes: list
    shown_scheme_id: Optional[str]
    shown_scheme_name: Optional[str]
    shown_scheme_lang: Optional[str]
    # Which entry in shown_schemes (by scheme_name) THIS followup message
    # refers to, resolved by extract_need_node from the message text itself
    # (ordinal - "pehli"/"first" - or topical - "education wali") rather than
    # guessed positionally in code. null if only one scheme was shown, or the
    # message refers to all of them together.
    followup_target: Optional[str]
    pending_need_text: Optional[str]  # set when we asked a clarifying question, so a terse reply ("no"/"haan") next turn doesn't lose what was being asked about
    already_shown_scheme_ids: list  # every scheme_id ever set as shown_scheme_id this conversation - show_more_node excludes all of these, not just the current one

    # How many consecutive clarify/clarify_identity questions have been asked
    # for the CURRENT unresolved need without it ever reaching stage_b. Reset
    # to 0 by extract_need_node whenever there was no pending clarification
    # going into the turn (a fresh need), and by stage_b_node once a real
    # answer is actually delivered. route_after_stage_a refuses to clarify a
    # second time in a row - found live (2026-08-10): a citizen who repeated
    # the same vague answer twice got the identical clarifying question back
    # verbatim, with no way to escape the loop.
    clarify_attempts: int

    # rolling cache of recently retrieved chunks (deduped by scheme_id+field_type+lang,
    # newest-touched first, capped at MAX_CACHED_CHUNKS in app/chat.py) - appended to by
    # every retrieval node (stage_b/stage_b_followup/show_more), so generate_node has
    # access to recent context even when this turn's OWN retrieval was narrow (e.g. a
    # documents-only fetch) or didn't run at all.
    chunk_cache: list  # list[(doc_text, metadata)]

    # Set by stage_b_followup_node ONLY when the citizen explicitly narrowed
    # a followup to one specific previously-shown scheme (via followup_target)
    # - tells generate_node to restrict chunk_cache's "fill in recent
    # context" merge to just that scheme, not every recently-discussed one.
    # Every other retrieval node (stage_b/show_more) explicitly resets this
    # to None, so a narrowing from an earlier turn can't linger and wrongly
    # suppress a later, genuinely multi-scheme answer. Added 2026-08-11:
    # "tell me about the last scheme" correctly narrowed its OWN fetch, but
    # chunk_cache still held the other previously-shown schemes and
    # generate_node silently merged them back in, defeating the narrowing.
    focused_scheme_ids: Optional[list]

    # rolling transcript of the last MAX_HISTORY_TURNS (raw_message, last_reply) pairs,
    # appended by finalize_node after every turn regardless of which path produced the
    # reply - gives generate_node real conversational context, not just this turn's
    # isolated need_text + chunks.
    conversation_history: list  # list[{"user": str, "bot": str}]

    # --- job-vacancy domain (parallel to the scheme fields above, kept
    # separate rather than overloading shortlist/shown_schemes - job
    # postings aren't identity-eligibility-gated schemes, they're
    # structured recruitment records (qualification/location/deadline/
    # vacancies), so they get their own small parallel state rather than
    # forcing the scheme machinery to serve two unrelated concerns. ---
    job_shortlist: list  # list[dict], full job records from the last jobs search
    shown_jobs: list  # [{"job_id", "title", "department"}, ...] shown last turn, in order presented
    shown_job_id: Optional[str]
    already_shown_job_ids: list  # every job_id ever shown this conversation - mirrors already_shown_scheme_ids
