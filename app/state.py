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
    # Overridden every turn by `locked_reply_lang` below, if the citizen
    # picked one - this per-turn value is only ever used when no lock exists.
    reply_lang: str

    # --- persistent, set once from the frontend's language picker (2026-08-12) ---
    # If the citizen picked a language up front, every future turn's
    # reply_lang is pinned to this instead of being freshly re-classified
    # from that turn's message - added because voice input's transcription
    # can come back as either Devanagari or Romanized Hindi for the SAME
    # spoken sentence depending on Whisper's own choice, which made
    # reply_lang flip between 'hindi' and 'hinglish' turn to turn even
    # though the citizen never actually changed how they were speaking.
    # None (unset) if the citizen never picked a language - falls back to
    # normal per-turn detection in that case, unchanged from before.
    locked_reply_lang: Optional[str]

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
    # Free-form durable memory alongside the fixed-schema `profile` above -
    # short factual sentences for anything with no profile field of its
    # own: family-member facts ("son is in class 10"), documents held/
    # lacking, schemes already applied to, land/assets, narrow scheme-
    # specific criteria (MSME/Udyam registration, tractor ownership, prior
    # subsidy history), and direct answers to the bot's own questions
    # (including "no" answers). Added 2026-08-18: the profile's 9 fixed
    # fields were the ONLY thing that outlived the rolling
    # conversation_history window, so every out-of-schema fact silently
    # expired with the window and got re-asked - the exact "asks the same
    # question again" failure seen live on 2026-08-16/17. Fed by TWO
    # capture channels appending into this one list (extract_profile_node's
    # ProfileExtraction.new_notes and extract_need_node's NeedExtraction.
    # stated_facts - the latter sees the clarify-answer context the profile
    # call doesn't). Appended-only (deduped, capped at MAX_CONVERSATION_
    # NOTES in app/chat.py), fed into every reply-generating prompt with an
    # explicit do-not-re-ask rule.
    conversation_notes: list  # list[str]
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
    # Which domain the pending clarifying question (above) was actually
    # about - "scheme" (clarify/clarify_identity/generate) or "job"
    # (clarify_jobs). Added 2026-08-18: route_after_extract's
    # was_answering_clarification override used to unconditionally route to
    # "stage_a" (the SCHEME search) the moment ANY clarification was
    # pending, with no awareness that a JOB clarification could be pending
    # instead - found live: "mere liye koi job btao" -> clarify_jobs asks
    # about qualification -> "10th pass hu" (answering it) got forced into a
    # fresh SCHEME search instead of continuing the job search, surfacing an
    # unrelated "Leather Training Scheme" with no connection to the job
    # request at all. None when no clarification is pending, same lifecycle
    # as pending_need_text (set together, reset together).
    pending_need_domain: Optional[str]
    # Ephemeral, ONE-turn-only companion to pending_need_domain above - set
    # by extract_need_node to whatever pending_need_domain was BEFORE it got
    # reset to None in that same call (pending_need_domain has to reset every
    # turn, same lifecycle as pending_need_text, but route_after_extract
    # still needs the OLD value THIS turn to decide "stage_a" vs
    # "stage_a_jobs" when was_answering_clarification fires). Not meant to be
    # read on any later turn - it gets overwritten (or left stale/unused)
    # the moment this turn's routing decision is made.
    clarification_domain: Optional[str]
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

    # How many consecutive turns generate_node ITSELF has asked a
    # clarifying question for the current unresolved need - separate from
    # clarify_attempts above, which only tracks clarify_node/
    # clarify_identity_node's OWN attempts before routing to stage_b. Reset
    # to 0 alongside clarify_attempts (a fresh need), and to 0 by
    # generate_node whenever it gives a real answer. Added 2026-08-17 after
    # a live conversation asked SEVEN consecutive rounds of narrowing
    # questions about a business-loan need without ever landing on an
    # answer, re-asking something already answered once it scrolled out of
    # conversation_history's rolling window - generate_node now hard-stops
    # and gives its best-effort answer once this reaches 2, regardless of
    # whether the model's own text still contains another question.
    generate_clarify_rounds: int

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

    # --- citizen_status domain: Jan Aadhaar OTP-consent verification sub-flow
    # (mock, 2026-08-14) - parallel to job_shortlist above for the same reason:
    # this isn't a scheme-eligibility search, it's a scripted 3-step identity
    # check, so it gets its own small state slice rather than overloading the
    # scheme/job machinery. ---
    # None outside the flow. Set the moment a citizen_status need is detected
    # and cleared the moment verification succeeds or is abandoned. Whenever
    # non-None, route_after_normalize sends the turn straight to
    # janaadhaar_handle_node, BYPASSING extract_profile/extract_need (both LLM
    # calls) entirely - this turn's raw_message is itself a Jan Aadhaar ID,
    # member choice, or OTP, and must never become model input.
    pending_janaadhaar_step: Optional[str]  # None | "awaiting_consent" | "awaiting_id" | "awaiting_member_choice" | "awaiting_otp"
    janaadhaar_member_id: Optional[str]  # the member_id OTP was generated for, needed to validate it
    janaadhaar_tid: Optional[str]  # transaction id from generate_otp, needed to validate the OTP
    janaadhaar_members: list  # masked member list from fetch_member_list, so "the second one" resolves against it
    # How many consecutive failed attempts at the CURRENT step (bad ID
    # format/no match/wrong OTP) - capped in janaadhaar_handle_node so a
    # citizen who can't complete verification isn't trapped forever; reset
    # whenever the flow advances to a new step.
    janaadhaar_attempts: int
    # Non-identifier status fields (e.g. pension_status) from the last
    # successful verification, kept in case a same-session followup asks
    # about it again without re-verifying.
    janaadhaar_status_data: Optional[dict]
    # WHO janaadhaar_status_data above is actually for - the subject_relation
    # (e.g. "maa"/"mother") of the need that triggered that verification, or
    # "self" if it was about the citizen themselves (2026-08-25, found live).
    # The cache-reuse path in janaadhaar_handle_node compares this against
    # the CURRENT need's own subject_relation before reusing it - without
    # this, any later citizen_status need in the same session (about ANY
    # family member) answered from whichever member was verified FIRST, with
    # no check that it's even the same person being asked about.
    janaadhaar_status_data_for: Optional[str]
    # Set by janaadhaar_handle_node on every turn IT processes (an ID/member
    # choice/OTP attempt), to a synthetic label like "[Jan Aadhaar ID
    # redacted]" instead of the real value. finalize_node persists THIS in
    # conversation_history in place of raw_message when set, and app/main.py's
    # _log_turn does the same for the server log - otherwise the identifier/
    # OTP would leak into a LATER turn's LLM prompt via history replay, or
    # sit in plaintext server logs, even though the CURRENT turn's flow never
    # sent it to a model. Reset to None by normalize_node at the start of
    # every turn, so a stale value can't wrongly redact an unrelated later
    # turn's conversation_history entry.
    redacted_raw_message: Optional[str]

    # Mock-only (2026-08-17): the simulated OTP text, for the frontend to
    # show as a "SMS arrived" popup instead of requiring a look at server
    # logs - set by janaadhaar_handle_node the moment it generates an OTP,
    # None every other turn. Reset by normalize_node at the start of every
    # turn for the same reason redacted_raw_message is - so a stale value
    # from an earlier turn can't linger and pop up again on an unrelated
    # later turn. Naturally empty once JANAADHAAR_MODE=real, since the real
    # transport's response never carries the OTP value at all.
    mock_sms_preview: Optional[str]
