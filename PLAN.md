# Rajasthan Scheme Chatbot — Build Plan

A conversational RAG chatbot for Rajasthan government scheme discovery, for a rural user base. Scheme data comes from https://www.myscheme.gov.in/search, filtered to Rajasthan (state schemes + central schemes applicable there). This is a from-scratch build under `chatbot_scratch/` — the plan below is derived only from the data source and the requirements, not from any other codebase.

## 1. Data acquisition — CONFIRMED LIVE (2026-08-05)
myscheme.gov.in is a Next.js SPA — the page itself carries no scheme data until its backend API responds. Inspected the site's JS bundles directly and confirmed the real API (this can change in the future — re-derive the same way if the scraper starts failing: view the bundle's requests in a browser Network tab, or grep the downloaded `_next/static/chunks/pages/_app-*.js` for `api.myscheme.gov.in`).

**Base API:** `https://api.myscheme.gov.in`, called with headers `x-api-key` (a public client-side key embedded in myScheme's own frontend bundle — every visitor's browser sends the same one, it's not a private credential), `Origin`/`Referer` set to `https://www.myscheme.gov.in` (requests are rejected with 401 without these).

**Search** — `GET /search/v6/schemes?lang=en&q={URL-encoded JSON filter array}&keyword=&sort=&from={n}&size={<=100}`. `size` is capped at 100 by the API (412 error above that) — Rajasthan state schemes paginate across 2 pages. Filters for Rajasthan state-specific schemes: `[{"identifier":"beneficiaryState","value":"Rajasthan"},{"identifier":"level","value":"State"}]` → confirmed **157 schemes**. Each hit returns `slug`, `schemeName`, `briefDescription`, `tags`, `schemeCategory`, `level`, `schemeFor`, `schemeCloseDate`.

**Note on central schemes:** central (all-India) schemes carry `beneficiaryState:["All"]`, not `"Rajasthan"` — filtering by `beneficiaryState=Rajasthan` only returns state-specific schemes. If central schemes applicable to every citizen (e.g. PM-Kisan, Ayushman Bharat) should be included too, that's a separate query (`level=Central`, ~714 schemes across all of India) and a **scope decision**: include all ~714, or a curated subset? Flagged as open below rather than decided unilaterally.

**Detail** — `GET /schemes/v6/public/schemes?slug={slug}&lang={en|hi}` → `data._id`, and `data.en`/`data.hi` (whichever lang was requested) containing `basicDetails` (name, level, category, department, dates, tags), `schemeContent` (`detailedDescription_md`, `benefits_md`, `exclusions_md`, `benefitTypes`), `applicationProcess` (per mode: online/offline, url, `process_md`), and `eligibilityCriteria`.

**Documents** — `GET /schemes/v6/public/schemes/{_id}/documents?lang={en|hi}` → `documentsRequired_md`.
**FAQs** — `GET /schemes/v6/public/schemes/{_id}/faqs?lang={en|hi}` → list of `{question, answer_md}`.

**Key discovery — structured eligibility comes free from the API, no LLM needed.** Verified across multiple schemes: the **`lang=hi`** detail response's `data.hi.eligibilityCriteria` includes full **machine-coded eligibility facets** — `age` (per category, gte/lte), `familyIncomeAnnual`/`individualIncomeAnnual`/`parentIncomeAnnual` (gte/lte), `gender[].value`, `caste[].value`, `maritalStatus[].value`, `residence[].value`, `beneficiaryState[].value`, plus `numberOfChildren`/`numberOfBoys`/`numberOfGirls`/`workExperience`/`education`/`numberOfDependents`/`nationality`. The `lang=en` response for the *same* scheme only returns free-text eligibility description under that key — no structured fields. The `.value` codes (e.g. `caste: "sc"`, `gender: "All"`) are stable identifiers, not translated text, so they're usable directly regardless of which language's text is shown to the user. This means **§2's "structured eligibility extraction" is a deterministic parse of data the source already provides, not an LLM batch-extraction job** — a meaningful simplification (cheaper, more accurate, no LLM cost at all for this step). See §2a below, revised accordingly.

**Scraper implemented:** `scripts/scrape_myscheme.py` — pages through the Rajasthan-state search, then for each slug fetches detail + documents + faqs in *both* `en` and `hi`, merges into one record (`{scheme_id, slug, en, hi, eligibility_structured, documents, faqs}`), and writes it to `data/raw/{scrape-date}/{slug}.json` (plus a `_listing.json` index) — skips a slug on re-run if its file already exists, so it's safe to resume after an interruption. Rate-limited (~0.4s between calls) as a courtesy to a public API that isn't ours.

## 2. Structuring the data for retrieval
Long free-text eligibility/benefit paragraphs are hard to match reliably with embeddings alone, so each scheme gets processed into two complementary forms:

**a. Structured eligibility fields** — **no LLM extraction needed** (revised per §1's discovery): myScheme's own API already returns machine-coded eligibility facets per scheme (`eligibility_structured` in the scraped record) — age range per category, income ceilings (family/individual/parent), gender/caste/marital-status/residence value-codes, beneficiaryState. `scripts/extract_eligibility.py` is therefore a **deterministic parser/normalizer** of that existing object into our own flat schema (e.g. `{caste: [...], gender: [...], age_min, age_max, income_ceiling, marital_status: [...], residence, ...}`), not a batch LLM job. Two gaps the API's structured object doesn't cover, which still need simple deterministic derivation from the scraped text/tags (not full LLM extraction): occupation/target-group (farmer, student, widow, differently-abled, unemployed) — derivable from `targetBeneficiaries`/`tags`/`schemeCategory` already in the scrape — and benefit type (scholarship, pension, loan, subsidy, equipment, training, insurance, housing) — myScheme has a coarse `benefitTypes` (cash/in-kind/composite) but not this finer split, so it needs a lightweight keyword/tag-based rule rather than an LLM call, given how cheap and mechanical the signal is (scheme name/category/tags almost always name the benefit type directly).

**b. Retrieval chunks**, split by field type (eligibility / benefits / how-to-apply / documents) per scheme, plus one short one-line summary per scheme ("scheme name — who it's for — what it gives — category") that's cheap to scan for shortlisting many candidates at once.

Store both in a local vector database (ChromaDB) with a multilingual embedding model, since queries will arrive in Hindi, Hinglish, English, or a mix.

**Scale check (measured against the actual scrape, 2026-08-05):** ~10 field-level chunks per scheme across both languages, averaging ~600 characters (~150 tokens) each → ~1,600-2,000 substantive chunks for all 157 Rajasthan schemes (more with FAQs/discovery cards, maybe ~3,000-4,000; ~15,000-20,000 if central schemes are added later per the open scope decision). This is small for a local embedding model + ChromaDB — a one-time batch job taking well under a minute on CPU, no ongoing cost, nowhere near the corpus size where vector-DB tuning or infra changes would matter.

### Chunking strategy, concretely
- **Unit = one scheme's one field, not a fixed-size sliding window.** myScheme's detail payload already arrives split into fields (Eligibility, Benefits, Application Process, Documents Required, Details/Description, etc.) — those field boundaries are the natural semantic unit. Splitting on them avoids the classic fixed-window problem of cutting an eligibility sentence in half; a 500-character sliding window has no idea "must be female, SC/ST, and below 21 years" is one clause.
- **Prefix each chunk with scheme name + field label before embedding** — e.g. `"{scheme_name} — Eligibility: {text}"` rather than bare text. Bare eligibility text across 150+ schemes is often generic ("must be a resident of Rajasthan, below poverty line...") and embeddings need the name/label as anchor context to disambiguate.
- **Secondary split only when a field is unusually long** (some Description fields run long): recursive split at ~300–400 tokens with ~15% overlap, so no chunk exceeds the embedding model's effective window and continuity isn't lost at the cut.
- **A second, separate chunk type — the discovery card**: one compact line per scheme (name, category, target group, benefit type, one-line benefit). Lives in its own small collection. This is not a replacement for field chunks — it's a cheap front door for scanning/shortlisting across all ~150+ schemes before drilling into any scheme's full detail.
- **Metadata on every chunk** (field chunks and discovery cards alike): `scheme_id`, `scheme_name`, `field_type`, `category`, `department`, `level` (state/central), plus the full structured eligibility record. Attaching the eligibility record to every chunk (not just a separate table) means any chunk can be filtered on it directly, no join needed.

## 3. Retrieval strategy
Two stages per query, not one flat semantic search over everything:

**Stage A — candidate shortlist (broad, cheap).** Query the *discovery-card* collection: embedding similarity on the stated need/category, combined with the eligibility filter (below) applied over each card's attached metadata. Output: a ranked shortlist of candidate `scheme_id`s (roughly top 10–20), not full scheme detail yet.

**Stage B — detail retrieval (narrow, precise).** For the top few shortlisted schemes (or a specific scheme already in focus), query the *field-chunk* collection restricted to those `scheme_id`s AND only the `field_type`(s) the user's actual ask calls for — a generic "what schemes are there for me" pulls discovery-card + benefits; "how do I apply" pulls application+documents only; a plain eligibility question pulls just the eligibility chunk. This keeps what eventually reaches the LLM small and targeted instead of dumping every field of every candidate.

**Eligibility filter mechanics (used in Stage A).** Chroma's metadata `where` clause only does flat equality/range matching — it can't express "exclude only on a real contradiction, otherwise don't touch." So the filter logic itself runs in Python, not as a single Chroma query: pull a coarse metadata-filtered set (e.g. by category) from Chroma, then in code compare each candidate's structured eligibility record against whatever is currently known about the citizen (see conversation memory below) and drop a scheme only on an explicit contradiction (e.g. scheme is women-only and the citizen has stated male; scheme's income ceiling is below the citizen's stated income). Any attribute not yet known is simply never checked — it can never cause an exclusion. This fail-open design matters because a rural user's profile is built up incrementally; the filter must never punish incomplete information.

**Ranking.** After the fail-open filter, rank survivors by embedding similarity to the current need, then apply a deterministic boost for an exact `category`/`benefit_type` metadata match — so a scheme that matches the stated need's *type* exactly outranks one that only matched loosely on wording.

## 4. Conversation memory model — the core conversational requirement
The bot must carry context across turns, and specifically must separate two different kinds of information:

**a. Citizen profile (persistent, person-level facts).** Anything the user has stated *about themselves* — caste/category, gender, age, income bracket, occupation, district, family/social status (farmer, widow, student, disabled, unemployed, etc.). Once stated, these stick for the rest of the conversation and apply to *every subsequent query*, regardless of topic. Example the user flagged: if someone establishes "I'm SC, female, income under 2 lakh" while asking about a scholarship, and later — in an unrelated message — asks about a farm-equipment subsidy, the bot must still apply SC/female/income when filtering *that* new question, without the user repeating themselves. If the user later restates or corrects a fact (e.g., a different income figure), the profile is updated, not appended.

**b. Query-level topic/need (resets per message).** What the user is asking about *right now* — which category, which need, which scheme. This should NOT persist onto unrelated future questions; asking about tractors doesn't mean every later question is about tractors.

**c. Shown-scheme focus (short-lived, for follow-ups).** The specific scheme(s) most recently presented to the user, kept only to resolve immediate pronoun follow-ups ("iske documents batao", "isme se pehla wala"). This focus is dropped as soon as the user's next message expresses a clearly different need/category — it must not leak into an unrelated new question the way the profile does.

Every incoming message runs through: (1) an extraction step that pulls out any new person-level facts (merged into the persistent profile) and the current message's need/category (query-level, not persisted), (2) a check of whether pronoun/follow-up resolution applies (uses shown-scheme focus), (3) retrieval that combines the *persistent profile filter* with the *current query's need*.

**Clarifying questions** are asked only when the current need is genuinely ambiguous *and* the missing information would actually change which schemes qualify — i.e., check how many candidates match everything known so far (profile + current need) before deciding whether a follow-up question is even necessary, and if so, which single attribute to ask for.

## 5. Answer generation
- Compose the reply strictly from the retrieved chunks for the schemes that survive filtering+ranking — no invented eligibility criteria, amounts, or steps.
- Always include, when relevant: scheme name, what it gives, who it's for, how/where to apply, documents needed.
- Reply in the language/script the user is using (Hindi/Devanagari, Hinglish/Roman, or English), detected per message or fixed after an initial choice.
- Handle multiple needs in one message (e.g., "housing aur beti ki padhai dono ke liye") by answering each need's matches separately rather than merging them into one answer.

## 6. Handling rural/vague/half queries
- Normalize input before intent detection: mixed Devanagari/Roman script in a single message, common misspellings/transliteration variants of scheme names, filler words, voice-transcription artifacts.
- Classify each message's intent (find a scheme / ask about a specific scheme / eligibility check / how to apply / follow-up on something already shown / small talk) tolerant of short fragments ("पेंशन चाहिए", "tractor ke liye paisa").
- When a message is too vague to retrieve anything meaningful even combined with the known profile, ask one clarifying question generated from what schemes are actually plausible — not a fixed script of questions asked in the same order every time.

## 7. Architecture (kept simple and cheap to run)
- Backend: Python FastAPI.
- LLM: used for intent classification, profile-fact extraction, clarifying-question generation, and final answer composition — keep the number of calls per turn small and deliberate.
- Vector DB: ChromaDB, local.
- Embeddings: local multilingual sentence-transformer model (no per-call API cost).
- Session/conversation state: in-memory to start (citizen profile + current need + shown-scheme focus + recent turns); move to Redis only once this needs to survive restarts or scale beyond a single process.
- Interface: a plain web chat UI first, for fast iteration; a messaging-channel integration (WhatsApp or similar) is a later phase once the conversation logic itself is solid.

### Mapped onto LangChain
Where each piece from §2/§3/§4 lands in LangChain's building blocks:

- **Loading & structuring** — a custom loader turns each scraped scheme's fields into `Document` objects (`page_content` = the field text with the scheme-name/field-label prefix, `metadata` = scheme_id/field_type/category/eligibility record from §2). No off-the-shelf LangChain document loader fits myScheme's API shape, so this stays hand-written.
- **Splitting** — `RecursiveCharacterTextSplitter` (or `TokenTextSplitter` for a token-accurate size), used *only* for the secondary split on unusually long fields (§2) — most field chunks are already the right size from field-based chunking and don't need splitting at all.
- **Embeddings** — `HuggingFaceEmbeddings` (via `langchain-huggingface`) wrapping the local multilingual sentence-transformer model, so no per-call API cost for indexing or query embedding.
- **Vector store** — `Chroma` (via `langchain-chroma`), as two separate collections/instances — one for discovery cards, one for field chunks — matching §2/§3 exactly.
- **Retrieval (Stage A + Stage B)** — a custom retriever subclassing `BaseRetriever`, built on top of the Chroma vectorstore's similarity search. This is deliberately NOT one of LangChain's off-the-shelf retrievers (`SelfQueryRetriever`, `EnsembleRetriever`, `ContextualCompressionRetriever`) — none of them model "fail-open eligibility filter in Python, then rank with a category-match boost" out of the box. Those prebuilt retrievers are worth revisiting later for reranking, but Stage A/B's core logic gets hand-written against the vectorstore API.
- **Structured extraction** — profile-fact extraction, intent classification, and clarifying-question generation each use `.with_structured_output(SomePydanticModel)` on the chat model, so the LLM returns a typed object directly instead of prose that needs parsing.
- **Conversation state (§4)** — recommend **LangGraph** (LangChain's companion state-machine library) over the legacy `ConversationBufferMemory`-style memory classes. Reasoning: §4's memory model isn't a flat message history — it's three distinct pieces of state (persistent citizen profile / per-turn need / shown-scheme focus) with conditional flow (ask a clarifying question vs. answer directly). LangGraph models this as an explicit typed state object threaded through nodes (`normalize → extract → retrieve_A → maybe retrieve_B → maybe_clarify → generate`) with conditional edges, and gives checkpointing for free if state ever needs to survive a restart. Plain LangChain `Runnable`/LCEL chains are linear and would need workarounds for the same branching. This adds `langgraph` as a dependency — flag if you'd rather keep to a hand-written state object and skip the extra library for a first cut.
- **Answer generation** — `ChatPromptTemplate` + LCEL composed as the final node in the graph, grounded strictly in the Stage B chunks passed into its context.

Packages this implies: `langchain`, `langchain-community`, `langchain-chroma`, `langchain-huggingface`, a provider package for whichever chat model, and `langgraph` if you take that recommendation.

## 8. Evaluation
- Build a small set of realistic test conversations that specifically exercise cross-topic profile carry-over (e.g., establish eligibility facts in turn 1 on scheme A's topic, then switch to an unrelated category in turn 2 and confirm the bot applies those facts without re-asking) alongside vague/short/mixed-script single-turn queries.
- Prefer deterministic, non-LLM checks wherever the thing being checked is a filter/rank decision rather than natural-language phrasing; reserve full live-conversation replays (which burn LLM calls) for pre-ship checks, run sparingly and deliberately.

## 9. Suggested folder layout
```
chatbot_scratch/
  PLAN.md
  scripts/
    scrape_myscheme.py
    extract_eligibility.py
    build_knowledge_base.py
  data/
    raw/            # dated raw scrape snapshots, never hand-edited
    processed/       # cleaned + chunked + structured
    chromadb/
  app/
    main.py          # FastAPI entry
    chat.py           # conversation state: profile, current need, shown-scheme focus
    rag_service.py     # filter + rank
    intent_service.py  # classification, extraction, clarifying questions
  eval/
    golden_set.py
    sample_conversations.md
```

## 10. Phased delivery — what happens to chunking/retrieval at each phase
**Phase 0 — spike.** No chunking/retrieval yet; just confirm the live myScheme API shape for a Rajasthan-filtered search (browser devtools).

**Phase 1 — data + knowledge base build.** This is where the chunking strategy from §2 gets fully implemented: extract structured eligibility fields per scheme, build the two Chroma collections (discovery cards, field chunks), embed everything with the multilingual model, attach metadata to every chunk. Deliverable: a queryable vector DB covering every Rajasthan scheme — nothing conversational yet.

**Phase 2 — core retrieval, single turn.** Implement Stage A (shortlist) + Stage B (detail) from §3 for one fully-specified query with no memory involved — whatever attributes/need are in *that one message* drive the filter+rank. Get shortlist precision and field-selection correctness right here, before conversation logic sits on top of it, so bugs aren't hidden by multi-turn interactions.

**Phase 3 — conversation memory layer. DONE (2026-08-05).** Built as a LangGraph state machine (`app/chat.py`, `app/state.py`) per §7/§11: `normalize -> extract -> [branch: followup? chitchat? fresh need?] -> (stage_a ->) [branch: no match? too vague? proceed?] -> (clarify | stage_b | stage_b_followup) -> generate -> END`, backed by `MemorySaver` (thread_id = session id) so `profile`/`shown_scheme_*` persist across turns while `need_*`/`intent`/`shortlist` are freshly overwritten each turn. LLM: Groq `llama-3.3-70b-versatile` (`.env`). Validated end-to-end (`scripts/chat_cli.py --debug`): cross-topic profile carry-over (caste stated during a scholarship question correctly applied to an unrelated farming question 2 turns later), follow-up resolution correctly targeting the most-recently-shown scheme (not an earlier one), the Stage B shortcut for follow-ups (skips Stage A, fetches only the relevant field type), chitchat handled without touching retrieval, and a vague need correctly triggering ONE clarifying question instead of a scheme dump.

Two real LLM-reliability issues found and fixed during validation (not retrieval/data bugs - documented since they matter for anyone adding more structured-output calls later): (1) Groq's tool-call validation is strict and rejects a `bool` field when the model emits `"false"` as a string instead of a JSON boolean - switched to `Literal["yes","no"]` and convert in code, since string enums prove far more reliable than raw booleans for this model's tool-calling. (2) A strict `Literal` enum for `need_category` broke when the model added a natural space into "Agriculture,Rural & Environment" (the stored category string's unusual comma-no-space formatting comes straight from myScheme's own data) - switched to free-text extraction + a `_normalize_to_known()` fuzzy (whitespace/case-insensitive) matcher in code, rather than requiring the model to emit a byte-perfect enum value. General lesson: normalize LLM output defensively in application code rather than relying on the provider's strict-mode validation to never reject harmless phrasing variance.

**Phase 4 — rural query normalization.** Sits upstream of Stage A — cleans/normalizes the query text (script mixing, typos, transliteration) before it's embedded — no change to the chunking or retrieval structure itself.

**Phase 5 — evaluation.** Test retrieval precision directly: does Stage A's shortlist contain the right `scheme_id`s for known queries, does Stage B pull the right `field_type`s for a given intent, does the fail-open filter correctly *not* exclude on a missing attribute and correctly exclude on a real contradiction — plus the cross-topic profile carry-over scenarios end-to-end.

## 11. Traceability / observability
Two layers, because LangSmith alone won't cover everything worth tracing here:

**a. LangSmith (LangChain/LangGraph-native).** Since the graph in §7 is built on LangGraph, turning on tracing (`LANGCHAIN_TRACING_V2` + project key) gives one trace per user query automatically, with every node (normalize / extract / retrieve-A / retrieve-B / clarify / generate) as a nested span — each LLM call's prompt, response, latency, and token count is captured for free, no extra instrumentation needed for the LLM-facing parts.

**b. A custom structured trace record for the Python-only decision logic.** LangSmith sees LLM calls and node boundaries, but the fail-open eligibility filter and the ranking boost (§3) are plain Python — LangSmith won't show *why* a scheme was dropped or *why* one outranked another unless that's explicitly logged. So each graph node writes into a `query_trace` object threaded through the state:
  - `profile_before` / `profile_after` (what changed this turn)
  - `stage_a`: candidate count before filtering, per-scheme filter verdict (kept / dropped + the exact contradicting attribute), similarity scores, final shortlist
  - `stage_b`: which `scheme_id`s and `field_type`s were fetched and why (which intent triggered which fields)
  - `clarify`: whether a clarifying question was asked, which attribute it targeted, and why (candidate count/ambiguity that triggered it)
  - `answer`: which chunks the final answer was grounded in
  This record is persisted (e.g. one JSON-lines file per day, or a small SQLite table) independent of LangSmith — useful for debugging without a LangSmith UI open, and it doubles as raw material for building the eval golden-set (§8) from real interactions later.

**c. Life of one query, concretely — what happens, in order:**
1. Message arrives → session resolved → existing state loaded (citizen profile, current need, shown-scheme focus, recent turns).
2. **Normalize**: clean script-mixing/typos/filler words → normalized text. (Span: input=raw message, output=normalized text.)
3. **Extract** (LLM, structured output): classify intent, pull any new profile facts from this message (merged into the persistent profile immediately), determine the current need, and — using the shown-scheme focus + last reply as context — decide if this is a follow-up. (Span: LangSmith captures the LLM call in full; `query_trace` records `profile_before`/`profile_after`.)
4. **Branch**: if it's a follow-up referring to an already-shown scheme, skip straight to step 6 for that specific `scheme_id`.
5. **Stage A** (retrieval): discovery-card similarity search + the fail-open filter using the full persistent profile + current need → ranked shortlist. (`query_trace.stage_a` recorded here.)
6. **Clarify check**: if the shortlist is still too ambiguous *and* a specific missing attribute would actually narrow it, generate one clarifying question (LLM, structured output) and return it as the reply — the turn ends here, state is saved, no Stage B call happens. (`query_trace.clarify` recorded.)
7. **Stage B** (retrieval): otherwise, pull only the field-type chunks the intent calls for, for the top shortlisted (or focused) scheme(s). (`query_trace.stage_b` recorded.)
8. **Generate** (LLM): compose the reply strictly from the Stage B chunks, in the user's script. (Span captured by LangSmith; `query_trace.answer` records which chunks grounded it.)
9. **State update**: shown-scheme focus updated to what was just presented, profile persisted, turn appended to recent history, and the full `query_trace` record for this turn is written to the trace store.
10. Reply returned.

**Open consideration:** the citizen profile carries sensitive attributes (caste, income, gender, disability status). Sending full traces (including these fields) to LangSmith's hosted cloud is worth a deliberate decision rather than a default — self-hosted LangSmith, or dropping/redacting those specific fields before they reach the cloud trace while keeping them in the local `query_trace` store, are both options. Flagging this now so it's a conscious choice before real user data flows through it, not an afterthought.

## Open decisions for you to confirm
- **Interface for the first demo**: web chat vs. a messaging channel from day one? Recommend web chat first for iteration speed.
- **LLM/embedding budget for this project**: which provider/key should this fresh build use?
- **LangSmith cloud vs. self-hosted / local-only tracing**, given the sensitive profile attributes noted above.
- **Central-scheme scope** (see §1): include all ~714 central schemes as "applicable to Rajasthan," a curated subset, or Rajasthan-state schemes only for the first cut?