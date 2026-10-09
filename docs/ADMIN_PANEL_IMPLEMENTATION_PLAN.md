# Jan Seva AI — Admin Panel Implementation Plan (Phase-wise)

## Context
The admin panel is the **control room for everything the citizen assistant is allowed to say**. If something isn't in the admin panel and approved, the bot must not say it.

Today the bot reads hand-built JSON files with no versioning, no sources, no review, no login and no audit. Data fixes are one-off scripts, and the knowledge base (KB) can't reload without a restart.

This plan delivers the panel in three phases, matching the priority list:
- **Phase 1:** must-have for the first version (items 1–6).
- **Phase 2:** once citizens are using it (items 7–10).
- **Phase 3:** when working with departments (items 11–13).

A privacy model runs through all phases.

Decisions already made:
- Rajasthan data for now; `state` is a column everywhere.
- PostgreSQL.
- A separate `admin/` React app.
- The backend now lives in `backend/`.

On approval, `docs/ADMIN_PANEL_IMPLEMENTATION_PLAN.md` is rewritten to match this plan.

## Guiding rules (apply to every phase)
1. **Only published records reach the bot.** When a published record is edited, the old version stays live until the new version is approved and published.
2. **Every field can be "unknown."** The bot never guesses. Unknown fields reach the bot as an explicit "Not verified — please check the official portal: <link>".
3. **Every record links to at least one source.** myScheme is a *discovery* source, never the final authority.
4. **Permissions are enforced in FastAPI** (`require_perm(...)`). The UI only hides what the server already forbids.
5. **Every change is audited** with who, what, when, and old and new values. So is every view of sensitive conversation data.
6. **Citizen privacy.** General admin roles never see phone numbers, Aadhaar or Jan Aadhaar IDs, OTPs or full conversations. They see masked, trimmed snippets only. Full transcripts are limited to a separate permission held by a few people, and every access is logged.
7. **Chat never depends on the admin system being up.** If Postgres is down, the bot answers from the last active build.

## Architecture
```
admin/ (:5175) ──► FastAPI /admin/api (JWT cookie + RBAC) ──► PostgreSQL (records, sources, versions, audit, settings, issues)
                                                                   │ approve → auto-publish
                                                                   ▼
                    adapter (canonical record → existing KB-builder input) → new KB build → smoke check → hot swap
                                                                   ▼
                     RagService / JobService / BotSettings ◄── /chat ──► chat_turns (redacted) ──► Issue review
```

## Existing code to reuse (no rewrites)
- **KB build functions:** `backend/scripts/pipeline/build_knowledge_base.py` (`build_discovery_cards`, `build_field_chunks`, `eligibility_metadata`, `base_metadata`)
- **myScheme client:** `backend/scripts/pipeline/scrape_myscheme.py` (`search_schemes`, `scrape_scheme`)
- **Eligibility mapping:** `backend/scripts/pipeline/extract_eligibility.py` (`extract_one`)
- **Rules engine:** `eligibility_verdict` and `OCCUPATION_KEYWORDS` in `backend/app/services/rag_service.py`
- **Job status from dates:** `_compute_status` in `backend/app/services/job_service.py`
- **Service getters:** `get_rag_service` / `get_job_service` in `backend/app/graph/chat.py`
- **PII helpers:** `scrub_jan_aadhaar_ids` in `backend/app/utils/pii_utils.py`; `redacted_raw_message` in the graph state
- **Hardcoded text that moves to settings:** `_NO_MATCH_TEXT` (`chat.py`), the error apology (`main.py`), `EmptyState.jsx` / `SuggestionsPanel.jsx` (frontend)
- **Regression baseline:** `backend/eval/golden_set.py` (92 pass, 1 known fail)

---

# PHASE 1 — Must-have for the first version (items 1–6)  ≈ 6–7 weeks for one developer

### P1.1 Foundation (~4 days)
- **Dependencies:** `sqlalchemy>=2`, `alembic`, `psycopg[binary]`, `pyjwt`, `argon2-cffi`, `pytest`, `httpx`. Add `docker-compose.yml` with Postgres 16.
- **Config:** `DATABASE_URL`, `ADMIN_JWT_SECRET`, `ADMIN_CORS_ORIGINS`, `ADMIN_BOOTSTRAP_EMAIL/PASSWORD`, `KB_ROOT`. Admin features turn off cleanly when `DATABASE_URL` is unset.
- **`backend/app/admin/` package:** `db.py`, `models.py`, `schemas.py`, `security.py` (argon2 + JWT httpOnly cookie), `deps.py`, `permissions.py`, `audit.py`, `routers/`, `services/`.
- **Migrations:** Alembic. **Tests:** pytest with a throwaway test database.

### P1.2 Records: schemes, services, jobs (item 1)
One canonical, bilingual, field-level record model. `kind` is `scheme` or `service` (certificates and public services), and jobs have their own table.

| Field group | Fields (each `{en, hi}` where it's text) |
|---|---|
| Identity | name, short name, kind, department, level, state, categories, tags |
| Content | description, benefits, eligibility (text), exclusions, required documents (list), how to apply (steps, mode online/offline), FAQs |
| Links and contact | official link, application link, helpline numbers / email |
| Dates | opening date, closing date, other important dates (label + date) |

- **Per-field status:** a `field_status` JSONB map holds `known` / `unknown` / `not_applicable` for each field.
  - Unknown fields are stored explicitly and shown as "Unknown" badges in the UI.
  - The KB adapter renders them as "Not verified — check official portal" chunks.
- **Snapshot:** the raw import (e.g. the myScheme JSON) is kept in `source_snapshot` for reference.
- **Jobs (v1):** today's fields (title, department, vacancies, qualification, location, start and last dates, salary, documents, portal), plus `field_status`. Extended job fields come in Phase 2 (item 9).
- **Seed** (`backend/scripts/tools/seed_admin_db.py`):
  - Maps the 157 myScheme records and the eligibility file into canonical records, and the 120 jobs.
  - Fields missing from the data, such as the helpline, are marked `unknown`.
  - Everything is loaded as **published v1**, so the bot's behaviour is unchanged on day one.

### P1.3 Structured eligibility rules (item 2)
Stored as fields, not sentences, in `eligibility_rules` on each version:

```
age:            {min, max, applies_to}            applies_to ∈ self|parent|spouse|child|guardian|household
income:         {family_max, individual_max, parent_max, currency_period: annual}
category/caste: [..] | ["all"]        gender: [..] | ["all"]        marital_status: [..] | ["all"]
residence:      rural|urban|all       districts: [..] | ["all"]
occupation:     [{group, applies_to}]           e.g. {group:"construction_worker_registered", applies_to:"parent"}
education_level, disability
other_conditions: [{text_en, text_hi}]          shown to citizens, not machine-checked
```
- **Example:** "age 6–14, parent is a registered construction worker" is stored as `age{6,14,self}` plus `occupation[{construction_worker_registered, parent}]`.
- **Allowed values:** enum lists live in `backend/app/admin/vocab.py`, reusing `OCCUPATION_KEYWORDS` and the existing value sets.
- **Engine compatibility:** an adapter `to_engine_facets(rules)` produces exactly today's flat 19-key record, so the current `eligibility_verdict` behaviour is unchanged.
- **Rules-engine extension (P1.8, the only intentional bot-logic change):**
  - New checks: `districts` and `applies_to`.
  - Fail-open: a scheme is only excluded when a known citizen fact contradicts a rule.
  - Covered by new golden-set cases.

### P1.4 Official source registry (item 3)
- **`sources` table:**
  - type: `portal` / `government_order` / `pdf` / `notification` / `website` / `api` / `manual`
  - authority: `official` / `discovery`
  - title, URL or uploaded PDF, GO number, issuing department, **issue date**, **last checked at / by**, notes
- **PDF and GO files** are stored on disk under `backend/data/uploads/` with a SHA-256 hash.
- **`record_sources` link table:** record version ↔ source, with which fields that source supports (optional).
- **Validation:**
  - Submitting a record needs at least one source.
  - A record with only discovery sources gets a **"Discovery-only"** warning badge.
  - `REQUIRE_OFFICIAL_SOURCE` (default off, so the seeded myScheme data can stay live) makes an official source mandatory for publishing.
- **Seed:** myScheme is registered as a `discovery` source and linked to all 157 schemes.

### P1.5 Review and approval workflow (item 4)
```
draft ──submit──► in_review ──approve──► published ──archive──► archived
  ▲                   │                      │
  └──── returned ◄────┘ (note required)      └─ edit = new draft; old version stays live until the new one is approved
```
- **One open draft per record**, enforced with a partial unique index.
- **Submit checks:** required fields present or explicitly marked unknown, valid enums, `age.min ≤ age.max`, non-negative incomes, at least one source.
- **Approve auto-publishes.** Approval queues a KB build. Builds are debounced (approvals within about 30 seconds share one build).
  - The build passes a smoke check, then is hot-swapped in, and only then does the version become `published`.
  - The previous version becomes `superseded` and stays in history.
  - If the build fails, the version stays `approved` and the old one stays live; the super admin is alerted on the dashboard.
- **Archive** removes a record from the next build.
- **Return** needs a note, which is shown in the researcher's draft view.
- **No self-approval:** a reviewer can't approve their own draft. `ALLOW_SELF_APPROVAL` exists for single-person demos only and is audited.

### P1.6 KB publish with hot swap (enables P1.5)
- **Build code:** move the build logic to `backend/app/services/kb_builder.py` as `build_kb(records, out_dir)`. Each build goes into a new `KB_ROOT/<build_id>/` and never deletes the live one. The script becomes a thin wrapper.
- **Adapter:** `backend/app/admin/services/kb_adapter.py` turns canonical records into today's `(raw, elig)` input shape, so retrieval and ranking stay unchanged.
- **Services:** `RagService(persist_dir=None)`, `JobService(jobs=None)`, and `swap_rag_service` / `swap_job_service` behind a lock in `chat.py`. An `ACTIVE` pointer file says which build is live; startup falls back to `data/chromadb`.
- **Smoke check:** collection counts plus 5 golden queries. A failure leaves the live build untouched. The last 5 builds are kept, and the super admin can roll back.

### P1.7 Roles, permissions, audit log (items 5, 6)
| Role | Can do |
|---|---|
| `researcher` | Create and edit drafts, manage sources, submit |
| `reviewer` | Approve (not their own), return with a note, archive |
| `super_admin` | Everything, plus users, rollback and settings |
| `auditor` | Read-only on everything, including the audit log; no edits |

- **Enforcement:** `permissions.py` holds the role → permission matrix, and every route uses `Depends(require_perm("record.approve"))`. Tests cover the full matrix.
- **`audit_log` table:** user, action (`record.draft.update`, `record.approve`, …), entity, **before/after JSONB**, IP, user agent, timestamp. It is append-only: the app's DB role has no UPDATE or DELETE on it.
- **Audit UI:** filter by user, record, action and date, with a field-level before/after diff viewer and CSV export (for government clients).
- **Login security:** rate limiting, lockout after repeated failures, Secure/SameSite cookies, and a CSRF header on admin writes.

### P1.8 Rules-engine extension for new fields (~3 days)
- **Districts:** extend `eligibility_verdict` with a district check. Only applies when the citizen's district is known; add district to profile extraction.
- **applies_to:** check occupation and age against the right person (self or a family member), using the existing `alt_citizen` path.
- **Tests:** new golden-set cases. Existing results must stay identical (92 pass, 1 known fail).

### P1.9 Admin UI for Phase 1 (`admin/`, ~7 days)
- **Stack:** React 18 + Vite + react-router + Mantine + TanStack Query, themed with the citizen app's colour tokens and fonts.
- **Pages:**
  - Login, Dashboard (review backlog, unknown-field counts, discovery-only records, last build)
  - **Records:** list filtered by kind, department, status, unknown or discovery-only
  - **Record editor:** tabs for Basics · Content (en/hi side by side) · Eligibility rules (structured form) · Documents · How to apply · Links & helplines · Dates · Sources · Versions (diff) · Audit. Each field has an "Unknown" toggle.
  - **Jobs** list and editor
  - **Sources** registry (with PDF upload)
  - **Review queue:** diff between the live and new version, Approve / Return (note) / Archive
  - **Publish status:** builds and rollback
  - **Users** (super admin), **Audit log**

**Phase 1 acceptance:**
- Seeded data gives an identical golden-set result.
- Researcher edits → reviewer approves → the bot answers with the new text and no restart, and the old version stays live until then.
- Unknown fields make the bot say "not verified, check <link>."
- Wrong role → 403.
- Every change appears in the audit log with old and new values.

---

# PHASE 2 — Once citizens are using it (items 7–10)  ≈ 4 weeks

### P2.1 Citizen issue review (item 7)
- **Log every turn** in `/chat` from a background task that never breaks chat:
  - `chat_turns` stores the message (`redacted_raw_message` + `scrub_jan_aadhaar_ids` + phone and OTP masking), reply, language, intent, `nodes_visited` (via `_graph.stream(stream_mode="updates")`; graph logic untouched), `no_match`, and the schemes shown.
  - The response gains a `turn_id`.
- **Citizen UI:** 👍/👎 with reasons (wrong info, outdated, didn't answer, other), and a "report a problem" option. `POST /feedback`.
- **`issues` table:**
  - Auto-created for `no_match` turns and 👎.
  - Created manually for wrong eligibility, outdated info and complaints.
  - Each issue has a category, language, a **masked snippet** (the turn plus one turn of context, trimmed to about 300 characters), linked record, status, assignee and resolution.
- **"Create draft fix"** opens a draft on the linked record.
- **"Top missing data"** groups unanswered questions by topic, giving the list of data to add next.
- **Privacy:** full transcripts need the `conversation.full_view` permission, granted per user by the super admin. Every view writes an audit row. Retention is configurable (default 90 days).
- **New role `quality_reviewer`:** sees issues and masked snippets only.

### P2.2 Source freshness alerts (item 8)
- **Scheduled checker:** APScheduler, running in-process and daily, or the same script via cron. For each source it:
  - flags it **stale** if not checked in N days (default 90, per-source override);
  - runs a **link check** (HTTP status, redirects);
  - runs **content-change detection** (normalized page or PDF hash compared with the last check).
- **Alerts:** each problem creates a `source_alert` row (stale, broken link, content changed).
  - Linked published records get a **"Needs re-verification"** flag. They stay live; the flag is an internal warning.
  - The alert creates a review task.
- **UI:** an Alerts page and dashboard counter. "Mark verified" updates `last_checked_at/by` and is audited.

### P2.3 Job and recruitment notices (item 9)
- **Extended job schema:** posts (a list of post name plus vacancies), total vacancies, qualification, **age limit** (min, max, relaxation note), **fee** (by category), start and last dates, **notification PDF link**, **application page link**, recruiting body, and source. Each field supports `unknown`.
- **Notices** (`kind=notification`): admit cards, results, corrigenda, linked to a job.
- **Auto-expiry:**
  - A daily job moves records whose last date has passed to `expired`.
  - Publishing excludes them.
  - The bot already computes status from dates (`_compute_status`) as a second safeguard, so a closed job is never shown as open, even between builds.

### P2.4 Bot response settings (item 10)
- **`bot_settings`:** versioned, with the same review workflow and audit. It holds, per language:
  - the welcome message and example questions (by category);
  - fallback replies, e.g. "I don't have verified information on this, please check [portal]";
  - the error apology and the no-match text;
  - the helpline directory (name, number, hours, department).
- **Backend:** a `SettingsService` with an in-memory cache, refreshed on publish. `_NO_MATCH_TEXT` and the `main.py` apology read from it, with **today's text as the default** so nothing changes until an admin edits it.
- **Frontend:** `EmptyState.jsx` and `SuggestionsPanel.jsx` load from `GET /public/settings`, keeping their current hardcoded values as a fallback.

---

# PHASE 3 — When working with departments (items 11–13)  ≈ 3–4 weeks

### P3.1 Analytics (item 11)
- **Data:** only aggregates from `chat_turns` / `issues` (nightly rollup tables). No per-citizen rows are exposed.
- **Small-count suppression:** counts under 5 are shown as "<5".
- **Views:** most-asked schemes and services, demand by district (needs the district capture from P1.8), common missing documents, unanswered-question rate over time, and language mix.
- **Export:** CSV of the aggregates.

### P3.2 Staff management (item 12)
- **Staff accounts:** a `staff` user type with a **department and district scope**. This is the management side only; the staff workspace itself is a separate project.
- **Who manages them:** the super admin and a new `department_admin` role, limited to their own department.
- **Audit:** every grant and revoke is logged.

### P3.3 Integration health (item 13)
- **Event recording:** `integration_events` records each outbound call (OpenRouter LLM, Sarvam STT, Jan Aadhaar, myScheme, and WhatsApp later) with status, latency and error class. Recording happens in thin wrappers at the existing call sites, and nothing about how the calls work changes.
- **Health page:** per-integration uptime, error counts and p95 latency (last hour / 24 hours), plus a last-error sample with no personal data.
- **Alerts:** threshold alerts appear on the dashboard (email or webhook later).

---

## What stays out of the admin panel
- **Phone numbers, Aadhaar or Jan Aadhaar IDs, OTPs:** never stored unmasked in `chat_turns`, and never shown to any role.
- **Full conversations:** only for `conversation.full_view` holders; every access is audited.
- **Analytics:** aggregates only, with small counts suppressed.
- **Logs:** citizen profile data stays in the redacted turn log only; it never appears in audit or integration logs.

## Critical files
- **New:**
  - `docker-compose.yml`
  - `backend/alembic/**`
  - `backend/app/admin/**`: models, routers, `services/{versioning, validation, kb_adapter, publish, turn_logger, freshness, settings}`
  - `backend/app/services/kb_builder.py`
  - `backend/scripts/tools/seed_admin_db.py`
  - `backend/tests/admin/**`
  - `admin/**`
- **Modified:**
  - `backend/app/main.py`: admin router, CORS, warm-up from the active build, turn logging, `turn_id`, `/feedback`, `/public/settings`.
  - `backend/app/graph/chat.py`: swap functions, settings lookup for fallback text, district in profile extraction (P1.8).
  - `backend/app/services/rag_service.py`: `persist_dir`; district and `applies_to` checks (P1.8).
  - `backend/app/services/job_service.py`: `jobs=` constructor argument.
  - `backend/app/core/config.py`, `requirements.txt`, `.env.example`, `README.md`.
  - Citizen frontend: `chatStore.js`, `MessageBubble.jsx`, `chatApi.js`, `EmptyState.jsx`, `SuggestionsPanel.jsx`, `vite.config.js`.
- **Docs:** `docs/ADMIN_PANEL_IMPLEMENTATION_PLAN.md`, rewritten to this structure.

## Verification
1. **Setup:** `docker compose up -d` → `alembic upgrade head` → seed. Expect 157 schemes and 120 jobs as published v1, and myScheme linked as a discovery source.
2. **Unit tests:** `pytest backend/tests/admin` covers the RBAC matrix, the workflow state machine (including "old version stays live"), unknown-field validation, source requirement, audit before/after, publish failure and rollback, PII masking, and full-view access auditing.
3. **Regression:** `python eval/golden_set.py` on a DB-built KB matches the baseline. After P1.8 it passes the new district and `applies_to` cases too. `eval/conversation_tests.py` still passes.
4. **Browser end-to-end:**
   1. Researcher edits a scheme and marks the helpline unknown.
   2. Reviewer returns it with a note.
   3. Researcher fixes and resubmits.
   4. Reviewer approves; the bot shows the new text without a restart and says "not verified" for the helpline.
   5. Audit shows every step.
5. **Phase 2:**
   - A 👎 shows up as an issue with a masked snippet.
   - A stale or broken source raises an alert and flags its records.
   - A job past its last date disappears from the bot.
   - Editing the welcome message changes the citizen UI with no deploy.
6. **Phase 3:**
   - Analytics show only aggregates, with "<5" suppression.
   - Staff scope limits are enforced.
   - Stopping the Sarvam key shows STT errors on the health page.
7. **Resilience:** with Postgres stopped, `/chat` still answers.
