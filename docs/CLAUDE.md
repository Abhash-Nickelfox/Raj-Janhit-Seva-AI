# CLAUDE.md

Working rules for Claude Code in **Raj Janhit Seva AI** (`Raj-Janhit-Seva-AI`). Read the [README](../README.md) for how both assistants work (section 4 covers the Government Staff app) and [PLAN.md](PLAN.md) for the build plan. Section 12 of the plan covers the citizen + government-staff product from `mermaid-diagram (2).png`.

## Hard rules
- **Never modify `backend/` or `frontend/`.** That is the citizen flow, and it must not change. Government Staff work goes only in `gov-staff/`. Reading citizen code to copy a pattern is fine. The unified frontend may import citizen screens read-only (`@citizen` alias) but never edits them.
- **Never touch `../Raj-Janhit-Seva-Admin`.** Don't read, edit or run anything there until the user explicitly says so.
- **Do only what the user asked.** When they say "do not change anything else", change only the files named. Plans and docs are written first; **implementation starts only when the user says so.**
- **Never print, log or echo secrets**: `.env` values, API keys, tokens, OTPs, passwords. Check configuration with yes/no or status codes only. Keep secrets out of docs and commits.
- **Don't commit or push unless asked.**
- **Don't break the citizen bot.** After any backend change, run `eval/golden_set.py` (no LLM calls). Known pre-existing failure: Section 9, "financial assistance" name-boost check. Run `eval/conversation_tests.py` (real LLM calls) only when the change touches conversation logic, and say so first because it costs tokens.
- **Personal data:** citizen profile facts (caste, income, gender, disability) and any uploaded document text never go to logs, traces or a knowledge base.

## Project map
```
backend/app/main.py            FastAPI: POST /chat, POST /transcribe, GET /health (port 8000)
backend/app/graph/chat.py      LangGraph conversation graph (the core; large file, read the relevant part only)
backend/app/graph/state.py     ConvState
backend/app/services/          rag_service.py (Chroma Stage A/B + fail-open eligibility), job_service.py, stt_service.py (Sarvam)
backend/app/integrations/      janaadhaar/ (mock by default)
backend/app/core/config.py     reads backend/.env
backend/scripts/pipeline/      scrape_myscheme → extract_eligibility → build_knowledge_base (offline)
backend/data/                  raw/, processed/, chromadb/ (gitignored, rebuildable)
backend/eval/                  golden_set.py, conversation_tests.py
frontend/                      React + Vite chat UI (Zustand store, components/)
docs/                          PLAN.md, this file, flow diagram
gov-staff/backend/             Government Staff API (port 8200): app/core, models, routers (health, departments,
                               auth, ask), app/auth (passwords, sessions, OTP, validation), alembic/, tests/, scripts/
gov-staff/frontend/            ONE frontend for both assistants (port 5180): / choose user type, /citizen
                               (citizen screens, read-only), /staff/register|verify|login, /staff chat
gov-staff/docs/ARCHITECTURE.md staff app design and the three isolation layers
```

## Commands (Windows; Git Bash or PowerShell)
```
cd backend
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
.venv/Scripts/python eval/golden_set.py
.venv/Scripts/python scripts/pipeline/build_knowledge_base.py      # rebuild data/chromadb
cd ../frontend && npm run dev

cd gov-staff/backend
.venv/Scripts/python -m pytest -q                       # staff tests (SQLite)
.venv/Scripts/python -m alembic upgrade head            # staff database
.venv/Scripts/python -m uvicorn app.main:app --port 8200
cd ../frontend && npm run build                         # compile check
npm run dev                                             # unified frontend, port 5180 (strictPort)
```
- **Ports:** citizen backend 8000; staff backend 8200; unified frontend 5180. The unified frontend proxies `/chat` and `/transcribe` to 8000 and `/api` to 8200. Override with `CITIZEN_API_TARGET` / `STAFF_API_TARGET`.
- **Never start servers on 8000, 8200 or 5180 yourself.** The user runs those, and a leftover process blocks their commands. Use spare ports (e.g. 8299 / 5199) for checks, and stop them afterwards.
- The staff backend runs without `--reload`: after backend changes, tell the user to restart it.
- Staff schema changes need both a model change and a new Alembic migration. Models must stay SQLite-compatible for tests; Postgres-only protections go in the migration. Postgres checks: `tests/test_postgres_isolation.py` with `TEST_POSTGRES_URL` (superuser, empty throwaway database).
- Always use `python -m <tool>`. `.exe` launchers such as `uvicorn.exe` are blocked by Windows Smart App Control.
- The user usually has their own dev server running on :8000 with `--reload`. Don't stop it; it picks up code changes by itself. Use another port for test instances.
- Set `PYTHONIOENCODING=utf-8` when printing Hindi; the console is cp1252.
- In Git Bash, prefix commands that pass `/paths` with `MSYS_NO_PATHCONV=1`. For multi-line edits, prefer the file editor or a script file, because heredocs can mangle quotes and backslashes.

## Conventions
- **Fail-open eligibility:** an unknown citizen attribute never excludes a scheme. Only an explicit contradiction does.
- **Grounded answers only:** never invent eligibility, amounts, steps or links. Links and references are built in code from retrieved metadata, not written by the LLM.
- **Normalise LLM output in code.** Use string enums plus fuzzy matching, not strict booleans or exact enum values (see PLAN §10, Phase 3 lessons).
- **Languages:** reply in the user's language (Hindi / Hinglish / English). Every UI label needs both Hindi and English.
- **Match the surrounding code style.** Comments explain *why*, not what.

## Staff assistant (PLAN §12, `gov-staff/`): non-negotiables
- `department_id` and `role` always come from the signed-in user's database record (`StaffActor`, `app/auth/deps.py`), never from the request, URL, header or prompt.
- **Isolation has three layers, and all three must hold:**
  - application checks
  - PostgreSQL Row-Level Security (`set_department_scope()` / `set_auth_lookup()` in `app/core/db.py`; the scope is re-applied after every commit)
  - a separate Chroma collection and file-storage prefix per department (P2)
- **Every new staff route** needs:
  - a session check (`current_actor`)
  - the `X-Requested-With: fetch` header on changes (`require_fetch_header`)
  - an audit entry (`write_audit`; the audit log is append-only, so never add update or delete paths)
  - tests, plus isolation coverage (two departments, public and private documents, tampered ids, deleted documents, prompt injection)
- **Request models** use `extra="forbid"`, so role, status and department can't be smuggled in. Field rules live in `app/auth/validation.py`, mirrored in `frontend/src/auth/validation.js`.
- **Secrets in the database:** passwords and OTP codes are stored only as hashes; never log them.
- **OTP test code** (`OTP_TEST_CODE`, default `1234`) is development only. The backend refuses to start in staging/production while it's set.
- **Staff code** lives only in `gov-staff/`. It never imports citizen backend code; small helpers are copied, not imported.
- **Grounded answers:** staff answers come only from retrieved passages of the user's own department. Citations are built in code from chunk metadata, never by the LLM.
- **Labels:** every staff UI label goes in both `gov-staff/frontend/src/i18n/en.json` and `hi.json`.

## How work is reviewed
Each phase is developed, then reviewed by a team-lead pass (code, security, plan conformance) and a QA pass (real repros), looping until the team lead is satisfied and QA passes. Reviewers report only; they don't edit.
