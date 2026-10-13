# Raj Janhit Seva AI — Chatbot (from scratch)

A conversational assistant that helps Rajasthan citizens discover **government welfare schemes** and **government job vacancies**, in Hindi, Hinglish, or English. Built from scratch — no reused code from any earlier project.

This README explains the full pipeline (how raw scheme data becomes something the bot can search) and the exact steps to get it running on your own machine.

The repository also contains the **Government Staff AI Assistant** in `gov-staff/`, a separate login-only app for government employees. See [section 4](#4-government-staff-ai-assistant-gov-staff).

---

## 1. How it actually works, end to end

There are two separate data domains, each with its own pipeline, feeding into one shared conversation layer:

All paths below are relative to `backend/` (except `frontend/`).

```
SCHEMES                                          JOBS
--------                                         ----
myScheme.gov.in API                              data/processed/jobs.json
      |  scrape_myscheme.py                            (provided directly,
      v                                                  no scraping step)
data/raw/{date}/*.json
      |  extract_eligibility.py
      v
data/processed/eligibility_structured.json
      |  build_knowledge_base.py
      v
data/chromadb/  (2 Chroma vector collections)          |
      |                                                 |
      v                                                 v
app/services/rag_service.py (RagService)         app/services/job_service.py (JobService)
      \                                                 /
       \_______________________  ________________________/
                               \/
                    app/graph/chat.py  (LangGraph conversation graph)
                               |
                    app/main.py  (FastAPI: POST /chat)
                               |
                    frontend/  (React chat UI)
```

**Why two pipelines?** Schemes are gated by citizen *identity* (caste, gender, marital status, income...) and live as long free-text eligibility descriptions — that needs semantic search + a fail-open eligibility filter. Jobs are small, fully structured records (title, department, qualification, deadline, vacancies) — that just needs fuzzy field matching, no embeddings at all. Forcing both through the same mechanism would have made either one worse, so they're separate services that plug into the same conversation graph.

### Step by step: turning scheme data into something searchable

1. **`scripts/pipeline/scrape_myscheme.py`** — pulls every Rajasthan scheme from myScheme.gov.in's public API (both English and Hindi versions of each scheme) and writes one raw JSON file per scheme into `data/raw/{scrape-date}/`.
2. **`scripts/pipeline/extract_eligibility.py`** — reads those raw files and produces one flat, structured record per scheme in `data/processed/eligibility_structured.json`: caste/gender/marital-status/age/income facets, occupation target group, benefit type, etc. Mostly mechanical (the source API already returns machine-coded facets); only `education_level` needs an LLM call (once per scheme, cached forever), since that fact usually only exists buried in free eligibility text.
3. **`scripts/pipeline/build_knowledge_base.py`** — reads the structured records plus the raw scraped text and builds two persisted Chroma vector collections in `data/chromadb/`:
   - `discovery_cards` — one short embeddable "card" per scheme, used for the initial semantic shortlist search.
   - `field_chunks` — each scheme's eligibility/benefits/documents/application text, chunked field-by-field (not by arbitrary character windows), used to answer follow-up questions about one specific scheme.

This is a **one-time offline pipeline** — it's not run live during a conversation. `data/chromadb/` is gitignored (regenerable, and large) — you rebuild it locally by re-running steps 1–3 (see Setup below).

### Step by step: a citizen's message becoming a reply

This is `app/graph/chat.py`'s [LangGraph](https://langchain-ai.github.io/langgraph/) state machine, run fresh on every message:

1. **`normalize`** — light cleanup of the raw message.
2. **`extract_profile`** — an LLM call that pulls out any *new, explicitly stated* facts about the citizen themselves (caste, gender, residence, education level...) and merges them into a persistent profile for this conversation. The same call also records free-form **conversation notes** — durable facts with no profile field of their own (a family member's class, documents held, answers to the bot's own questions) — so they never expire with the rolling chat history and are never re-asked.
3. **`extract_need`** — a second LLM call that classifies what this message actually wants: is it a fresh need, a follow-up on something just shown, or a request for more options? Is it about a scheme or a job? Is it for the citizen or a family member? What language should the reply be in?
4. Routing splits here:
   - **Scheme domain** → `stage_a` (semantic shortlist search + fail-open eligibility filter over `RagService`) → either `clarify`/`clarify_identity` (ask one short question if genuinely needed) or `stage_b` (fetch full detail for the best matching schemes) → `generate`.
   - **Job domain** → `stage_a_jobs` (fuzzy search over `JobService`) → `generate_jobs`.
   - **Follow-up on something already shown** → `stage_b_followup` / `stage_b_followup_jobs` (re-verify + re-fetch detail for the specific scheme/job referenced) → the same generate step.
5. **`generate` / `generate_jobs`** — the final LLM call that composes the actual reply from the retrieved data, in the citizen's own language/register (Hindi / Hinglish / English), using conversation history so follow-ups and short answers ("nahi", "haan") are understood in context.
6. **`finalize`** — appends this turn to the conversation history and returns.

All of this state is kept in memory per `session_id` (see `app/main.py` — no database, this is a from-scratch prototype, not a production deployment).

---

## 2. What's in each folder

```
janseva-ai/
├── backend/                          # Everything Python (FastAPI app, pipeline, eval, data)
│   ├── app/                          # The actual application
│   │   ├── main.py                   #   FastAPI app - POST /chat, GET /health, POST /transcribe
│   │   ├── core/
│   │   │   └── config.py             #   loads backend/.env (OpenRouter API key/model, Sarvam, Jan Aadhaar)
│   │   ├── graph/
│   │   │   ├── chat.py               #   the LangGraph conversation graph - the core logic
│   │   │   └── state.py              #   the conversation's shared state shape (ConvState)
│   │   ├── services/
│   │   │   ├── rag_service.py        #   Stage A/B retrieval over the scheme vector store (RagService)
│   │   │   ├── job_service.py        #   structured fuzzy search over jobs.json (JobService)
│   │   │   └── stt_service.py        #   speech-to-text via Sarvam
│   │   ├── integrations/
│   │   │   └── janaadhaar/           #   Jan Aadhaar client, crypto, service + local mock server routes
│   │   └── utils/
│   │       └── pii_utils.py          #   regex PII extraction (Jan Aadhaar ID, OTP, consent)
│   │
│   ├── data/
│   │   ├── raw/                      #   one JSON file per scheme, straight from myScheme's API (scrape output)
│   │   ├── processed/                #   eligibility_structured.json (schemes) + jobs.json (job postings)
│   │   └── chromadb/                 #   ⚠ gitignored - the built vector store, regenerate locally (see Setup)
│   │
│   ├── scripts/                      # One-time/offline tooling, and manual test harnesses
│   │   ├── pipeline/
│   │   │   ├── scrape_myscheme.py         #   step 1 of the scheme pipeline
│   │   │   ├── extract_eligibility.py     #   step 2
│   │   │   └── build_knowledge_base.py    #   step 3
│   │   ├── fixes/                         #   one-time data patches (fix_*.py)
│   │   └── tools/
│   │       ├── query_kb.py                #   debug: raw semantic search only, no filtering
│   │       ├── query_retrieval.py         #   debug: full Stage A/B retrieval (filter + rank boost)
│   │       └── chat_cli.py                #   a full interactive multi-turn conversation in your terminal
│   │
│   ├── eval/
│   │   ├── golden_set.py             #   regression test suite (retrieval + eligibility filter correctness) - no LLM calls, deterministic
│   │   └── conversation_tests.py     #   multi-turn conversation replays (real LLM calls)
│   │
│   ├── requirements.txt              # Python dependencies
│   └── .env.example                  # copy to .env and fill in your own key
│
├── frontend/                         # React + Vite chat UI
│   └── src/
│       ├── store/chatStore.js        #   conversation state (Zustand) + calls the API
│       ├── api/chatApi.js            #   POST /chat wrapper
│       └── components/               #   ChatWindow, MessageBubble (markdown rendering), InputBar
│
└── PLAN.md                           # the original design/architecture notes this was built against
```

---

## 3. Setup — running this on a new machine

### Prerequisites
- Python 3.11–3.12
- Node.js 18+ (for the frontend)
- An [OpenRouter](https://openrouter.ai/) API key (used for every LLM call — extraction, retrieval reasoning, and reply generation) - **required**, the app won't start without it.
- A [Sarvam AI](https://www.sarvam.ai/) API key - **optional**, only needed for the mic/voice-input button (speech-to-text). Everything else works fine without it.

### 3.1 Backend

```bash
cd backend
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd.exe)
.venv\Scripts\activate.bat
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

# Add your API key(s)
cp .env.example .env        # Windows: copy .env.example .env
# then edit .env and paste your OPENROUTER_API_KEY (required)
# and, if you want voice input, your SARVAM_API_KEY (optional)
```

### 3.2 Build the knowledge base (one-time, ~few minutes)

`data/processed/eligibility_structured.json` is already included, so you can skip straight to building the vector store from it:

```bash
python scripts/pipeline/build_knowledge_base.py
```

This creates `data/chromadb/` locally. It downloads a multilingual sentence-transformer embedding model on first run.

If you ever need to re-scrape fresh scheme data from myScheme.gov.in instead of using the included snapshot, run the first two pipeline steps before this one:
```bash
python scripts/pipeline/scrape_myscheme.py
python scripts/pipeline/extract_eligibility.py
```

`data/processed/jobs.json` (job postings) needs no build step — `JobService` reads it directly.

### 3.3 Start the backend server

```bash
# from backend/ with the venv active
uvicorn app.main:app --reload --port 8000
```

Wait for `"Warm-up complete - ready to serve requests."` in the log — that means the embedding model and both services have loaded. Check `http://localhost:8000/health` returns `{"status": "ok"}`.

### 3.4 Start the frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL it prints (default `http://localhost:5174`) and start chatting.

### 3.5 (Optional) Test without the frontend

```bash
# from backend/ with the venv active
# Interactive terminal conversation
python scripts/tools/chat_cli.py

# Regression suite (no LLM calls, fast, deterministic)
python -m eval.golden_set
```

---

## 4. Government Staff AI Assistant (`gov-staff/`)

A separate, login-only assistant for **verified government employees**. Staff upload their department's documents and ask questions; answers come **only** from documents their own department may see, with references to the exact passages. Every department's documents are strictly isolated from every other department. Full plan: [`docs/PLAN.md`](docs/PLAN.md), section 12.

`gov-staff/` is **self-contained**: its own backend, frontend, `.env`, database and (later) knowledge bases. It shares no backend code, database or login with the citizen app, and it never modifies `backend/` or `frontend/`.

### Status

| Phase | Scope | State |
|---|---|---|
| P0 Foundation | backend + frontend skeletons, staff database and migrations, department isolation in the database, audit log, health check | Done |
| P1 Onboarding | registration (name, email, phone, department, designation, password), OTP verification by email or phone (test code `1234`, nothing sent yet), sign-in with email or phone, sessions, audit | Done (testing mode) |
| One frontend | start screen "Choose user type"; citizen and staff assistants in one app | Done |
| P2 Documents | upload (public/private), validation, malware scan, text extraction + OCR, per-department indexing | Next |
| P3 Q&A | department-only retrieval, grounded answers with references, isolation test suite | Planned (the staff chat replies "being set up" until then) |
| P6 Hardening | real email/SMS OTP, rate limits, security review, backups, runbook | Planned |

Still to decide (later phase): how to confirm that a person really belongs to the department they choose at registration (staff list or department access code, PLAN section 12.5).

### Structure
```
gov-staff/
├── backend/     FastAPI staff API (port 8200)
│                app/core (config, db, audit), app/models, app/routers (health, departments, auth, ask),
│                app/auth (passwords, sessions, OTP, validation), alembic/ migrations, tests/, scripts/,
│                data/ (gitignored: dev database)
├── frontend/    React + Vite, one app for both assistants (port 5180)
│                pages/ (choose user type), citizen/ (citizen app, read-only), auth/ (register, OTP, login),
│                staff/chat/ (staff chat), i18n/ (English, Hindi)
└── docs/        ARCHITECTURE.md
```

### One frontend for both assistants
Run only `gov-staff/frontend` (http://localhost:5180):
- `/` asks **Choose user type**.
- `/citizen` shows the citizen assistant, using the citizen app's own screens from `frontend/src` read-only, never edited.
- `/staff/register`, `/staff/verify` and `/staff/login` handle staff registration, OTP and sign-in; `/staff` is the staff chat, with the same look as the citizen chat.

The frontend sends citizen calls (`/chat`, `/transcribe`) to the citizen backend on **8000**, and staff calls (`/api`) to the staff backend on **8200**, so run both backends. For other addresses, set `CITIZEN_API_TARGET` / `STAFF_API_TARGET` before `npm run dev`.

### Run it (Windows, PowerShell)
Citizen backend: see section 3.3 (port 8000).

Staff backend:
```powershell
cd gov-staff\backend
python -m venv .venv                                  # first time only
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env                                # first time only
.venv\Scripts\python -m alembic upgrade head          # create / update the database
.venv\Scripts\python -m uvicorn app.main:app --port 8200
```
Without `DATABASE_URL`, development uses SQLite at `gov-staff/backend/data/staff.db`; production must use PostgreSQL (see `gov-staff/docs/ARCHITECTURE.md`). Restart the staff backend after code changes, or add `--reload`.

Frontend (one for both assistants):
```powershell
cd gov-staff\frontend
npm install                                           # first time only
npm run dev                                           # http://localhost:5180
```

Test accounts (development only; create or reset with `.venv\Scripts\python -m scripts.create_dummy_users`):

| Email | Password | Department |
|---|---|---|
| abhi@buildyourtech.com | admin123 | Medical & Health |
| ravi@buildyourtech.com | admin321 | Agriculture |

These passwords are public. Never use these accounts anywhere shared; the script refuses to run in staging or production.

### Tests
```powershell
cd gov-staff\backend
.venv\Scripts\python -m pytest -q
```
Tests build a throwaway SQLite database with the real migrations. To also check the PostgreSQL protections (row-level security, restricted role, append-only audit log), point `TEST_POSTGRES_URL` at an **empty throwaway** database as a **superuser** (or a role with BYPASSRLS + CREATEROLE):
```powershell
$env:TEST_POSTGRES_URL = "postgresql+psycopg://postgres:<password>@localhost:5432/staff_test"
.venv\Scripts\python -m pytest -q tests\test_postgres_isolation.py
```

### Security in short
- Passwords and OTP codes are stored only as hashes.
- Sessions use an httpOnly cookie. The department always comes from the database, never from the request.
- Every sign-in, registration, OTP step and question is recorded in an append-only audit log.
- The staff backend refuses to start in staging/production with unsafe settings: a weak session secret, non-https origins, SQLite, or the OTP test code still set.
