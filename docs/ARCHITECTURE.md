# Architecture — Government Staff AI Assistant

## Separation from the citizen app
The citizen assistant (`../backend`, `../frontend`) is public and anonymous. This staff assistant is login-only and handles internal department documents. They run as **separate processes** with separate `.env` files, databases, files, knowledge bases and ports. They share no code (small helpers are copied, never imported). The only planned link is a "Government employee" button on the citizen landing page that opens this app's URL. That button is on hold until the citizen app may be changed.

```
staff browser (React, :5180) ──cookie session──► staff API (FastAPI, :8200)
                                                    ├──► PostgreSQL (staff DB; RLS by department; restricted app role)
                                                    ├──► files: data/files/<department_id>/ (encrypted)       [P2]
                                                    ├──► Chroma: one collection per department (dept_<id>_docs) [P2]
                                                    └──► LLM (OpenRouter) for grounded answers                  [P3]
```

## Backend layout (`backend/app`)
| Package | Contents |
|---|---|
| `core/` | `config.py` (settings plus a production safety guard), `db.py` (engine, `get_db`, `set_department_scope`, `verify_app_role`), `audit.py` (`write_audit`, append-only) |
| `models/` | `Department`, `StaffUser`, `OtpCode`, `VerificationRequest`, `AuditLog` (P0); `Document`, `DocumentAccess` (P2) |
| `routers/` | `health` (`GET /api/health`), `departments` (`GET /api/departments`, public, names only); then auth/admin (P1), docs (P2), ask (P3) |
| `schemas/` | Pydantic response/request models |
| `auth/` (P1) | OTP, affiliation rules, sessions, `StaffActor`, permissions |
| `ingest/`, `storage/` (P2) | validation, malware scan, extraction/OCR, chunking, indexing; encrypted file store |
| `qa/` (P3) | authorized retrieval, grounded answers with references, staff chat memory |

## Data model (P0)
- **departments**: the tenants (code, English/Hindi name, active).
- **staff_users**: email (unique), name, designation, employee ID, department, role (`super_admin`, `dept_admin`, `officer`, `viewer`) and status (`pending_email` → `pending_approval` → `active`, or `rejected` / `suspended`). Database checks enforce the allowed values, and that every user except the super admin has a department.
- **otp_codes**: hashed one-time codes with expiry and attempt counts (used from P1).
- **verification_requests**: affiliation approvals by a Department Admin or the Super Admin.
- **audit_log**: append-only record of security-relevant actions.

## Department isolation (three layers)
1. **Application:** the department and role come only from the authenticated user record (P1 `StaffActor`). Requests can't choose a department.
2. **Database (PostgreSQL):** migration 0001 sets up:
   - a NOLOGIN role `staff_app_rw` with only the privileges the app needs, and nothing granted to PUBLIC
   - `FORCE ROW LEVEL SECURITY` on `staff_users`, `verification_requests` and `audit_log`; rows are visible only when `department_id = current_setting('app.department_id')`, or when `app.is_super_admin = 'true'`
   - writes into another department refused by `WITH CHECK`
   - a `staff_users_guard` trigger: every new account starts pending and below admin unless the super admin creates it; only the super admin can move a user to another department or grant `super_admin`; any other role change or activation (`status` → `active`) needs an admin context (a department scope or the super admin)
   - departments that the app role can read but not change
   - one narrow **self** policy on `staff_users`: before the department is known (login, OTP, loading a session), `set_auth_lookup(email=…)` or `set_auth_lookup(user_id=…)` shows exactly that one row and nothing else. In this mode the app may read and update that row only: no inserts, no role change, no self-activation. The OTP step (`pending_email` → `pending_approval`) is allowed.
   - `audit_log` uses client-generated ids, so it can be appended to before a user is known; reading it stays scoped

   `set_department_scope()` and `set_auth_lookup()` store the scope on the SQLAlchemy session, and an `after_begin` listener re-applies it on every transaction, so a commit can't drop it. Registration is the one place a request-chosen department is used: the new user's row is inserted under that department's scope, and the account stays `pending` until that department's admin approves it. At startup, `verify_app_role()` checks that the app role is a member of `staff_app_rw` and is not a superuser, a BYPASSRLS role, or (a member of) the table owner; in staging/production it refuses to start otherwise.

   Because of `FORCE`, **even the table owner is subject to RLS** unless it is a superuser or BYPASSRLS. Admin scripts (seed, create super admin; P1) connect with the app role and set the super-admin scope explicitly, and every such action is audited.
3. **Storage (P2):** a separate Chroma collection and file prefix per department. The name is derived on the server from the user's department.

The audit log refuses UPDATE, DELETE and TRUNCATE through triggers, on PostgreSQL and on SQLite.

## Setting up PostgreSQL (staging/production)
1. Create an empty database. Run migrations as the owner: `MIGRATION_DATABASE_URL=... python -m alembic upgrade head`. The owner needs CREATEROLE the first time, because migration 0001 creates `staff_app_rw`; otherwise create that role beforehand.
2. Create the app's login role and add it to the restricted role:
   `CREATE ROLE staff_app LOGIN PASSWORD '<strong random>' IN ROLE staff_app_rw;`
3. Set `DATABASE_URL` to `staff_app`. The app must never connect as the owner.
4. Check with `tests/test_postgres_isolation.py` against an empty throwaway database, as a superuser (see README).

## Frontend (`frontend/src`)
React 18 + Vite, port 5180, React Router, and a small English/Hindi dictionary (`i18n/`). `api/client.js` sends cookies (`credentials: "include"`) and never stores tokens in JavaScript. `layout/StaffShell.jsx` is the frame; from P1 its header always shows the user's department.
