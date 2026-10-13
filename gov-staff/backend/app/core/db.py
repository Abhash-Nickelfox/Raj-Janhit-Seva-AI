"""Database engine, sessions and the per-request department scope.

Isolation layer 2 (docs/ARCHITECTURE.md): on PostgreSQL every request runs
with `app.department_id` / `app.is_super_admin` set from the *authenticated*
user, and Row-Level Security policies only show that department's rows.
The department never comes from the request itself.
"""

import logging
from collections.abc import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import DATA_DIR, get_settings

log = logging.getLogger("staff.db")


def _make_engine():
    settings = get_settings()
    if settings.is_sqlite:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        return engine
    return create_engine(settings.database_url, pool_pre_ping=True, pool_size=5, max_overflow=5)


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _apply_scope(db: Session, connection=None) -> None:
    scope = db.info.get("rls_scope")
    if not scope:
        return
    conn = connection if connection is not None else db.connection()
    if conn.dialect.name != "postgresql":
        return
    conn.execute(text(
        "SELECT set_config('app.department_id', :department_id, true), set_config('app.is_super_admin', :is_super_admin, true), "
        "set_config('app.auth_email', :auth_email, true), set_config('app.user_id', :user_id, true), "
        "set_config('app.auth_phone', :auth_phone, true)"), scope)


@event.listens_for(Session, "after_begin")
def _reapply_scope(session, _transaction, connection):
    # set_config(..., true) lasts one transaction; re-apply it on every new one,
    # so a commit can't silently drop the scope.
    _apply_scope(session, connection)


def set_department_scope(db: Session, department_id: str | None, is_super_admin: bool = False) -> None:
    """Scopes this session to one department for RLS (PostgreSQL only).

    Call only with values from the authenticated user record. The scope stays
    for the session's lifetime (re-applied after every commit)."""
    db.info["rls_scope"] = {"department_id": department_id or "", "is_super_admin": "true" if is_super_admin else "false",
                            "auth_email": "", "user_id": "", "auth_phone": ""}
    if db.in_transaction():
        _apply_scope(db)


def set_auth_lookup(db: Session, *, email: str | None = None, user_id: str | None = None,
                    phone: str | None = None) -> None:
    """Before the department is known (login, OTP, loading a session): lets the
    'self' RLS policy show exactly the one staff_users row with this email, phone or id."""
    db.info["rls_scope"] = {"department_id": "", "is_super_admin": "false",
                            "auth_email": (email or "").lower(), "user_id": user_id or "", "auth_phone": phone or ""}
    if db.in_transaction():
        _apply_scope(db)


def check_database() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 - reported as unhealthy, details in the log
        log.exception("database check failed")
        return False


def verify_app_role() -> None:
    """On PostgreSQL the app must connect as a member of staff_app_rw, and never
    as a superuser, a BYPASSRLS role, or the owner (or a member of the owner role)
    of the protected tables - those can bypass or switch off RLS.
    Refuses to start in staging/production."""
    if engine.dialect.name != "postgresql":
        return
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT r.rolsuper OR r.rolbypassrls,
                   EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                           WHERE n.nspname = current_schema()
                             AND c.relname IN ('staff_users', 'verification_requests', 'audit_log')
                             AND pg_has_role(current_user, c.relowner, 'MEMBER')),
                   CASE WHEN EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'staff_app_rw')
                        THEN pg_has_role(current_user, 'staff_app_rw', 'USAGE') ELSE false END
            FROM pg_roles r WHERE r.rolname = current_user""")).one()
    problems = []
    if row[0]:
        problems.append("connects as a superuser or BYPASSRLS role")
    if row[1]:
        problems.append("connects as (a member of) the table owner")
    if not row[2]:
        problems.append("is not a member of staff_app_rw")
    if problems:
        message = "DATABASE_URL " + " and ".join(problems) + " - Row-Level Security would not protect departments."
        if get_settings().is_production:
            raise RuntimeError(message)
        log.warning(message + " (allowed only in development)")
