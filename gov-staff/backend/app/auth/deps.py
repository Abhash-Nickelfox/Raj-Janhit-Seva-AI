"""The signed-in staff member (StaffActor). Department and role come only from the
database record - never from the request."""

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.sessions import COOKIE, read_token
from app.core.db import get_db, set_auth_lookup, set_department_scope
from app.models import Department, StaffUser


@dataclass
class StaffActor:
    user: StaffUser
    department: Department | None

    @property
    def department_id(self) -> str | None:
        return self.user.department_id


def require_fetch_header(request: Request) -> None:
    """CSRF guard for state-changing calls: browsers can't add this header cross-site without CORS approval."""
    if request.headers.get("X-Requested-With") != "fetch":
        raise HTTPException(403, detail={"code": "csrf", "message": "Missing request header."})


def current_actor(request: Request, db: Session = Depends(get_db)) -> StaffActor:
    user_id = read_token(request.cookies.get(COOKIE))
    if not user_id:
        raise HTTPException(401, detail={"code": "not_signed_in", "message": "Please sign in."})
    set_auth_lookup(db, user_id=user_id)
    user = db.scalar(select(StaffUser).where(StaffUser.id == user_id))
    if user is None or user.status != "active":
        raise HTTPException(401, detail={"code": "not_signed_in", "message": "Please sign in."})
    # from here on, everything this request reads is limited to the user's own department
    set_department_scope(db, user.department_id, is_super_admin=user.role == "super_admin")
    department = db.get(Department, user.department_id) if user.department_id else None
    return StaffActor(user=user, department=department)
