from datetime import datetime

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdType, UTCDateTime, new_id, utcnow

ROLES = ("super_admin", "dept_admin", "officer", "viewer")
STATUSES = ("pending_email", "pending_approval", "active", "rejected", "suspended")


class StaffUser(Base):
    """A government employee. Belongs to exactly one department (super_admin: none).

    department_id and role are the only source of truth for what a user may
    see - requests never choose a department."""

    __tablename__ = "staff_users"

    id: Mapped[str] = mapped_column(IdType, primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(254), unique=True)  # stored lower-case
    full_name: Mapped[str] = mapped_column(String(200))
    mobile: Mapped[str | None] = mapped_column(String(20), unique=True)  # 10 digits, e.g. 9876543210
    designation: Mapped[str] = mapped_column(String(200))
    employee_id: Mapped[str | None] = mapped_column(String(60))
    department_id: Mapped[str | None] = mapped_column(IdType, ForeignKey("departments.id"), index=True)
    role: Mapped[str] = mapped_column(String(20), default="viewer")
    status: Mapped[str] = mapped_column(String(20), default="pending_email")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    approved_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    approved_by_id: Mapped[str | None] = mapped_column(IdType, ForeignKey("staff_users.id"))
    last_login_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    password_hash: Mapped[str | None] = mapped_column(String(255))  # salted scrypt, never the password
