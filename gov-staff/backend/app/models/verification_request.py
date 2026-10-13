from datetime import datetime

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdType, UTCDateTime, new_id, utcnow


class VerificationRequest(Base):
    """Government-affiliation check (P1): a Department Admin (or the Super Admin)
    approves or rejects a registration. One open request per user."""

    __tablename__ = "verification_requests"

    id: Mapped[str] = mapped_column(IdType, primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(IdType, ForeignKey("staff_users.id"), index=True)
    department_id: Mapped[str] = mapped_column(IdType, ForeignKey("departments.id"), index=True)
    reason: Mapped[str | None] = mapped_column(Text)  # the applicant's note, e.g. why the email domain differs
    status: Mapped[str] = mapped_column(String(20), default="open")  # open | approved | rejected
    decision_note: Mapped[str | None] = mapped_column(Text)
    decided_by_id: Mapped[str | None] = mapped_column(IdType, ForeignKey("staff_users.id"))
    decided_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
