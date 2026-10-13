from datetime import datetime

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdType, UTCDateTime, new_id, utcnow


class OtpCode(Base):
    """One-time email code (P1). Only a hash is stored; short-lived, attempt-limited."""

    __tablename__ = "otp_codes"

    id: Mapped[str] = mapped_column(IdType, primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(254), index=True)
    purpose: Mapped[str] = mapped_column(String(20))  # register | login
    code_hash: Mapped[str] = mapped_column(String(128))
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)
    consumed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
