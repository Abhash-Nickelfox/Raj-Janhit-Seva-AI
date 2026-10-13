from datetime import datetime

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdType, UTCDateTime, new_id, utcnow


class Department(Base):
    """A tenant. Every staff user, document and knowledge base belongs to exactly one."""

    __tablename__ = "departments"

    id: Mapped[str] = mapped_column(IdType, primary_key=True, default=new_id)
    code: Mapped[str] = mapped_column(String(40), unique=True)  # stable short code, e.g. "agriculture"
    name_en: Mapped[str] = mapped_column(String(200))
    name_hi: Mapped[str] = mapped_column(String(200))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
