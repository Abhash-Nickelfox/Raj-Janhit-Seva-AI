from datetime import datetime

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdType, JSONType, UTCDateTime, new_id, utcnow


class AuditLog(Base):
    """Append-only. UPDATE, DELETE and TRUNCATE are refused by database triggers (migration 0001). Order by `at`."""

    __tablename__ = "audit_log"

    # client-generated: no RETURNING needed, so it can be written before a user is identified
    id: Mapped[str] = mapped_column(IdType, primary_key=True, default=new_id)
    at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    actor_user_id: Mapped[str | None] = mapped_column(IdType, index=True)
    department_id: Mapped[str | None] = mapped_column(IdType, index=True)
    action: Mapped[str] = mapped_column(String(80), index=True)
    entity_type: Mapped[str | None] = mapped_column(String(40))
    entity_id: Mapped[str | None] = mapped_column(String(64))
    ip: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict] = mapped_column(JSONType, default=dict)
