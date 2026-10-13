"""Append-only audit log. Every security-relevant action writes one row in the
same transaction as the change it records. The table refuses UPDATE and DELETE
(database triggers), so history can't be rewritten."""

from sqlalchemy.orm import Session

from app.models import AuditLog


def write_audit(db: Session, action: str, *, actor_id: str | None = None, department_id: str | None = None,
                entity_type: str | None = None, entity_id: str | None = None, ip: str | None = None,
                details: dict | None = None) -> AuditLog:
    row = AuditLog(action=action, actor_user_id=actor_id, department_id=department_id, entity_type=entity_type,
                   entity_id=entity_id, ip=ip, details=details or {})
    db.add(row)
    return row
