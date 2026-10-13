"""Public list of active departments - feeds the registration dropdown (P1).
Exposes only names and codes, nothing about members or documents."""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models import Department
from app.schemas.department import DepartmentOut

router = APIRouter(prefix="/api", tags=["departments"])


@router.get("/departments", response_model=list[DepartmentOut])
def list_departments(db: Session = Depends(get_db)):
    return db.scalars(select(Department).where(Department.active.is_(True)).order_by(Department.name_en)).all()
