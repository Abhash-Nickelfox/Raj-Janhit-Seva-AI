from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.db import check_database

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health():
    db_ok = check_database()
    body = {"status": "ok" if db_ok else "degraded", "database": "ok" if db_ok else "unavailable"}
    if not get_settings().is_production:
        body["environment"] = get_settings().environment
    return JSONResponse(body, status_code=200 if db_ok else 503)
