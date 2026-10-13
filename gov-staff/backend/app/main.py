"""Government Staff AI Assistant - API (port 8200).

Separate from the citizen app: own process, database, files and knowledge bases.
Run from gov-staff/backend:  .venv/Scripts/python -m uvicorn app.main:app --port 8200
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.db import verify_app_role
from app.routers import ask, auth, departments, health

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    verify_app_role()
    yield


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
    docs_url="/api/docs" if settings.show_docs else None,
    redoc_url=None,
    openapi_url="/api/openapi.json" if settings.show_docs else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,  # session cookie (P1)
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-Requested-With"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cache-Control", "no-store")
    if settings.is_production:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError):
    """422 as {code, fields: {field: message}} so forms can show each message under its field."""
    fields = {}
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", []) if p != "body"]
        field = loc[0] if loc else "_form"
        msg = str(err.get("msg", "Invalid value")).removeprefix("Value error, ")
        fields.setdefault(field, msg)
    return JSONResponse(status_code=422, content={"detail": {"code": "validation", "message": "Please correct the highlighted fields.",
                                                             "fields": fields}})


app.include_router(health.router)
app.include_router(departments.router)
app.include_router(auth.router)
app.include_router(ask.router)
