"""Settings for the Government Staff AI Assistant, read from gov-staff/backend/.env.

This app is fully separate from the citizen app (../../backend): its own .env,
database, files and knowledge bases. Nothing here is shared with it.
"""

from functools import lru_cache
from urllib.parse import urlparse
from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    environment: Literal["development", "test", "staging", "production"] = "development"
    app_name: str = "Raj Janhit Seva AI - Government Staff Assistant"

    # The app connects as a restricted role (Postgres) - never the owner.
    # Migrations use MIGRATION_DATABASE_URL (owner). Development may use SQLite.
    database_url: str = f"sqlite:///{(DATA_DIR / 'staff.db').as_posix()}"
    migration_database_url: str = ""

    # Browser origins allowed to call the API (the staff frontend, port 5180).
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:5180"]

    # Signs session tokens (P1). Must be long and random outside development.
    session_secret: str = ""

    # Official email domains that may register (P1), e.g. rajasthan.gov.in,nic.in
    allowed_email_domains: Annotated[list[str], NoDecode] = []

    # Email OTP delivery (P1). Empty SMTP_HOST in development = codes are logged
    # to the console only, never in staging/production.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""

    # Testing only: every OTP is this code and nothing is sent. Must be empty in
    # staging/production (real email/SMS delivery comes later).
    otp_test_code: str = "1234"

    # Document storage (P2): local disk in development, S3-compatible bucket later.
    storage_backend: Literal["local", "s3"] = "local"
    storage_dir: str = str(DATA_DIR / "files")

    @field_validator("cors_origins", "allowed_email_domains", mode="before")
    @classmethod
    def split_csv(cls, value):
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @field_validator("allowed_email_domains")
    @classmethod
    def normalise_domains(cls, value):
        return [d.lower().lstrip("@") for d in value]

    @model_validator(mode="after")
    def production_guard(self):
        if self.environment in ("staging", "production"):
            problems = []
            if not self.database_url.startswith("postgresql"):
                problems.append("DATABASE_URL must be PostgreSQL")
            secret = self.session_secret
            if len(secret) < 32 or secret.startswith("#") or " " in secret or len(set(secret)) < 12:
                problems.append("SESSION_SECRET must be 32+ random characters (no spaces or comments)")
            if not self.cors_origins or "*" in self.cors_origins:
                problems.append("CORS_ORIGINS must list the staff frontend origin(s), never *")
            elif any((u := urlparse(o)).scheme != "https" or not u.hostname or u.path not in ("", "/") or u.query or u.fragment
                     for o in self.cors_origins):
                problems.append("CORS_ORIGINS must be https origins (scheme + host only)")
            if self.otp_test_code:
                problems.append("OTP_TEST_CODE must be empty (test codes are for development only)")
            if not self.allowed_email_domains:
                problems.append("ALLOWED_EMAIL_DOMAINS must be set")
            if problems:
                raise ValueError("Unsafe settings for " + self.environment + ": " + "; ".join(problems))
        return self

    @property
    def is_production(self) -> bool:
        return self.environment in ("staging", "production")

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def show_docs(self) -> bool:
        return not self.is_production


@lru_cache
def get_settings() -> Settings:
    return Settings()
