"""Development only: create the Healthcare and Agriculture departments and one
test staff login in each. Refuses to run in staging/production.

Run from gov-staff/backend:  .venv/Scripts/python -m scripts.create_dummy_users
Safe to re-run: existing departments/users are updated, not duplicated.
Only salted scrypt password hashes are kept here - never the passwords.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.core.audit import write_audit  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.db import SessionLocal, set_department_scope  # noqa: E402
from app.models import Department, StaffUser  # noqa: E402

DEPARTMENTS = {
    "healthcare": ("Medical & Health", "चिकित्सा एवं स्वास्थ्य"),
    "agriculture": ("Agriculture", "कृषि"),
}
TEST_USERS = [
    # email, department code, full name, designation, password hash (scrypt)
    ("abhi@buildyourtech.com", "healthcare", "Abhi (test)", "Officer",
     "scrypt$16384$8$1$dcnMxeIZe3kZVf82VRbSKg==$0eU2RN18ZONhQKYbGBLWIMGZVd8nEbCX9JbfIeWzCRo="),
    ("ravi@buildyourtech.com", "agriculture", "Ravi (test)", "Officer",
     "scrypt$16384$8$1$1eYpoohZsOwdE1fZ0tXXdQ==$TheLVSsn/QPTTFTz1562dOIiEXzqABXyzCt6yr3xhME="),
]


def main() -> None:
    if get_settings().is_production:
        raise SystemExit("create_dummy_users is for development only.")
    with SessionLocal() as db:
        set_department_scope(db, None, is_super_admin=True)  # seeding is a super-admin action
        depts = {}
        for code, (name_en, name_hi) in DEPARTMENTS.items():
            dept = db.scalar(select(Department).where(Department.code == code))
            if dept is None:
                dept = Department(code=code, name_en=name_en, name_hi=name_hi)
                db.add(dept)
                db.flush()
            depts[code] = dept
        for email, code, name, designation, password_hash in TEST_USERS:
            user = db.scalar(select(StaffUser).where(StaffUser.email == email))
            if user is None:
                user = StaffUser(email=email, full_name=name, designation=designation)
                db.add(user)
            user.department_id = depts[code].id
            user.role, user.status = "officer", "active"
            user.approved_at = datetime.now(timezone.utc)
            user.password_hash = password_hash
            db.flush()
            write_audit(db, "user.test_account_set", department_id=user.department_id,
                        entity_type="staff_user", entity_id=user.id, details={"email": email, "department": code})
            print(f"ok  {email:28s} -> {code}")
        db.commit()


if __name__ == "__main__":
    main()
