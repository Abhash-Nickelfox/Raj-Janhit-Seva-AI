"""Staff chat: POST /api/ask.

Signed-in staff only; the department comes from the session, never the request.
Document Q&A (search the department's own documents, answer with references)
arrives in P2/P3 - until then the reply says so, in the user's language.
Every question is audited (PLAN 12.7)."""

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.auth.deps import StaffActor, current_actor, require_fetch_header
from app.core.audit import write_audit
from app.core.db import get_db

router = APIRouter(prefix="/api", tags=["ask"])


class AskIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)
    preferred_lang: str | None = Field(default=None, pattern="^(english|hindi|hinglish)$")


def _not_ready_reply(lang: str | None, dept_en: str, dept_hi: str) -> str:
    if lang == "hindi":
        return (f"अभी **{dept_hi}** विभाग के दस्तावेज़ों से उत्तर देने की सुविधा तैयार हो रही है।\n\n"
                "जल्द ही आप अपने विभाग के दस्तावेज़ अपलोड करके उनसे प्रश्न पूछ सकेंगे, "
                "और उत्तर सटीक दस्तावेज़ व पृष्ठ के संदर्भ के साथ मिलेगा।")
    if lang == "hinglish":
        return (f"Abhi **{dept_en}** department ke documents se jawab dene ki suvidha taiyaar ho rahi hai.\n\n"
                "Jald hi aap apne department ke documents upload karke unse sawaal pooch sakenge, "
                "aur jawab document aur page ke reference ke saath milega.")
    return (f"Answering from the **{dept_en}** department's documents is being set up.\n\n"
            "Soon you'll be able to upload your department's documents and ask questions about them - "
            "every answer will cite the exact document and page.")


@router.post("/ask", dependencies=[Depends(require_fetch_header)])
def ask(data: AskIn, request: Request, actor: StaffActor = Depends(current_actor), db: Session = Depends(get_db)):
    dept = actor.department
    write_audit(db, "qa.ask", actor_id=actor.user.id, department_id=actor.department_id, entity_type="question",
                ip=request.client.host if request.client else None,
                details={"question": data.message.strip()[:500], "session_id": data.session_id, "answered": False})
    db.commit()
    reply = _not_ready_reply(data.preferred_lang, dept.name_en if dept else "your", dept.name_hi if dept else "आपके")
    return {"session_id": data.session_id, "reply": reply, "references": []}
