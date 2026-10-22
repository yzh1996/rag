"""Feedback triage, human handoff and reviewed test-case promotion."""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth import AuthUser, require_admin_or_local, get_current_user
from core.database import get_db
from core.db_models import Feedback, QAInteraction
from core.feedback_service import feedback_query, promote_feedback

router = APIRouter(prefix="/feedback", tags=["用户反馈"])


class ReviewRequest(BaseModel):
    status: Literal["open", "pending_handoff", "in_progress", "resolved", "closed"]
    assignee: str | None = Field(default=None, max_length=128)
    resolution_notes: str | None = Field(default=None, max_length=20000)
    corrected_answer: str | None = Field(default=None, max_length=20000)


class PromoteRequest(BaseModel):
    dataset_name: str = Field(default="feedback-regression", min_length=1, max_length=128)
    expected_answer: str = Field(min_length=1, max_length=20000)


def payload(item, interaction):
    return {"id": item.id, "interaction_id": item.interaction_id, "question": interaction.question,
        "answer": interaction.answer, "rating": item.rating, "category": item.category, "comment": item.comment,
        "corrected_answer": item.corrected_answer, "is_resolved": item.is_resolved,
        "handoff_requested": item.handoff_requested, "status": item.status, "assignee": item.assignee,
        "resolution_notes": item.resolution_notes, "promoted_case_id": item.promoted_case_id,
        "created_at": item.created_at, "updated_at": item.updated_at}


def find(db, user, feedback_id):
    item = db.scalar(feedback_query(user).where(Feedback.id == feedback_id).with_for_update())
    if not item:
        raise HTTPException(404, "反馈不存在")
    return item


@router.get("")
async def list_feedback(status: str | None = None, limit: int = 50, db: Session = Depends(get_db), user: AuthUser = Depends(require_admin_or_local)):
    from core.db_models import Feedback

    query = feedback_query(user).order_by(Feedback.created_at.desc())
    if status:
        query = query.where(Feedback.status == status)
    items = db.scalars(query.limit(max(1, min(limit, 200)))).all()
    return {"items": [payload(item, db.get(QAInteraction, item.interaction_id)) for item in items]}


@router.patch("/{feedback_id}")
async def review_feedback(feedback_id: str, request: ReviewRequest, db: Session = Depends(get_db), user: AuthUser = Depends(require_admin_or_local)):
    item = find(db, user, feedback_id)
    for key, value in request.model_dump(exclude_unset=True).items():
        setattr(item, key, value)
    item.is_resolved = request.status == "resolved"
    db.commit()
    return payload(item, db.get(QAInteraction, item.interaction_id))


@router.get("/mine")
async def my_feedback(limit: int = 50, db: Session = Depends(get_db), user: AuthUser = Depends(get_current_user)):
    items = db.scalars(feedback_query(user).where(QAInteraction.username == user.username)
        .order_by(Feedback.updated_at.desc()).limit(max(1, min(limit, 200)))).all()
    return {"items": [payload(item, db.get(QAInteraction, item.interaction_id)) for item in items]}


@router.post("/{feedback_id}/promote")
async def promote(feedback_id: str, request: PromoteRequest, db: Session = Depends(get_db), user: AuthUser = Depends(require_admin_or_local)):
    item = find(db, user, feedback_id)
    try:
        case = promote_feedback(db, item, request.expected_answer, request.dataset_name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"case_id": case.id, "dataset_name": case.dataset_name}
