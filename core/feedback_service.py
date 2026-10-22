"""Tenant-scoped feedback review and reviewed regression-case promotion."""
from fastapi import HTTPException
from sqlalchemy import select

from core.auth import AuthUser
from core.db_models import EvalCase, Feedback, QAInteraction


def assert_interaction_access(interaction: QAInteraction, user: AuthUser):
    if interaction.tenant_id != user.tenant_id:
        raise HTTPException(404, "问答记录不存在")
    if interaction.username != user.username and user.role not in {"admin", "knowledge_admin"}:
        raise HTTPException(403, "只能反馈自己的问答")


def promote_feedback(db, feedback: Feedback, expected_answer: str, dataset_name: str):
    if not expected_answer.strip() or not dataset_name.strip():
        raise ValueError("人工确认的参考答案和测试集名称不能为空")
    if feedback.promoted_case_id:
        return db.get(EvalCase, feedback.promoted_case_id)
    interaction = db.get(QAInteraction, feedback.interaction_id)
    case = EvalCase(tenant_id=interaction.tenant_id, dataset_name=dataset_name.strip(),
        question=interaction.question, expected_answer=expected_answer.strip(),
        history=interaction.history or [],
        category=feedback.category, tags=["user-feedback", "human-reviewed"],
        expected_sources=[source.get("source", "") for source in interaction.sources if source.get("source")])
    db.add(case)
    db.flush()
    feedback.promoted_case_id = case.id
    db.commit()
    return case


def feedback_query(user: AuthUser):
    return select(Feedback).join(QAInteraction).where(QAInteraction.tenant_id == user.tenant_id)
