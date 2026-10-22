"""第一阶段的持久化服务。"""
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.db_models import EvalCase, EvalResult, EvalRun, Feedback, QAInteraction


def get_interaction(db: Session, interaction_id: str) -> QAInteraction | None:
    return db.get(QAInteraction, interaction_id)


def save_feedback(
    db: Session,
    interaction_id: str,
    rating: str,
    category: str | None = None,
    comment: str | None = None,
    corrected_answer: str | None = None,
    is_resolved: bool | None = None,
    handoff_requested: bool = False,
) -> Feedback:
    # Serialize feedback edits per interaction in PostgreSQL.
    db.scalar(select(QAInteraction).where(QAInteraction.id == interaction_id).with_for_update())
    existing = db.scalar(
        select(Feedback)
        .where(Feedback.interaction_id == interaction_id)
        .order_by(Feedback.created_at.desc())
    )
    if existing:
        existing.rating = rating
        existing.category = category
        existing.comment = comment
        existing.corrected_answer = corrected_answer
        existing.is_resolved = is_resolved
        existing.handoff_requested = handoff_requested
        existing.status = "pending_handoff" if handoff_requested else "resolved" if is_resolved else "open"
        db.commit()
        db.refresh(existing)
        return existing

    feedback = Feedback(
        interaction_id=interaction_id,
        rating=rating,
        category=category,
        comment=comment,
        corrected_answer=corrected_answer,
        is_resolved=is_resolved,
        handoff_requested=handoff_requested,
        status="pending_handoff" if handoff_requested else "resolved" if is_resolved else "open",
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


def create_eval_case(db: Session, **data: Any) -> EvalCase:
    case = EvalCase(**data)
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


def create_eval_cases(db: Session, cases: list[dict[str, Any]]) -> list[EvalCase]:
    items = [EvalCase(**case) for case in cases]
    db.add_all(items)
    db.commit()
    for item in items:
        db.refresh(item)
    return items


def set_eval_case_active(db: Session, case_id: str, active: bool) -> EvalCase | None:
    case = db.get(EvalCase, case_id)
    if not case:
        return None
    case.active = active
    db.commit()
    db.refresh(case)
    return case


def list_eval_cases(
    db: Session,
    active_only: bool = True,
    dataset_name: str | None = None,
    tenant_id: str | None = None,
) -> list[EvalCase]:
    statement = select(EvalCase).order_by(EvalCase.created_at.asc())
    if active_only:
        statement = statement.where(EvalCase.active.is_(True))
    if dataset_name:
        statement = statement.where(EvalCase.dataset_name == dataset_name)
    if tenant_id is not None:
        statement = statement.where(EvalCase.tenant_id == tenant_id)
    return list(db.scalars(statement).all())


def list_eval_datasets(db: Session, active_only: bool = False, tenant_id: str | None = None) -> list[dict[str, Any]]:
    statement = (
        select(EvalCase.dataset_name, func.count(EvalCase.id))
        .group_by(EvalCase.dataset_name)
        .order_by(EvalCase.dataset_name.asc())
    )
    if active_only:
        statement = statement.where(EvalCase.active.is_(True))
    if tenant_id is not None:
        statement = statement.where(EvalCase.tenant_id == tenant_id)
    return [
        {"name": name or "default", "case_count": count}
        for name, count in db.execute(statement).all()
    ]


def create_eval_run(db: Session, dataset_name: str, config: dict[str, Any], tenant_id: str = "default") -> EvalRun:
    from core.evaluation_contract import freeze_cases, fingerprint, runtime_config
    cases = list_eval_cases(db, dataset_name=dataset_name, tenant_id=tenant_id)
    snapshot = freeze_cases(cases)
    config = dict(config)
    if config.get("index_id"):
        from core.db_models import IndexVersion
        item = db.get(IndexVersion, config["index_id"])
        if not item or item.tenant_id != tenant_id or item.status not in {"candidate", "canary", "published", "archived"}:
            raise ValueError("候选索引不存在")
        config["artifact_hash"] = item.config.get("artifact_hash")
    run = EvalRun(dataset_name=dataset_name, config={**config, "case_snapshot": snapshot,
        "dataset_revision": fingerprint(snapshot), "runtime": runtime_config()}, tenant_id=tenant_id)
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def list_eval_runs(db: Session, limit: int = 50, tenant_id: str | None = None) -> list[EvalRun]:
    statement = select(EvalRun).order_by(EvalRun.created_at.desc()).limit(limit)
    if tenant_id is not None:
        statement = statement.where(EvalRun.tenant_id == tenant_id)
    return list(db.scalars(statement).all())


def get_eval_run(db: Session, run_id: str) -> EvalRun | None:
    return db.get(EvalRun, run_id)


def list_eval_results(db: Session, run_id: str) -> list[EvalResult]:
    statement = (
        select(EvalResult)
        .where(EvalResult.run_id == run_id)
        .order_by(EvalResult.created_at.asc())
    )
    return list(db.scalars(statement).all())
