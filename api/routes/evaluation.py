"""测试集和评测运行管理接口。"""
from typing import Any
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.database import get_db
from core.db_models import EvalCase
from core.phase1_services import (
    create_eval_case,
    create_eval_cases,
    create_eval_run,
    get_eval_run,
    list_eval_datasets,
    list_eval_cases,
    list_eval_results,
    list_eval_runs,
    set_eval_case_active,
)
from core.evaluation_service import execute_eval_run
from core.evaluation_import import parse_eval_cases
from core.auth import AuthUser, get_current_user, require_admin_or_local, actor_payload

router = APIRouter(prefix="/evaluation", tags=["RAG评测"])


def _run_progress(run) -> dict[str, Any]:
    progress = (run.config or {}).get("progress") or {}
    total_cases = progress.get("total_cases", run.total_cases or 0)
    completed_cases = progress.get("completed_cases")
    if completed_cases is None:
        completed_cases = total_cases if run.status == "completed" else 0
    return {
        "completed_cases": completed_cases,
        "total_cases": total_cases,
        "phase": progress.get("phase") or ("completed" if run.status == "completed" else None),
        "current_case": progress.get("current_case"),
    }


class EvalCaseRequest(BaseModel):
    dataset_name: str = Field(default="default", min_length=1, max_length=128)
    question: str = Field(min_length=1)
    expected_answer: str | None = None
    expected_sources: list[str] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    expected_behavior: Literal["answer", "refuse", "clarify"] = "answer"
    require_citations: bool = False
    access_context: dict[str, Any] = Field(default_factory=dict)


class EvalRunRequest(BaseModel):
    dataset_name: str = "default"
    config: dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=0, ge=0, le=1000)


class EvalCaseStatusRequest(BaseModel):
    active: bool


@router.post("/cases")
async def add_eval_case(request: EvalCaseRequest, db: Session = Depends(get_db), user: AuthUser = Depends(require_admin_or_local)):
    data = request.model_dump()
    data["tenant_id"] = user.tenant_id
    data["dataset_name"] = data["dataset_name"].strip()
    if not data["dataset_name"]:
        raise HTTPException(400, "测试集名称不能为空")
    if any(item.get("role") not in {"user", "human", "assistant", "ai"} or not isinstance(item.get("content"), str) for item in request.history):
        raise HTTPException(400, "历史消息需要有效角色和文字内容")
    if set(request.access_context) - {"username", "department"} or any(not isinstance(value, str) for value in request.access_context.values()):
        raise HTTPException(400, "测试身份只能设置本租户普通用户名和部门")
    case = create_eval_case(db, **data)
    return {"id": case.id, "status": "created"}


@router.post("/cases/import")
async def import_eval_cases(
    file: UploadFile = File(...),
    dataset_name: str = Form("default"),
    db: Session = Depends(get_db),
    user: AuthUser = Depends(require_admin_or_local),
):
    dataset_name = dataset_name.strip()
    if not dataset_name:
        raise HTTPException(400, "测试集名称不能为空")
    if len(dataset_name) > 128:
        raise HTTPException(400, "测试集名称不能超过128个字符")

    content = await file.read()
    try:
        cases = parse_eval_cases(file.filename or "", content, dataset_name)
        for case in cases:
            case["tenant_id"] = user.tenant_id
        imported = create_eval_cases(db, cases)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"测试集导入失败：{exc}") from exc

    return {
        "status": "created",
        "dataset_name": dataset_name,
        "imported_count": len(imported),
    }


@router.get("/cases")
async def get_eval_cases(active_only: bool = True, db: Session = Depends(get_db), user: AuthUser = Depends(get_current_user)):
    cases = list_eval_cases(db, active_only=active_only, tenant_id=user.tenant_id)
    return {
        "items": [
            {
                "id": case.id,
                "dataset_name": case.dataset_name,
                "question": case.question,
                "expected_answer": case.expected_answer,
                "expected_sources": case.expected_sources,
                "history": case.history,
                "category": case.category,
                "tags": case.tags,
                "expected_behavior": case.expected_behavior,
                "require_citations": case.require_citations,
                "access_context": case.access_context,
                "active": case.active,
            }
            for case in cases
        ]
    }


@router.get("/datasets")
async def get_eval_datasets(db: Session = Depends(get_db), user: AuthUser = Depends(get_current_user)):
    return {"items": list_eval_datasets(db, tenant_id=user.tenant_id)}


@router.patch("/cases/{case_id}")
async def update_eval_case_status(
    case_id: str,
    request: EvalCaseStatusRequest,
    db: Session = Depends(get_db),
    user: AuthUser = Depends(require_admin_or_local),
):
    target = db.get(EvalCase, case_id)
    if not target or target.tenant_id != user.tenant_id:
        raise HTTPException(404, "测试题不存在")
    case = set_eval_case_active(db, case_id, request.active)
    if not case:
        raise HTTPException(404, "测试题不存在")
    return {"id": case.id, "active": case.active}


@router.post("/runs")
async def create_evaluation_run(
    request: EvalRunRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: AuthUser = Depends(require_admin_or_local),
):
    try:
        run = create_eval_run(db, request.dataset_name,
            {**request.config, "limit": request.limit, "actor": actor_payload(user)}, tenant_id=user.tenant_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    from core.task_queue import enqueue_evaluation
    enqueue_evaluation(run.id, request.limit, user)
    return {"id": run.id, "dataset_name": run.dataset_name, "status": "pending"}


@router.post("/runs/{run_id}/start")
async def start_evaluation_run(
    run_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: AuthUser = Depends(require_admin_or_local),
):
    run = get_eval_run(db, run_id)
    if not run or run.tenant_id != user.tenant_id:
        raise HTTPException(404, "评测运行不存在")
    if run.status in {"pending", "running"}:
        raise HTTPException(409, "评测已经在运行")
    # 手动再评测创建新记录，旧结果保持可追溯。
    run = create_eval_run(db, run.dataset_name,
        {"index_id": run.config.get("index_id"), "actor": actor_payload(user), "limit": 0},
        tenant_id=user.tenant_id)
    from core.task_queue import enqueue_evaluation
    enqueue_evaluation(run.id, 0, user)
    return {"id": run.id, "status": "pending"}


@router.get("/runs")
async def get_evaluation_runs(limit: int = 50, db: Session = Depends(get_db), user: AuthUser = Depends(get_current_user)):
    runs = list_eval_runs(db, max(1, min(limit, 200)), tenant_id=user.tenant_id)
    return {
        "items": [
            {
                "id": run.id,
                "dataset_name": run.dataset_name,
                "status": run.status,
                "metrics": run.metrics,
                "total_cases": run.total_cases,
                "passed_cases": run.passed_cases,
                "progress": _run_progress(run),
                "created_at": run.created_at,
                "completed_at": run.completed_at,
            }
            for run in runs
        ]
    }


@router.get("/runs/{run_id}")
async def get_evaluation_run(run_id: str, db: Session = Depends(get_db), user: AuthUser = Depends(get_current_user)):
    run = get_eval_run(db, run_id)
    if not run or run.tenant_id != user.tenant_id:
        raise HTTPException(404, "评测运行不存在")
    results = list_eval_results(db, run_id)
    result_case_ids = [item.case_id for item in results]
    cases = {
        case.id: case
        for case in db.query(EvalCase).filter(EvalCase.id.in_(result_case_ids)).all()
    }
    return {
        "id": run.id,
        "dataset_name": run.dataset_name,
        "status": run.status,
        "config": run.config,
        "metrics": run.metrics,
        "total_cases": run.total_cases,
        "passed_cases": run.passed_cases,
        "progress": _run_progress(run),
        "results": [
            {
                "id": result.id,
                "case_id": result.case_id,
                "answer": result.answer,
                "retrieved_sources": result.retrieved_sources,
                "metrics": result.metrics,
                "passed": result.passed,
                "error": result.error,
                "question": (result.metrics.get("case_snapshot") or {}).get("question") or (cases[result.case_id].question if result.case_id in cases else None),
                "expected_answer": (result.metrics.get("case_snapshot") or {}).get("expected_answer"),
            }
            for result in results
        ],
    }
