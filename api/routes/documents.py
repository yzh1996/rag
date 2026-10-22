"""文档生命周期和入库任务接口。"""
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.dependencies import get_db_dep
from config.settings import settings
from core.auth import AuthUser, can_access_document, get_current_user, require_authenticated, require_admin
from core.file_safety import save_upload
from core.knowledge_scope import parse_permission_scope
from core.db_models import Document, DocumentVersion, IngestionJob, KnowledgeBase, BackgroundJob
from core.ingestion_service import (
    create_document_update,
    create_ingestion_job,
    retry_ingestion_job,
)

router = APIRouter(prefix="/documents", tags=["文档生命周期"])
jobs_router = APIRouter(prefix="/jobs", tags=["入库任务"])


class DocumentStatusRequest(BaseModel):
    permission_scope: dict[str, Any] = Field(default_factory=dict)


@router.patch("/{document_id}/permissions")
async def update_document_permissions(document_id: str, request: DocumentStatusRequest,
    db: Session = Depends(get_db_dep), user: AuthUser = Depends(require_admin)):
    document = db.get(Document, document_id)
    if not document or document.tenant_id != user.tenant_id:
        raise HTTPException(404, "文档不存在")
    try:
        scope = parse_permission_scope(request.permission_scope)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    # 保留原知识分组以避免修改已冻结索引；收紧权限由查询前文档过滤立即执行。
    knowledge_base = db.get(KnowledgeBase, document.knowledge_base_id)
    group_scope = knowledge_base.permission_scope if knowledge_base else {}
    for field in ("tenant_ids", "users", "roles", "departments"):
        if group_scope.get(field) and (scope.get("public") or not scope.get(field) or not set(scope[field]).issubset(group_scope[field])):
            raise HTTPException(409, "文档权限只能在原知识分组范围内收紧；扩大范围请重新上传并评测发布")
    document.permission_scope = scope
    db.commit()
    return _document_payload(document)


def _document_payload(document: Document) -> dict[str, Any]:
    return {
        "id": document.id,
        "source_uri": document.source_uri,
        "original_filename": document.original_filename,
        "category": document.category,
        "knowledge_base_id": document.knowledge_base_id,
        "owner_id": document.owner_id,
        "tenant_id": document.tenant_id,
        "permission_scope": document.permission_scope,
        "upload_metadata": document.upload_metadata or {},
        "status": document.status,
        "current_version": document.current_version,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
    }


def _require_document_write(db, document, user):
    if not document or document.tenant_id != user.tenant_id:
        raise HTTPException(404, "文档不存在")
    if user.role in {"admin", "knowledge_admin"}:
        return
    owned = document.owner_id is not None and document.owner_id == user.user_id
    authored = db.scalar(select(DocumentVersion.id).where(DocumentVersion.document_id == document.id,
        DocumentVersion.created_by == user.username).limit(1))
    if not owned and not authored:
        raise HTTPException(403, "只有上传者或管理员可以修改文档")


@router.get("")
async def list_documents(
    db: Session = Depends(get_db_dep),
    user: AuthUser = Depends(get_current_user),
):
    documents = list(db.scalars(select(Document).where(Document.tenant_id == user.tenant_id).order_by(Document.updated_at.desc())).all())
    return {"items": [_document_payload(item) for item in documents if user.role in {"admin", "knowledge_admin"} or can_access_document(item.permission_scope, user)]}


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    db: Session = Depends(get_db_dep),
    user: AuthUser = Depends(get_current_user),
):
    document = db.get(Document, document_id)
    if not document or document.tenant_id != user.tenant_id or (user.role not in {"admin", "knowledge_admin"} and not can_access_document(document.permission_scope, user)):
        raise HTTPException(404, "文档不存在")
    versions = db.scalars(
        select(DocumentVersion)
        .where(DocumentVersion.document_id == document_id)
        .order_by(DocumentVersion.version.desc())
    ).all()
    payload = _document_payload(document)
    payload["versions"] = [
        {
            "id": version.id,
            "version": version.version,
            "content_hash": version.content_hash,
            "status": version.status,
            "chunking_strategy": version.chunking_strategy,
            "chunking_params": version.chunking_params,
            "effective_at": version.effective_at,
            "expired_at": version.expired_at,
            "created_at": version.created_at,
        }
        for version in versions
    ]
    return payload


@router.post("/{document_id}/versions")
async def upload_document_version(
    document_id: str,
    file: UploadFile = File(...),
    category: str = Form("general"),
    chunking_strategy: str = Form("recursive"),
    chunking_params: str = Form("{}"),
    db: Session = Depends(get_db_dep),
    user: AuthUser = Depends(require_authenticated),
):
    document = db.get(Document, document_id)
    _require_document_write(db, document, user)
    try:
        params = json.loads(chunking_params or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "分块参数格式错误") from exc
    if not isinstance(params, dict):
        raise HTTPException(400, "分块参数必须是对象")
    file_id = uuid.uuid4().hex[:8]
    filename = Path(file.filename or "uploaded_file").name
    save_path = settings.UPLOAD_DIR / f"{file_id}_{filename}"
    try:
        save_upload(file, save_path)
        updated_document, version = create_document_update(
            document_id,
            str(save_path),
            original_filename=filename,
            category=category,
            user=user,
            chunking_strategy=chunking_strategy,
            chunking_params=params,
        )
        job = create_ingestion_job(updated_document.id, version.id, "version_update")
    except PermissionError as exc:
        save_path.unlink(missing_ok=True)
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        save_path.unlink(missing_ok=True)
        raise HTTPException(409, str(exc)) from exc
    except Exception:
        save_path.unlink(missing_ok=True)
        raise
    return {"document_id": updated_document.id, "version_id": version.id, "job_id": job.id, "status": job.status}


@router.post("/{document_id}/deprecate")
async def deprecate_document(
    document_id: str,
    db: Session = Depends(get_db_dep),
    user: AuthUser = Depends(require_authenticated),
):
    document = db.get(Document, document_id)
    _require_document_write(db, document, user)
    document.status = "deprecated"
    for version in document.versions:
        if version.status == "published":
            version.status = "deprecated"
            version.expired_at = datetime.now(timezone.utc)
    db.commit()
    return {"id": document.id, "status": document.status}


@router.post("/{document_id}/restore")
async def restore_document(
    document_id: str,
    db: Session = Depends(get_db_dep),
    user: AuthUser = Depends(require_authenticated),
):
    document = db.get(Document, document_id)
    _require_document_write(db, document, user)
    if document.status != "deprecated":
        raise HTTPException(409, "只能恢复已下线文档")
    latest = db.scalar(
        select(DocumentVersion)
        .where(DocumentVersion.document_id == document_id, DocumentVersion.version == document.current_version,
            DocumentVersion.status.in_(["deprecated", "archived", "published"]))
        .order_by(DocumentVersion.version.desc())
    )
    if not latest:
        raise HTTPException(409, "文档没有可恢复的版本")
    document.status = "published"
    document.current_version = latest.version
    for previous in document.versions:
        if previous.id != latest.id and previous.status == "published":
            previous.status = "archived"
    latest.status = "published"
    latest.expired_at = None
    latest.effective_at = datetime.now(timezone.utc)
    db.commit()
    return {"id": document.id, "status": document.status, "version": latest.version}


@jobs_router.get("")
async def list_jobs(
    limit: int = 50,
    db: Session = Depends(get_db_dep),
    user: AuthUser = Depends(get_current_user),
):
    jobs = db.scalars(select(IngestionJob).order_by(IngestionJob.created_at.desc()).limit(max(1, min(limit, 200)))).all()
    visible = []
    for job in jobs:
        document = db.get(Document, job.document_id)
        if document and document.tenant_id == user.tenant_id and (user.role in {"admin", "knowledge_admin"} or can_access_document(document.permission_scope, user)):
            visible.append(job)
    return {"items": [_job_payload(job) for job in visible]}


def _can_view_task(db, job, user):
    if job.kind == "ingest":
        ingestion = db.get(IngestionJob, job.payload.get("job_id"))
        document = db.get(Document, ingestion.document_id) if ingestion else None
        return document and document.tenant_id == user.tenant_id and (user.role in {"admin", "knowledge_admin"} or can_access_document(document.permission_scope, user))
    actor = job.payload.get("user") or {}
    return actor.get("tenant_id") == user.tenant_id and (actor.get("username") == user.username or user.role in {"admin", "knowledge_admin"})


@jobs_router.get("/queue/list")
async def list_background_tasks(db: Session = Depends(get_db_dep), user: AuthUser = Depends(get_current_user)):
    tasks = db.scalars(select(BackgroundJob).order_by(BackgroundJob.created_at.desc()).limit(200)).all()
    return {"items": [{"id": job.id, "kind": job.kind, "status": job.status, "attempts": job.attempts,
        "error_message": job.error_message, "available_at": job.available_at, "updated_at": job.updated_at}
        for job in tasks if _can_view_task(db, job, user)]}


@jobs_router.post("/queue/{task_id}/retry")
async def retry_background_task(task_id: str, db: Session = Depends(get_db_dep), user: AuthUser = Depends(require_authenticated)):
    job = db.get(BackgroundJob, task_id)
    if not job or not _can_view_task(db, job, user):
        raise HTTPException(404, "任务不存在")
    if job.status != "failed":
        raise HTTPException(409, "只能手动重试已失败的任务")
    from core.task_queue import enqueue
    enqueue(job.kind, job.task_key, job.payload)
    return {"id": job.id, "status": "pending"}


def _job_payload(job: IngestionJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "document_id": job.document_id,
        "version_id": job.version_id,
        "job_type": job.job_type,
        "status": job.status,
        "retry_count": job.retry_count,
        "error_message": job.error_message,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "created_at": job.created_at,
    }


@jobs_router.get("/{job_id}")
async def get_job(
    job_id: str,
    db: Session = Depends(get_db_dep),
    user: AuthUser = Depends(get_current_user),
):
    job = db.get(IngestionJob, job_id)
    document = db.get(Document, job.document_id) if job else None
    if not job or not document or document.tenant_id != user.tenant_id or (user.role not in {"admin", "knowledge_admin"} and not can_access_document(document.permission_scope, user)):
        raise HTTPException(404, "入库任务不存在")
    return _job_payload(job)


@jobs_router.post("/{job_id}/retry")
async def retry_job(
    job_id: str,
    db: Session = Depends(get_db_dep),
    user: AuthUser = Depends(require_authenticated),
):
    job = db.get(IngestionJob, job_id)
    document = db.get(Document, job.document_id) if job else None
    if not job or not document or document.tenant_id != user.tenant_id:
        raise HTTPException(404, "入库任务不存在")
    _require_document_write(db, document, user)
    try:
        retried = retry_ingestion_job(job_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _job_payload(retried)
