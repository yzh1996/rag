"""持久化文档入库任务服务。

任务进入持久化数据库队列，可由本地线程或独立worker执行。
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from config.settings import settings
from core.auth import AuthUser
from core.database import init_db, session_scope
from core.db_models import Document, DocumentVersion, IngestionJob, UploadTask
from core.document_loader import DocumentLoader
from core.vector_store import get_vector_store_manager
from core.knowledge_scope import ensure_knowledge_base
from core.metrics import increment

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def file_sha256(file_path: str) -> str:
    """计算文件内容的指纹，用来发现重复上传或预览后被改动的文件。"""
    digest = hashlib.sha256()
    with open(file_path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _permission_scope(user: AuthUser, scope: dict[str, Any] | None) -> dict[str, Any]:
    if scope:
        return scope
    if not user.authenticated:
        return {}
    return {"tenant_ids": [user.tenant_id], "users": [user.username]}


def create_document_version(
    file_path: str,
    *,
    original_filename: str,
    category: str,
    user: AuthUser,
    chunking_strategy: str = "recursive",
    chunking_params: dict[str, Any] | None = None,
    permission_scope: dict[str, Any] | None = None,
    initial_status: str = "pending",
) -> tuple[Document, DocumentVersion]:
    """登记文档和版本，并对内容 Hash 做重复提交保护。

    Document 表示一份逻辑文档，DocumentVersion 表示这份文档的某个具体版本；
    分开保存后，更新文件时可以保留旧版本用于追溯。
    """
    init_db()
    content_hash = file_sha256(file_path)
    with session_scope() as db:
        reserved = db.scalar(select(UploadTask.id).where(UploadTask.tenant_id == user.tenant_id,
            UploadTask.content_hash == content_hash, UploadTask.reservation_key.is_not(None)))
        if reserved:
            raise ValueError("文档内容已存在上传记录，请在上传处理页面继续处理")
        duplicate = db.scalar(
            select(DocumentVersion)
            .join(Document).where(Document.tenant_id == user.tenant_id)
            .where(DocumentVersion.content_hash == content_hash)
            .where(DocumentVersion.status.in_(["published", "candidate", "processing", "pending", "preview"]))
        )
        if duplicate:
            raise ValueError("文档内容已存在，本次未重复处理")

        document_scope = _permission_scope(user, permission_scope)
        knowledge_base = ensure_knowledge_base(db, user.tenant_id, category, document_scope)
        document = Document(
            source_uri=str(Path(file_path).resolve()),
            original_filename=original_filename,
            category=category,
            owner_id=user.user_id,
            tenant_id=user.tenant_id,
            permission_scope=document_scope,
            knowledge_base_id=knowledge_base.id,
            status=initial_status,
        )
        db.add(document)
        db.flush()
        version = DocumentVersion(
            document_id=document.id,
            version=1,
            content_hash=content_hash,
            file_path=str(Path(file_path).resolve()),
            chunking_strategy=chunking_strategy,
            chunking_params=chunking_params or {},
            created_by=user.username,
            status=initial_status,
        )
        db.add(version)
        db.flush()
        return document, version


def create_ingestion_job(document_id: str, version_id: str, job_type: str = "ingest") -> IngestionJob:
    """创建持久化入库任务并放入后台队列，返回任务记录供页面查询进度。"""
    init_db()
    with session_scope() as db:
        version = db.scalar(select(DocumentVersion).where(DocumentVersion.id == version_id,
            DocumentVersion.document_id == document_id).with_for_update())
        if not version:
            raise ValueError("文档版本不存在")
        job = db.scalar(select(IngestionJob).where(IngestionJob.version_id == version_id))
        if job:
            return job
        job = IngestionJob(document_id=document_id, version_id=version_id, job_type=job_type)
        version.status = "pending"
        db.add(job)
        db.flush()
        job_id = job.id
    from core.task_queue import enqueue_ingestion
    enqueue_ingestion(job_id)
    with session_scope() as db:
        return db.get(IngestionJob, job_id)


def create_document_update(
    document_id: str,
    file_path: str,
    *,
    original_filename: str,
    category: str,
    user: AuthUser,
    chunking_strategy: str = "recursive",
    chunking_params: dict[str, Any] | None = None,
) -> tuple[Document, DocumentVersion]:
    """为已有文档登记新版本，旧版本仍保留用于追溯。"""
    content_hash = file_sha256(file_path)
    with session_scope() as db:
        document = db.get(Document, document_id)
        if not document:
            raise ValueError("文档不存在")
        if document.tenant_id != user.tenant_id:
            raise PermissionError("没有修改该文档的权限")
        duplicate = db.scalar(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .where(DocumentVersion.content_hash == content_hash)
        )
        if duplicate:
            raise ValueError("文档内容没有变化，无需创建新版本")
        latest = db.scalar(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version.desc())
        )
        version = DocumentVersion(
            document_id=document.id,
            version=(latest.version if latest else 0) + 1,
            content_hash=content_hash,
            file_path=str(Path(file_path).resolve()),
            chunking_strategy=chunking_strategy,
            chunking_params=chunking_params or {},
            status="pending",
            created_by=user.username,
        )
        document.original_filename = original_filename
        document.category = category
        document.knowledge_base_id = ensure_knowledge_base(
            db, document.tenant_id, category, document.permission_scope or {}
        ).id
        if document.current_version == 0:
            document.status = "pending"
        db.add(version)
        db.flush()
        return document, version


def enqueue_document(
    file_path: str,
    *,
    original_filename: str,
    category: str,
    user: AuthUser,
    chunking_strategy: str = "recursive",
    chunking_params: dict[str, Any] | None = None,
    permission_scope: dict[str, Any] | None = None,
) -> tuple[Document, DocumentVersion, IngestionJob]:
    """便捷入口：登记文档版本，然后立即创建入库任务。"""
    document, version = create_document_version(
        file_path,
        original_filename=original_filename,
        category=category,
        user=user,
        chunking_strategy=chunking_strategy,
        chunking_params=chunking_params,
        permission_scope=permission_scope,
    )
    job = create_ingestion_job(document.id, version.id)
    return document, version, job


def process_ingestion_job(job_id: str) -> None:
    """执行一个文档入库任务。

    流程是：读取任务和文档信息 -> 重新解析或使用预览分块 -> 写入向量库 -> 更新任务状态。
    每个任务状态都会保存到数据库，因此服务重启后可以识别并恢复未完成任务。
    """
    init_db()
    with session_scope() as db:
        job = db.get(IngestionJob, job_id)
        if not job:
            return
        if job.status == "completed":
            return
        job.status = "running"
        job.started_at = _now()
        job.error_message = None
        version = db.get(DocumentVersion, job.version_id)
        document = db.get(Document, job.document_id)
        if not version or not document:
            job.status = "failed"
            job.error_message = "文档或版本记录不存在"
            job.finished_at = _now()
            return
        if document.status == "deprecated":
            job.status, job.error_message, job.finished_at = "failed", "文档已下线，入库已停止", _now()
            return
        version.status = "processing"
        if document.current_version == 0:
            document.status = "processing"
        file_path = version.file_path
        metadata = {
            "document_id": document.id,
            "knowledge_base_id": document.knowledge_base_id,
            "version_id": version.id,
            "document_version": version.version,
            "permission_scope": json.dumps(document.permission_scope or {}, ensure_ascii=False),
            "category": document.category,
            "original_filename": document.original_filename,
            "content_hash": version.content_hash,
        }
        metadata.update(document.upload_metadata or {})
        metadata["tenant_id"] = document.tenant_id
        prepared_chunks = version.prepared_chunks
        strategy = version.chunking_strategy
        params = version.chunking_params or {}

    # 处理文件和写向量库可能失败，因此与数据库状态更新分开处理。
    # 失败时记录错误并标记 failed，成功后再统一标记为 candidate 或 published。
    try:
        if file_sha256(file_path) != metadata["content_hash"]:
            raise ValueError("原文件在预览后已变化，请重新上传")
        if prepared_chunks is not None:
            from core.upload_service import verify_upload_ingestion
            verify_upload_ingestion(job.version_id)
            from langchain_core.documents import Document as Chunk
            documents = [Chunk(page_content=item["content"], metadata={**item["metadata"], **metadata})
                for item in prepared_chunks]
        else:
            documents = DocumentLoader().load_file(file_path, metadata, strategy, params)
        if not documents:
            raise ValueError("文件没有可用内容")
        added_count = get_vector_store_manager().add_documents(documents)
        if added_count != len(documents):
            raise RuntimeError(f"向量入库数量不完整：{added_count}/{len(documents)}")
    except Exception as exc:
        increment("ingestion_failed_total")
        logger.exception("入库任务失败: %s", job_id)
        with session_scope() as db:
            job = db.get(IngestionJob, job_id)
            if not job:
                return
            job.status = "failed"
            job.retry_count += 1
            job.error_message = str(exc)
            job.finished_at = _now()
            version = db.get(DocumentVersion, job.version_id)
            document = db.get(Document, job.document_id)
            if version:
                version.status = "failed"
            if document and document.current_version == 0:
                document.status = "failed"
        return

    with session_scope() as db:
        job = db.get(IngestionJob, job_id)
        version = db.get(DocumentVersion, job.version_id) if job else None
        document = db.get(Document, job.document_id) if job else None
        if not job or not version or not document:
            return
        job.status = "completed"
        job.finished_at = _now()
        if settings.KNOWLEDGE_REVIEW_REQUIRED:
            version.status = "candidate"
            if document.current_version == 0 and document.status != "deprecated":
                document.status = "candidate"
        else:
            if document.status == "deprecated":
                version.status = "candidate"
                return
            version.status = "published"
            version.effective_at = _now()
            for previous in db.scalars(select(DocumentVersion).where(
                DocumentVersion.document_id == document.id, DocumentVersion.id != version.id,
                DocumentVersion.status == "published",
            )):
                previous.status = "archived"
                previous.expired_at = _now()
            document.status = "published"
            document.current_version = version.version


def retry_ingestion_job(job_id: str) -> IngestionJob | None:
    init_db()
    with session_scope() as db:
        job = db.get(IngestionJob, job_id)
        if not job:
            return None
        if job.status not in {"failed", "pending"}:
            raise ValueError("只有失败或等待中的任务可以重试")
        if job.retry_count >= settings.INGESTION_MAX_RETRIES:
            raise ValueError("任务已达到最大重试次数")
        job.status = "pending"
        job.error_message = None
        job.finished_at = None
    from core.task_queue import enqueue_ingestion
    enqueue_ingestion(job_id)
    with session_scope() as db:
        return db.get(IngestionJob, job_id)


def recover_pending_jobs() -> int:
    """服务启动时恢复 pending/running 任务。"""
    init_db()
    # 补偿登记版本后、创建任务前进程退出留下的待入库版本；预览不会自动确认。
    with session_scope() as db:
        orphaned = db.scalars(select(DocumentVersion.id).where(DocumentVersion.status == "pending",
            ~select(IngestionJob.id).where(IngestionJob.version_id == DocumentVersion.id).exists())).all()
    for version_id in orphaned:
        with session_scope() as db:
            version = db.get(DocumentVersion, version_id)
        create_ingestion_job(version.document_id, version.id)
    with session_scope() as db:
        jobs = list(
            db.scalars(
                select(IngestionJob).where(IngestionJob.status.in_(["pending", "running"]))
            ).all()
        )
        job_ids = [job.id for job in jobs]
    for job_id in job_ids:
        from core.task_queue import enqueue_ingestion
        enqueue_ingestion(job_id)
    return len(job_ids)
