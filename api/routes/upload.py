"""文件上传和预览接口。"""
import json
import logging
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, Query
from fastapi.params import Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.dependencies import get_vector_store_dep
from config.settings import settings
from core.auth import AuthUser, get_current_user
from core.chunking_strategies import STRATEGY_LABELS
from core.document_loader import DocumentLoader
from core.embedding import get_embedding_model
from core.ingestion_service import create_document_version, create_ingestion_job, enqueue_document
from core.vector_store import VectorStoreManager
from core.file_safety import save_upload
from core.database import session_scope
from core.db_models import Document, DocumentVersion, UploadTask
from sqlalchemy import select

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["文件上传"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx", ".xls"}


def process_file_background(file_path: str, metadata: dict, vector_store: VectorStoreManager):
    """兼容旧调用方；新上传流程使用持久化入库任务。"""
    try:
        docs = DocumentLoader().load_file(file_path, metadata)
        if docs:
            vector_store.add_documents(docs)
    except Exception as exc:
        logger.error("后台处理文件失败: %s, 错误: %s", file_path, exc)


def _require_upload_user(user: AuthUser) -> None:
    if settings.AUTH_REQUIRED and not user.authenticated:
        raise HTTPException(401, "上传文档前请先登录")


@router.post("/file")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form("general"),
    vector_store: VectorStoreManager = Depends(get_vector_store_dep),
    user: AuthUser = Depends(get_current_user),
):
    """上传文件并创建可追踪的入库任务。

    接口先保存原文件并登记数据库记录，再创建后台任务；实际解析和向量化由
    core/ingestion_service.py 执行，避免上传接口承担耗时工作。
    """
    _require_upload_user(user)
    filename = Path(file.filename or "uploaded_file").name
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件格式: {file_ext}")

    unique_id = uuid.uuid4().hex[:8]
    save_path = settings.UPLOAD_DIR / f"{unique_id}_{filename}"
    try:
        save_upload(file, save_path)
        document, version, job = enqueue_document(
            str(save_path),
            original_filename=filename,
            category=category,
            user=user,
        )
    except ValueError as exc:
        save_path.unlink(missing_ok=True)
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        logger.exception("文件入库任务创建失败")
        raise HTTPException(500, f"文件入库任务创建失败：{exc}") from exc

    return JSONResponse(
        {
            "status": job.status,
            "message": f"文件 {filename} 上传成功，已创建持久化入库任务",
            "file_id": unique_id,
            "filename": save_path.name,
            "document_id": document.id,
            "version_id": version.id,
            "job_id": job.id,
        }
    )


@router.post("/file/preview")
async def preview_file(
    file: UploadFile = File(...),
    category: str = Form("general"),
    chunking_strategy: str = Form("recursive"),
    chunking_params: str = Form("{}"),
    user: AuthUser = Depends(get_current_user),
):
    """上传文件并返回分块结果，确认后才进入持久化入库任务。

    预览让用户先检查切分效果；此时文件内容还没有写入向量库。
    用户确认后由 /file/confirm 创建后台入库任务。
    """
    _require_upload_user(user)
    filename = Path(file.filename or "uploaded_file").name
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件格式: {file_ext}")
    if chunking_strategy not in STRATEGY_LABELS:
        raise HTTPException(400, f"不支持的分块方式: {chunking_strategy}")
    try:
        strategy_params = json.loads(chunking_params or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "分块参数格式错误") from exc
    if not isinstance(strategy_params, dict):
        raise HTTPException(400, "分块参数必须是对象")

    unique_id = uuid.uuid4().hex[:8]
    save_path = settings.UPLOAD_DIR / f"{unique_id}_{filename}"
    document = version = None
    try:
        save_upload(file, save_path)
        # 兼容入口也先登记去重，再进行昂贵的解析和分块。
        document, version = create_document_version(
            str(save_path), original_filename=filename, category=category, user=user,
            chunking_strategy=chunking_strategy, chunking_params=strategy_params, initial_status="preview",
        )
        metadata = {"category": category, "original_filename": filename}
        embeddings = get_embedding_model() if chunking_strategy == "semantic" else None
        # 复用正式入库时的解析和分块逻辑，保证用户预览的内容就是后续准备入库的内容。
        original_content, documents = DocumentLoader().load_file_with_original(
            str(save_path), metadata, chunking_strategy, strategy_params, embeddings
        )
        if not documents:
            raise ValueError("文件没有可用内容，无法生成分割预览")
    except ValueError as exc:
        save_path.unlink(missing_ok=True)
        if version:
            with session_scope() as db:
                db.get(DocumentVersion, version.id).status = "failed"
                db.get(Document, document.id).status = "failed"
        raise HTTPException(400 if "没有可用内容" in str(exc) else 409, str(exc)) from exc
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        if version:
            with session_scope() as db:
                db.get(DocumentVersion, version.id).status = "failed"
                db.get(Document, document.id).status = "failed"
        logger.exception("文件预览失败")
        raise HTTPException(500, f"文件预览失败: {exc}") from exc

    unique_id = version.id
    return JSONResponse(
        {
            "status": "preview",
            "file_id": unique_id,
            "filename": filename,
            "document_id": document.id,
            "version_id": version.id,
            "chunking_strategy": chunking_strategy,
            "chunking_params": strategy_params,
            "original_content": original_content,
            "chunk_count": len(documents),
            "chunks": [
                {
                    "index": index,
                    "content": document.page_content,
                    "parent_index": document.metadata.get("parent_index"),
                    "child_index": document.metadata.get("child_index"),
                    "structure_header": document.metadata.get("structure_header"),
                }
                for index, document in enumerate(documents, 1)
            ],
        }
    )


@router.post("/file/confirm")
async def confirm_file(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...),
    vector_store: VectorStoreManager = Depends(get_vector_store_dep),
    user: AuthUser = Depends(get_current_user),
):
    """确认预览结果并创建持久化向量入库任务。"""
    _require_upload_user(user)
    with session_scope() as db:
        version = db.get(DocumentVersion, file_id)
        document = db.get(Document, version.document_id) if version else None
        if not document or document.tenant_id != user.tenant_id or version.created_by != user.username:
            raise HTTPException(404, "预览不存在或不属于当前用户")
        if version.status not in {"preview", "pending", "processing", "candidate", "published"}:
            raise HTTPException(409, "预览状态已变化，请在任务页面查看")
        enterprise_task = db.scalar(select(UploadTask).where(UploadTask.version_id == version.id))
    # 先确认这份预览属于当前用户，再将它交给持久化任务队列。
    try:
        if enterprise_task:
            from core.upload_service import confirm_upload
            job = confirm_upload(enterprise_task.id, user)
        else:
            job = create_ingestion_job(document.id, version.id)
    except (LookupError, ValueError, PermissionError) as exc:
        raise _upload_error(exc) from exc
    except Exception as exc:
        raise HTTPException(500, f"创建入库任务失败：{exc}") from exc
    return JSONResponse(
        {
            "status": job.status,
            "file_id": file_id,
            "document_id": document.id,
            "version_id": version.id,
            "job_id": job.id,
            "message": "已确认，正在后台向量化入库",
        }
    )


@router.post("/batch")
async def upload_batch_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    category: str = Form("general"),
    vector_store: VectorStoreManager = Depends(get_vector_store_dep),
    user: AuthUser = Depends(get_current_user),
):
    """批量上传文件并为每个文件创建独立任务。"""
    _require_upload_user(user)
    results = []
    for file in files:
        filename = Path(file.filename or "uploaded_file").name
        save_path = None
        try:
            file_ext = Path(filename).suffix.lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                results.append({"filename": filename, "status": "skipped", "reason": "不支持的文件格式"})
                continue
            unique_id = uuid.uuid4().hex[:8]
            save_path = settings.UPLOAD_DIR / f"{unique_id}_{filename}"
            save_upload(file, save_path)
            document, version, job = enqueue_document(
                str(save_path),
                original_filename=filename,
                category=category,
                user=user,
            )
            results.append(
                {
                    "filename": filename,
                    "status": job.status,
                    "file_id": unique_id,
                    "document_id": document.id,
                    "version_id": version.id,
                    "job_id": job.id,
                }
            )
        except Exception as exc:
            if save_path:
                save_path.unlink(missing_ok=True)
            results.append({"filename": filename, "status": "error", "reason": str(exc)})
    return JSONResponse({"results": results})


class ChunkRequest(BaseModel):
    strategy: str = "recursive"
    params: dict = Field(default_factory=dict)


def _upload_error(exc):
    if isinstance(exc, LookupError):
        return HTTPException(404, str(exc))
    return HTTPException(403 if isinstance(exc, PermissionError) else 400, str(exc))


def get_upload_actor(
    tenant_id: str | None = Query(None, max_length=128),
    user: AuthUser = Depends(get_current_user),
):
    from core.upload_service import resolve_upload_actor
    if settings.AUTH_REQUIRED and not user.authenticated:
        raise HTTPException(401, "上传资料前请先登录")
    try:
        return resolve_upload_actor(user, tenant_id)
    except (ValueError, PermissionError) as exc:
        raise _upload_error(exc) from exc


@router.get("/options")
def enterprise_upload_options(user: AuthUser = Depends(get_upload_actor)):
    from core.upload_service import upload_options
    try:
        return upload_options(user)
    except PermissionError as exc:
        raise _upload_error(exc) from exc


@router.post("/tasks", status_code=202)
def create_upload_task(
    file: UploadFile = File(...),
    knowledge_base_id: str = Form(...),
    metadata: str = Form("{}"),
    user: AuthUser = Depends(get_upload_actor),
):
    from core.upload_service import submit_upload, task_payload, UploadAdmissionError
    try:
        values = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "上传信息格式不正确，请检查表单后重试") from exc
    try:
        task_id = submit_upload(file, knowledge_base_id, values, user)
        return task_payload(task_id, user, include_content=False)
    except UploadAdmissionError as exc:
        return JSONResponse(exc.payload, status_code=exc.status_code)
    except PermissionError as exc:
        raise _upload_error(exc) from exc


@router.get("/tasks")
def enterprise_upload_tasks(user: AuthUser = Depends(get_upload_actor)):
    from core.upload_service import list_uploads
    try:
        return {"items": list_uploads(user)}
    except PermissionError as exc:
        raise _upload_error(exc) from exc


@router.get("/tasks/{task_id}")
def enterprise_upload_task(task_id: str, user: AuthUser = Depends(get_upload_actor)):
    from core.upload_service import task_payload
    try:
        return task_payload(task_id, user)
    except (LookupError, PermissionError) as exc:
        raise _upload_error(exc) from exc


@router.post("/tasks/{task_id}/chunks", status_code=202)
def enterprise_upload_chunks(task_id: str, request: ChunkRequest, user: AuthUser = Depends(get_upload_actor)):
    from core.upload_service import request_chunks, task_payload
    try:
        request_chunks(task_id, request.strategy, request.params, user)
        return task_payload(task_id, user)
    except (LookupError, ValueError, PermissionError) as exc:
        raise _upload_error(exc) from exc


@router.delete("/tasks/{task_id}")
def enterprise_delete_upload_record(task_id: str, user: AuthUser = Depends(get_upload_actor)):
    from core.upload_service import delete_upload_record
    try:
        return delete_upload_record(task_id, user)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (LookupError, PermissionError) as exc:
        raise _upload_error(exc) from exc


@router.post("/tasks/{task_id}/confirm")
def enterprise_upload_confirm(task_id: str, user: AuthUser = Depends(get_upload_actor)):
    from core.upload_service import confirm_upload
    try:
        job = confirm_upload(task_id, user)
        return {"job_id": job.id, "status": job.status,
            "message": "已确认分块，正在后台向量化；审核发布请进入版本与发布页面"}
    except (LookupError, ValueError, PermissionError) as exc:
        raise _upload_error(exc) from exc


@router.post("/tasks/{task_id}/retry", status_code=202)
def enterprise_upload_retry(task_id: str, user: AuthUser = Depends(get_upload_actor)):
    from core.upload_service import retry_upload
    try:
        retry_upload(task_id, user)
        return {"message": "已安排从失败阶段重试，之前完成的处理结果已保留"}
    except (LookupError, ValueError, PermissionError) as exc:
        raise _upload_error(exc) from exc


@router.post("/tasks/{task_id}/retry-ingestion", status_code=202)
def enterprise_ingestion_retry(task_id: str, user: AuthUser = Depends(get_upload_actor)):
    from core.upload_service import retry_upload_ingestion
    try:
        retry_upload_ingestion(task_id, user)
        return {"message": "已安排重新入库"}
    except (LookupError, ValueError, PermissionError) as exc:
        raise _upload_error(exc) from exc
