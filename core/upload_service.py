"""企业上传：公司隔离、准入检查、持久化预处理和确认后分块。"""
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document as Chunk
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from config.settings import settings
from core.auth import AuthUser, actor_payload, can_access_document
from core.database import session_scope
from core.db_models import Document, DocumentVersion, KnowledgeBase, UploadTask, User, IngestionJob, BackgroundJob
from core.file_safety import save_upload
from core.ingestion_service import file_sha256, create_ingestion_job, retry_ingestion_job
from core.knowledge_scope import parse_permission_scope

logger = logging.getLogger(__name__)
DEPARTMENTS = {"finance": "财务部", "customer_service": "客服部"}
ADMIN_ROLES = {"admin", "knowledge_admin"}
LOCAL_ADMIN_SCOPE = "upload:local-admin"
LOCAL_ADMIN_USERNAME = "local_admin"
STAGES = (
    ("permissions", "权限与归属校验"), ("safety", "基础文件安全校验"),
    ("dedup", "完全相同文件去重"), ("parse", "文档解析"),
    ("ocr", "按需OCR识别"), ("clean", "内容清洗"),
    ("chunks", "内容确认与分块"), ("ingest", "向量化入库"),
)
ACTIONS = {
    "permissions": "请调整归属或可见范围，选择有上传权限的知识库；如需授权，请联系管理员。",
    "safety": "请检查文件格式、大小或加密情况，修正后重新上传。",
    "dedup": "无需重复上传；如需调整已有资料的权限或归属，请联系管理员。",
    "parse": "请重新导出文件；若文件正常，请联系管理员检查解析环境后重试。",
    "ocr": "请上传更清晰的文件，或联系管理员检查OCR识别环境后重试。",
    "clean": "请检查提取的文字，文件需要包含可用内容；必要时重新上传。",
    "chunks": "请调整分块参数后重新生成；语义分块还需检查模型服务。",
    "ingest": "请重试入库；若仍失败，请联系管理员检查模型和存储服务。",
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def _local_admin(user):
    return (not settings.AUTH_REQUIRED and not user.authenticated and user.role == "admin"
        and user.username == LOCAL_ADMIN_USERNAME and LOCAL_ADMIN_SCOPE in user.scopes)


def resolve_upload_actor(user, tenant_id=None):
    """免登录时使用上传专用管理员身份；正式身份不能自行跨公司。"""
    if user.authenticated:
        tenant = _text(tenant_id, "公司编号", 128, user.tenant_id) if tenant_id is not None else user.tenant_id
        if tenant != user.tenant_id:
            raise PermissionError("当前登录身份没有切换到该公司的授权")
        return user
    if settings.AUTH_REQUIRED:
        raise PermissionError("上传资料前请先登录")
    tenant = _text(tenant_id, "公司编号", 128, "default") if tenant_id is not None else (
        user.tenant_id if _local_admin(user) else "default")
    return AuthUser(username=LOCAL_ADMIN_USERNAME, role="admin", tenant_id=tenant,
        authenticated=False, scopes={LOCAL_ADMIN_SCOPE})


def _identity(user):
    if not user.authenticated and not _local_admin(user):
        raise PermissionError("企业上传需要登录后才能确定所属公司和部门")


def _active_actor(db, user):
    _identity(user)
    account = db.get(User, user.user_id) if user.user_id else None
    if account:
        if not account.is_active or account.tenant_id != user.tenant_id or account.username != user.username:
            raise PermissionError("账号已停用或公司授权已变化，请重新登录")
        return AuthUser(username=account.username, role=account.role, department=account.department,
            tenant_id=account.tenant_id, user_id=account.id, authenticated=True)
    return user


def _templates(user):
    items = []
    for code, name in DEPARTMENTS.items():
        items.append({"id": hashlib.sha256(f"department:{user.tenant_id}:{code}".encode()).hexdigest(),
            "name": name + "知识库", "tenant_id": user.tenant_id, "owner_department": name,
            "permission_scope": {"tenant_ids": [user.tenant_id], "departments": [name]},
            "upload_scope": {"tenant_ids": [user.tenant_id], "departments": [name]}})
    items.append({"id": hashlib.sha256(f"shared:{user.tenant_id}".encode()).hexdigest(),
        "name": "公司共享知识库", "tenant_id": user.tenant_id, "owner_department": None,
        "permission_scope": {"tenant_ids": [user.tenant_id]},
        "upload_scope": {"tenant_ids": [user.tenant_id], "roles": sorted(ADMIN_ROLES)}})
    return items


def _can_upload(kb, user):
    if kb.tenant_id != user.tenant_id:
        return False
    if user.role in ADMIN_ROLES:
        return True
    # 未配置写入权限的历史知识库不自动赋予普通用户写入权限。
    return bool(kb.upload_scope) and can_access_document(kb.upload_scope, user)


def _kb(db, kb_id, user, create=False):
    user = _active_actor(db, user)
    item = db.get(KnowledgeBase, kb_id)
    if item is None:
        template = next((value for value in _templates(user) if value["id"] == kb_id), None)
        if template:
            item = KnowledgeBase(**template)
            if create:
                try:
                    with db.begin_nested():
                        db.add(item)
                        db.flush()
                except IntegrityError:
                    item = db.get(KnowledgeBase, kb_id)
    if item is None or not _can_upload(item, user):
        raise PermissionError("目标知识库不存在，或你没有向该知识库上传资料的权限")
    return item


def _users(db, user):
    statement = select(User).where(User.tenant_id == user.tenant_id, User.is_active.is_(True))
    if user.role not in ADMIN_ROLES:
        statement = statement.where(User.department == user.department)
    names = {item.username for item in db.scalars(statement)}
    names.add(user.username)
    return sorted(names)


def upload_options(user):
    _identity(user)
    with session_scope() as db:
        user = _active_actor(db, user)
        existing = list(db.scalars(select(KnowledgeBase).where(KnowledgeBase.tenant_id == user.tenant_id)))
        bases = {item.id: item for item in existing}
        for value in _templates(user):
            bases.setdefault(value["id"], KnowledgeBase(**value))
        departments = [name for name in DEPARTMENTS.values()
            if user.role in ADMIN_ROLES or name == user.department]
        return {"tenant_id": user.tenant_id, "username": user.username,
            "tenant_editable": _local_admin(user),
            "departments": departments, "users": _users(db, user),
            "business_owners": _users(db, user) if user.role in ADMIN_ROLES else [user.username],
            "knowledge_bases": [{"id": item.id, "name": item.name,
                "owner_department": item.owner_department,
                "read_departments": (item.permission_scope or {}).get("departments", list(DEPARTMENTS.values()))}
                for item in bases.values() if _can_upload(item, user)],
            "max_upload_mb": settings.MAX_UPLOAD_MB}


def _text(value, label, maximum, default=""):
    if not isinstance(value, str):
        raise ValueError(f"{label}需要填写文字")
    result = value.strip() or default
    if len(result) > maximum:
        raise ValueError(f"{label}不能超过{maximum}个字符")
    return result


def _attributes(db, values, filename, kb, user):
    if not isinstance(values, dict):
        raise ValueError("上传信息格式不正确")
    supplied_tenant = _text(values.get("tenant_id", user.tenant_id), "公司编号", 128, user.tenant_id)
    if supplied_tenant != user.tenant_id:
        raise PermissionError("不能向未经授权的公司上传资料")
    department = values.get("owner_department", user.department)
    if department not in DEPARTMENTS.values():
        raise ValueError("归属部门请选择财务部或客服部")
    if user.role not in ADMIN_ROLES and department != user.department:
        raise PermissionError("你没有代表该部门上传资料的权限")
    if kb.owner_department and department != kb.owner_department:
        raise ValueError("归属部门与目标部门知识库不一致，请重新选择")
    owner = _text(values.get("business_owner", user.username), "业务负责人", 128, user.username)
    allowed_users = _users(db, user)
    if owner not in allowed_users or (user.role not in ADMIN_ROLES and owner != user.username):
        raise PermissionError("你不能将资料负责人设置为该人员")
    scope = parse_permission_scope(kb.permission_scope)
    # 新上传永远属于当前公司；历史public标记不继承到新资料。
    scope.pop("public", None)
    scope["tenant_ids"] = [user.tenant_id]
    visibility = values.get("visibility", "inherit")
    if visibility == "self":
        if scope.get("users") and user.username not in scope["users"] and user.user_id not in scope["users"]:
            raise PermissionError("当前知识库的阅读范围不包含你，请调整可见范围")
        scope["users"] = [user.username]
    elif visibility in {"departments", "users"}:
        field = "departments" if visibility == "departments" else "users"
        chosen = values.get("allowed_" + field, [])
        allowed = list(DEPARTMENTS.values()) if field == "departments" else allowed_users
        if not isinstance(chosen, list) or not chosen or any(value not in allowed for value in chosen):
            raise ValueError("请选择有效的允许部门或人员")
        if scope.get(field) and not set(chosen).issubset(scope[field]):
            raise PermissionError("文档可见范围不能超出目标知识库的允许范围")
        scope[field] = sorted(set(chosen))
    elif visibility != "inherit":
        raise ValueError("可见范围请选择继承知识库、仅本人、指定部门或指定人员")
    department_id = next(code for code, name in DEPARTMENTS.items() if name == department)
    attributes = {"title": _text(values.get("title", ""), "文档标题", 512, filename),
        "category": _text(values.get("category", "general"), "文档分类", 128, "general"),
        "tenant_id": user.tenant_id, "owner_department": department, "department_id": department_id,
        "business_owner": owner, "source_type": "manual_upload",
        "source_reference": _text(values.get("source_reference", ""), "资料来源", 1024),
        "visibility": visibility, "uploaded_by": user.username, "uploaded_at": timestamp()}
    return attributes, scope


def _stage(task, key, status, detail="", stats=None):
    items = [dict(value) for value in task.stages]
    item = next(value for value in items if value["key"] == key)
    item.update(status=status, detail=detail, next_action=ACTIONS[key] if status == "failed" else "")
    if status == "running":
        item["started_at"], item["finished_at"] = timestamp(), None
    elif status in {"passed", "skipped", "failed"}:
        item["finished_at"] = timestamp()
        if item.get("started_at"):
            item["duration_ms"] = round((datetime.fromisoformat(item["finished_at"]) -
                datetime.fromisoformat(item["started_at"])).total_seconds() * 1000)
    if stats is not None:
        item["stats"] = stats
    task.stages, task.updated_at = items, datetime.now(timezone.utc)


class UploadAdmissionError(ValueError):
    """返回上传失败原因，不登记处理记录。"""

    def __init__(self, task, key, message, status_code=400):
        super().__init__(message)
        _stage(task, key, "failed", message)
        self.status_code = status_code
        self.payload = {"id": None, "filename": task.filename, "status": "failed",
            "stages": task.stages, "record_saved": False}


def submit_upload(upload, kb_id, values, user):
    _identity(user)
    filename = Path((upload.filename or "uploaded_file").replace("\\", "/")).name[:512]
    task = UploadTask(id=uuid.uuid4().hex, tenant_id=user.tenant_id, username=user.username,
        filename=filename, knowledge_base_id=kb_id,
        stages=[{"key": key, "label": label, "status": "pending", "detail": ""} for key, label in STAGES])
    key = "permissions"
    path = settings.UPLOAD_DIR / f"{task.id}{Path(filename).suffix.lower()}"
    try:
        _stage(task, key, "running")
        with session_scope() as db:
            user = _active_actor(db, user)
            kb = _kb(db, kb_id, user)
            task.upload_metadata, task.permission_scope = _attributes(db, values, filename, kb, user)
        _stage(task, key, "passed", "公司、归属部门、上传权限及可见范围已校验")
        key = "safety"
        _stage(task, key, "running")
        save_upload(upload, path)
        task.file_path = str(path.resolve())
        file_metadata = {"file_size_bytes": path.stat().st_size, "file_type": path.suffix.lower().lstrip(".")}
        _stage(task, key, "passed", "文件大小、基础格式和适用的解压安全检查通过；不包含病毒扫描")
        key = "dedup"
        _stage(task, key, "running")
        digest = file_sha256(str(path))
        with session_scope() as db:
            # 登记前再次核对身份和归属，检查期间发生的授权变化也会拦截。
            key = "permissions"
            user = _active_actor(db, user)
            kb = _kb(db, kb_id, user, create=True)
            task.upload_metadata, task.permission_scope = _attributes(db, values, filename, kb, user)
            task.upload_metadata = {**task.upload_metadata, **file_metadata}
            task.actor = actor_payload(user)
            key = "dedup"
            duplicate = db.scalar(select(DocumentVersion.id).join(Document)
                .where(Document.tenant_id == user.tenant_id, DocumentVersion.content_hash == digest,
                    DocumentVersion.status.in_(["preview", "pending", "processing", "candidate", "published"])))
            if duplicate:
                raise ValueError("该文件与已有资料完全相同，本次未重复处理")
            task.content_hash = digest
            task.reservation_key = hashlib.sha256(json.dumps([user.tenant_id, digest]).encode()).hexdigest()
            _stage(task, key, "passed", "未发现本公司已有的完全相同文件")
            task.status = "queued"
            db.add(task)
            db.flush()  # 唯一约束阻止并发提交相同内容；失败时整条登记回滚。
    except IntegrityError as exc:
        path.unlink(missing_ok=True)
        raise UploadAdmissionError(task, "dedup", "该文件与已有提交完全相同，本次未重复处理", 409) from exc
    except (ValueError, PermissionError) as exc:
        path.unlink(missing_ok=True)
        raise UploadAdmissionError(task, key, str(exc), 403 if isinstance(exc, PermissionError) else
            409 if key == "dedup" else 400) from exc
    except Exception as exc:
        logger.exception("上传准入失败：%s", task.id)
        path.unlink(missing_ok=True)
        raise UploadAdmissionError(task, key, "处理暂时失败，请修正后重新上传，或联系管理员并提供文件名", 500) from exc
    _enqueue(task.id, 0, "preprocess")
    return task.id


def _enqueue(task_id, generation, phase):
    from core.task_queue import enqueue
    return enqueue("upload_" + phase, f"upload:{task_id}:{phase}:{generation}",
        {"task_id": task_id, "generation": generation})


def _owned(db, task_id, user):
    _identity(user)
    task = db.scalar(select(UploadTask).where(UploadTask.id == task_id,
        UploadTask.tenant_id == user.tenant_id, UploadTask.username == user.username,
        UploadTask.deleted_at.is_(None)).with_for_update())
    if not task:
        raise LookupError("处理记录不存在或不属于当前用户")
    _reconcile_queue(db, task)
    return task


def _current_actor(db, task):
    user = _active_actor(db, AuthUser(**task.actor))
    _kb(db, task.knowledge_base_id, user)
    return user


def _reconcile_queue(db, task):
    if task.status not in {"queued", "processing", "chunking"}:
        return
    phase = "chunks" if task.status == "chunking" else "preprocess"
    queue_job = db.scalar(select(BackgroundJob).where(
        BackgroundJob.task_key == f"upload:{task.id}:{phase}:{task.generation}"))
    if queue_job and queue_job.status == "failed":
        key = next((value["key"] for value in task.stages if value["status"] == "running"),
            "chunks" if phase == "chunks" else "parse")
        _stage(task, key, "failed", "任务中断且自动恢复次数已用完，请检查服务后从失败阶段重试")
        task.status = "failed"


def _require_content_access(db, task, user):
    user = _active_actor(db, user)
    kb = _kb(db, task.knowledge_base_id, user)
    document = db.get(Document, task.document_id) if task.document_id else None
    if not _can_read_upload(kb, document, task.permission_scope, user):
        raise PermissionError("当前阅读权限不允许检查正文，不能确认分块或入库，请联系管理员")


def _can_read_upload(kb, document, scope, user):
    """管理员可审核本公司上传内容，文档面向业务用户的阅读范围仍保留。"""
    if not kb or kb.tenant_id != user.tenant_id or document and document.tenant_id != user.tenant_id:
        return False
    return user.role in ADMIN_ROLES or (
        can_access_document(kb.permission_scope, user)
        and can_access_document(document.permission_scope if document else scope, user))


def _serialize(documents):
    return [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]


def _deserialize(items):
    return [Chunk(page_content=value["content"], metadata=dict(value["metadata"])) for value in items or []]


def _snapshot(task_id, generation):
    with session_scope() as db:
        task = db.get(UploadTask, task_id)
        if not task or task.deleted_at is not None or task.generation != generation:
            return None
        _current_actor(db, task)
        if file_sha256(task.file_path) != task.content_hash:
            raise ValueError("原文件已变化，请重新上传")
        return task


def _checkpoint(task_id, generation, key, state, message="", stats=None, **values):
    with session_scope() as db:
        task = db.get(UploadTask, task_id)
        if not task or task.deleted_at is not None or task.generation != generation:
            return False
        _stage(task, key, state, message, stats)
        if state == "failed" and task.file_path:
            path = Path(task.file_path)
            if not path.exists() or file_sha256(task.file_path) != task.content_hash:
                task.reservation_key = None
                if task.version_id:
                    version = db.get(DocumentVersion, task.version_id)
                    if version and version.status == "preview":
                        version.status = "failed"
        for name, value in values.items():
            setattr(task, name, value)
        return True


def process_preprocessing(task_id, generation):
    from core.document_loader import DocumentLoader
    from core.content_processing import recognize_documents, clean_documents
    key = "parse"
    try:
        task = _snapshot(task_id, generation)
        if task is None or task.status not in {"queued", "processing"}:
            return
        if task.extracted_documents is None:
            _checkpoint(task_id, generation, key, "running", "正在提取文字、页码和扫描页信息", status="processing")
            documents = DocumentLoader().load_raw_file(task.file_path, task.upload_metadata)
            _checkpoint(task_id, generation, key, "passed", f"已解析{len(documents)}个页面或内容单元",
                {"units": len(documents)}, extracted_documents=_serialize(documents))
            task = _snapshot(task_id, generation)
        key = "ocr"
        if task.raw_documents is None:
            documents = _deserialize(task.extracted_documents)
            required = sum(bool(doc.metadata.get("ocr_required")) for doc in documents)
            if required:
                _checkpoint(task_id, generation, key, "running", f"正在识别{required}个扫描页")
                documents = recognize_documents(task.file_path, documents)
            _checkpoint(task_id, generation, key, "passed" if required else "skipped",
                f"已识别{required}个扫描页" if required else "无需OCR，使用已提取的文字",
                {"ocr_pages": required}, raw_documents=_serialize(documents))
            task = _snapshot(task_id, generation)
        key = "clean"
        if task.cleaned_documents is None:
            _checkpoint(task_id, generation, key, "running", "正在整理空白、控制字符和空内容")
            raw = _deserialize(task.raw_documents)
            cleaned = clean_documents(raw)
            if not cleaned or not any(doc.page_content.strip() for doc in cleaned):
                raise ValueError("处理后没有可用文字，无法进入内容确认和分块")
            before = sum(len(value["content"]) for value in task.raw_documents)
            after = sum(len(doc.page_content) for doc in cleaned)
            _checkpoint(task_id, generation, key, "passed" if settings.DOCUMENT_CLEANING_ENABLED else "skipped",
                "清洗完成，请检查处理前后内容" if settings.DOCUMENT_CLEANING_ENABLED else "清洗未启用，已保留可用文字",
                {"before_chars": before, "after_chars": after, "removed_empty_units": len(raw) - len(cleaned)},
                cleaned_documents=_serialize(cleaned))
        with session_scope() as db:
            task = db.get(UploadTask, task_id)
            if task.generation == generation:
                task.status = "ready"
    except Exception as exc:
        logger.exception("文档预处理失败：%s，阶段：%s", task_id, key)
        message = str(exc) if isinstance(exc, (ValueError, PermissionError)) else "处理暂时失败，请联系管理员检查服务后重试"
        _checkpoint(task_id, generation, "permissions" if isinstance(exc, PermissionError) else key,
            "failed", message, status="failed")


def request_chunks(task_id, strategy, params, user):
    from core.chunking_strategies import STRATEGY_LABELS
    if strategy not in STRATEGY_LABELS or not isinstance(params, dict):
        raise ValueError("请选择有效的分块方式和参数")
    with session_scope() as db:
        task = _owned(db, task_id, user)
        _require_content_access(db, task, user)
        if task.status not in {"ready", "preview", "failed"} or task.cleaned_documents is None:
            raise ValueError("请先完成内容处理并检查文字，再生成分块")
        if task.version_id:
            version = db.get(DocumentVersion, task.version_id)
            if version.status not in {"preview", "failed"}:
                raise ValueError("文档已经进入入库，不能重新分块")
        task.strategy, task.params = strategy, params
        task.generation += 1
        task.status = "chunking"
        _stage(task, "chunks", "pending", "已确认处理后的内容，等待生成分块")
        generation = task.generation
    _enqueue(task_id, generation, "chunks")


def process_chunks(task_id, generation):
    from core.chunking_strategies import split_documents
    try:
        task = _snapshot(task_id, generation)
        if task is None or task.status != "chunking":
            return
        _checkpoint(task_id, generation, "chunks", "running", "正在根据确认后的内容生成分块")
        embeddings = None
        if task.strategy == "semantic":
            from core.embedding import get_embedding_model
            embeddings = get_embedding_model()
        chunks = split_documents(_deserialize(task.cleaned_documents), task.strategy, task.params,
            Path(task.filename).suffix.lower(), embeddings)
        if not chunks:
            raise ValueError("没有生成有效分块，请检查内容和参数")
        with session_scope() as db:
            item = db.get(UploadTask, task_id)
            if not item or item.deleted_at is not None or item.generation != generation:
                return
            user = _current_actor(db, item)
            if not item.version_id:
                document = Document(source_uri=item.file_path, original_filename=item.filename,
                    category=item.upload_metadata["category"], owner_id=user.user_id, tenant_id=item.tenant_id,
                    knowledge_base_id=item.knowledge_base_id, permission_scope=item.permission_scope,
                    upload_metadata=item.upload_metadata, status="preview")
                db.add(document)
                db.flush()
                version = DocumentVersion(document_id=document.id, version=1, content_hash=item.content_hash,
                    file_path=item.file_path, created_by=item.username, status="preview")
                db.add(version)
                db.flush()
                item.document_id, item.version_id = document.id, version.id
            else:
                version = db.get(DocumentVersion, item.version_id)
            version.chunking_strategy, version.chunking_params = item.strategy, item.params
            version.status = "preview"
            version.prepared_chunks = _serialize(chunks)
            _stage(item, "chunks", "passed", f"已生成{len(chunks)}个片段，请预览后确认入库", {"chunk_count": len(chunks)})
            item.status = "preview"
    except Exception as exc:
        logger.exception("文档分块失败：%s", task_id)
        message = str(exc) if isinstance(exc, (ValueError, PermissionError)) else "分块服务暂时失败，请检查服务后重试"
        for technical, plain in (("chunk_overlap", "重叠大小"), ("chunk_size", "分块大小"),
            ("child_overlap", "子块重叠"), ("child_size", "子块大小"), ("parent_size", "父块大小"),
            ("min_chunk_size", "最小片段大小"), ("max_chunk_size", "最大片段大小")):
            message = message.replace(technical, plain)
        _checkpoint(task_id, generation, "permissions" if isinstance(exc, PermissionError) else "chunks",
            "failed", message, status="failed")


def confirm_upload(task_id, user):
    with session_scope() as db:
        task = _owned(db, task_id, user)
        _require_content_access(db, task, user)
        if task.status not in {"preview", "submitted", "candidate", "published"} or not task.version_id:
            raise ValueError("请先生成并检查分块预览，再确认入库")
        version = db.get(DocumentVersion, task.version_id)
        if not version.prepared_chunks:
            raise ValueError("分块结果不存在，请重新生成")
        if file_sha256(task.file_path) != task.content_hash:
            raise ValueError("原文件已变化，请重新上传")
        if task.status == "preview":
            task.status = "submitted"
            _stage(task, "ingest", "pending", "已确认分块，等待后台向量化；发布状态请在版本与发布页面查看")
        document_id, version_id = task.document_id, task.version_id
    job = create_ingestion_job(document_id, version_id)
    return job


def retry_upload(task_id, user):
    with session_scope() as db:
        task = _owned(db, task_id, user)
        _kb(db, task.knowledge_base_id, user)
        failed = next((item["key"] for item in task.stages if item["status"] == "failed"), None)
        if task.status != "failed" or failed not in {"parse", "ocr", "clean", "chunks"} or not task.file_path:
            raise ValueError("该问题需要修正后重新上传，不能直接重试")
        if not Path(task.file_path).exists() or file_sha256(task.file_path) != task.content_hash:
            raise ValueError("源文件已经丢失或变化，请重新上传")
        task.reservation_key = hashlib.sha256(json.dumps([task.tenant_id, task.content_hash]).encode()).hexdigest()
        task.generation += 1
        generation = task.generation
        phase = "chunks" if failed == "chunks" else "preprocess"
        task.status = "chunking" if phase == "chunks" else "queued"
        _stage(task, failed, "pending", "已安排重试")
    _enqueue(task_id, generation, phase)


def retry_upload_ingestion(task_id, user):
    with session_scope() as db:
        task = _owned(db, task_id, user)
        _kb(db, task.knowledge_base_id, user)
        job = db.scalar(select(IngestionJob).where(IngestionJob.version_id == task.version_id))
        if not job or job.status not in {"failed", "pending"}:
            raise ValueError("当前没有可重试的入库任务")
        job_id = job.id
    retry_ingestion_job(job_id)


def _record_busy(db, task):
    if task.status in {"checking", "queued", "processing", "chunking"}:
        return True
    job = db.scalar(select(IngestionJob).where(IngestionJob.version_id == task.version_id)) if task.version_id else None
    if (task.status == "submitted" and not job) or (job and job.status in {"pending", "running"}):
        return True
    keys = BackgroundJob.task_key.startswith(f"upload:{task.id}:", autoescape=True)
    if job:
        keys = keys | (BackgroundJob.task_key == "ingest:" + job.id)
    return db.scalar(select(BackgroundJob.id).where(keys,
        BackgroundJob.status.in_(["pending", "running"])).limit(1)) is not None


def delete_upload_record(task_id, user):
    with session_scope() as db:
        user = _active_actor(db, user)
        task = _owned(db, task_id, user)
        if _record_busy(db, task):
            raise ValueError("该文件仍在处理或等待处理，请完成后再删除记录")
        # 从处理入口移除，保留关联资料及其授权信息，避免误删已入库内容。
        task.deleted_at = datetime.now(timezone.utc)
        version = db.get(DocumentVersion, task.version_id) if task.version_id else None
        if not version or version.status in {"preview", "failed"}:
            task.reservation_key = None
            if version:
                version.status = "failed"
                document = db.get(Document, task.document_id)
                if document and document.current_version == 0 and document.status == "preview":
                    document.status = "failed"
    return {"id": task_id, "deleted": True, "message": "处理记录已删除；未入库流程已结束，已入库资料保留"}


def task_payload(task_id, user, include_content=True):
    with session_scope() as db:
        user = _active_actor(db, user)
        task = _owned(db, task_id, user)
        result = {"id": task.id, "filename": task.filename, "tenant_id": task.tenant_id,
            "status": task.status, "stages": task.stages, "metadata": task.upload_metadata,
            "permission_scope": task.permission_scope, "document_id": task.document_id,
            "version_id": task.version_id, "created_at": task.created_at.isoformat(),
            "strategy": task.strategy, "params": task.params}
        result["can_delete"] = not _record_busy(db, task)
        job = db.scalar(select(IngestionJob).where(IngestionJob.version_id == task.version_id)) if task.version_id else None
        if job:
            version = db.get(DocumentVersion, task.version_id)
            stages = [dict(value) for value in task.stages]
            stage = next(value for value in stages if value["key"] == "ingest")
            stage.update(status={"pending": "pending", "running": "running", "failed": "failed",
                "completed": "passed"}.get(job.status, "pending"),
                detail="向量化完成，待审核发布" if version.status == "candidate" else
                    "文档版本已发布；问答使用范围以索引发布为准" if job.status == "completed" else
                    "向量化入库失败，请重试或联系管理员" if job.status == "failed" else "等待或正在后台向量化",
                next_action=ACTIONS["ingest"] if job.status == "failed" else "",
                started_at=job.started_at.isoformat() if job.started_at else None,
                finished_at=job.finished_at.isoformat() if job.finished_at else None)
            result.update(stages=stages, job_id=job.id, ingestion_status=job.status)
            if job.status == "completed":
                result["status"] = version.status
        result["can_retry"] = task.status == "failed" and any(value["key"] in {"parse", "ocr", "clean", "chunks"}
            and value["status"] == "failed" for value in task.stages) and bool(task.file_path)
        result["can_retry_ingestion"] = bool(job and job.status == "failed")
        if result["can_retry_ingestion"]:
            result["status"] = "ingestion_failed"
        # 公司范围先校验；管理员可审核，普通用户仍须满足当前阅读权限。
        kb = db.get(KnowledgeBase, task.knowledge_base_id) if task.knowledge_base_id else None
        document = db.get(Document, task.document_id) if task.document_id else None
        readable = _can_read_upload(kb, document, task.permission_scope, user)
        result["can_read_content"] = readable
        if include_content and readable:
            result["before_content"] = "\n\n".join(value["content"] for value in task.raw_documents or [])
            result["after_content"] = "\n\n".join(value["content"] for value in task.cleaned_documents or [])
            version = db.get(DocumentVersion, task.version_id) if task.version_id else None
            result["chunks"] = [{"index": number, **value} for number, value in enumerate(version.prepared_chunks or [], 1)] if version else []
        return result


def list_uploads(user, limit=50):
    _identity(user)
    with session_scope() as db:
        ids = list(db.scalars(select(UploadTask.id).where(UploadTask.tenant_id == user.tenant_id,
            UploadTask.username == user.username, UploadTask.deleted_at.is_(None))
            .order_by(UploadTask.created_at.desc()).limit(limit)))
    records = []
    for task_id in ids:
        try:
            records.append(task_payload(task_id, user, include_content=False))
        except LookupError:
            continue  # 自动刷新与删除同时发生时，跳过刚移除的记录。
    return records


def recover_upload_tasks():
    with session_scope() as db:
        pending = list(db.scalars(select(UploadTask).where(UploadTask.deleted_at.is_(None),
            UploadTask.status.in_(["queued", "processing", "chunking", "submitted"]))))
        for task in pending:
            _reconcile_queue(db, task)
        pending = [task for task in pending if task.status != "failed"]
    for task in pending:
        if task.status == "submitted" and task.version_id:
            create_ingestion_job(task.document_id, task.version_id)
        else:
            _enqueue(task.id, task.generation, "chunks" if task.status == "chunking" else "preprocess")
    return len(pending)


def verify_upload_ingestion(version_id):
    with session_scope() as db:
        task = db.scalar(select(UploadTask).where(UploadTask.version_id == version_id))
        if task:
            if task.deleted_at is not None:
                raise PermissionError("处理记录已删除，本次入库流程已结束")
            _current_actor(db, task)
