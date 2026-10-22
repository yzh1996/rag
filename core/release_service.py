"""版本化候选索引、审核、灰度和指针回退。"""
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document as Chunk
from sqlalchemy import select

from config.settings import settings
from core.auth import ANONYMOUS_USER, AuthUser, actor_payload, can_access_document
from core.database import init_db, session_scope
from core.db_models import Document, DocumentVersion, EvalRun, IndexVersion, ReleaseState
from core.evaluation_contract import fingerprint, runtime_config
from core.vector_store import VectorStoreManager, get_vector_store_manager

_stores = {}


def _now():
    return datetime.now(timezone.utc)


def artifact_hash(manager):
    payload = manager.get_documents()
    return fingerprint(sorted([{"content": text, "metadata": metadata} for text, metadata in
        zip(payload.get("documents", []), payload.get("metadatas", []))], key=fingerprint))


def release_store(item):
    if item.embedding_model != settings.EMBEDDING_MODEL_NAME or item.vector_store != settings.VECTOR_STORE_TYPE:
        raise ValueError("版本与当前模型或检索服务不兼容")
    if item.id not in _stores:
        _stores[item.id] = VectorStoreManager(backend=item.vector_store,
            persist_directory=Path(item.snapshot_path) if item.snapshot_path else None,
            index_name=item.config.get("index_name"), immutable=True)
    return _stores[item.id]


def register_index_build(config=None, user=None):
    init_db()
    actor = user or ANONYMOUS_USER
    name, manifest = "index-" + uuid.uuid4().hex, {}
    with session_scope() as db:
        documents = db.scalars(select(Document).where(Document.tenant_id == actor.tenant_id,
            Document.status.in_(["published", "candidate"]))).all()
        for document in documents:
            if actor.role not in {"admin", "knowledge_admin"} and not can_access_document(document.permission_scope, actor):
                continue
            version = db.scalar(select(DocumentVersion).where(DocumentVersion.document_id == document.id,
                DocumentVersion.status.in_(["published", "candidate"])).order_by(DocumentVersion.version.desc()))
            if version:
                manifest[document.id] = version.id
    if not manifest:
        raise ValueError("没有可构建的已入库文档，请先上传并等待处理完成")
    destination = settings.INDEX_SNAPSHOT_DIR / name
    index_name = settings.OPENSEARCH_INDEX + "-" + name if settings.VECTOR_STORE_TYPE == "opensearch" else None
    with session_scope() as db:
        item = IndexVersion(version=name, tenant_id=actor.tenant_id, embedding_model=settings.EMBEDDING_MODEL_NAME,
            vector_store=settings.VECTOR_STORE_TYPE, snapshot_path=str(destination), status="building",
            config={**(config or {}), "manifest": manifest, "index_name": index_name,
                "runtime": runtime_config(), "actor": actor_payload(actor), "review": None})
        db.add(item)
        db.flush()
        return item


def build_index_job(index_id, user):
    item = get_candidate(index_id, user)
    if item.status == "candidate":
        return item
    if item.status not in {"building", "failed"}:
        raise ValueError("该版本不能再次构建")
    try:
        if item.config["runtime"] != runtime_config():
            raise ValueError("构建期间配置已变化，请创建新的候选版本")
        source = get_vector_store_manager()
        from core.knowledge_scope import accessible_knowledge_filter
        current_filter = accessible_knowledge_filter(user)
        legacy_filter = {"tenant_id": user.tenant_id}
        if current_filter:
            legacy_filter = {"$or": [legacy_filter, current_filter]}
        legacy = source.get_documents(legacy_filter)
        if any(not metadata.get("version_id") and metadata.get("source") != "init" for metadata in legacy.get("metadatas", [])):
            raise ValueError("存在尚未登记版本的旧知识，请先重新上传；构建已停止以避免发布后遗漏旧知识")
        payload = source.get_documents({"version_id": {"$in": list(item.config["manifest"].values())}})
        chunks = [Chunk(page_content=text, metadata=dict(metadata)) for text, metadata in
            zip(payload.get("documents", []), payload.get("metadatas", []))]
        if {chunk.metadata.get("version_id") for chunk in chunks} != set(item.config["manifest"].values()):
            raise ValueError("候选文档片段不完整，不能构建索引")
        target = VectorStoreManager(backend=item.vector_store, persist_directory=Path(item.snapshot_path),
            index_name=item.config.get("index_name"), embedding_model=source.embedding_model)
        if target.add_documents(chunks) != len(chunks):
            raise ValueError("候选索引构建失败，未影响线上版本")
        with session_scope() as db:
            result = db.get(IndexVersion, item.id)
            result.status = "candidate"
            result.config = {**result.config, "artifact_hash": artifact_hash(target), "error": None}
        _stores[item.id] = target
        return result
    except Exception as exc:
        with session_scope() as db:
            result = db.get(IndexVersion, item.id)
            result.status = "failed"
            result.config = {**result.config, "error": str(exc)}
        raise


def create_index_snapshot(config=None, user=None):
    """同步入口供离线工具使用；网页构建使用持久化任务。"""
    item = register_index_build(config, user)
    return build_index_job(item.id, user or ANONYMOUS_USER)


def enqueue_index_build(config=None, user=None):
    from core.task_queue import enqueue
    item = register_index_build(config, user)
    enqueue("build_index", "build:" + item.id, {"index_id": item.id, "user": item.config["actor"]})
    return item


def recover_index_builds():
    from core.task_queue import enqueue
    with session_scope() as db:
        items = db.scalars(select(IndexVersion).where(IndexVersion.status == "building")).all()
    for item in items:
        if item.config.get("actor"):
            enqueue("build_index", "build:" + item.id, {"index_id": item.id, "user": item.config["actor"]})


def get_candidate(index_id, user):
    with session_scope() as db:
        item = db.get(IndexVersion, index_id)
        if not item or item.tenant_id != user.tenant_id:
            raise ValueError("索引版本不存在")
        if not item.config.get("manifest"):
            raise ValueError("旧快照没有版本清单，请重新构建候选索引")
        return item


def review_index(index_id, user, notes):
    if not notes.strip():
        raise ValueError("审核结论不能为空")
    with session_scope() as db:
        item = db.get(IndexVersion, index_id)
        if not item or item.tenant_id != user.tenant_id or item.status != "candidate":
            raise ValueError("只能审核本租户的候选版本")
        item.config = {**item.config, "review": {"reviewer": user.username, "notes": notes.strip(),
            "artifact_hash": item.config["artifact_hash"], "at": _now().isoformat()}}
        return item


def _quality_gate_passed(db, eval_run_id, item=None):
    run = db.get(EvalRun, eval_run_id) if eval_run_id else None
    if not run or run.status != "completed" or not run.metrics.get("quality_gate_passed") or item is None or not run.total_cases:
        return False
    return bool(run.tenant_id == item.tenant_id and run.config.get("index_id") == item.id
        and run.config.get("artifact_hash") == item.config.get("artifact_hash")
        and run.config.get("runtime") == runtime_config() == item.config.get("runtime")
        and run.metrics.get("complete_suite") and run.total_cases == run.passed_cases
        and run.metrics.get("dataset_revision") == run.config.get("dataset_revision"))


def _validate_publication(db, item, eval_run_id):
    if item.status not in {"candidate", "canary"}:
        raise ValueError("只能发布候选或灰度版本")
    if not _quality_gate_passed(db, eval_run_id, item):
        raise ValueError("必须使用本候选版本、当前配置和完整测试集的合格评测")
    if (item.config.get("review") or {}).get("artifact_hash") != item.config.get("artifact_hash"):
        raise ValueError("候选版本尚未完成审核")
    if artifact_hash(release_store(item)) != item.config["artifact_hash"]:
        raise ValueError("候选索引已变化，请重新构建并评测")
    for doc_id, version_id in item.config["manifest"].items():
        doc, version = db.get(Document, doc_id), db.get(DocumentVersion, version_id)
        if not doc or doc.tenant_id != item.tenant_id or doc.status == "deprecated" or not version or version.document_id != doc_id or version.status not in {"published", "candidate", "archived"}:
            raise ValueError("候选知识状态已变化，请重新构建")


def _activate_manifest(db, item):
    for doc_id, version_id in item.config["manifest"].items():
        doc, version = db.get(Document, doc_id), db.get(DocumentVersion, version_id)
        if doc and version and doc.status != "deprecated":
            for old in doc.versions:
                if old.id != version_id and old.status == "published":
                    old.status, old.expired_at = "archived", _now()
            version.status, version.effective_at, version.expired_at = "published", _now(), None
            doc.status, doc.current_version = "published", version.version


def publish_index(index_id, eval_run_id=None, user=None, canary_percent=0):
    actor = user or ANONYMOUS_USER
    with session_scope() as db:
        state = db.get(ReleaseState, actor.tenant_id, with_for_update=True)
        if state is None:
            state = ReleaseState(tenant_id=actor.tenant_id)
            db.add(state)
            db.flush()
        item = db.get(IndexVersion, index_id)
        if not item or item.tenant_id != actor.tenant_id:
            raise ValueError("索引版本不存在")
        _validate_publication(db, item, eval_run_id)
        if canary_percent:
            if not state.active_index_id:
                raise ValueError("首次发布需要先建立正式版本，随后才能灰度")
            if not 1 <= canary_percent <= 50:
                raise ValueError("灰度比例必须在1到50之间")
            if state.canary_index_id and state.canary_index_id != item.id:
                raise ValueError("已有其他灰度版本，请先结束或回退")
            state.canary_index_id, state.canary_percent = item.id, canary_percent
            item.status = "canary"
        else:
            previous = db.get(IndexVersion, state.active_index_id) if state.active_index_id else None
            if previous:
                previous.status = "archived"
            state.previous_index_id, state.active_index_id = state.active_index_id, item.id
            state.canary_index_id, state.canary_percent = None, 0
            item.status, item.published_at = "published", _now()
            _activate_manifest(db, item)
        item.config = {**item.config, "eval_run_id": eval_run_id}
        return item


def rollback_index(index_id, user=None):
    actor = user or ANONYMOUS_USER
    with session_scope() as db:
        state = db.get(ReleaseState, actor.tenant_id, with_for_update=True)
        item = db.get(IndexVersion, index_id)
        if not state or not item or item.tenant_id != actor.tenant_id or not item.published_at:
            raise ValueError("只能回退到本租户以前正式发布过的版本")
        if artifact_hash(release_store(item)) != item.config.get("artifact_hash"):
            raise ValueError("旧索引完整性检查失败，不能回退")
        if item.config.get("runtime") != runtime_config():
            raise ValueError("旧版本配置与当前服务不同，请先恢复对应配置")
        for current_id in (state.active_index_id, state.canary_index_id):
            current = db.get(IndexVersion, current_id) if current_id else None
            if current and current.id != item.id:
                current.status = "rolled_back" if current.published_at else "candidate"
        state.previous_index_id, state.active_index_id = state.active_index_id, item.id
        state.canary_index_id, state.canary_percent = None, 0
        item.status = "published"
        _activate_manifest(db, item)
        return item


def resolve_release(user, session_id, index_id=None):
    if index_id:
        return get_candidate(index_id, user)
    with session_scope() as db:
        state = db.get(ReleaseState, user.tenant_id)
        if not state or not state.active_index_id:
            return None
        bucket = int(hashlib.sha256(f"{user.tenant_id}:{user.username}:{session_id}".encode()).hexdigest(), 16) % 100
        selected = state.canary_index_id if state.canary_index_id and bucket < state.canary_percent else state.active_index_id
        item = db.get(IndexVersion, selected)
        if not item:
            raise ValueError("当前发布版本不存在，已停止无版本回退")
        return item


def list_index_versions(limit=50, user=None):
    actor = user or ANONYMOUS_USER
    with session_scope() as db:
        return list(db.scalars(select(IndexVersion).where(IndexVersion.tenant_id == actor.tenant_id)
            .order_by(IndexVersion.created_at.desc()).limit(limit)).all())
