"""Knowledge-base grouping and request-scoped read permissions."""
import hashlib
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from core.auth import AuthUser, can_access_document
from core.database import session_scope
from core.db_models import Document, DocumentVersion, KnowledgeBase


def parse_permission_scope(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    scope = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(scope, dict):
        raise ValueError("Invalid document permission scope")
    for field in ("tenant_ids", "users", "roles", "departments"):
        if field in scope and (not isinstance(scope[field], list) or any(not isinstance(item, str) or not item.strip() for item in scope[field])):
            raise ValueError("Invalid permission list: " + field)
    if "public" in scope and not isinstance(scope["public"], bool):
        raise ValueError("public必须为布尔值")
    return scope


def ensure_knowledge_base(db, tenant_id: str, category: str, scope: dict[str, Any]) -> KnowledgeBase:
    scope = parse_permission_scope(scope)
    normalized = {
        key: sorted(set(value)) if isinstance(value, list) else value
        for key, value in scope.items()
    }
    signature = json.dumps([tenant_id, category, normalized], sort_keys=True, ensure_ascii=True)
    knowledge_base_id = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
    if knowledge_base is None:
        knowledge_base = KnowledgeBase(
            id=knowledge_base_id,
            name=category,
            tenant_id=tenant_id,
            permission_scope=normalized,
        )
        try:
            with db.begin_nested():
                db.add(knowledge_base)
                db.flush()
        except IntegrityError:
            # Concurrent uploads with the same permissions share one knowledge base.
            knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
            if knowledge_base is None:
                raise
    return knowledge_base


def assign_document_knowledge_bases(db) -> None:
    documents = db.scalars(select(Document).where(Document.knowledge_base_id.is_(None))).all()
    for document in documents:
        knowledge_base = ensure_knowledge_base(
            db, document.tenant_id, document.category, document.permission_scope or {}
        )
        document.knowledge_base_id = knowledge_base.id


def accessible_knowledge_filter(user: AuthUser, version_manifest: dict | None = None, include_candidates: bool = False) -> dict[str, Any] | None:
    # None means no authorized knowledge, never an unrestricted search.
    with session_scope() as db:
        knowledge_bases = db.scalars(select(KnowledgeBase)).all()
        allowed_ids = {
            item.id for item in knowledge_bases
            if can_access_document(item.permission_scope, user)
        }
        if not allowed_ids:
            return None
        # Narrow document permissions also apply before Top-K selection.
        documents = db.scalars(select(Document)).all()
        blocked_ids = [
            item.id for item in documents
            if item.knowledge_base_id in allowed_ids and (
                not can_access_document(item.permission_scope, user)
                or item.status == "deprecated"
                or (item.status != "published" and not (include_candidates and item.status == "candidate"))
            )
        ]
        by_id = {item.id: item for item in documents}
        versions = db.scalars(select(DocumentVersion)).all()
        blocked_versions = [item.id for item in versions if (
            item.document_id not in by_id
            or item.version != by_id[item.document_id].current_version
            or item.status != "published"
        )]
    knowledge_filter = {"knowledge_base_id": {"$in": sorted(allowed_ids)}}
    filters = [knowledge_filter]
    if blocked_ids:
        filters.append({"document_id": {"$nin": sorted(blocked_ids)}})
    if version_manifest is not None:
        if not version_manifest:
            return None
        filters.append({"version_id": {"$in": sorted(set(version_manifest.values()))}})
    elif blocked_versions:
        filters.append({"version_id": {"$nin": sorted(blocked_versions)}})
    return {"$and": filters} if len(filters) > 1 else knowledge_filter
