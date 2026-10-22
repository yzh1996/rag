"""索引版本发布和回滚接口。"""
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.auth import AuthUser, require_admin, require_admin_or_local
from core.index_service import create_index_snapshot, list_index_versions, publish_index, rollback_index, review_index
from core.release_service import enqueue_index_build

router = APIRouter(prefix="/index", tags=["索引发布"])


class IndexSnapshotRequest(BaseModel):
    chunking_strategy: str = "recursive"
    config: dict[str, Any] = Field(default_factory=dict)


class IndexPublishRequest(BaseModel):
    eval_run_id: str
    canary_percent: int = Field(default=0, ge=0, le=50)


class IndexReviewRequest(BaseModel):
    notes: str = Field(min_length=1, max_length=20000)


def _payload(item):
    return {
        "id": item.id,
        "version": item.version,
        "embedding_model": item.embedding_model,
        "tenant_id": item.tenant_id,
        "chunking_strategy": item.chunking_strategy,
        "vector_store": item.vector_store,
        "snapshot_path": item.snapshot_path,
        "status": item.status,
        "config": item.config,
        "created_at": item.created_at,
        "published_at": item.published_at,
    }


@router.get("/versions")
async def get_index_versions(user: AuthUser = Depends(require_admin_or_local)):
    return {"items": [_payload(item) for item in list_index_versions(user=user)]}


@router.post("/versions/snapshot")
async def snapshot_index(
    request: IndexSnapshotRequest,
    user: AuthUser = Depends(require_admin_or_local),
):
    try:
        item = enqueue_index_build({**request.config, "chunking_strategy": request.chunking_strategy}, user=user)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _payload(item)


@router.post("/versions/{index_id}/publish")
async def publish_index_version(
    index_id: str,
    request: IndexPublishRequest,
    user: AuthUser = Depends(require_admin),
):
    try:
        item = publish_index(index_id, request.eval_run_id, user=user, canary_percent=request.canary_percent)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _payload(item)


@router.post("/versions/{index_id}/rollback")
async def rollback_index_version(
    index_id: str,
    user: AuthUser = Depends(require_admin),
):
    try:
        item = rollback_index(index_id, user=user)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _payload(item)


@router.post("/versions/{index_id}/review")
async def review_index_version(index_id: str, request: IndexReviewRequest, user: AuthUser = Depends(require_admin)):
    try:
        item = review_index(index_id, user, request.notes)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _payload(item)
