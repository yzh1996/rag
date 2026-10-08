"""
知识库管理API
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from api.dependencies import get_vector_store_dep
from core.vector_store import VectorStoreManager

router = APIRouter(prefix="/knowledge", tags=["知识库管理"])


class DeleteRequest(BaseModel):
    source_path: str


@router.get("/stats")
async def get_knowledge_stats(
    vector_store: VectorStoreManager = Depends(get_vector_store_dep)
):
    """
    获取知识库统计信息
    """
    stats = vector_store.get_collection_stats()
    return stats


@router.post("/delete")
async def delete_document(
    request: DeleteRequest,
    vector_store: VectorStoreManager = Depends(get_vector_store_dep)
):
    """
    根据源文件路径删除文档
    """
    deleted_count = vector_store.delete_by_source(request.source_path)
    return {
        "status": "success",
        "deleted_count": deleted_count,
        "source_path": request.source_path
    }


@router.post("/clear")
async def clear_knowledge_base(
    vector_store: VectorStoreManager = Depends(get_vector_store_dep)
):
    """
    清空整个知识库
    """
    vector_store.clear_all()
    return {"status": "success", "message": "知识库已清空"}