"""
系统状态API
"""
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from sqlalchemy import text

from config.settings import settings
from core.database import engine
from core.metrics import prometheus_text, snapshot

# 必须定义名为 router 的 APIRouter 实例
router = APIRouter(prefix="/system", tags=["系统状态"])


@router.get("/health")
async def health_check():
    """健康检查"""
    database_status = "healthy"
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        database_status = "unhealthy"
    search_status = "not_configured"
    if settings.VECTOR_STORE_TYPE == "opensearch":
        try:
            from core.vector_store import get_vector_store_manager

            store = get_vector_store_manager()._store
            search_status = "healthy" if store.client.ping() and store.client.indices.exists(index=store.index) else "unhealthy"
        except Exception:
            search_status = "unhealthy"
    return {
        "status": "healthy" if database_status == "healthy" and search_status != "unhealthy" else "degraded",
        "service": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "database": database_status,
        "search": search_status,
    }


@router.get("/metrics")
async def metrics():
    """Prometheus 文本格式指标。"""
    return PlainTextResponse(prometheus_text(), media_type="text/plain; version=0.0.4")


@router.get("/metrics/json")
async def metrics_json():
    return snapshot()


@router.get("/config")
async def get_config():
    """获取当前配置（隐藏敏感信息）"""
    return {
        "vector_store_type": settings.VECTOR_STORE_TYPE,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.OLLAMA_MODEL_NAME if settings.LLM_PROVIDER == "ollama" else settings.OPENAI_MODEL_NAME,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "search_top_k": settings.SEARCH_TOP_K,
    }
