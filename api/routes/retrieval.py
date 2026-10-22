"""独立检索 API；保持现有问答 API 行为不变。"""
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

from core.auth import AuthUser, require_authenticated
from core.retrieval_service import RetrievalUnavailable, search_published


class RetrievalRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def wrapped(request):
            try:
                return await handler(request)
            except (HTTPException, RequestValidationError, RetrievalUnavailable) as exc:
                if isinstance(exc, RetrievalUnavailable):
                    code, message, status = exc.code, exc.message, exc.status
                elif isinstance(exc, RequestValidationError):
                    code, message, status = "INVALID_REQUEST", "检索参数格式不正确", 422
                else:
                    status = exc.status_code
                    code = {401: "AUTH_REQUIRED", 403: "ACCESS_DENIED"}.get(status, "REQUEST_REJECTED")
                    message = "请提供有效的知识库登录凭据" if status == 401 else "检索请求被拒绝"
                return JSONResponse(status_code=status, content={"trace_id": uuid.uuid4().hex,
                    "error": {"code": code, "message": message, "retryable": status >= 500}})
            except Exception:
                return JSONResponse(status_code=503, content={"trace_id": uuid.uuid4().hex,
                    "error": {"code": "KNOWLEDGE_RETRIEVAL_FAILED", "message": "知识检索暂不可用",
                              "retryable": True}})
        return wrapped


router = APIRouter(prefix="/retrieval", tags=["独立检索"], route_class=RetrievalRoute)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10, strict=True)
    task_id: str = Field(min_length=1, max_length=128)


class SearchItem(BaseModel):
    content: str
    document_id: str | None
    version_id: str | None
    source: str
    page: int | str | None
    truncated: bool


class SearchResponse(BaseModel):
    trace_id: str
    index_version: str
    knowledge_view: str
    items: list[SearchItem]
    truncated: bool


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, user: AuthUser = Depends(require_authenticated)):
    return search_published(payload.query, payload.top_k, payload.task_id, user)
