"""混合检索、权限过滤和可选重排。"""
import logging
import re
import time
from typing import Any

from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from config.settings import settings
from core.auth import ANONYMOUS_USER, AuthUser, can_access_document
from core.knowledge_scope import accessible_knowledge_filter, parse_permission_scope
from core.vector_store import get_vector_store_manager
from core.metrics import increment, observe
from core.telemetry import traced

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseDocumentCompressor):
    """基于交叉编码器的重排序器。"""

    model_name: str
    top_k: int
    model: object = None

    def _initialize_model(self):
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(self.model_name)
                logger.info("交叉编码器模型已初始化: %s", self.model_name)
            except Exception as exc:
                logger.error("初始化交叉编码器模型失败: %s", exc)
                self.model = None

    @traced("rag.rerank")
    def compress_documents(self, documents, query, **kwargs):
        if not documents:
            return []
        self._initialize_model()
        if self.model is None:
            return documents[: self.top_k]
        try:
            pairs = [[query, doc.page_content] for doc in documents]  ##交叉重排序
            scores = self.model.predict(pairs)
            sorted_docs = sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
            if settings.RERANK_MIN_SCORE is not None:
                sorted_docs = [(doc, score) for doc, score in sorted_docs if score >= settings.RERANK_MIN_SCORE]
            return [doc for doc, _ in sorted_docs[: self.top_k]]
        except Exception as exc:
            logger.error("重排序失败: %s", exc)
            return documents[: self.top_k]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", (text or "").lower()))


def _metadata_key(document) -> str:
    return f"{document.metadata.get('source', '')}:{document.metadata.get('page', '')}:{document.page_content}"


def _permission_scope(metadata: dict[str, Any]) -> dict[str, Any]:
    return parse_permission_scope(metadata.get("permission_scope"))


def expand_parent_context(documents):
    """子块完成排序后补回同版本的父段落，保留排序并去重。"""
    from langchain_core.documents import Document
    result, seen = [], set()
    for doc in documents:
        parent = doc.metadata.get("parent_content")
        key = (doc.metadata.get("document_id"), doc.metadata.get("version_id"),
            doc.metadata.get("source"), doc.metadata.get("page"), doc.metadata.get("parent_id")) if parent else _metadata_key(doc)
        if key in seen:
            continue
        seen.add(key)
        result.append(Document(page_content=parent, metadata={**doc.metadata, "parent_expanded": True}) if parent else doc)
    return result


class HybridRetriever(BaseRetriever):
    """把向量候选和关键词候选合并后交给重排器。

    这是 LangChain 需要的适配层：底层 RAGRetriever 负责具体检索，
    这个类把它包装成 LangChain 的 retriever 接口。
    """

    owner: Any = Field(exclude=True)
    search_kwargs: dict[str, Any] = Field(default_factory=dict)
    user: AuthUser | None = Field(default=None, exclude=True)
    version_manifest: dict | None = Field(default=None, exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        return self.owner.retrieve(
            query,
            top_k=self.search_kwargs.get("k", settings.SEARCH_TOP_K),
            filter_dict=self.search_kwargs.get("filter"),
            user=self.user,
            version_manifest=self.version_manifest,
            apply_rerank=True,
        )


class RAGRetriever:
    """向量+关键词混合召回，支持用户权限过滤和重排。

    向量检索擅长理解相近含义，关键词检索擅长匹配专有名词；两者合并后通常比只用一种方式稳定。
    """

    def __init__(self, vector_store=None):
        self.vector_store = vector_store or get_vector_store_manager()
        self.reranker = self._initialize_reranker()

    def _initialize_reranker(self):
        if not settings.USE_RERANKER:
            logger.info("未启用Reranker")
            return None
        try:
            model_name = settings.RERANKER_MODEL_NAME
            local_model_path = settings.MODELS_DIR / model_name.replace("/", "_")
            model_path = str(local_model_path) if local_model_path.exists() else model_name
            return CrossEncoderReranker(model_name=model_path, top_k=settings.SEARCH_TOP_K)
        except Exception as exc:
            logger.error("初始化Reranker失败: %s", exc)
            return None

    def _can_access(self, document, user: AuthUser | None) -> bool:
        try:
            if document.metadata.get("document_id"):
                from core.database import session_scope
                from core.db_models import Document
                with session_scope() as db:
                    stored = db.get(Document, document.metadata["document_id"])
                    if stored:
                        return can_access_document(stored.permission_scope, user or ANONYMOUS_USER)
            return can_access_document(_permission_scope(document.metadata), user or ANONYMOUS_USER)
        except (ValueError, TypeError):
            return False

    def _keyword_candidates(self, query: str, user: AuthUser | None, limit: int, filter_dict: dict | None = None) -> list[tuple[Any, float]]:
        if settings.VECTOR_STORE_TYPE == "opensearch":
            return self.vector_store._store.keyword_search(query, limit, filter_dict)
        try:
            payload = self.vector_store.get_documents(filter_dict=filter_dict)
        except Exception as exc:
            logger.warning("关键词候选读取失败，将只使用向量检索: %s", exc)
            return []
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        candidates = []
        for content, metadata in zip(payload.get("documents", []), payload.get("metadatas", [])):
            metadata = metadata or {}
            document_tokens = _tokens(content)
            overlap = len(query_tokens & document_tokens)
            if overlap <= 0:
                continue
            from langchain_core.documents import Document

            document = Document(page_content=content or "", metadata=metadata)
            if self._can_access(document, user):
                candidates.append((document, overlap / max(1, len(query_tokens))))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:limit]

    @traced("rag.retrieve")
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_dict: dict | None = None,
        return_score: bool = False,
        user: AuthUser | None = None,
        apply_rerank: bool = True,
        version_manifest: dict | None = None,
    ) -> list:
        # 先召回一批候选，再按权限过滤和综合分数排序，最后可选地用重排模型精排。
        # 这样既能兼顾检索速度，也给精排模型保留足够的候选范围。
        final_k = top_k or settings.SEARCH_TOP_K
        started_at = time.perf_counter()
        candidate_k = max(final_k * 4, 20)
        include_tool_catalog = bool(filter_dict and filter_dict.get("category") == "assistant_tools")
        user = user or ANONYMOUS_USER
        permission_filter = accessible_knowledge_filter(user) if version_manifest is None else accessible_knowledge_filter(user, version_manifest, include_candidates=True)
        if permission_filter is None:
            return []
        filter_dict = (
            {"$and": [permission_filter, filter_dict]} if filter_dict else permission_filter
        )
        #langchain封装的隐藏掉了很多参数，实际生产的一般使用向量数据库自己原生的api接口
        try:
            vector_results = self.vector_store.similarity_search_with_score(
                query, k=candidate_k, filter_dict=filter_dict
            )
        except Exception:
            increment("retrieval_errors_total")
            raise
        if settings.VECTOR_STORE_TYPE == "opensearch":
            vector_results = [(doc, distance) for doc, distance in vector_results if 1.0 - distance >= settings.SEARCH_MIN_VECTOR_SIMILARITY]
        ranked: dict[str, dict[str, Any]] = {}
        for rank, (document, distance) in enumerate(vector_results):
            if (not include_tool_catalog and document.metadata.get("category") == "assistant_tools"):
                continue
            if self._can_access(document, user):
                ranked[_metadata_key(document)] = {
                    "document": document,
                    "vector_score": 1.0 / (60 + rank + 1) if settings.VECTOR_STORE_TYPE == "opensearch" else 1.0 / (rank + 1),
                    "keyword_score": 0.0,
                }
        for keyword_rank, (document, keyword_score) in enumerate(self._keyword_candidates(query, user, candidate_k, filter_dict)):
            if not include_tool_catalog and document.metadata.get("category") == "assistant_tools":
                continue
            if not self._can_access(document, user):
                continue
            if settings.VECTOR_STORE_TYPE == "opensearch":
                keyword_score = 1.0 / (60 + keyword_rank + 1)
            key = _metadata_key(document)
            item = ranked.setdefault(
                key,
                {"document": document, "vector_score": 0.0, "keyword_score": 0.0},
            )
            item["keyword_score"] = max(item["keyword_score"], keyword_score)
        items = sorted(
            ranked.values(),
            key=lambda item: 0.65 * item["vector_score"] + 0.35 * item["keyword_score"],
            reverse=True,
        )
        candidates = [item["document"] for item in items[:candidate_k]]
        if apply_rerank and self.reranker:
            candidates = self.reranker.compress_documents(candidates, query)
        elif apply_rerank:
            candidates = candidates[:final_k]
        observe("retrieval", (time.perf_counter() - started_at) * 1000)
        if not candidates:
            increment("retrieval_empty_total")
        if return_score:
            scores = {id(item["document"]): 0.65 * item["vector_score"] + 0.35 * item["keyword_score"] for item in items}
            return [(expand_parent_context([document])[0], scores.get(id(document), 0.0)) for document in candidates[:final_k]]
        if apply_rerank:
            candidates = expand_parent_context(candidates)
        return candidates if not apply_rerank else candidates[:final_k]

    def get_compresstion_retriever(self, search_kwargs: dict, user: AuthUser | None = None, version_manifest=None):
        search_kwargs = search_kwargs or {"k": settings.SEARCH_TOP_K}
        base_retriever = HybridRetriever(owner=self, search_kwargs=search_kwargs, user=user, version_manifest=version_manifest)
        return base_retriever


_rag_retriever_instance = None


def get_rag_retriever() -> RAGRetriever:
    global _rag_retriever_instance
    if _rag_retriever_instance is None:
        _rag_retriever_instance = RAGRetriever()
    return _rag_retriever_instance
