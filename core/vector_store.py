"""
向量库管理模块——封装chroma或者faiss操作，提供统一接口
通过embedding模块获取嵌入模型，转换为向量后，使用chroma或者faiss存储到本地向量库持久化存储，方便后续的检索与查询等操作。
"""
import logging
import operator
import shutil
import hashlib
import json
from langchain.schema import Document
from langchain_community.vectorstores import Chroma, FAISS
from config.settings import settings
from core.embedding import get_embedding_model

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """向量库管理器。

    它把 Chroma、FAISS、OpenSearch 的差异藏在内部，对外提供统一的添加、删除和检索方法。
    """
    def __init__(self, *, persist_directory=None, backend=None, index_name=None, embedding_model=None, immutable=False):
        self.embedding_model = embedding_model or get_embedding_model()
        self.backend = backend or settings.VECTOR_STORE_TYPE
        self.persist_directory = persist_directory or settings.VECTOR_DB_DIR
        self.index_name = index_name
        self.immutable = immutable
        self._store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """初始化向量库"""
        # 启动时打开已有索引；如果索引还不存在，FAISS 模式会先创建一个空的初始索引。
        if self.backend == "chroma":
            self._store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embedding_model
            )
            logger.info(f"Chroma向量库已初始化，存储路径: {settings.VECTOR_DB_DIR}")
        elif self.backend == "faiss":
            faiss_index_path = self.persist_directory / "index.faiss"
            if faiss_index_path.exists():
                self._store = FAISS.load_local(
                    str(self.persist_directory),
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"FAISS向量库已加载，存储路径: {settings.VECTOR_DB_DIR}")
            else:
                self._store = FAISS.from_documents(
                    [Document(page_content="初始化向量库", metadata={"source": "init"})],
                    self.embedding_model
                )
                self._save_faiss()
                logger.info(f"FAISS向量库已保存，存储路径: {settings.VECTOR_DB_DIR}")
        elif self.backend == "opensearch":
            from core.opensearch_store import OpenSearchStore

            self._store = OpenSearchStore(self.embedding_model, index_name=self.index_name)
        else:
            raise ValueError(f"不支持的向量库类型: {self.backend}")
        if not self.immutable:
            self._migrate_knowledge_metadata()

    def reload(self):
        """发布或回滚索引后重新加载当前向量库。"""
        self._initialize_vector_store()

    def _assign_knowledge_metadata(self, metadatas: list[dict]) -> list[dict]:
        from sqlalchemy import select

        from core.database import session_scope
        from core.db_models import Document as StoredDocument
        from core.knowledge_scope import ensure_knowledge_base, parse_permission_scope

        result = []
        with session_scope() as db:
            stored_documents = db.scalars(select(StoredDocument)).all()
            by_id = {item.id: item for item in stored_documents}
            by_source = {item.source_uri: item for item in stored_documents}
            for metadata in metadatas:
                metadata = dict(metadata or {})
                document = by_id.get(metadata.get("document_id"))
                if document is None:
                    document = by_source.get(metadata.get("source"))
                if document is not None:
                    metadata["document_id"] = document.id
                    metadata["knowledge_base_id"] = document.knowledge_base_id
                    metadata["tenant_id"] = document.tenant_id
                else:
                    scope = parse_permission_scope(metadata.get("permission_scope"))
                    tenant_id = metadata.get("tenant_id") or "default"
                    category = metadata.get("category") or "general"
                    metadata["tenant_id"] = tenant_id
                    metadata["knowledge_base_id"] = ensure_knowledge_base(
                        db, tenant_id, category, scope
                    ).id
                result.append(metadata)
        return result

    def _migrate_knowledge_metadata(self) -> None:
        from core.database import init_db

        init_db()
        if self.backend == "opensearch":
            return
        if self.backend == "faiss":
            documents = list(self._store.docstore._dict.values())
            assigned = self._assign_knowledge_metadata([doc.metadata for doc in documents])
            changed = False
            for document, metadata in zip(documents, assigned):
                if document.metadata != metadata:
                    document.metadata = metadata
                    changed = True
            if changed:
                self._save_faiss()
            return
        payload = self._store.get(include=["metadatas"])
        ids = payload.get("ids", [])
        if not ids:
            return
        original = payload["metadatas"]
        assigned = self._assign_knowledge_metadata(original)
        updates = [(chunk_id, metadata) for chunk_id, before, metadata in zip(ids, original, assigned) if before != metadata]
        # Metadata-only updates preserve the existing embeddings and chunk IDs.
        for offset in range(0, len(updates), 500):
            batch = updates[offset:offset + 500]
            self._store._collection.update(
                ids=[item[0] for item in batch],
                metadatas=[item[1] for item in batch],
            )
        
    def _save_faiss(self):
        """保存FAISS向量库"""
        if self.backend == "faiss":
            self._store.save_local(str(self.persist_directory))
    
    def add_documents(self, documents: list) -> int:
        """
        添加文档到向量库
        :param documents: 文档列表
        :return: 添加的文档数量
        """
        # 每个 Document 都包含正文 page_content 和来源等 metadata。
        # 这里先补齐权限与知识库信息，再把正文转换为向量并持久化。
        if not documents:
            return 0
        try:
            metadatas = self._assign_knowledge_metadata([doc.metadata for doc in documents])
            for document, metadata in zip(documents, metadatas):
                document.metadata = metadata
            ids = [hashlib.sha256(json.dumps([document.metadata.get("version_id"), document.metadata.get("source"),
                document.page_content, index], sort_keys=True, ensure_ascii=True).encode()).hexdigest()
                for index, document in enumerate(documents)]
            if self.backend == "faiss":
                remaining = [(document, chunk_id) for document, chunk_id in zip(documents, ids)
                    if chunk_id not in self._store.docstore._dict]
                if remaining:
                    self._store.add_documents([item[0] for item in remaining], ids=[item[1] for item in remaining])
            elif self.backend == "chroma":
                self._store.add_documents(documents, ids=ids)
            else:
                self._store.add_documents(documents)
            #faiss add_doc 只是写入到内存，所以需要额外的save_faiss 落到磁盘
            if self.backend == "faiss":
                self._save_faiss()
            logger.info(f"成功添加 {len(documents)} 个文档到向量库")
            return len(documents)
        except Exception as e:
            logger.error(f"添加文档到向量库失败: {e}")
            return 0
        
    def delete_by_source(self, source: str) -> int:
        """
        根据源文件删除文档
        :param source: 源文件路径
        :return: 是否删除成功
        """
        if self.backend == "opensearch":
            return self._store.delete_by_source(source)
        if self.backend == "chroma":
            results = self._store.get(where={"source": source})
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                self._store.delete(ids=ids_to_delete)
                logger.info(f"成功删除源文件' {source} '的 {len(ids_to_delete)} 个片段")
                return len(ids_to_delete)
            return 0
        else:
            logger.warning(f"FAISS模式下删除功能比较慢，建议使用Chroma")
            return self._delete_by_source_faiss(source)
        
    def _delete_by_source_faiss(self, source: str) -> int:
        """
        FAISS模式的删除，是需要重构索引的
        """
        doc_ids = list(self._store.docstore._dict.keys())
        to_keep = []
        deleted_count = 0
        for doc_id in doc_ids:
            doc = self._store.docstore.search(doc_id)
            if doc.metadata.get("source") == source:
                deleted_count += 1
            else:
                to_keep.append(doc_id)
        if deleted_count > 0:
            # 重构索引
            self._store = FAISS.from_documents(
                to_keep,
                self.embedding_model
            )
            self._save_faiss()
            logger.info(f"成功删除源文件' {source} '的 {deleted_count} 个片段")
        return deleted_count
    
    def similarity_search(self, query: str, k: int = None, filter_dict: dict = None) -> list:
        """相似度检索：把用户问题转换成向量，找出最接近的 k 个文档片段。"""
        k = k or settings.SEARCH_TOP_K
        if self.backend == "opensearch":
            return self._store.similarity_search(query, k=k, filter=filter_dict)
        if filter_dict and self.backend == "chroma":
            return self._store.similarity_search(query, k=k, filter=filter_dict)
        else:
            return self._store.similarity_search(query, k=k)
        
    def similarity_search_with_score(self, query: str, k: int = None, filter_dict: dict = None) -> list:
        """相似度检索并返回相关性分数，供混合检索和重排使用。"""
        k = k or settings.SEARCH_TOP_K
        if self.backend == "opensearch":
            return self._store.similarity_search_with_score(query, k=k, filter=filter_dict)
        if filter_dict and self.backend == "faiss":
            return self._filtered_faiss_search(query, k, filter_dict)
        if filter_dict and self.backend == "chroma":
            return self._store.similarity_search_with_score(query, k=k, filter=filter_dict)
        else:
            return self._store.similarity_search_with_score(query, k=k)

    def get_documents(self, filter_dict: dict = None) -> dict:
        if self.backend == "opensearch":
            return self._store.get(where=filter_dict)
        if self.backend == "chroma":
            return self._store.get(where=filter_dict, include=["documents", "metadatas"])
        documents = [doc for doc in self._store.docstore._dict.values() if self._metadata_matches(doc.metadata, filter_dict)]
        return {
            "documents": [doc.page_content for doc in documents],
            "metadatas": [doc.metadata for doc in documents],
        }

    @staticmethod
    def _metadata_matches(metadata: dict, filter_dict: dict | None) -> bool:
        comparisons = {
            "$eq": operator.eq, "$ne": operator.ne,
            "$gt": operator.gt, "$gte": operator.ge,
            "$lt": operator.lt, "$lte": operator.le,
        }
        for key, expected in (filter_dict or {}).items():
            if key in {"$and", "$or"}:
                matches = [VectorStoreManager._metadata_matches(metadata, part) for part in expected]
                if not (all(matches) if key == "$and" else any(matches)):
                    return False
                continue
            actual = metadata.get(key)
            if not isinstance(expected, dict):
                if actual != expected:
                    return False
                continue
            for operation, value in expected.items():
                if operation in {"$in", "$nin"}:
                    if (actual in value) != (operation == "$in"):
                        return False
                elif operation in comparisons:
                    try:
                        if not comparisons[operation](actual, value):
                            return False
                    except TypeError:
                        return False
                else:
                    raise ValueError("Unsupported metadata filter: " + operation)
        return True

    def _filtered_faiss_search(self, query: str, k: int, filter_dict: dict) -> list:
        import faiss
        import numpy as np
        from langchain_community.docstore.in_memory import InMemoryDocstore

        selected = []
        for index, doc_id in self._store.index_to_docstore_id.items():
            document = self._store.docstore.search(doc_id)
            if self._metadata_matches(document.metadata, filter_dict):
                selected.append((index, doc_id, document))
        if not selected:
            return []
        # Search only allowed vectors; the original index is never modified.
        metric = self._store.index.metric_type
        if metric == faiss.METRIC_L2:
            index = faiss.IndexFlatL2(self._store.index.d)
        elif metric == faiss.METRIC_INNER_PRODUCT:
            index = faiss.IndexFlatIP(self._store.index.d)
        else:
            raise ValueError("Unsupported FAISS metric for permission prefiltering")
        index.add(np.asarray([self._store.index.reconstruct(item[0]) for item in selected], dtype=np.float32))
        subset = FAISS(
            embedding_function=self.embedding_model,
            index=index,
            docstore=InMemoryDocstore({item[1]: item[2] for item in selected}),
            index_to_docstore_id={offset: item[1] for offset, item in enumerate(selected)},
            normalize_L2=self._store._normalize_L2,
            distance_strategy=self._store.distance_strategy,
        )
        return subset.similarity_search_with_score(query, k=k)

    def get_collection_stats(self) -> dict:
        """ 获取向量库统计信息 """
        if self.backend == "opensearch":
            return {"total_documents": self._store.count(), "vector_store_type": "opensearch", "index": self._store.index}
        if self.backend == "chroma":
            count =  self._store._collection.count()
            return {
                "total_documents": count,
                "vector_store_type": "chroma",
                "persist_directory": str(self.persist_directory)
            }
        else:
            count = self._store.index.ntotal
            return {
                "total_documents": count,
                "vector_store_type": "faiss",
                "persist_directory": str(self.persist_directory)
            }
        
    def clear_all(self):
        """ 清空向量库 """
        if self.backend == "opensearch":
            self._store.clear()
        elif self.backend == "chroma":
            self._store.delete_collection()
            self._initialize_vector_store()
        else:
            shutil.rmtree(self.persist_directory, ignore_errors=True)
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._initialize_vector_store()
        logger.info("向量库已清空")

# 全局单例
_vector_store_instance: VectorStoreManager = None

def get_vector_store_manager() -> VectorStoreManager:
    """ 获取向量库管理器单例 """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreManager()
    return _vector_store_instance
