"""
向量库管理模块——封装chroma或者faiss操作，提供统一接口
通过embedding模块获取嵌入模型，转换为向量后，使用chroma或者faiss存储到本地向量库持久化存储，方便后续的检索与查询等操作。
"""
import logging
import shutil
from langchain.schema import Document
from langchain_community.vectorstores import Chroma, FAISS
from config.settings import settings
from core.embedding import get_embedding_model

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """向量库管理器"""
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self._store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """初始化向量库"""
        if settings.VECTOR_STORE_TYPE == "chroma":
            self._store = Chroma(
                persist_directory=str(settings.VECTOR_DB_DIR),
                embedding_function=self.embedding_model
            )
            logger.info(f"Chroma向量库已初始化，存储路径: {settings.VECTOR_DB_DIR}")
        elif settings.VECTOR_STORE_TYPE == "faiss":
            faiss_index_path = settings.VECTOR_DB_DIR / "index.faiss"
            if faiss_index_path.exists():
                self._store = FAISS.load_local(
                    str(settings.VECTOR_DB_DIR),
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"FAISS向量库已加载，存储路径: {settings.VECTOR_DB_DIR}")
            else:
                self._store = FAISS.from_documents(
                    [Document(page_content="初始化向量库", metadata={"source", "init"})],
                    self.embedding_model
                )
                self._save_faiss()
                logger.info(f"FAISS向量库已保存，存储路径: {settings.VECTOR_DB_DIR}")
        else:
            raise ValueError(f"不支持的向量库类型: {settings.VECTOR_STORE_TYPE}")
        
    def _save_faiss(self):
        """保存FAISS向量库"""
        if settings.VECTOR_STORE_TYPE == "faiss":
            self._store.save_local(str(settings.VECTOR_DB_DIR))
    
    def add_documents(self, documents: list) -> int:
        """
        添加文档到向量库
        :param documents: 文档列表
        :return: 添加的文档数量
        """
        if not documents:
            return 0
        try:
            self._store.add_documents(documents)
            if settings.VECTOR_STORE_TYPE == "faiss":
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
        if settings.VECTOR_STORE_TYPE == "chroma":
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
        """ 相似度检索 """
        k = k or settings.SEARCH_TOP_KNone
        if filter_dict and settings.VECTOR_STORE_TYPE == "chroma":
            return self._store.similarity_search(query, k=k, filter=filter_dict)
        else:
            return self._store.similarity_search(query, k=k)
        
    def similarity_search_with_score(self, query: str, k: int = None, filter_dict: dict = None) -> list:
        """ 相似度检索并返回相关性分数 """
        k = k or settings.SEARCH_TOP_K
        if filter_dict and settings.VECTOR_STORE_TYPE == "chroma":
            return self._store.similarity_search_with_score(query, k=k, filter=filter_dict)
        else:
            return self._store.similarity_search_with_score(query, k=k)

    def get_collection_stats(self) -> dict:
        """ 获取向量库统计信息 """
        if settings.VECTOR_STORE_TYPE == "chroma":
            count =  self._store._collection.count()
            return {
                "total_documents": count,
                "vector_store_type": "chroma",
                "persist_directory": str(settings.VECTOR_DB_DIR)
            }
        else:
            count = self._store.index.ntotal
            return {
                "total_documents": count,
                "vector_store_type": "faiss",
                "persist_directory": str(settings.VECTOR_DB_DIR)
            }
        
    def clear_all(self):
        """ 清空向量库 """
        if settings.VECTOR_STORE_TYPE == "chroma":
            self._store.delete_collection()
            self._initialize_vector_store()
        else:
            shutil.rmtree(settings.VECTOR_DB_DIR, ignore_errors=True)
            settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
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