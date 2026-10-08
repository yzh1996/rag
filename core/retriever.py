"""
检索模块 —— 支持相似度检索 + 可选的重排序
主要功能：
    1. 基于向量相似度进行文档检索
    2. 可选的文档重排序（使用Reranker）
        使用重排序的原因是为了提高检索结果的相关性和准确性。
        这属于对RAG技术的优化，主要就是为了提高检索结果的质量。
        但是会增加一些额外的开销。
"""
import logging
from config.settings import settings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents.compressor import BaseDocumentCompressor
from core.vector_store import get_vector_store_manager

logger = logging.getLogger(__name__)

class CrossEncoderReranker(BaseDocumentCompressor):
    """基于交叉编码器的重排序器"""
    model_name: str
    top_k: int
    model: object = None  # 延迟加载模型
    # def __init__(self, model_name: str, top_k: int = 5):
    #     """初始化重排序器"""
    #     super().__init__()
    #     self.model_name = model_name or settings.RERANKER_MODEL_NAME
    #     self.top_k = top_k
    #     # 延迟初始化模型，避免启动时加载耗时！！！！！！
        
    def _initialize_model(self):
        """初始化交叉编码器模型，使用时才第一次加载"""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.model_name)
                logger.info(f"交叉编码器模型已初始化: {self.model_name}")
            except Exception as e:
                logger.error(f"初始化交叉编码器模型失败: {str(e)}")
                self.model = None

    def compress_documents(self, documents, query, **kwargs):
        """重排序核心方法：对文档列表按与查询query的相关性排序"""
        if not documents:
            return []
        # 初始化模型
        self._initialize_model()
        if self.model is None:
            # 模型加载失败，直接返回原文档，不做排序
            return documents[:self.top_k]
        try:
            # 构建 查询-文档对 理解为qa对
            pairs = [[query, doc.page_content] for doc in documents]
            # 使用交叉编码器模型 进行打分
            scores = self.model.predict(pairs)
            # 按分数降序排序
            sorted_docs = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True
            )
            # 返回前top_k个文档
            return [doc for doc, _ in sorted_docs[:self.top_k]]
        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            return documents[:self.top_k]
        
    

class RAGRetriever:
    """RAG检索器，封装向量检索和（可选的）重排序功能"""
    def __init__(self):
        self.vector_store = get_vector_store_manager()
        self.reranker = self._initialize_reranker()

    def _initialize_reranker(self):
        """初始化Reranker"""
        if not settings.USE_RERANKER:
            logging.info("未启用Reranker")
            return None
        try:
            # 尝试本地加载，本地没有的话再huggingface下载
            model_name = settings.RERANKER_MODEL_NAME
            local_model_path = settings.MODELS_DIR / model_name.replace("/", "_")
            if local_model_path.exists():
                model_path = str(local_model_path)
                logger.info(f"从本地加载Reranker模型: {local_model_path}")
            else:
                model_path = model_name
                logger.info(f"从HuggingFace下载Reranker模型: {model_name}")
            # 创建重排序实例
            reranker = CrossEncoderReranker(
                model_name=model_path,
                top_k=settings.SEARCH_TOP_K
            )
            logger.info(f"Reranker已初始化: {settings.RERANKER_MODEL_NAME}")
            return reranker
        except Exception as e:
            logging.error(f"初始化Reranker失败: {str(e)}")
            return None
        
    def retrieve(self, query: str, top_k: int = None, filter_dict: dict = None, return_score: bool = False) -> list:
        """
        检索文档
        :param query: 查询文本
        :param top_k: 返回的文档数量
        :param filter_dict: 过滤条件
        :param return_score: 是否返回相关性分数
        :return: 检索到的文档列表
        """
        top_k = top_k or settings.SEARCH_TOP_K
        # 1. 执行向量检索
        docs_with_score = self.vector_store.similarity_search_with_score(
            query,
            k=top_k,
            filter_dict=filter_dict
        )
        # 2. 如果启用了Reranker，则进行重排序
        if self.reranker:
            raw_docs = [doc for doc, _ in docs_with_score]
            ranked_docs = self.reranker.compress_documents(raw_docs, query)
            # 重新匹配原分数
            ranked_docs_with_score = [
                (doc, next(score for doc_, score in docs_with_score if doc_ == doc))
                for doc in ranked_docs
            ]
        else:
            ranked_docs_with_score = docs_with_score
    
        # 3. 返回结果
        if return_score:
            return ranked_docs_with_score
        else:
            return [doc for doc, _ in ranked_docs_with_score]
        
    def get_compresstion_retriever(self, search_kwargs: dict):
        """
        获取适配Langchain链的检索器
        """
        search_kwargs = search_kwargs or {"k": settings.SEARCH_TOP_K}
        # 获取基础向量检索器
        base_retriever = self.vector_store._store.as_retriever(
            search_kwargs=search_kwargs
        )
        # 如果启用了Reranker，则创建压缩检索器
        if self.reranker:
            return ContextualCompressionRetriever(
                base_compressor=self.reranker,
                base_retriever=base_retriever
            )
        else:
            return base_retriever


# 单例实例（全局唯一，避免重复加载）
_rag_retriever_instance = None

def get_rag_retriever() -> RAGRetriever:
    """获取全局唯一的RAG检索器实例"""
    global _rag_retriever_instance
    if _rag_retriever_instance is None:
        _rag_retriever_instance = RAGRetriever()
    return _rag_retriever_instance