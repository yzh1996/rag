"""
FastAPI依赖注入模块
"""
from core.vector_store import get_vector_store_manager
from core.rag_chain import get_rag_chain
from core.document_loader import DocumentLoader


def get_vector_store_dep():
    """获取向量库依赖"""
    return get_vector_store_manager()


def get_rag_chain_dep():
    """获取RAG链依赖"""
    return get_rag_chain()


def get_document_loader_dep():
    """获取文档加载器依赖"""
    return DocumentLoader()