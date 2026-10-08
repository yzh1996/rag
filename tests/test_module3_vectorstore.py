"""
模块3：验证嵌入模型和向量库的存储、删除、检索功能
"""
# 想要导入自定义包或者模块，建议将项目根目录加入系统路径
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from config.logging_config import setup_logging
from core.vector_store import get_vector_store_manager
from core.embedding import get_embedding_model
from core.document_loader import DocumentLoader

setup_logging()

def test_embedding():
    """测试嵌入模型"""
    print("\n=== 测试嵌入模型 ===")
    model = get_embedding_model()
    text = "这是一段测试文本"
    embedding = model.embed_query(text)
    print(f"嵌入向量维度: {len(embedding)}")
    print(f"嵌入向量前5个值: {embedding[:5]}")
    return embedding

def test_vectorstore():
    """测试向量库存储、删除、检索功能"""
    print("\n=== 测试向量库 ===")
    # 加载测试文档并分块
    sample_file = Path(__file__).parent / "sample_docs" / "test.docx"
    # sample_docs = Path(__file__).parent / "sample_docs"
    loader = DocumentLoader()
    docs = loader.load_file(str(sample_file))
    print(f"加载了 {len(docs)} 个文档片段")

    # 初始化向量库
    vector_store = get_vector_store_manager()

    # 清空旧数据
    vector_store.clear_all()

    # 添加文档
    print("\n=== 添加文档 ===")
    added = vector_store.add_documents(docs)
    print(f"成功添加 {added} 个文档片段")

    # 测试检索
    print("\n=== 测试检索 ===")
    query = "加班"
    print(f"检索的问题: {query}")
    results = vector_store.similarity_search(query, k=3)
    if results:
        print("\n检索结果:")
        for i, doc in enumerate(results, 1):
            print(f"\n--- 结果 {i} ---")
            print(f"内容: {doc.page_content[:100]}...")
            print(f"来源: {doc.metadata.get('source', '未知')}")
    else:
        print("未找到相关文档")



if __name__ == "__main__":
    test_embedding()
    test_vectorstore()

