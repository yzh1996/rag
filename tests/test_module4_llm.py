"""
模块4：测试大模型客户端 与 统一接口检索器是否生效
"""
# 想要导入自定义包或者模块，建议将项目根目录加入系统路径
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from config.logging_config import setup_logging
from core.llm_client import get_llm
from core.retriever import get_rag_retriever

setup_logging()

def test_llm_client():
    """测试大模型客户端"""
    print("\n=== 测试大模型客户端 ===")
    llm = get_llm()
    if llm:
        print("✅ 大模型客户端初始化成功！")
        print(f"当前使用的模型: {settings.LLM_PROVIDER}")
        response = llm.invoke("请用一句话介绍一下人工智能")
        print(f"模型回复: {response.content}")
    else:
        print("❌ 大模型客户端初始化失败！")


def test_retriever():
    """测试统一接口检索器"""
    print("\n=== 测试统一接口检索器 ===")
    retriever = get_rag_retriever()
    if retriever:
        print("✅ 检索器初始化成功！")
        query = "企业内部规章制度"
        print(f"\n检索问题: {query}")
        results = retriever.retrieve(query, top_k=3)
        if results:
            print("\n检索结果:")
            for i, doc in enumerate(results, 1):
                print(f"\n--- 结果 {i} ---")
                print(f"内容: {doc.page_content[:100]}...")
                print(f"来源: {doc.metadata.get('source', '未知')}")
        else:
            print("未找到相关文档")
    else:
        print("❌ 检索器初始化失败！")

if __name__ == "__main__":
    test_llm_client()
    test_retriever()