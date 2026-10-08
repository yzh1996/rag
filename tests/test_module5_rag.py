"""
测试模块5：RAG整合链的功能
"""
# 想要导入自定义包或者模块，建议将项目根目录加入系统路径
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rag_chain import get_rag_chain

def test_rag_chain_ask(question: str):
    """测试RAG问答链的基本功能"""
    print("\n=== 测试RAG问答链 ===")
    rag_chain = get_rag_chain()
    result = rag_chain.ask(question)
    print(f"\n问题: {question}")
    print(f"答案: {result['answer']}")
    print("\n来源:")
    for source in result['sources']:
        print(f"- {source['source']} (页码: {source['page']})")
        print(f"  内容: {source['content']}")

if __name__ == "__main__":
    test_rag_chain_ask("我是张三，我想加班")
    print("-"*100, "\n")
    test_rag_chain_ask("怎么申请？")
    print("-"*100, "\n")
    test_rag_chain_ask("还记得我的名字吗？我加班找谁审批啊？")