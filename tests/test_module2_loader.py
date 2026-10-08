"""
模块2测试：验证文档加载器和文本分块的功能
"""
# 想要导入自定义包或者模块，建议将项目根目录加入系统路径
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.document_loader import DocumentLoader
from config.settings import settings
from config.logging_config import setup_logging
setup_logging()

def test_document_loader():
    """测试文档加载器功能"""
    print("\n=== 测试文档加载器 ===")

    sample_file = Path(__file__).parent / "sample_docs" / "test.docx"

    if not sample_file.exists():
        print("⚠️ 测试文件不存在，请先运行模块1测试创建。")
        return

    # 初始化加载器
    loader = DocumentLoader(chunk_size=200, chunk_overlap=20)
    # 加载文件
    docs = loader.load_file(str(sample_file))
    print(f"📄 加载了 {len(docs)} 个文档片段")

    # 打印前3个片段
    for i, doc in enumerate(docs[:3], 1):
        print(f"\n--- 片段 {i} ---")
        print(f"内容: {doc.page_content[:100]}...")
        print(f"来源: {doc.metadata.get('source', '未知')}")

if __name__ == "__main__":
    test_document_loader()
    