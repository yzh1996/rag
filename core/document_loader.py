"""
文档加载器模块——支持 pdf word excel txt markdown 等多种格式的文档加载
步骤：
1. 加载文档：使用合适的加载器，可以选择是否添加元数据。 
    什么是元数据？——就是描述数据的数据，例如：文件名、文件类型、作者等等等
    添加元数据，优点就是能更好的管理数据。
2. 分割文档：将文档分割成多个片段，以便于后续的处理。
"""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from config.settings import settings
import logging
logger = logging.getLogger(__name__)
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    PDFPlumberLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.chunking_strategies import split_documents

class DocumentLoader:
    """统一文档加载器类"""
    # 支持的文档扩展名映射
    LOADER_MAP= {
        ".pdf": PyMuPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader
    }

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            length_function=len
        )

    def load_file(self, file_path: str, metadata: dict = None, strategy: str = "recursive", params: dict = None, loader_name: str = None) -> list:
        """
        加载单个文件
        :param file_path: 文件路径
        :param metadata: 可选的元数据字典
        :return: 文档片段列表
        """
        # 普通分块不需要嵌入模型；只有“语义切分”会在切分阶段比较句子含义。
        embeddings = None
        if strategy == "semantic":
            from core.embedding import get_embedding_model
            embeddings = get_embedding_model()
        _, split_docs = self.load_file_with_original(file_path, metadata, strategy, params, embeddings, loader_name)
        return split_docs

    def load_raw_file(self, file_path: str, metadata: dict = None, loader_name: str = None) -> list:
        """只解析文件；OCR、清洗和分块由调用方分别执行。"""
        from core.file_safety import validate_file
        from core.content_processing import extract_pdf
        path = Path(file_path)
        validate_file(path)
        ext = path.suffix.lower()
        # 根据扩展名选择解析器，把 PDF、Word、Excel 等格式统一转换成 Document 对象。
        loader_class = self.LOADER_MAP[ext]
        if ext == ".pdf":
            documents = extract_pdf(str(path), loader_name=loader_name)
            parser_name = next((doc.metadata.get("loader") for doc in documents if doc.metadata.get("loader")), loader_class.__name__)
        else:
            loader = loader_class(str(path), encoding="utf-8") if ext == ".txt" else loader_class(str(path))
            documents = loader.load()
            parser_name = loader_class.__name__
        base_metadata = {"source": str(path), "file_name": path.name, "file_type": ext[1:], "loader": parser_name}
        if ext == ".docx":
            with ZipFile(path) as archive:
                header_text = "".join(
                    "".join(node.text or "" for node in ET.fromstring(archive.read(name)).iter(
                        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
                    ))
                    for name in archive.namelist()
                    if re.fullmatch(r"word/header[0-9]+\.xml", name)
                ).lstrip()
            if header_text:
                base_metadata["_docx_header_prefix"] = header_text
        base_metadata.update(metadata or {})
        for doc in documents:
            doc.metadata.update(base_metadata)
        return documents

    def load_file_with_original(
        self,
        file_path: str,
        metadata: dict = None,
        strategy: str = "recursive",
        params: dict = None,
        embeddings=None,
        loader_name: str = None,
    ) -> tuple:
        """
        加载文件并返回原文和分块结果。
        :return: (原文内容, 文档片段列表)
        """
        # 这是上传流程使用的总入口：先读取原文，再清洗、补充元数据，最后分块。
        # 返回原文是为了让前端展示预览；返回 split_docs 是为了后续写入向量库。
        # 1. 先检查文件是否存在
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return "", []
        # 2. 检查文件类型
        ext = file_path.suffix.lower()
        if ext not in self.LOADER_MAP:
            logger.error(f"不支持的文件类型: {ext}")
            return "", []
        try:
            from core.content_processing import recognize_documents, clean_documents
            # 3. 加载文档
            documents = self.load_raw_file(str(file_path), metadata, loader_name=loader_name)
            documents = recognize_documents(str(file_path), documents)
            documents = clean_documents(documents)
            logger.info(f"成功加载文件: {file_path}， 原始文档书段数: {len(documents)}")
            # 4. 添加自定义元数据
            base_metadata= {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": ext[1:],       # 过滤 .txt 的 . 保留 txt
            }
            if metadata:
                base_metadata.update(metadata)
            for doc in documents:
                doc.metadata.update(base_metadata)
            original_content = "\n\n".join(doc.page_content for doc in documents)
            # 5. 按指定策略分割文档
            if strategy == "recursive" and not params:
                split_docs = self.text_splitter.split_documents(documents)
            else:
                split_docs = split_documents(
                    documents,
                    strategy=strategy,
                    params=params,
                    file_ext=ext,
                    embeddings=embeddings,
                )
            logger.info(f"成功分割文档: {file_path}， 分割后的文档数: {len(split_docs)}")
            return original_content, split_docs
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"加载文件失败: {file_path}, 错误: {e}")
            raise ValueError(f"文件解析失败，请检查格式和解析依赖：{e}") from e
        
    def load_directory(self, dir_path: str, recursive: bool = True) -> list:
        """
        加载整个目录下的所有文件
        :param dir_path: 目录路径
        :param recursive: 是否递归加载子目录
        :return: 文档片段列表
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.error(f"目录不存在: {dir_path}")
            return []
        # 遍历目录下的所有文件
        all_docs = []
        patten = "**/*" if recursive else "*"
        for file_path in dir_path.rglob(patten):
            if file_path.is_file() and file_path.suffix.lower() in self.LOADER_MAP:
                docs = self.load_file(str(file_path))
                all_docs.extend(docs)
        logger.info(f"成功加载目录: {dir_path}, 共 {len(all_docs)} 个文档片段")
        return all_docs

