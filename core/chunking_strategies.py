"""文档分块策略。"""
import uuid
import re
import hashlib
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


STRATEGY_LABELS = {
    "recursive": "递归字符切分",
    "fixed": "固定长度切分",
    "structure": "结构感知切分",
    "semantic": "语义切分",
    "parent_child": "父子分块",
}

DEFAULT_PARAMS = {
    "recursive": {"chunk_size": 500, "chunk_overlap": 50},
    "fixed": {"chunk_size": 500, "chunk_overlap": 50},
    "structure": {"chunk_size": 800, "chunk_overlap": 50, "preserve_headers": True},
    "semantic": {"breakpoint_threshold": 75, "min_chunk_size": 200, "max_chunk_size": 800},
    "parent_child": {"parent_size": 1000, "child_size": 300, "child_overlap": 50},
}

_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]


def _positive_int(params: dict, name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是整数")
    if value < minimum:
        raise ValueError(f"{name} 必须大于等于 {minimum}")
    return value


def _splitter(size: int, overlap: int, splitter_cls=RecursiveCharacterTextSplitter):
    if overlap >= size:
        raise ValueError("重叠大小必须小于分块大小")
    kwargs = {
        "chunk_size": size,
        "chunk_overlap": overlap,
        "length_function": len,
        "strip_whitespace": False,
    }
    if splitter_cls is RecursiveCharacterTextSplitter:
        kwargs["separators"] = _SEPARATORS
    else:
        kwargs["separator"] = ""
    return splitter_cls(**kwargs)


def _limit_chunk_size(documents: list[Document], max_size: int, overlap: int) -> list[Document]:
    if max_size <= 0:
        return documents
    splitter = _splitter(max_size, min(overlap, max_size - 1))
    result = []
    for document in documents:
        if len(document.page_content) <= max_size:
            result.append(document)
            continue
        result.extend(splitter.split_documents([document]))
    return result


def _structure_split(documents: list[Document], file_ext: str, params: dict) -> list[Document]:
    max_size = _positive_int(params, "chunk_size", DEFAULT_PARAMS["structure"]["chunk_size"])
    overlap = _positive_int(params, "chunk_overlap", DEFAULT_PARAMS["structure"]["chunk_overlap"], 0)
    preserve_headers = bool(params.get("preserve_headers", True))

    if file_ext == ".md":
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=not preserve_headers,
        )
        structured = []
        for document in documents:
            parts = header_splitter.split_text(document.page_content)
            for part in parts:
                part.metadata.update(document.metadata)
                structured.append(part)
    else:
        structured = []
        heading_pattern = re.compile(
            r"^(?:第.+?(?:章|节|部分|篇)|[一二三四五六七八九十百千万]+[、.]|\d+[.、]|[A-Z][.、])"
        )
        for document in documents:
            current_lines = []
            current_heading = ""

            def flush_section():
                if not current_lines:
                    return
                part = Document(
                    page_content="\n".join(current_lines).strip(),
                    metadata=dict(document.metadata),
                )
                if current_heading:
                    part.metadata["structure_header"] = current_heading
                structured.append(part)

            for line in document.page_content.splitlines():
                stripped = line.strip()
                if not stripped:
                    flush_section()
                    current_lines = []
                    continue
                is_heading = bool(heading_pattern.match(stripped))
                if is_heading and current_lines:
                    flush_section()
                    current_lines = []
                if is_heading:
                    current_heading = stripped
                if preserve_headers or not is_heading:
                    current_lines.append(stripped)
            flush_section()

        if not structured:
            structured = documents

    return _limit_chunk_size(structured, max_size, overlap)


def _parent_child_split(documents: list[Document], params: dict) -> list[Document]:
    parent_size = _positive_int(params, "parent_size", DEFAULT_PARAMS["parent_child"]["parent_size"])
    child_size = _positive_int(params, "child_size", DEFAULT_PARAMS["parent_child"]["child_size"])
    child_overlap = _positive_int(
        params, "child_overlap", DEFAULT_PARAMS["parent_child"]["child_overlap"], 0
    )
    if child_size > parent_size:
        raise ValueError("子块大小不能大于父块大小")

    parent_splitter = _splitter(parent_size, 0)
    child_splitter = _splitter(child_size, child_overlap)
    children = []
    for parent_index, parent in enumerate(parent_splitter.split_documents(documents), 1):
        parent_id = hashlib.sha256(f"{parent.metadata.get('source')}:{parent_index}:{parent.page_content}".encode()).hexdigest()[:24]
        parent_content = parent.page_content
        for child_index, child in enumerate(child_splitter.split_documents([parent]), 1):
            child.metadata.update({
                "parent_id": parent_id,
                "parent_index": parent_index,
                "child_index": child_index,
                "parent_content": parent_content,
            })
            children.append(child)
    return children


def split_documents(
    documents: list[Document],
    strategy: str = "recursive",
    params: Optional[dict] = None,
    file_ext: str = "",
    embeddings=None,
) -> list[Document]:
    """使用指定策略切分文档，返回可直接入向量库的子文档。

    文档不能整篇直接塞给模型，通常要先切成较小片段。切得太大可能超出模型上下文，
    切得太小又可能丢失上下文，所以这里把不同切分策略统一成一个入口。
    """
    params = params or {}
    if strategy not in STRATEGY_LABELS:
        raise ValueError(f"不支持的分块方式: {strategy}")

    if strategy == "recursive":
        # 优先按段落、换行和标点切分，尽量让一个片段保持自然语义。
        size = _positive_int(params, "chunk_size", DEFAULT_PARAMS[strategy]["chunk_size"])
        overlap = _positive_int(params, "chunk_overlap", DEFAULT_PARAMS[strategy]["chunk_overlap"], 0)
        return _splitter(size, overlap).split_documents(documents)

    if strategy == "fixed":
        # 固定长度切分更简单可控，但不一定尊重原文的段落边界。
        size = _positive_int(params, "chunk_size", DEFAULT_PARAMS[strategy]["chunk_size"])
        overlap = _positive_int(params, "chunk_overlap", DEFAULT_PARAMS[strategy]["chunk_overlap"], 0)
        return _splitter(size, overlap, CharacterTextSplitter).split_documents(documents)

    if strategy == "structure":
        # 结构感知切分会利用 Markdown 标题或中文章节标题保留文档结构。
        return _structure_split(documents, file_ext, params)

    if strategy == "parent_child":
        # 父子分块：用较小的子块参与检索，命中后再补回较大的父块作为上下文。
        return _parent_child_split(documents, params)

    # 语义切分需要嵌入模型比较相邻句子的语义变化。
    if embeddings is None:
        raise ValueError("语义切分需要可用的嵌入模型")
    from langchain_experimental.text_splitter import SemanticChunker

    threshold = float(params.get("breakpoint_threshold", DEFAULT_PARAMS["semantic"]["breakpoint_threshold"]))
    if not 0 < threshold < 100:
        raise ValueError("语义阈值必须在 0 到 100 之间")
    min_size = _positive_int(params, "min_chunk_size", DEFAULT_PARAMS["semantic"]["min_chunk_size"])
    max_size = _positive_int(params, "max_chunk_size", DEFAULT_PARAMS["semantic"]["max_chunk_size"])
    if min_size > max_size:
        raise ValueError("语义切分的最小片段大小不能大于最大片段大小")
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold,
        min_chunk_size=min_size,
        sentence_split_regex=r"(?<=[。！？.!?])",
    )
    return _limit_chunk_size(semantic_splitter.split_documents(documents), max_size, 0)
