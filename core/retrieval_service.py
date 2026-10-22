"""供外部 Agent 使用的已发布知识检索；不生成答案、不接受候选索引参数。"""
import ntpath
import uuid


class RetrievalUnavailable(Exception):
    def __init__(self, code, message, status=503):
        self.code, self.message, self.status = code, message, status
        super().__init__(message)


def search_published(query, top_k, task_id, user):
    # Lazy imports avoid preloading embeddings before an authenticated request.
    from core.release_service import release_store, resolve_release
    from core.retriever import RAGRetriever

    try:
        release = resolve_release(user, task_id)
    except ValueError:
        raise RetrievalUnavailable("KNOWLEDGE_RELEASE_UNAVAILABLE", "当前知识发布版本不可用", 409)
    if (release is None or not isinstance((release.config or {}).get("manifest"), dict)
            or not release.config["manifest"]):
        raise RetrievalUnavailable("KNOWLEDGE_RELEASE_UNAVAILABLE", "尚无可用的已发布知识版本", 409)
    manifest = release.config["manifest"]
    try:
        documents = RAGRetriever(vector_store=release_store(release)).retrieve(
            query, top_k=top_k, user=user, version_manifest=manifest)
    except Exception:
        raise RetrievalUnavailable("KNOWLEDGE_RETRIEVAL_FAILED", "知识检索暂不可用，请稍后重试")
    items, remaining, truncated = [], 24000, False
    for document in documents[:top_k]:
        metadata = document.metadata
        # Defence in depth: never return a chunk outside the selected release manifest.
        if (not metadata.get("document_id") or not metadata.get("version_id")
                or manifest.get(metadata["document_id"]) != metadata["version_id"]):
            continue
        if remaining <= 0:
            truncated = True
            break
        content = document.page_content[:min(6000, remaining)]
        clipped = len(content) < len(document.page_content)
        source = str(metadata.get("original_filename") or metadata.get("file_name") or metadata.get("source") or "未知")
        # Both Windows and POSIX paths are reduced to a display filename.
        source = ntpath.basename(source.replace("\\", "/"))[:256] or "未知"
        page = metadata.get("page")
        if not isinstance(page, (int, str)) or isinstance(page, bool):
            page = None
        items.append({"content": content, "document_id": str(metadata["document_id"]),
            "version_id": str(metadata["version_id"]), "source": source, "page": page, "truncated": clipped})
        remaining -= len(content)
        truncated = truncated or clipped
    return {"trace_id": uuid.uuid4().hex, "index_version": release.version,
            "knowledge_view": "release", "items": items, "truncated": truncated}
