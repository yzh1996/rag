"""固定评测输入，并把可重复的程序检查与模型评分分开保存。"""
import hashlib
import json
import re
from pathlib import Path

from config.settings import settings


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode()).hexdigest()


def runtime_config():
    keys = ("LLM_PROVIDER", "OPENAI_MODEL_NAME", "SILICONFLOW_MODEL_NAME", "OLLAMA_MODEL_NAME",
        "EMBEDDING_MODEL_NAME", "VECTOR_STORE_TYPE", "SEARCH_TOP_K", "USE_RERANKER", "RERANKER_MODEL_NAME",
        "SEARCH_MIN_VECTOR_SIMILARITY", "SEARCH_MIN_BM25_SCORE", "RERANK_MIN_SCORE", "QUALITY_GUARD_ENABLED",
        "REQUIRE_EVIDENCE", "ALLOW_GENERAL_ANSWERS", "EVIDENCE_MIN_SOURCES", "CHUNK_SIZE", "CHUNK_OVERLAP",
        "EVAL_MIN_FAITHFULNESS", "EVAL_MIN_ANSWER_RELEVANCY", "EVAL_MIN_CONTEXT_PRECISION",
        "EVAL_MIN_CONTEXT_RECALL", "EVAL_MIN_ANSWER_CORRECTNESS", "OCR_ENABLED", "OCR_PROVIDER", "OCR_LANGUAGE",
        "DOCUMENT_CLEANING_ENABLED", "REMOVE_REPEATED_PAGE_EDGES", "CLARIFICATION_ENABLED", "OCR_DPI", "OCR_MIN_TEXT_CHARS",
        "AGENT_ENABLED", "AGENT_MAX_ROUNDS", "AGENT_MAX_TOOL_CALLS", "AGENT_TIMEOUT_SECONDS")
    values = {key: getattr(settings, key) for key in keys}
    values["pipeline_revision"] = fingerprint({name: (Path(__file__).parent / name).read_text(encoding="utf-8")
        for name in ("rag_chain.py", "conversation_agent.py", "agent_tools.py", "quality_guard.py", "retriever.py", "document_loader.py", "chunking_strategies.py", "content_processing.py", "clarification.py")})
    return values


def freeze_cases(cases):
    fields = ("id", "question", "expected_answer", "expected_sources", "history", "category", "tags",
        "expected_behavior", "require_citations", "access_context")
    return [{key: getattr(case, key) for key in fields} for case in cases]


def contract_checks(case, result):
    sources = result.get("sources", [])
    found = {str(item.get(key)) for item in sources for key in ("source", "document_id", "version_id") if item.get(key)}
    expected = set(case.get("expected_sources") or [])
    behavior = result.get("response_type", "answer" if result.get("evidence_sufficient") else "refuse")
    citations = {int(item) for item in re.findall(r"\[(\d+)\]", result.get("answer", ""))}
    checks = {
        "execution_ok": result.get("status", "completed") == "completed",
        "behavior_matches": behavior == case.get("expected_behavior", "answer"),
        "expected_sources_found": not expected or expected.issubset(found),
        "citations_supported": not case.get("require_citations") or (
            result.get("citation_status") == "verified" and bool(citations)
            and citations.issubset({item.get("citation_id") for item in sources})),
        "refusal_has_no_unsupported_answer": behavior != "refuse" or result.get("evidence_sufficient") is False,
    }
    return checks
