"""Langfuse追踪适配层。

Langfuse未配置或SDK不可用时，所有方法都会降级为空操作，不影响在线问答。
"""
import logging
import time
import uuid
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


class LangfuseTracker:
    def __init__(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ):
        self.trace_id = trace_id or uuid.uuid4().hex
        self._client = None
        self._trace = None
        self._callback_handler = None
        self._started_at = time.perf_counter()
        self._session_id = session_id
        self._metadata = metadata or {}

        if not settings.LANGFUSE_ENABLED:
            return
        if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
            logger.warning("LANGFUSE_ENABLED=true但密钥未配置，已跳过追踪")
            return
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
            self._trace = self._client.trace(
                id=self.trace_id,
                name="rag.ask",
                session_id=session_id,
                metadata=self._metadata,
            )
            from langfuse.callback import CallbackHandler

            self._callback_handler = CallbackHandler(
                stateful_client=self._trace,
                update_stateful_client=True,
                enabled=True,
            )
        except Exception as exc:
            logger.warning("Langfuse初始化失败，已降级为本地记录: %s", exc)
            self._client = None
            self._trace = None

    def finish(
        self,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        model_name: str | None,
        error: str | None = None,
        first_token_ms: float | None = None,
    ) -> None:
        if not self._trace:
            return
        try:
            # 与监控一致：记录后台接收请求至首段可见正文的耗时，不记录问题内容作为指标。
            timings = {"first_token_ms": first_token_ms} if first_token_ms is not None else {}
            capture_content = settings.LANGFUSE_CAPTURE_CONTENT
            if capture_content:
                max_length = settings.LANGFUSE_MAX_CONTENT_LENGTH
                input_value = question[:max_length]
                output_value = answer[:max_length]
            else:
                input_value = {"length": len(question)}
                output_value = {"length": len(answer)}
            generation = self._trace.generation(
                name="rag.answer",
                model=model_name,
                input=input_value,
                output=output_value,
                metadata={"source_count": len(sources), "error": error},
            )
            generation.end(
                output=output_value,
                metadata={"latency_ms": round((time.perf_counter() - self._started_at) * 1000, 2), **timings},
            )
            self._trace.update(
                output=output_value,
                metadata={"source_count": len(sources), "error": error, **timings},
            )
            if self._callback_handler:
                self._callback_handler.flush()
            self._client.flush()
        except Exception as exc:
            logger.warning("Langfuse记录问答失败: %s", exc)

    def callbacks(self) -> list[Any]:
        """返回LangChain回调；未启用Langfuse时返回空列表。"""
        return [self._callback_handler] if self._callback_handler else []

    def score_feedback(self, rating: str, comment: str | None = None) -> None:
        if not self._client:
            return
        try:
            self._client.score(
                trace_id=self.trace_id,
                name="user_feedback",
                value=1 if rating == "positive" else 0,
                comment=comment,
            )
            self._client.flush()
        except Exception as exc:
            logger.warning("Langfuse记录反馈失败: %s", exc)
