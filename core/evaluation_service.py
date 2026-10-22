"""Ragas评测执行服务。

该服务同时被命令行脚本和评测页面调用，确保两种入口使用同一套逻辑。
"""
import logging
import math
from datetime import datetime, timezone
from typing import Any

from ragas import EvaluationDataset, evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from config.settings import settings
from core.database import init_db, session_scope
from core.db_models import EvalResult, EvalRun
from core.phase1_services import list_eval_cases
from core.rag_chain import get_rag_chain
from core.auth import AuthUser

logger = logging.getLogger(__name__)


def _safe_metrics(values: dict[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key, value in values.items():
        try:
            numeric = float(value)
            result[key] = numeric if math.isfinite(numeric) else None
        except (TypeError, ValueError):
            result[key] = None
    return result


def _case_passed(score: dict[str, Any], has_reference: bool) -> bool:
    """按配置阈值判断单题是否通过，避免把非空评分对象误判为通过。"""
    metrics = _safe_metrics(score or {})
    thresholds = {
        "faithfulness": settings.EVAL_MIN_FAITHFULNESS,
        "answer_relevancy": settings.EVAL_MIN_ANSWER_RELEVANCY,
    }
    if has_reference:
        thresholds.update(
            {
                "context_precision": settings.EVAL_MIN_CONTEXT_PRECISION,
                "context_recall": settings.EVAL_MIN_CONTEXT_RECALL,
            }
        )
    return all(metrics.get(name) is not None and metrics[name] >= threshold for name, threshold in thresholds.items())


def _merge_progress(config: dict[str, Any] | None, **updates: Any) -> dict[str, Any]:
    merged = dict(config or {})
    progress = dict(merged.get("progress") or {})
    progress.update(updates)
    merged["progress"] = progress
    return merged


def _update_progress(run_id: str, **updates: Any) -> None:
    with session_scope() as db:
        run = db.get(EvalRun, run_id)
        if run:
            run.config = _merge_progress(run.config, **updates)


def _mark_failed(run_id: str, error: str) -> None:
    with session_scope() as db:
        run = db.get(EvalRun, run_id)
        if run:
            run.status = "failed"
            config = _merge_progress(
                run.config,
                phase="failed",
                current_case=None,
            )
            config["error"] = error
            run.config = config
            run.completed_at = datetime.now(timezone.utc)


def execute_eval_run(run_id: str, limit: int = 0, user: AuthUser | None = None) -> None:
    """Execute the frozen evaluation suite using full retrieval evidence."""
    from core.evaluation_runner import run_evaluation
    run_evaluation(run_id, limit, user)
