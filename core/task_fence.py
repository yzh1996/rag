"""过期任务不能提交业务状态，避免旧执行进程覆盖新的结果。"""
from contextvars import ContextVar
from datetime import datetime, timezone

from sqlalchemy import select

current_task = ContextVar("rag_current_task", default=None)


def verify_lease(db):
    value = current_task.get()
    if value is None:
        return
    from core.db_models import BackgroundJob
    job_id, token = value
    job = db.scalar(select(BackgroundJob).where(BackgroundJob.id == job_id,
        BackgroundJob.status == "running", BackgroundJob.lease_token == token)
        .execution_options(populate_existing=True).with_for_update())
    if not job or not job.lease_until or job.lease_until.replace(tzinfo=timezone.utc) <= datetime.now(timezone.utc):
        raise RuntimeError("任务租约已失效，取消旧任务提交")
