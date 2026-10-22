"""持久化数据库任务队列，支持独立进程、租约、自动重试和恢复。"""
import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, or_, select, update
from sqlalchemy.exc import IntegrityError

from config.settings import settings
from core.database import session_scope
from core.db_models import BackgroundJob

logger = logging.getLogger(__name__)
_stop = threading.Event()
_thread = None


def now():
    return datetime.now(timezone.utc)


def enqueue(kind, key, payload):
    with session_scope() as db:
        existing = db.scalar(select(BackgroundJob).where(BackgroundJob.task_key == key))
        if existing:
            if existing.status == "failed":
                existing.status, existing.attempts = "pending", 0
                existing.available_at, existing.error_message = now(), None
            return existing.id
        try:
            with db.begin_nested():
                job = BackgroundJob(task_key=key, kind=kind, payload=payload)
                db.add(job)
                db.flush()
        except IntegrityError:
            job = db.scalar(select(BackgroundJob).where(BackgroundJob.task_key == key))
        return job.id


def enqueue_ingestion(job_id):
    return enqueue("ingest", "ingest:" + job_id, {"job_id": job_id})


def enqueue_evaluation(run_id, limit=0, user=None):
    from core.auth import actor_payload
    actor = actor_payload(user) if user else None
    return enqueue("evaluate", "evaluate:" + run_id, {"run_id": run_id, "limit": limit, "user": actor})


def claim():
    timestamp, token = now(), uuid.uuid4().hex
    eligible = or_(and_(BackgroundJob.status == "pending", BackgroundJob.available_at <= timestamp),
        and_(BackgroundJob.status == "running", BackgroundJob.lease_until < timestamp))
    with session_scope() as db:
        db.execute(update(BackgroundJob).where(eligible, BackgroundJob.attempts >= settings.TASK_MAX_ATTEMPTS)
            .values(status="failed", error_message="任务重试次数已用完"))
        candidate = db.scalar(select(BackgroundJob.id).where(eligible, BackgroundJob.attempts < settings.TASK_MAX_ATTEMPTS)
            .order_by(BackgroundJob.created_at).limit(1))
        if not candidate:
            return None
        claimed = db.execute(update(BackgroundJob).where(BackgroundJob.id == candidate, eligible,
            BackgroundJob.attempts < settings.TASK_MAX_ATTEMPTS).values(status="running", lease_token=token,
                lease_until=timestamp + timedelta(seconds=settings.TASK_LEASE_SECONDS), attempts=BackgroundJob.attempts + 1))
        if claimed.rowcount != 1:
            return None
        return db.get(BackgroundJob, candidate)


def heartbeat(job_id, token):
    with session_scope() as db:
        return db.execute(update(BackgroundJob).where(BackgroundJob.id == job_id,
            BackgroundJob.status == "running", BackgroundJob.lease_token == token, BackgroundJob.lease_until > now())
            .values(lease_until=now() + timedelta(seconds=settings.TASK_LEASE_SECONDS))).rowcount == 1


def finish(job, error=None):
    with session_scope() as db:
        values = {"lease_token": None, "lease_until": None, "error_message": str(error)[:4000] if error else None}
        if error:
            values["status"] = "pending" if job.attempts < settings.TASK_MAX_ATTEMPTS else "failed"
            values["available_at"] = now() + timedelta(seconds=min(300, 5 * 2 ** job.attempts))
        else:
            values["status"] = "completed"
        return db.execute(update(BackgroundJob).where(BackgroundJob.id == job.id,
            BackgroundJob.lease_token == job.lease_token, BackgroundJob.status == "running",
            BackgroundJob.lease_until > now()).values(**values)).rowcount == 1


def dispatch(job):
    if job.kind in {"upload_preprocess", "upload_chunks"}:
        from core.upload_service import process_preprocessing, process_chunks
        handler = process_chunks if job.kind == "upload_chunks" else process_preprocessing
        handler(job.payload["task_id"], job.payload["generation"])
    elif job.kind == "ingest":
        from core.ingestion_service import process_ingestion_job
        from core.db_models import IngestionJob
        process_ingestion_job(job.payload["job_id"])
        with session_scope() as db:
            item = db.get(IngestionJob, job.payload["job_id"])
            if not item or item.status != "completed":
                raise RuntimeError(item.error_message if item else "入库任务不存在")
    elif job.kind == "evaluate":
        from core.auth import AuthUser
        from core.evaluation_service import execute_eval_run
        from core.db_models import EvalRun
        actor = AuthUser(**job.payload["user"]) if job.payload.get("user") else None
        execute_eval_run(job.payload["run_id"], job.payload.get("limit", 0), actor)
        with session_scope() as db:
            item = db.get(EvalRun, job.payload["run_id"])
            if not item or item.status != "completed":
                raise RuntimeError((item.config or {}).get("error", "评测失败") if item else "评测不存在")
    elif job.kind == "build_index":
        from core.auth import AuthUser
        from core.release_service import build_index_job
        build_index_job(job.payload["index_id"], AuthUser(**job.payload["user"]))
    else:
        raise ValueError("未知后台任务类型")


def run_once(handler=None):
    job = claim()
    if not job:
        return False
    done = threading.Event()
    def renew():
        while not done.wait(max(1, settings.TASK_LEASE_SECONDS / 3)):
            try:
                if not heartbeat(job.id, job.lease_token):
                    return
            except Exception:
                logger.exception("任务续约失败")
    renewer = threading.Thread(target=renew, daemon=True)
    renewer.start()
    try:
        from core.task_fence import current_task
        context_token = current_task.set((job.id, job.lease_token))
        try:
            (handler or dispatch)(job)
        finally:
            current_task.reset(context_token)
    except Exception as exc:
        logger.exception("后台任务失败: %s", job.task_key)
        finish(job, exc)
    else:
        finish(job)
    finally:
        done.set()
        renewer.join(timeout=2)
    return True


def run_worker(stop=None):
    signal = stop or _stop
    while not signal.is_set():
        try:
            worked = run_once()
        except Exception:
            logger.exception("任务队列暂时不可用")
            worked = False
        if not worked:
            signal.wait(settings.TASK_POLL_SECONDS)


def start_local_worker():
    global _thread
    if settings.TASK_EXECUTION_MODE == "inline" and (_thread is None or not _thread.is_alive()):
        _stop.clear()
        _thread = threading.Thread(target=run_worker, daemon=True, name="rag-task-worker")
        _thread.start()


def recover_evaluations():
    from core.db_models import EvalRun
    with session_scope() as db:
        runs = db.scalars(select(EvalRun).where(EvalRun.status.in_(["pending", "running"]))).all()
    for run in runs:
        if (run.config or {}).get("case_snapshot"):
            enqueue("evaluate", "evaluate:" + run.id, {"run_id": run.id,
                "limit": (run.config or {}).get("limit", 0), "user": (run.config or {}).get("actor")})


def stop_local_worker():
    _stop.set()
    if _thread:
        _thread.join(timeout=5)
