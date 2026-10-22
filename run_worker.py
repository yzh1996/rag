"""启动独立后台任务服务，使用与API相同的数据库和文件目录。"""
import signal
import threading

from core.database import init_db
from core.task_queue import run_worker


if __name__ == "__main__":
    init_db()
    from core.ingestion_service import recover_pending_jobs
    from core.task_queue import recover_evaluations
    from core.release_service import recover_index_builds
    from core.upload_service import recover_upload_tasks
    recover_pending_jobs()
    recover_evaluations()
    recover_index_builds()
    recover_upload_tasks()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    run_worker(stop)
