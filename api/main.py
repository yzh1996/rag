"""
FastAPI应用入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config.settings import settings
from config.logging_config import setup_logging
from api.routes import (
    upload_router,
    qa_router,
    knowledge_router,
    system_router,
    evaluation_router,
    auth_router,
    documents_router,
    jobs_router,
    index_router,
    feedback_router,
    operations_router,
)
import time
from core.metrics import increment, observe
from api.routes.retrieval import router as retrieval_router

# 配置日志
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # lifespan 会在 FastAPI 启动时先执行初始化代码，yield 之后的代码负责关闭服务。
    # 这里集中准备数据库、向量库和后台任务，所以接口收到请求时依赖已经就绪。
    from core.database import init_db
    from core.telemetry import initialize, shutdown
    initialize()
    # 启动时初始化
    init_db()
    from core.vector_store import get_vector_store_manager
    get_vector_store_manager()  # 预加载向量库
    from core.ingestion_service import recover_pending_jobs
    recover_pending_jobs()
    from core.upload_service import recover_upload_tasks
    recover_upload_tasks()
    from core.task_queue import start_local_worker, stop_local_worker, recover_evaluations
    recover_evaluations()
    from core.release_service import recover_index_builds
    recover_index_builds()
    start_local_worker()
    try:
        yield
    finally:
        shutdown()
        stop_local_worker()
    # 关闭时清理
    pass


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.middleware("http")
async def collect_request_metrics(request, call_next):
    from core.telemetry import span
    from opentelemetry.propagate import extract

    started_at = time.perf_counter()
    # 流式首字耗时从请求进入后台开始计时，包含请求校验、依赖初始化和知识检索。
    request.state.started_at = started_at
    try:
        with span("http.request", context=extract(dict(request.headers))) as current:
            response = await call_next(request)
            current.set_attribute("http.request.method", request.method)
            current.set_attribute("http.response.status_code", response.status_code)
            route = request.scope.get("route")
            if route:
                current.set_attribute("http.route", route.path)
        increment("http_requests_total")
        if response.status_code >= 500:
            increment("http_server_errors_total")
        return response
    except Exception:
        increment("http_requests_total")
        increment("http_server_errors_total")
        raise
    finally:
        observe("http_request", (time.perf_counter() - started_at) * 1000)

# 配置CORS
"""
全称是跨域资源共享，它是浏览器的一种安全机制。默认情况下，在http://localhost:8501上运行的前端应用程序
不能访问在http://localhost:8000上运行的后端API，因为它们两个属于不同的源（端口不一样）。
CORS配置的作用就是告诉浏览器 "我这个后端服务器允许来自于localhost:8501 的前端应用程序访问我的API"
人话：公司的门禁系统，CORS就是允许访问的白名单。
我们常说的 加白 就是这个意思。
"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
# 路由可以理解为“网址到处理函数的映射”。统一加上 /api/v1，便于版本管理。
app.include_router(upload_router, prefix="/api/v1")
app.include_router(qa_router, prefix="/api/v1")
app.include_router(knowledge_router, prefix="/api/v1")
app.include_router(system_router, prefix="/api/v1")
app.include_router(evaluation_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")
app.include_router(index_router, prefix="/api/v1")
app.include_router(feedback_router, prefix="/api/v1")
app.include_router(operations_router, prefix="/api/v1")
app.include_router(retrieval_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": f"欢迎使用 {settings.PROJECT_NAME}",
        "version": settings.PROJECT_VERSION,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
