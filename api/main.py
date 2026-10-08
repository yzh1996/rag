"""
FastAPI应用入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config.settings import settings
from config.logging_config import setup_logging
from api.routes import upload_router, qa_router, knowledge_router, system_router

# 配置日志
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    from core.vector_store import get_vector_store_manager
    get_vector_store_manager()  # 预加载向量库
    yield
    # 关闭时清理
    pass


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

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
app.include_router(upload_router, prefix="/api/v1")
app.include_router(qa_router, prefix="/api/v1")
app.include_router(knowledge_router, prefix="/api/v1")
app.include_router(system_router, prefix="/api/v1")


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