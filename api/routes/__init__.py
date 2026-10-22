from .upload import router as upload_router
from .qa import router as qa_router
from .knowledge import router as knowledge_router
from .system import router as system_router
from .evaluation import router as evaluation_router
from .auth import router as auth_router
from .documents import router as documents_router, jobs_router
from .index import router as index_router
from .feedback import router as feedback_router
from .operations import router as operations_router

__all__ = [
    "upload_router",
    "qa_router",
    "knowledge_router",
    "system_router",
    "evaluation_router",
    "auth_router",
    "documents_router",
    "jobs_router",
    "index_router",
    "feedback_router",
    "operations_router",
]
