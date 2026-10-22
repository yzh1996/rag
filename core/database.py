"""RAG闭环业务数据库。

支持本机MySQL、SQLite和PostgreSQL；MySQL业务数据使用字段字典中的表。
"""
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from config.settings import settings

logger = logging.getLogger(__name__)
Base = declarative_base()

_engine_kwargs = {"pool_pre_ping": True}
if settings.DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(settings.DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def init_db() -> None:
    """创建第一阶段所需的业务表。"""
    from core import db_models  # noqa: F401

    tables = [table for table in Base.metadata.sorted_tables
              if engine.dialect.name != "mysql" or table.name not in {"live_sales", "video_sales"}]
    Base.metadata.create_all(bind=engine, tables=tables)
    _ensure_schema_compatibility()
    from core.knowledge_scope import assign_document_knowledge_bases

    with session_scope() as db:
        assign_document_knowledge_bases(db)
    logger.info("RAG业务数据库已初始化")


def _ensure_schema_compatibility() -> None:
    """为已有本地数据库补充新增字段，避免升级后旧数据无法读取。"""
    inspector = inspect(engine)
    additions = {
        "chat_session_states": {"latest_interaction_id": "VARCHAR(32)"},
        "qa_interactions": {"username": "VARCHAR(128) NOT NULL DEFAULT 'anonymous'", "tenant_id": "VARCHAR(128) NOT NULL DEFAULT 'default'", "input_tokens": "INTEGER NOT NULL DEFAULT 0", "output_tokens": "INTEGER NOT NULL DEFAULT 0", "cost_usd": "FLOAT NOT NULL DEFAULT 0", "query_details": "JSON"},
        "qa_feedback": {"corrected_answer": "TEXT", "is_resolved": "BOOLEAN", "handoff_requested": "BOOLEAN NOT NULL DEFAULT FALSE", "status": "VARCHAR(32) NOT NULL DEFAULT 'open'", "assignee": "VARCHAR(128)", "resolution_notes": "TEXT", "promoted_case_id": "VARCHAR(32)", "updated_at": "TIMESTAMP"},
        "eval_cases": {"tenant_id": "VARCHAR(128) NOT NULL DEFAULT 'default'", "expected_behavior": "VARCHAR(16) NOT NULL DEFAULT 'answer'", "require_citations": "BOOLEAN NOT NULL DEFAULT FALSE", "access_context": "JSON"},
        "eval_runs": {"tenant_id": "VARCHAR(128) NOT NULL DEFAULT 'default'"},
        "index_versions": {"tenant_id": "VARCHAR(128) NOT NULL DEFAULT 'default'"},
        "document_versions": {"created_by": "VARCHAR(128)", "prepared_chunks": "JSON"},
        "documents": {"upload_metadata": "JSON"},
        "knowledge_bases": {"upload_scope": "JSON", "owner_department": "VARCHAR(128)"},
        "upload_tasks": {"deleted_at": "TIMESTAMP"},
    }
    with engine.begin() as connection:
        if inspector.has_table("qa_interactions") and "history" not in {column["name"] for column in inspector.get_columns("qa_interactions")}:
            connection.execute(text("ALTER TABLE qa_interactions ADD COLUMN history JSON"))
        for table, fields in additions.items():
            if not inspector.has_table(table):
                continue
            existing = {column["name"] for column in inspector.get_columns(table)}
            for field, definition in fields.items():
                if field not in existing:
                    connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {field} {definition}"))
    if inspector.has_table("documents"):
        document_columns = {column["name"] for column in inspector.get_columns("documents")}
        if "knowledge_base_id" not in document_columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE documents ADD COLUMN knowledge_base_id VARCHAR(64)"))
    if not inspector.has_table("eval_cases"):
        return

    columns = {column["name"] for column in inspector.get_columns("eval_cases")}
    if "dataset_name" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE eval_cases "
                    "ADD COLUMN dataset_name VARCHAR(128) NOT NULL DEFAULT 'default'"
                )
            )


def get_db() -> Generator[Session, None, None]:
    """FastAPI数据库依赖。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """后台任务或服务层使用的事务会话。"""
    db = SessionLocal()
    try:
        yield db
        from core.task_fence import verify_lease
        verify_lease(db)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
