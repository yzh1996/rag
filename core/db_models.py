"""RAG闭环第一阶段的数据模型。"""
from datetime import date, datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import (JSON, Boolean, CheckConstraint, Date, DateTime, DDL, Float, ForeignKey,
                        Index, Integer, String, Text, UniqueConstraint, event)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid4().hex


class QAInteraction(Base):
    __tablename__ = "qa_interactions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    trace_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    query_details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    history: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="completed")
    model_provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    username: Mapped[str] = mapped_column(String(128), default="anonymous")
    tenant_id: Mapped[str] = mapped_column(String(128), default="default", index=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    feedback: Mapped[list["Feedback"]] = relationship(
        back_populates="interaction",
        cascade="all, delete-orphan",
    )


class LiveSales(Base):
    """一个房间下某位主播在同一近30天快照内的销量，单位：件。"""

    __tablename__ = "live_sales"
    __table_args__ = (CheckConstraint("sales_quantity >= 0"), CheckConstraint("period_end >= period_start"))

    room_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    anchor_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    sales_quantity: Mapped[int] = mapped_column(Integer)
    period_start: Mapped[date] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class VideoSales(Base):
    """一个视频在同一近30天快照内的销量，单位：件。"""

    __tablename__ = "video_sales"
    __table_args__ = (CheckConstraint("sales_quantity >= 0"), CheckConstraint("period_end >= period_start"))

    video_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    anchor_id: Mapped[str] = mapped_column(String(64), index=True)
    sales_quantity: Mapped[int] = mapped_column(Integer)
    period_start: Mapped[date] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Feedback(Base):
    __tablename__ = "qa_feedback"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    interaction_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("qa_interactions.id", ondelete="CASCADE"), index=True
    )
    rating: Mapped[str] = mapped_column(String(16))
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_resolved: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    handoff_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(32), default="open")
    assignee: Mapped[str | None] = mapped_column(String(128), nullable=True)
    resolution_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    promoted_case_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    interaction: Mapped[QAInteraction] = relationship(back_populates="feedback")


class EvalCase(Base):
    __tablename__ = "eval_cases"

    tenant_id: Mapped[str] = mapped_column(String(128), default="default", index=True)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    dataset_name: Mapped[str] = mapped_column(
        String(128), default="default", server_default="default", index=True
    )
    question: Mapped[str] = mapped_column(Text)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_sources: Mapped[list[str]] = mapped_column(JSON, default=list)
    history: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    expected_behavior: Mapped[str] = mapped_column(String(16), default="answer")
    require_citations: Mapped[bool] = mapped_column(Boolean, default=False)
    access_context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class EvalRun(Base):
    __tablename__ = "eval_runs"

    tenant_id: Mapped[str] = mapped_column(String(128), default="default", index=True)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    dataset_name: Mapped[str] = mapped_column(String(128), default="default")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    total_cases: Mapped[int] = mapped_column(Integer, default=0)
    passed_cases: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    results: Mapped[list["EvalResult"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True
    )
    case_id: Mapped[str] = mapped_column(String(32), ForeignKey("eval_cases.id"), index=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_sources: Mapped[list[str]] = mapped_column(JSON, default=list)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    run: Mapped[EvalRun] = relationship(back_populates="results")


class User(Base):
    """用户和知识权限的最小持久化模型。"""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    role: Mapped[str] = mapped_column(String(64), default="user")
    department: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tenant_id: Mapped[str] = mapped_column(String(128), default="default", index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    tenant_id: Mapped[str] = mapped_column(String(128), default="default", index=True)
    permission_scope: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    upload_scope: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    owner_department: Mapped[str | None] = mapped_column(String(128), nullable=True)


class Document(Base):
    """文档的逻辑实体，负责生命周期和当前版本指针。"""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    source_uri: Mapped[str] = mapped_column(String(1024))
    original_filename: Mapped[str] = mapped_column(String(512))
    category: Mapped[str] = mapped_column(String(128), default="general")
    knowledge_base_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    owner_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(128), default="default", index=True)
    permission_scope: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    upload_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    current_version: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    jobs: Mapped[list["IngestionJob"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


# utf8mb4 makes a 1024-character unique index exceed MySQL's 3072-byte limit.
# Preserve full paths and enforce uniqueness through a generated SHA-256 value.
_source_uri_unique = UniqueConstraint("source_uri", name="uq_documents_source_uri")
_source_uri_unique.ddl_if(dialect=("sqlite", "postgresql"))
Document.__table__.append_constraint(_source_uri_unique)
Index("ix_documents_source_uri", Document.source_uri).ddl_if(dialect=("sqlite", "postgresql"))
event.listen(Document.__table__, "after_create", DDL(
    "ALTER TABLE documents ADD COLUMN source_uri_sha256 CHAR(64) "
    "GENERATED ALWAYS AS (SHA2(source_uri,256)) STORED, "
    "ADD UNIQUE INDEX uq_documents_source_uri_sha256 (source_uri_sha256)"
).execute_if(dialect="mysql"))


class DocumentVersion(Base):
    """可追踪、可下线、可恢复的文档版本。"""

    __tablename__ = "document_versions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    file_path: Mapped[str] = mapped_column(String(1024))
    chunking_strategy: Mapped[str] = mapped_column(String(64), default="recursive")
    chunking_params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    prepared_chunks: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    effective_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    document: Mapped[Document] = relationship(back_populates="versions")


class IngestionJob(Base):
    """入库任务状态，服务重启后可以继续处理。"""

    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    version_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("document_versions.id", ondelete="CASCADE"), index=True
    )
    job_type: Mapped[str] = mapped_column(String(64), default="ingest")
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    document: Mapped[Document] = relationship(back_populates="jobs")


class UploadTask(Base):
    """通过上传检查后的内容处理和分块检查点。"""

    __tablename__ = "upload_tasks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    username: Mapped[str] = mapped_column(String(128), index=True)
    actor: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    filename: Mapped[str] = mapped_column(String(512))
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    knowledge_base_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    upload_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    permission_scope: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="checking", index=True)
    stages: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    reservation_key: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    extracted_documents: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    raw_documents: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    cleaned_documents: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    document_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    version_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    strategy: Mapped[str] = mapped_column(String(64), default="recursive")
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    generation: Mapped[int] = mapped_column(Integer, default=0)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class IndexVersion(Base):
    """索引候选、发布和回滚记录。"""

    __tablename__ = "index_versions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(128), default="default", index=True)
    version: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    embedding_model: Mapped[str] = mapped_column(String(255))
    chunking_strategy: Mapped[str] = mapped_column(String(64), default="recursive")
    vector_store: Mapped[str] = mapped_column(String(64), default="chroma")
    snapshot_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="candidate", index=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ReleaseState(Base):
    __tablename__ = "release_states"
    tenant_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    active_index_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    previous_index_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    canary_index_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    canary_percent: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class BackgroundJob(Base):
    __tablename__ = "background_jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    task_key: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    lease_token: Mapped[str | None] = mapped_column(String(32), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class ChatSessionState(Base):
    __tablename__ = "chat_session_states"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    username: Mapped[str] = mapped_column(String(128), index=True)
    session_id: Mapped[str] = mapped_column(String(128))
    cleared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    latest_interaction_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class OperationalAlert(Base):
    __tablename__ = "operational_alerts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    fingerprint: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(128))
    severity: Mapped[str] = mapped_column(String(32))
    summary: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
