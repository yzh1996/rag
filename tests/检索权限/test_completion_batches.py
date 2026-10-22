"""闭环补齐的隔离回归测试，不读取线上模型或业务数据。"""
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_core.documents import Document as Chunk

from config.settings import settings
from core import database
from core.auth import AuthUser
from core.db_models import Document, DocumentVersion, KnowledgeBase
from core.knowledge_scope import accessible_knowledge_filter
from core.vector_store import VectorStoreManager


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'closed-loop.db'}", connect_args={"check_same_thread": False})
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(engine, expire_on_commit=False))
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", False)
    monkeypatch.setattr(settings, "VECTOR_STORE_TYPE", "chroma")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "VECTOR_DB_DIR", tmp_path / "vectors")
    monkeypatch.setattr(settings, "INDEX_SNAPSHOT_DIR", tmp_path / "snapshots")
    database.init_db()
    yield
    engine.dispose()


def seed_document(status="published"):
    with database.session_scope() as db:
        db.add(KnowledgeBase(id="kb", name="policies", tenant_id="a", permission_scope={"tenant_ids": ["a"]}))
        doc = Document(id="doc", source_uri="policy.txt", original_filename="policy.txt", tenant_id="a",
            knowledge_base_id="kb", permission_scope={"tenant_ids": ["a"]}, status=status, current_version=2)
        db.add(doc)
        db.flush()
        db.add_all([
            DocumentVersion(id="v1", document_id=doc.id, version=1, content_hash="1" * 64, file_path="old.txt", status="archived"),
            DocumentVersion(id="v2", document_id=doc.id, version=2, content_hash="2" * 64, file_path="new.txt", status="published"),
        ])


def test_only_effective_version_is_retrievable_and_downline_takes_effect(isolated):
    seed_document()
    user = AuthUser(username="alice", tenant_id="a")
    filters = accessible_knowledge_filter(user)
    assert not VectorStoreManager._metadata_matches({"knowledge_base_id": "kb", "document_id": "doc", "version_id": "v1"}, filters)
    assert VectorStoreManager._metadata_matches({"knowledge_base_id": "kb", "document_id": "doc", "version_id": "v2"}, filters)
    with database.session_scope() as db:
        db.get(Document, "doc").status = "deprecated"
    assert not VectorStoreManager._metadata_matches({"knowledge_base_id": "kb", "document_id": "doc", "version_id": "v2"}, accessible_knowledge_filter(user))


def test_same_session_name_cannot_read_or_clear_other_users(isolated):
    from core.memory_manager import MemoryManager, session_key
    from core.rag_chain import RAGChain
    chain = object.__new__(RAGChain)
    chain.memory_manager = MemoryManager()
    alice = AuthUser(username="alice", tenant_id="a")
    bob = AuthUser(username="bob", tenant_id="a")
    from core.db_models import QAInteraction
    with database.session_scope() as db:
        db.add(QAInteraction(session_id="same", username="alice", tenant_id="a", question="private", answer="answer"))
    assert chain.get_chat_history("same", user=bob) == []
    chain.clear_session("same", user=bob)
    assert chain.get_chat_history("same", user=alice)[0]["content"] == "private"


def test_history_and_clear_require_token_when_auth_enabled(isolated, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routes.qa import router
    from api.dependencies import get_rag_chain_dep
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_rag_chain_dep] = lambda: SimpleNamespace()
    client = TestClient(app)
    assert client.get("/qa/history").status_code == 401
    assert client.post("/qa/clear_memory").status_code == 401


def test_semantic_ingestion_supplies_embedding_model(isolated, monkeypatch):
    from core.document_loader import DocumentLoader
    from core import embedding
    model = object()
    monkeypatch.setattr(embedding, "get_embedding_model", lambda: model)
    received = {}
    def preview(self, path, metadata, strategy, params, embeddings):
        received["model"] = embeddings
        return "sample", [Chunk(page_content="sample")]
    monkeypatch.setattr(DocumentLoader, "load_file_with_original", preview)
    assert DocumentLoader().load_file("sample.txt", strategy="semantic")
    assert received["model"] is model


def test_frozen_evaluation_uses_history_full_evidence_and_behavior_checks(isolated, monkeypatch):
    from core import evaluation_runner
    from core.db_models import EvalCase, EvalRun, EvalResult
    from core.phase1_services import create_eval_run
    from sqlalchemy import select
    received = []
    full = "evidence " * 200
    class Chain:
        llm = object()
        retriever = SimpleNamespace(vector_store=SimpleNamespace(embedding_model=object()))
        def ask(self, question, **kwargs):
            received.append((question, kwargs))
            if question == "deny":
                return {"answer": "refused", "sources": [], "response_type": "refuse", "evidence_sufficient": False}
            return {"answer": "supported [1]", "sources": [{"document_id": "doc", "citation_id": 1}],
                "retrieved_contexts": [full], "response_type": "answer", "evidence_sufficient": True, "citation_status": "verified"}
    with database.session_scope() as db:
        db.add_all([EvalCase(tenant_id="a", question="answer", expected_answer="supported", expected_sources=["doc"],
            require_citations=True, history=[{"role": "user", "content": "previous"}]),
            EvalCase(tenant_id="a", question="deny", expected_behavior="refuse")])
    with database.session_scope() as db:
        run = create_eval_run(db, "default", {}, tenant_id="a")
        run_id = run.id
        db.scalar(select(EvalCase).where(EvalCase.question == "answer")).question = "changed later"
    monkeypatch.setattr(evaluation_runner, "get_rag_chain", Chain)
    def score(dataset, **kwargs):
        assert dataset.samples[0].retrieved_contexts == [full]
        return SimpleNamespace(scores=[dict(faithfulness=1, answer_relevancy=1, context_precision=1, context_recall=1, answer_correctness=1)])
    monkeypatch.setattr(evaluation_runner, "evaluate", score)
    evaluation_runner.run_evaluation(run_id)
    assert received[0][0] == "answer"
    assert received[0][1]["history"] == [{"role": "user", "content": "previous"}]
    assert received[0][1]["persist"] is False
    with database.session_scope() as db:
        assert db.get(EvalRun, run_id).metrics["quality_gate_passed"] is True
        assert len(db.scalars(select(EvalResult)).all()) == 2


def test_missing_expected_source_and_citation_cannot_pass():
    from core.evaluation_contract import contract_checks
    checks = contract_checks({"expected_sources": ["correct"], "require_citations": True},
        {"answer": "unsupported [2]", "sources": [{"document_id": "other", "citation_id": 1}],
            "response_type": "answer", "citation_status": "verified"})
    assert checks["expected_sources_found"] is False
    assert checks["citations_supported"] is False


def test_candidates_are_isolated_gate_is_bound_and_canary_can_roll_back(isolated, monkeypatch):
    from langchain_core.embeddings import Embeddings
    from core import release_service
    from core.db_models import EvalRun, ReleaseState
    from core.evaluation_contract import runtime_config
    from core.retriever import RAGRetriever
    class FakeEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[1.0, 0.0] for text in texts]
        def embed_query(self, text):
            return [1.0, 0.0]
    monkeypatch.setattr(settings, "USE_RERANKER", False)
    monkeypatch.setattr(release_service, "_stores", {})
    seed_document()
    source = VectorStoreManager(embedding_model=FakeEmbeddings())
    source.add_documents([Chunk(page_content="old policy", metadata={"document_id": "doc", "version_id": "v2"})])
    monkeypatch.setattr(release_service, "get_vector_store_manager", lambda: source)
    user = AuthUser(username="alice", tenant_id="a", role="knowledge_admin", authenticated=True)
    first = release_service.create_index_snapshot(user=user)
    release_service._stores.clear()
    monkeypatch.setattr(__import__("core.vector_store", fromlist=["get_embedding_model"]), "get_embedding_model", FakeEmbeddings)
    assert release_service.artifact_hash(release_service.release_store(first)) == first.config["artifact_hash"]
    with pytest.raises(ValueError):
        release_service.publish_index(first.id, user=user)
    def passing_run(item):
        with database.session_scope() as db:
            run = EvalRun(tenant_id="a", status="completed", total_cases=1, passed_cases=1,
                config={"index_id": item.id, "artifact_hash": item.config["artifact_hash"], "runtime": runtime_config(), "dataset_revision": "fixed"},
                metrics={"quality_gate_passed": True, "complete_suite": True, "dataset_revision": "fixed"})
            db.add(run)
            db.flush()
            return run.id
    first_run = passing_run(first)
    with pytest.raises(ValueError):
        release_service.publish_index(first.id, first_run, user=user)
    release_service.review_index(first.id, user, "checked")
    release_service.publish_index(first.id, first_run, user=user)
    with database.session_scope() as db:
        db.add(DocumentVersion(id="v3", document_id="doc", version=3, content_hash="3"*64, file_path="candidate.txt", status="candidate"))
    source.add_documents([Chunk(page_content="new policy", metadata={"document_id": "doc", "version_id": "v3"})])
    assert [chunk.metadata["version_id"] for chunk in RAGRetriever(source).retrieve("policy", user=user)] == ["v2"]
    second = release_service.create_index_snapshot(user=user)
    assert first.snapshot_path != second.snapshot_path
    release_service.review_index(second.id, user, "checked new")
    with pytest.raises(ValueError):
        release_service.publish_index(second.id, first_run, user=user)
    second_run = passing_run(second)
    release_service.publish_index(second.id, second_run, user=user, canary_percent=20)
    assert release_service.resolve_release(user, "same").id == release_service.resolve_release(user, "same").id
    release_service.publish_index(second.id, second_run, user=user)
    assert release_service.resolve_release(user, "same").id == second.id
    release_service.rollback_index(first.id, user=user)
    assert release_service.resolve_release(user, "same").id == first.id
    with database.session_scope() as db:
        assert db.get(Document, "doc").current_version == 2
        assert db.get(ReleaseState, "a").canary_index_id is None


def test_partial_evaluation_never_opens_release_gate(isolated, monkeypatch):
    from core import evaluation_runner
    from core.db_models import EvalCase, EvalRun
    from core.phase1_services import create_eval_run
    with database.session_scope() as db:
        db.add_all([EvalCase(question="one", expected_behavior="refuse"), EvalCase(question="two", expected_behavior="refuse")])
    with database.session_scope() as db:
        run_id = create_eval_run(db, "default", {}).id
    monkeypatch.setattr(evaluation_runner, "get_rag_chain", lambda: SimpleNamespace(ask=lambda *args, **kwargs:
        {"answer": "refusal", "sources": [], "response_type": "refuse", "evidence_sufficient": False}))
    evaluation_runner.run_evaluation(run_id, limit=1)
    with database.session_scope() as db:
        assert db.get(EvalRun, run_id).passed_cases == 1
        assert db.get(EvalRun, run_id).metrics["quality_gate_passed"] is False


def test_queue_recovers_expired_leases_and_rejects_stale_commits(isolated, monkeypatch):
    from datetime import timedelta
    from core import task_queue as queue
    from core.task_fence import current_task
    from core.db_models import BackgroundJob, QAInteraction
    key = queue.enqueue("test", "unique", {})
    assert queue.enqueue("test", "unique", {}) == key
    first = queue.claim()
    assert queue.claim() is None
    with database.session_scope() as db:
        db.get(BackgroundJob, key).lease_until = queue.now() - timedelta(seconds=1)
    second = queue.claim()
    assert second.lease_token != first.lease_token
    assert not queue.finish(first)
    token = current_task.set((first.id, first.lease_token))
    try:
        with pytest.raises(RuntimeError, match="租约"):
            with database.session_scope() as db:
                db.add(QAInteraction(session_id="stale", question="q", answer="a"))
    finally:
        current_task.reset(token)
    assert queue.finish(second, RuntimeError("retry"))
    with database.session_scope() as db:
        job = db.get(BackgroundJob, key)
        assert job.status == "pending"
        job.available_at = queue.now() - timedelta(seconds=1)
    assert queue.run_once(lambda job: None)
    with database.session_scope() as db:
        assert db.get(BackgroundJob, key).status == "completed"
        assert not db.query(QAInteraction).filter_by(session_id="stale").all()


def test_persistent_history_survives_restart_and_clear_invalidates_other_process_memory(isolated):
    from core.memory_manager import MemoryManager, session_key
    from core.rag_chain import RAGChain
    from core.db_models import QAInteraction
    actor = AuthUser(username="alice", tenant_id="a")
    with database.session_scope() as db:
        db.add(QAInteraction(session_id="same", username="alice", tenant_id="a", question="private", answer="answer"))
    first, second = object.__new__(RAGChain), object.__new__(RAGChain)
    first.memory_manager, second.memory_manager = MemoryManager(), MemoryManager()
    second.memory_manager.add_exchange(session_key("same", actor), "stale", "stale")
    assert second.get_chat_history("same", actor)[0]["content"] == "private"
    first.clear_session("same", actor)
    assert second.get_chat_history("same", actor) == []


def test_manual_evaluation_restart_creates_new_frozen_run(isolated, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routes.evaluation import router
    from core.auth import require_admin_or_local
    from core.db_models import EvalCase, EvalRun, BackgroundJob
    actor = AuthUser(username="admin", role="admin", tenant_id="a", authenticated=True)
    with database.session_scope() as db:
        db.add(EvalCase(tenant_id="a", question="refuse", expected_behavior="refuse"))
        run = EvalRun(tenant_id="a", status="completed", config={}, metrics={"old": True})
        db.add(run)
        db.flush()
        old_id = run.id
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_admin_or_local] = lambda: actor
    response = TestClient(app).post(f"/evaluation/runs/{old_id}/start")
    assert response.status_code == 200
    new_id = response.json()["id"]
    assert new_id != old_id
    with database.session_scope() as db:
        assert db.get(EvalRun, old_id).metrics == {"old": True}
        assert db.get(EvalRun, new_id).config["actor"]["username"] == "admin"
        assert db.query(BackgroundJob).filter_by(task_key="evaluate:" + new_id).one().status == "pending"


def test_pdf_text_and_scan_detection_keep_page_evidence(isolated, tmp_path, monkeypatch):
    import fitz
    from core import content_processing
    from core.document_loader import DocumentLoader
    from PIL import Image
    import io
    path = tmp_path / "mixed.pdf"
    image = io.BytesIO()
    Image.new("RGB", (120, 80), "white").save(image, format="PNG")
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((30, 50), "This is a policy with enough embedded text.")
        page = pdf.new_page()
        page.insert_image(fitz.Rect(0, 0, 120, 80), stream=image.getvalue())
        pdf.save(path)
    received = []
    actual_ocr = content_processing.ocr_page
    monkeypatch.setattr(content_processing, "ocr_page", lambda page: received.append(page.number) or "OCR policy")
    original, chunks = DocumentLoader().load_file_with_original(str(path))
    assert received == [1]
    assert "OCR policy" in original
    assert [doc.metadata["page"] for doc in chunks] == [1, 2]
    assert chunks[1].metadata["ocr_used"] is True
    monkeypatch.setattr(settings, "OCR_ENABLED", False)
    with pytest.raises(ValueError, match="OCR"):
        actual_ocr(SimpleNamespace())


def test_cleaning_preserves_body_and_optional_repeated_edges(isolated, monkeypatch):
    from core.content_processing import clean_documents
    pages = [Chunk(page_content=f"HEADER\nbody {number}\x00\nFOOTER", metadata={"page": number}) for number in range(1, 4)]
    assert "HEADER" in clean_documents(pages)[0].page_content
    monkeypatch.setattr(settings, "REMOVE_REPEATED_PAGE_EDGES", True)
    cleaned = clean_documents(pages)
    assert [doc.page_content for doc in cleaned] == ["body 1", "body 2", "body 3"]
    assert [doc.metadata["page"] for doc in cleaned] == [1, 2, 3]


def test_upload_blocks_forged_type_oversize_and_archive_expansion(isolated, tmp_path, monkeypatch):
    from io import BytesIO
    from zipfile import ZipFile, ZIP_DEFLATED
    from core.file_safety import save_upload, validate_file
    bad = tmp_path / "fake.pdf"
    with pytest.raises(ValueError, match="PDF"):
        save_upload(SimpleNamespace(file=BytesIO(b"not a pdf")), bad)
    assert not bad.exists()
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)
    large = tmp_path / "large.txt"
    with pytest.raises(ValueError, match="上限"):
        save_upload(SimpleNamespace(file=BytesIO(b"x" * (1024 * 1024 + 1))), large)
    assert not large.exists()
    archive = tmp_path / "bomb.docx"
    monkeypatch.setattr(settings, "MAX_ARCHIVE_EXPANDED_MB", 1)
    with ZipFile(archive, "w", ZIP_DEFLATED) as output:
        output.writestr("word/document.xml", "x" * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="解压"):
        validate_file(archive)


def test_preview_can_confirm_after_process_restart_and_cannot_be_stolen(isolated, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routes.upload import router
    from api.dependencies import get_vector_store_dep
    from core.auth import get_current_user
    from core.db_models import IngestionJob
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    actor = AuthUser(username="alice", tenant_id="a", authenticated=True)
    def application():
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: actor
        app.dependency_overrides[get_vector_store_dep] = lambda: object()
        return TestClient(app)
    preview = application().post("/upload/file/preview", files={"file": ("policy.txt", b"annual leave policy", "text/plain")})
    assert preview.status_code == 200
    file_id = preview.json()["file_id"]
    actor = AuthUser(username="bob", tenant_id="a", authenticated=True)
    assert application().post("/upload/file/confirm", data={"file_id": file_id}).status_code == 404
    actor = AuthUser(username="alice", tenant_id="a", authenticated=True)
    first = application().post("/upload/file/confirm", data={"file_id": file_id})
    second = application().post("/upload/file/confirm", data={"file_id": file_id})
    assert first.status_code == second.status_code == 200
    assert first.json()["job_id"] == second.json()["job_id"]
    with database.session_scope() as db:
        assert db.query(IngestionJob).count() == 1


def test_parent_context_expands_after_rerank_and_deduplicates_versions(isolated, monkeypatch):
    from core.retriever import RAGRetriever
    import core.retriever as module
    monkeypatch.setattr(module, "accessible_knowledge_filter", lambda actor: {"knowledge_base_id": {"$in": ["kb"]}})
    chunks = [Chunk(page_content=f"child {number}", metadata={"document_id": "missing", "version_id": version,
        "parent_id": "same", "parent_content": "complete " + version}) for number, version in enumerate(["v1", "v1", "v2"])]
    received = []
    reranker = SimpleNamespace(compress_documents=lambda documents, query: received.extend(doc.page_content for doc in documents) or documents)
    instance = object.__new__(RAGRetriever)
    instance.vector_store = SimpleNamespace(similarity_search_with_score=lambda *args, **kwargs: [(doc, 0.1) for doc in chunks],
        get_documents=lambda *args, **kwargs: {"documents": [], "metadatas": []})
    instance.reranker = reranker
    result = instance.get_compresstion_retriever({"k": 5}).invoke("policy")
    assert received == ["child 0", "child 1", "child 2"]
    assert [doc.page_content for doc in result] == ["complete v1", "complete v2"]


def test_ambiguous_question_clarifies_without_generating_answer(isolated, monkeypatch):
    monkeypatch.setattr(settings, "AGENT_ENABLED", False)
    from core.rag_chain import RAGChain
    from core.memory_manager import MemoryManager
    from core.clarification import clarification_for
    from core.intent_router import IntentDecision
    chain = object.__new__(RAGChain)
    # 模型模式下，先由分类器提出追问；这里仅检查后续不会调用业务答案生成器。
    chain.intent_router = SimpleNamespace(classify=lambda *args, **kwargs: IntentDecision(
        route="clarification", intent="待澄清", confidence=0.95, need_clarification=True,
        clarification_message="请说明你想办理哪项业务。", source="llm"))
    chain.memory_manager = MemoryManager()
    chain.intent_recognizer = SimpleNamespace(recognize=lambda question: "操作指导")
    chain._rag_chain = SimpleNamespace(invoke=lambda *args, **kwargs: pytest.fail("ambiguous question must not generate"))
    result = chain.ask("那个怎么弄？", history=[], persist=False)
    assert result["response_type"] == "clarify"
    assert result["status"] == "completed"
    assert clarification_for("那个怎么弄", [object()]) is None
    assert clarification_for("公司报销怎么申请", []) is None


def test_user_can_query_own_feedback_result_only(isolated):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from core.auth import get_current_user
    from core.db_models import QAInteraction, Feedback
    from api.routes.feedback import router
    with database.session_scope() as db:
        for username, tenant in (("alice", "a"), ("bob", "a"), ("alice", "b")):
            row = QAInteraction(session_id="s", question=username, answer="old", username=username, tenant_id=tenant)
            db.add(row)
            db.flush()
            db.add(Feedback(interaction_id=row.id, rating="negative", status="resolved", corrected_answer="reviewed", resolution_notes="fixed"))
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: AuthUser(username="alice", tenant_id="a", authenticated=True)
    items = TestClient(app).get("/feedback/mine").json()["items"]
    assert len(items) == 1
    assert items[0]["corrected_answer"] == "reviewed"
    assert items[0]["resolution_notes"] == "fixed"


def test_independent_workers_claim_one_job_and_retry_budget_is_enforced(isolated, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from datetime import timedelta
    from core import task_queue as queue
    from core.db_models import BackgroundJob
    key = queue.enqueue("test", "parallel", {})
    barrier = threading.Barrier(2)
    def attempt():
        barrier.wait(timeout=5)
        return queue.claim()
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = list(pool.map(lambda unused: attempt(), range(2)))
    assert sum(job is not None for job in jobs) == 1
    job = next(job for job in jobs if job)
    monkeypatch.setattr(settings, "TASK_MAX_ATTEMPTS", 1)
    assert queue.finish(job, RuntimeError("temporary failure"))
    with database.session_scope() as db:
        assert db.get(BackgroundJob, key).status == "failed"
    assert queue.claim() is None


def test_ingestion_retry_does_not_duplicate_vectors(isolated, tmp_path, monkeypatch):
    from langchain_core.embeddings import Embeddings
    from core import ingestion_service
    from core.db_models import IngestionJob
    from core import task_queue
    class Embedding(Embeddings):
        def embed_documents(self, texts):
            return [[1.0, 0.0] for text in texts]
        def embed_query(self, text):
            return [1.0, 0.0]
    source = VectorStoreManager(embedding_model=Embedding())
    monkeypatch.setattr(ingestion_service, "get_vector_store_manager", lambda: source)
    path = tmp_path / "ingest.txt"
    path.write_text("annual leave policy", encoding="utf-8")
    actor = AuthUser(username="alice", tenant_id="a", authenticated=True)
    doc, version = ingestion_service.create_document_version(str(path), original_filename="policy.txt", category="policies", user=actor)
    assert ingestion_service.recover_pending_jobs() == 1
    with database.session_scope() as db:
        job = db.query(IngestionJob).filter_by(version_id=version.id).one()
        job_id = job.id
    assert task_queue.run_once()
    count = len(source.get_documents()["documents"])
    with database.session_scope() as db:
        db.get(IngestionJob, job_id).status = "pending"
    ingestion_service.process_ingestion_job(job_id)
    assert count == len(source.get_documents()["documents"]) == 1


def test_candidate_build_queue_recovers_registration_and_runs_independently(isolated, monkeypatch):
    from langchain_core.embeddings import Embeddings
    from core import release_service, task_queue
    from core.db_models import IndexVersion, BackgroundJob
    class Embedding(Embeddings):
        def embed_documents(self, texts):
            return [[1.0, 0.0] for text in texts]
        def embed_query(self, text):
            return [1.0, 0.0]
    seed_document()
    source = VectorStoreManager(embedding_model=Embedding())
    source.add_documents([Chunk(page_content="policy", metadata={"document_id": "doc", "version_id": "v2"})])
    monkeypatch.setattr(release_service, "get_vector_store_manager", lambda: source)
    monkeypatch.setattr(release_service, "_stores", {})
    actor = AuthUser(username="admin", role="admin", tenant_id="a", authenticated=True)
    item = release_service.register_index_build(user=actor)
    release_service.recover_index_builds()
    release_service.recover_index_builds()
    with database.session_scope() as db:
        assert db.get(IndexVersion, item.id).status == "building"
        assert db.query(BackgroundJob).count() == 1
    assert task_queue.run_once()
    with database.session_scope() as db:
        assert db.get(IndexVersion, item.id).status == "candidate"
        assert db.query(BackgroundJob).one().status == "completed"


def test_permission_changes_take_effect_in_existing_candidate_without_rewrite(isolated, monkeypatch):
    from core.retriever import RAGRetriever
    seed_document()
    before = AuthUser(username="alice", tenant_id="a")
    with database.session_scope() as db:
        db.get(Document, "doc").permission_scope = {"tenant_ids": ["a"], "users": ["bob"]}
    chunk = Chunk(page_content="policy", metadata={"document_id": "doc", "permission_scope": '{"users":["alice"]}'})
    instance = object.__new__(RAGRetriever)
    assert not instance._can_access(chunk, before)
    assert instance._can_access(chunk, AuthUser(username="bob", tenant_id="a"))
    assert not VectorStoreManager._metadata_matches({"knowledge_base_id": "kb", "document_id": "doc", "version_id": "v2"},
        accessible_knowledge_filter(before, {"doc": "v2"}, include_candidates=True))


def test_feedback_promotion_keeps_history_for_multiturn_regression(isolated):
    from core.db_models import QAInteraction
    from core.phase1_services import save_feedback
    from core.feedback_service import promote_feedback
    with database.session_scope() as db:
        history = [{"role": "user", "content": "annual leave"}, {"role": "assistant", "content": "policy"}]
        row = QAInteraction(session_id="s", username="alice", tenant_id="a", question="how many days", answer="old", history=history)
        db.add(row)
        db.flush()
        feedback = save_feedback(db, row.id, "negative")
        assert promote_feedback(db, feedback, "five", "regression").history == history


def test_saved_session_list_is_scoped_to_actor(isolated):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routes.qa import router
    from core.auth import get_current_user
    from core.db_models import QAInteraction
    with database.session_scope() as db:
        for username, tenant, name in (("alice", "a", "mine"), ("bob", "a", "other-user"), ("alice", "b", "other-tenant")):
            db.add(QAInteraction(username=username, tenant_id=tenant, session_id=name, question="policy", answer="a"))
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: AuthUser(username="alice", tenant_id="a", authenticated=True)
    items = TestClient(app).get("/qa/sessions").json()["items"]
    assert [row["session_id"] for row in items] == ["mine"]
