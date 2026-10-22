"""Isolated contract tests; no production services, paid models or live data."""
import json
import sqlite3
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from core import database
from core.auth import AuthUser, ANONYMOUS_USER, get_current_user
from core.backup_service import create_backup, restore_backup, verify_backup
from core.db_models import EvalCase, Feedback, QAInteraction
from core.feedback_service import assert_interaction_access, feedback_query, promote_feedback
from core.metrics import increment, observe, prometheus_text
from core.opensearch_store import OpenSearchStore, search_filter
from core.phase1_services import save_feedback, list_eval_cases
from core.quality_guard import Verdict, guarded_chain, valid_verdict
from core.retriever import RAGRetriever


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", factory)
    database.init_db()
    with factory() as session:
        yield session
    engine.dispose()


def interaction(db, tenant="a", username="alice"):
    item = QAInteraction(session_id="session", question="leave policy", answer="old answer", username=username,
        tenant_id=tenant, sources=[{"source": "policy.txt"}])
    db.add(item)
    db.commit()
    return item


def test_feedback_handoff_correction_and_idempotent_promotion(db):
    item = interaction(db)
    feedback = save_feedback(db, item.id, "negative", category="answer", corrected_answer="correct answer",
        is_resolved=False, handoff_requested=True)
    assert feedback.status == "pending_handoff"
    assert feedback.corrected_answer == "correct answer"
    assert_interaction_access(item, AuthUser(username="alice", tenant_id="a"))
    case = promote_feedback(db, feedback, "reviewed answer", "regression")
    assert case.tenant_id == "a"
    assert case.expected_answer == "reviewed answer"
    assert promote_feedback(db, feedback, "different answer", "other").id == case.id
    assert len(list_eval_cases(db, tenant_id="b")) == 0
    assert len(db.scalars(select(EvalCase)).all()) == 1
    with pytest.raises(ValueError):
        promote_feedback(db, feedback, " ", "regression")


def test_feedback_owner_and_tenant_isolation(db):
    item = interaction(db)
    save_feedback(db, item.id, "positive")
    with pytest.raises(HTTPException) as exc:
        assert_interaction_access(item, AuthUser(username="bob", tenant_id="a"))
    assert exc.value.status_code == 403
    with pytest.raises(HTTPException) as exc:
        assert_interaction_access(item, AuthUser(username="alice", role="admin", tenant_id="b"))
    assert exc.value.status_code == 404
    assert db.scalars(feedback_query(AuthUser(username="admin", tenant_id="b"))).all() == []


def test_feedback_admin_api_requires_role_and_scopes_records(db, monkeypatch):
    from api.routes.feedback import router

    item = interaction(db)
    feedback = save_feedback(db, item.id, "negative", handoff_requested=True)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[database.get_db] = lambda: db
    actor = AuthUser(username="admin", role="admin", tenant_id="a", authenticated=True)
    app.dependency_overrides[get_current_user] = lambda: actor
    client = TestClient(app)
    assert client.get("/feedback").json()["items"][0]["id"] == feedback.id
    assert client.patch("/feedback/" + feedback.id, json={"status": "resolved", "assignee": "agent"}).status_code == 200
    assert client.post("/feedback/" + feedback.id + "/promote", json={"dataset_name": "regression", "expected_answer": "reviewed"}).status_code == 200
    actor = AuthUser(username="admin", role="admin", tenant_id="b", authenticated=True)
    assert client.get("/feedback").json()["items"] == []
    assert client.patch("/feedback/" + feedback.id, json={"status": "closed"}).status_code == 404
    actor = ANONYMOUS_USER
    monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
    assert client.get("/feedback").status_code == 200
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    assert client.get("/feedback").status_code == 401
    actor = AuthUser(username="alice", role="user", tenant_id="a", authenticated=True)
    assert client.patch("/feedback/" + feedback.id, json={"status": "closed"}).status_code == 403


def test_knowledge_admin_api_allows_local_access_only_when_auth_disabled(db, monkeypatch):
    from api.dependencies import get_vector_store_dep
    from api.routes.knowledge import router

    class VectorStore:
        def delete_by_source(self, source_path):
            return 1

        def clear_all(self):
            return None

    monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[database.get_db] = lambda: db
    app.dependency_overrides[get_vector_store_dep] = lambda: VectorStore()
    client = TestClient(app)
    assert client.post("/knowledge/delete", json={"source_path": "policy.txt"}).status_code == 200
    assert client.post("/knowledge/clear").status_code == 200

    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    assert client.post("/knowledge/delete", json={"source_path": "policy.txt"}).status_code == 401
    assert client.post("/knowledge/clear").status_code == 401


class Embeddings:
    def embed_query(self, text):
        return [1.0, 0.0]


class SearchClient:
    def __init__(self):
        self.requests = []
        self.created = None
        self.indices = self

    def exists(self, index):
        return False

    def create(self, index, body):
        self.created = body

    def search(self, index, body):
        self.requests.append(body)
        return {"hits": {"hits": [{"_score": 3.0, "_source": {"content": "policy", "metadata": {"knowledge_base_id": "allowed"}, "embedding": [1.0, 0.0]}}]}}


def test_opensearch_bm25_mapping_and_prefilter_on_both_queries(monkeypatch):
    monkeypatch.setattr(settings, "OPENSEARCH_DIMENSION", 2)
    client = SearchClient()
    store = OpenSearchStore(Embeddings(), client)
    mapping = client.created["mappings"]["properties"]
    assert mapping["content"]["similarity"] == "BM25"
    assert mapping["content"]["analyzer"] == "cjk"
    assert mapping["embedding"]["method"]["engine"] == "lucene"
    scope = {"$and": [{"knowledge_base_id": {"$in": ["allowed"]}}, {"document_id": {"$nin": ["blocked"]}}]}
    result = store.similarity_search_with_score("policy", 20, scope)
    assert result[0][1] == 0.0
    assert client.requests[-1]["query"]["knn"]["embedding"]["filter"] == search_filter(scope)
    assert store.keyword_search("policy", 20, scope)[0][1] == 3.0
    assert client.requests[-1]["query"]["bool"]["filter"] == [search_filter(scope)]
    with pytest.raises(ValueError, match="authorized"):
        store.keyword_search("policy", 20, None)
    with pytest.raises(ValueError, match="authorized"):
        store.similarity_search_with_score("policy", 20, {"category": "manual"})


def test_opensearch_filter_rejects_unknown_operators():
    with pytest.raises(ValueError):
        search_filter({"knowledge_base_id": {"$invalid": ["allowed"]}})
    with pytest.raises(ValueError):
        search_filter({"metadata.inject": "bad"})


def test_opensearch_hybrid_thresholds_and_no_full_document_scan(monkeypatch):
    import core.retriever as module

    monkeypatch.setattr(settings, "VECTOR_STORE_TYPE", "opensearch")
    monkeypatch.setattr(settings, "SEARCH_MIN_VECTOR_SIMILARITY", 0.5)
    monkeypatch.setattr(module, "accessible_knowledge_filter", lambda user: {"knowledge_base_id": {"$in": ["allowed"]}})
    good = Document(page_content="policy", metadata={"source": "a"})
    bad = Document(page_content="irrelevant", metadata={"source": "b"})
    keyword_filters = []
    manager = SimpleNamespace(_store=SimpleNamespace(keyword_search=lambda q, k, f: keyword_filters.append(f) or [(good, 3.0)]),
        similarity_search_with_score=lambda q, k, filter_dict: [(bad, 0.9), (good, 0.1)])
    retriever = object.__new__(RAGRetriever)
    retriever.vector_store = manager
    retriever.reranker = None
    assert retriever.retrieve("policy", user=ANONYMOUS_USER) == [good]
    assert keyword_filters == [{"knowledge_base_id": {"$in": ["allowed"]}}]


@pytest.mark.parametrize("answer,citations,valid", [("answer [1]", [1], True), ("answer [2]", [2], False), ("answer", [], False), ("answer [1]", [], False)])
def test_citation_verdict_range_and_missing_checks(answer, citations, valid):
    assert valid_verdict(Verdict(safe=True, evidence_sufficient=True, all_claims_supported=True, citations=citations), answer, 1) is valid


def test_no_evidence_skips_generation(monkeypatch):
    monkeypatch.setattr(settings, "EVIDENCE_MIN_SOURCES", 1)
    prompt = ChatPromptTemplate.from_template("{context}\n{input}")
    chain = guarded_chain(RunnableLambda(lambda inputs: pytest.fail("LLM must not be called")), RunnableLambda(lambda inputs: []), prompt)
    result = chain.invoke({"input": "policy", "chat_history": []})
    assert result["evidence_sufficient"] is False
    assert result["citation_status"] == "missing"


def test_unsupported_claim_verdict_rejects():
    assert not valid_verdict(Verdict(safe=True, evidence_sufficient=True, all_claims_supported=False, citations=[1]), "answer [1]", 1)


@pytest.mark.parametrize("supported,status", [(True, "verified"), (False, "rejected")])
def test_guarded_generation_verifies_or_refuses(supported, status):
    class ReviewModel(FakeListChatModel):
        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(lambda inputs: Verdict(safe=True, evidence_sufficient=True,
                all_claims_supported=supported, citations=[1]))

    model = ReviewModel(responses=["reviewed answer [1]"])
    prompt = ChatPromptTemplate.from_template("{context}\n{input}")
    documents = [Document(page_content="reviewed answer")]
    chain = guarded_chain(model, RunnableLambda(lambda inputs: documents), prompt)
    result = chain.invoke({"input": "policy", "chat_history": []})
    assert result["citation_status"] == status
    assert result["evidence_sufficient"] is supported
    if not supported:
        assert result["answer"] != "reviewed answer [1]"


def test_metrics_use_histogram_without_sensitive_labels():
    increment("test_enterprise_total")
    observe("test_enterprise", 125)
    value = prometheus_text()
    assert "rag_test_enterprise_seconds_bucket" in value
    assert "rag_test_enterprise_total 1.0" in value
    assert "username=" not in value
    assert "question=" not in value


def backup_config(tmp_path, monkeypatch):
    database_path = tmp_path / "source.db"
    with sqlite3.connect(database_path) as db:
        db.execute("CREATE TABLE sample (value TEXT)")
        db.execute("INSERT INTO sample VALUES ('original')")
    upload = tmp_path / "uploads"
    upload.mkdir()
    (upload / "policy.txt").write_text("policy", encoding="utf-8")
    vectors = tmp_path / "vectors"
    vectors.mkdir()
    (vectors / "index.bin").write_bytes(b"vector")
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setattr(settings, "UPLOAD_DIR", upload)
    monkeypatch.setattr(settings, "VECTOR_DB_DIR", vectors)
    monkeypatch.setattr(settings, "INDEX_SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(settings, "VECTOR_STORE_TYPE", "chroma")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "app-data")
    return database_path, upload, vectors


def test_backup_checksums_and_refuses_live_restore(tmp_path, monkeypatch):
    backup_config(tmp_path, monkeypatch)
    output = tmp_path / "test.backup.zip"
    with pytest.raises(ValueError, match="offline"):
        create_backup(output, False)
    result = create_backup(output, True)
    assert len(result["sha256"]) == 64
    with pytest.raises(ValueError, match="not empty"):
        restore_backup(output, True, execute=True)
    with pytest.raises(ValueError, match="offline"):
        restore_backup(output, False)
    manifest = verify_backup(output, tmp_path / "verified")
    assert manifest["database"] == "sqlite"
    assert "uploads/policy.txt" in manifest["files"]


def test_backup_rejects_path_traversal_and_tampering(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../outside", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        verify_backup(archive, tmp_path / "verify")
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("manifest.json", json.dumps({"format": 1, "files": {"data": "0" * 64}}))
        output.writestr("data", "tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_backup(archive, tmp_path / "verify")


def test_backup_restores_to_empty_environment_only(tmp_path, monkeypatch):
    _, upload, vectors = backup_config(tmp_path, monkeypatch)
    archive = tmp_path / "recovery.backup.zip"
    create_backup(archive, True)
    # Remove only the known fixture files; no application workspace data is touched.
    (upload / "policy.txt").unlink()
    (vectors / "index.bin").unlink()
    recovery = tmp_path / "restored.db"
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{recovery}")
    assert restore_backup(archive, True)["execute"] is False
    assert not recovery.exists()
    assert not (upload / "policy.txt").exists()
    assert restore_backup(archive, True, execute=True)["execute"] is True
    connection = sqlite3.connect(recovery)
    try:
        assert connection.execute("SELECT value FROM sample").fetchone()[0] == "original"
    finally:
        connection.close()
    assert (upload / "policy.txt").read_text(encoding="utf-8") == "policy"
    assert (vectors / "index.bin").read_bytes() == b"vector"


def test_old_feedback_schema_migrates_without_losing_records(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    monkeypatch.setattr(database, "engine", engine)
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE qa_feedback (id VARCHAR(32) PRIMARY KEY, interaction_id VARCHAR(32), rating VARCHAR(16), category VARCHAR(64), comment TEXT, created_at DATETIME)"))
        connection.execute(text("INSERT INTO qa_feedback (id, rating) VALUES ('old', 'positive')"))
    database._ensure_schema_compatibility()
    assert "corrected_answer" in {column["name"] for column in inspect(engine).get_columns("qa_feedback")}
    with engine.connect() as connection:
        assert connection.execute(text("SELECT rating, status FROM qa_feedback WHERE id='old'")).first() == ("positive", "open")
    engine.dispose()


def test_alert_receiver_requires_separate_token_and_persists(db, monkeypatch):
    from api.routes.operations import router
    from core.db_models import OperationalAlert

    monkeypatch.setattr(settings, "ALERT_RECEIVER_TOKEN", "test-token")
    monkeypatch.setattr(settings, "ALERT_NOTIFICATION_URL", "")
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[database.get_db] = lambda: db
    client = TestClient(app)
    payload = {"alerts": [{"status": "firing", "fingerprint": "one", "labels": {"alertname": "RAGUnavailable", "severity": "critical"}, "annotations": {"summary": "service unavailable"}}]}
    assert client.post("/operations/alerts", json=payload).status_code == 401
    assert client.post("/operations/alerts", json=payload, headers={"Authorization": "Bearer test-token"}).status_code == 200
    assert db.scalar(select(OperationalAlert)).name == "RAGUnavailable"


def test_windows_document_paths_can_be_remapped_without_escape():
    from core.business_migration import remap_file

    assert remap_file(r"E:\project\upload\doc.txt", r"E:\project\upload", "/app/upload") == "/app/upload/doc.txt"
    with pytest.raises(ValueError, match="outside"):
        remap_file(r"E:\private\secret.txt", r"E:\project\upload", "/app/upload")
