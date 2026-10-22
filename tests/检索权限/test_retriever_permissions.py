"""Permission prefilter tests with isolated SQLite and real Chroma."""
import json

import numpy as np
import pytest
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import create_engine, select, inspect, text
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from core import database, retriever as retriever_module, vector_store as vector_module
from core.auth import ANONYMOUS_USER, AuthUser
from core.db_models import Document as StoredDocument, KnowledgeBase
from core.knowledge_scope import accessible_knowledge_filter, ensure_knowledge_base
from core.retriever import RAGRetriever
from core.vector_store import VectorStoreManager


class TestEmbeddings(Embeddings):
    __test__ = False

    def embed_documents(self, texts):
        return [[0.0, 0.0, 0.0] if item.startswith("restricted") else [1.0, 0.0, 0.0] for item in texts]

    def embed_query(self, text):
        return [0.0, 0.0, 0.0]


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'permissions.db'}")
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", factory)
    monkeypatch.setattr(settings, "VECTOR_STORE_TYPE", "chroma")
    monkeypatch.setattr(settings, "VECTOR_DB_DIR", tmp_path / "vectors")
    monkeypatch.setattr(vector_module, "get_embedding_model", TestEmbeddings)
    database.init_db()
    yield factory
    engine.dispose()


def make_retriever(manager):
    instance = object.__new__(RAGRetriever)
    instance.vector_store = manager
    instance.reranker = None
    return instance


def add_stored_document(scope, *, tenant="tenant-a", category="policies"):
    with database.session_scope() as db:
        knowledge_base = ensure_knowledge_base(db, tenant, category, scope)
        document = StoredDocument(
            source_uri="source-" + knowledge_base.id + "-" + str(len(db.scalars(select(StoredDocument)).all())),
            original_filename="policy.txt",
            category=category,
            tenant_id=tenant,
            permission_scope=scope,
            knowledge_base_id=knowledge_base.id,
            status="published",
        )
        db.add(document)
        db.flush()
        return document.id, knowledge_base.id


def test_permission_prefilter_finds_result_after_twenty_restricted_chunks(isolated_db, monkeypatch):
    allowed_id, allowed_kb = add_stored_document({"tenant_ids": ["tenant-a"], "users": ["alice"]})
    blocked_id, _ = add_stored_document({"tenant_ids": ["tenant-b"]}, tenant="tenant-b")
    manager = VectorStoreManager()
    chunks = [Document(page_content=f"restricted-{index}", metadata={"document_id": blocked_id, "permission_scope": '{"tenant_ids": ["tenant-b"]}'}) for index in range(25)]
    chunks.append(Document(page_content="allowed benefits", metadata={"document_id": allowed_id, "permission_scope": '{"tenant_ids": ["tenant-a"], "users": ["alice"]}'}))
    assert manager.add_documents(chunks) == 26
    assert all(doc.metadata["document_id"] == blocked_id for doc, _ in manager.similarity_search_with_score("annual leave", k=20))

    observed = {}
    search = manager.similarity_search_with_score
    read = manager.get_documents

    def track_search(query, k, filter_dict=None):
        observed["vector_filter"] = filter_dict
        return search(query, k, filter_dict)

    def track_read(filter_dict=None):
        observed["keyword_filter"] = filter_dict
        payload = read(filter_dict)
        observed["keyword_documents"] = payload["documents"]
        return payload

    monkeypatch.setattr(manager, "similarity_search_with_score", track_search)
    monkeypatch.setattr(manager, "get_documents", track_read)
    result = make_retriever(manager).retrieve("annual leave", top_k=1, user=AuthUser(username="alice", tenant_id="tenant-a", authenticated=True))
    assert [doc.metadata["document_id"] for doc in result] == [allowed_id]
    assert observed["vector_filter"] == {"knowledge_base_id": {"$in": [allowed_kb]}}
    assert observed["keyword_filter"] == observed["vector_filter"]
    assert observed["keyword_documents"] == ["allowed benefits"]


def test_anonymous_user_is_prefiltered(isolated_db):
    public_id, public_kb = add_stored_document({})
    private_id, _ = add_stored_document({"users": ["alice"]})
    manager = VectorStoreManager()
    assert manager.add_documents([
        Document(page_content="public policy", metadata={"document_id": public_id}),
        Document(page_content="restricted policy", metadata={"document_id": private_id, "permission_scope": '{"users": ["alice"]}'}),
    ]) == 2
    assert accessible_knowledge_filter(ANONYMOUS_USER) == {"knowledge_base_id": {"$in": [public_kb]}}
    result = make_retriever(manager).retrieve("policy", user=ANONYMOUS_USER)
    assert [doc.metadata["document_id"] for doc in result] == [public_id]


def test_no_permission_skips_both_searches(isolated_db, monkeypatch):
    add_stored_document({"users": ["alice"]})
    manager = VectorStoreManager()

    def unexpected(*args, **kwargs):
        pytest.fail("Search must not run without permission")

    monkeypatch.setattr(manager, "similarity_search_with_score", unexpected)
    monkeypatch.setattr(manager, "get_documents", unexpected)
    assert make_retriever(manager).retrieve("policy", user=AuthUser(username="bob")) == []


def test_permission_lookup_failure_does_not_search(isolated_db, monkeypatch):
    manager = VectorStoreManager()

    def unavailable(user):
        raise RuntimeError("permission database unavailable")

    def unexpected(*args, **kwargs):
        pytest.fail("Must not fall back to unrestricted search")

    monkeypatch.setattr(retriever_module, "accessible_knowledge_filter", unavailable)
    monkeypatch.setattr(manager, "similarity_search_with_score", unexpected)
    monkeypatch.setattr(manager, "get_documents", unexpected)
    with pytest.raises(RuntimeError, match="unavailable"):
        make_retriever(manager).retrieve("policy", user=AuthUser(username="alice"))


def test_requested_filter_applies_to_vector_and_keyword_search(isolated_db):
    manager = VectorStoreManager()
    assert manager.add_documents([
        Document(page_content="manual policy", metadata={"category": "manual"}),
        Document(page_content="restricted policy", metadata={"category": "finance"}),
    ]) == 2
    result = make_retriever(manager).retrieve("policy", filter_dict={"category": "manual"}, user=ANONYMOUS_USER)
    assert [doc.metadata["category"] for doc in result] == ["manual"]
    chain_retriever = make_retriever(manager).get_compresstion_retriever({"k": 5, "filter": {"category": "manual"}}, user=ANONYMOUS_USER)
    assert [doc.metadata["category"] for doc in chain_retriever.invoke("policy")] == ["manual"]


def test_document_restriction_is_applied_before_top_k(isolated_db):
    scope = {"users": ["alice"]}
    allowed_id, kb_id = add_stored_document(scope)
    blocked_id, _ = add_stored_document(scope)
    with database.session_scope() as db:
        db.get(StoredDocument, blocked_id).permission_scope = {"users": ["bob"]}
    manager = VectorStoreManager()
    # Vector metadata intentionally contains the stale, formerly allowed scope.
    assert manager.add_documents([
        *[Document(page_content=f"restricted-{index}", metadata={"document_id": blocked_id, "permission_scope": json.dumps(scope)}) for index in range(25)],
        Document(page_content="allowed benefits", metadata={"document_id": allowed_id, "permission_scope": json.dumps(scope)}),
    ]) == 26
    permission_filter = accessible_knowledge_filter(AuthUser(username="alice"))
    assert permission_filter == {"$and": [{"knowledge_base_id": {"$in": [kb_id]}}, {"document_id": {"$nin": [blocked_id]}}]}
    result = make_retriever(manager).retrieve("annual leave", top_k=1, user=AuthUser(username="alice"))
    assert [doc.metadata["document_id"] for doc in result] == [allowed_id]


def test_legacy_metadata_migration_preserves_embeddings_and_ids(isolated_db):
    legacy = Chroma(persist_directory=str(settings.VECTOR_DB_DIR), embedding_function=TestEmbeddings())
    legacy.add_documents([
        Document(page_content="public policy", metadata={"source": "old-public.txt"}),
        Document(page_content="restricted policy", metadata={"source": "old-private.txt", "permission_scope": '{"users": ["alice"]}'}),
    ], ids=["public-chunk", "private-chunk"])
    before = legacy.get(include=["embeddings", "documents", "metadatas"])
    manager = VectorStoreManager()
    after = manager._store.get(include=["embeddings", "documents", "metadatas"])
    assert before["ids"] == after["ids"]
    assert before["documents"] == after["documents"]
    assert np.array_equal(before["embeddings"], after["embeddings"])
    assert all(metadata.get("knowledge_base_id") for metadata in after["metadatas"])
    result = make_retriever(manager).retrieve("policy", user=ANONYMOUS_USER)
    assert [doc.metadata["source"] for doc in result] == ["old-public.txt"]
    manager.reload()
    assert manager._store.get(include=["metadatas"])["metadatas"] == after["metadatas"]


def test_same_permissions_group_together_but_tenants_and_categories_do_not(isolated_db):
    with database.session_scope() as db:
        first = ensure_knowledge_base(db, "tenant-a", "policies", {"users": ["alice", "bob"]})
        reordered = ensure_knowledge_base(db, "tenant-a", "policies", {"users": ["bob", "alice"]})
        other_tenant = ensure_knowledge_base(db, "tenant-b", "policies", {"users": ["alice", "bob"]})
        other_category = ensure_knowledge_base(db, "tenant-a", "finance", {"users": ["alice", "bob"]})
        assert first.id == reordered.id
        assert len({first.id, other_tenant.id, other_category.id}) == 3


@pytest.mark.parametrize("scope", ["broken json", "[]", '{"users": "alice"}'])
def test_malformed_permissions_are_not_treated_as_public(isolated_db, scope):
    manager = VectorStoreManager()
    assert manager.add_documents([Document(page_content="restricted", metadata={"permission_scope": scope})]) == 0
    assert not make_retriever(manager)._can_access(Document(page_content="restricted", metadata={"permission_scope": scope}), ANONYMOUS_USER)


@pytest.mark.parametrize("strategy", [DistanceStrategy.EUCLIDEAN_DISTANCE, DistanceStrategy.MAX_INNER_PRODUCT])
def test_faiss_prefilters_allowed_vectors_without_changing_index(isolated_db, monkeypatch, strategy):
    allowed_id, _ = add_stored_document({"users": ["alice"]})
    blocked_id, _ = add_stored_document({"users": ["bob"]})
    chunks = [Document(page_content=f"restricted-{index}", metadata={"document_id": blocked_id, "permission_scope": '{"users": ["bob"]}'}) for index in range(25)]
    chunks.append(Document(page_content="allowed benefits", metadata={"document_id": allowed_id, "permission_scope": '{"users": ["alice"]}'}))
    legacy = FAISS.from_documents(chunks, TestEmbeddings(), distance_strategy=strategy)
    legacy.save_local(str(settings.VECTOR_DB_DIR))
    before = legacy.index.reconstruct_n(0, legacy.index.ntotal).copy()
    monkeypatch.setattr(settings, "VECTOR_STORE_TYPE", "faiss")
    manager = VectorStoreManager()
    result = make_retriever(manager).retrieve("annual leave", top_k=1, user=AuthUser(username="alice"))
    assert [doc.metadata["document_id"] for doc in result] == [allowed_id]
    assert len(manager.get_documents(accessible_knowledge_filter(AuthUser(username="alice")))["documents"]) == 1
    assert manager._store.index.ntotal == 26
    assert np.array_equal(before, manager._store.index.reconstruct_n(0, 26))
    assert manager.add_documents([Document(page_content="public policy", metadata={"source": "new-public.txt"})]) == 1
    assert [doc.metadata["source"] for doc in make_retriever(manager).retrieve("policy", user=ANONYMOUS_USER)] == ["new-public.txt"]


def test_missing_user_identity_does_not_bypass_permissions(isolated_db):
    manager = VectorStoreManager()
    assert manager.add_documents([
        Document(page_content="public policy", metadata={"source": "public.txt"}),
        Document(page_content="restricted policy", metadata={"source": "private.txt", "permission_scope": '{"users": ["alice"]}'}),
    ]) == 2
    result = make_retriever(manager).retrieve("policy")
    assert [doc.metadata["source"] for doc in result] == ["public.txt"]


def test_legacy_public_chunks_survive_narrow_document_permissions(isolated_db):
    document_id, _ = add_stored_document({})
    with database.session_scope() as db:
        db.get(StoredDocument, document_id).permission_scope = {"users": ["alice"]}
    manager = VectorStoreManager()
    assert manager.add_documents([
        Document(page_content="restricted policy", metadata={"document_id": document_id}),
        Document(page_content="public policy", metadata={"source": "legacy.txt"}),
    ]) == 2
    result = make_retriever(manager).retrieve("policy", user=ANONYMOUS_USER)
    assert [doc.metadata["source"] for doc in result] == ["legacy.txt"]


def test_knowledge_base_revocation_takes_effect_without_reindexing(isolated_db):
    document_id, kb_id = add_stored_document({"users": ["alice"]})
    manager = VectorStoreManager()
    assert manager.add_documents([Document(page_content="allowed policy", metadata={"document_id": document_id, "permission_scope": '{"users": ["alice"]}'})]) == 1
    retriever = make_retriever(manager)
    assert len(retriever.retrieve("policy", user=AuthUser(username="alice"))) == 1
    with database.session_scope() as db:
        db.get(KnowledgeBase, kb_id).permission_scope = {"users": ["bob"]}
    assert retriever.retrieve("policy", user=AuthUser(username="alice")) == []


def test_new_ingestion_preserves_permissions_and_writes_knowledge_base_id(isolated_db, tmp_path, monkeypatch):
    from core import ingestion_service
    from core.db_models import IngestionJob
    monkeypatch.setattr(settings, "KNOWLEDGE_REVIEW_REQUIRED", False)

    file_path = tmp_path / "sample.txt"
    file_path.write_text("annual leave", encoding="utf-8")
    user = AuthUser(username="alice", tenant_id="tenant-a", authenticated=True)
    document, version = ingestion_service.create_document_version(
        str(file_path), original_filename="sample.txt", category="policies", user=user
    )
    with database.session_scope() as db:
        job = IngestionJob(document_id=document.id, version_id=version.id)
        db.add(job)
        db.flush()
        job_id = job.id
    captured = {}

    class RecordingLoader:
        def load_file(self, file_path, metadata, strategy, params):
            captured.update(metadata)
            return [Document(page_content="annual leave", metadata=metadata)]

    manager = VectorStoreManager()
    monkeypatch.setattr(ingestion_service, "DocumentLoader", RecordingLoader)
    monkeypatch.setattr(ingestion_service, "get_vector_store_manager", lambda: manager)
    ingestion_service.process_ingestion_job(job_id)
    assert captured["knowledge_base_id"] == document.knowledge_base_id
    assert json.loads(captured["permission_scope"]) == {"tenant_ids": ["tenant-a"], "users": ["alice"]}
    with isolated_db() as db:
        assert db.get(IngestionJob, job_id).status == "completed"
    assert len(make_retriever(manager).retrieve("annual leave", user=user)) == 1


def test_legacy_database_gets_knowledge_base_column(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", factory)
    with engine.begin() as connection:
        connection.execute(text("""CREATE TABLE documents (
            id VARCHAR(32) PRIMARY KEY, source_uri VARCHAR(1024), original_filename VARCHAR(512),
            category VARCHAR(128), owner_id VARCHAR(32), tenant_id VARCHAR(128), permission_scope JSON,
            status VARCHAR(32), current_version INTEGER, created_at DATETIME, updated_at DATETIME
        )"""))
        connection.execute(text("""INSERT INTO documents (id, source_uri, original_filename, category, tenant_id,
            permission_scope, status, current_version) VALUES ('legacy-doc', 'old.txt', 'old.txt', 'general',
            'default', '{}', 'published', 1)"""))
    database.init_db()
    assert "knowledge_base_id" in {column["name"] for column in inspect(engine).get_columns("documents")}
    with factory() as db:
        document = db.get(StoredDocument, "legacy-doc")
        assert document.knowledge_base_id
        assert db.get(KnowledgeBase, document.knowledge_base_id).permission_scope == {}
    database.init_db()
    engine.dispose()
