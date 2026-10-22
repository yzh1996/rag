"""独立检索接口隔离测试；不连接真实知识库、模型或集群。"""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from api.routes.retrieval import router
from core.auth import AuthUser, get_current_user
from core.retrieval_service import search_published


@pytest.fixture
def isolated_api(monkeypatch):
    from core import release_service, retriever
    calls = []
    state = {"release": SimpleNamespace(version="release-test", config={"manifest": {"doc": "v1"}}),
             "documents": [Document(page_content="test passage", metadata={"document_id": "doc",
                 "version_id": "v1", "source": "E:/private/path/runbook.md"})], "error": False}

    def resolve(user, task_id):
        calls.append(("resolve", user.username, task_id))
        return state["release"]

    class Retriever:
        def __init__(self, vector_store=None):
            assert vector_store == "published-test-store"

        def retrieve(self, query, **kwargs):
            calls.append(("retrieve", query, kwargs))
            if state["error"]:
                raise RuntimeError("private/internal/path and credentials")
            return state["documents"]

    monkeypatch.setattr(release_service, "resolve_release", resolve)
    monkeypatch.setattr(release_service, "release_store", lambda item: "published-test-store")
    monkeypatch.setattr(retriever, "RAGRetriever", Retriever)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    user = AuthUser(username="test-user", authenticated=True)
    app.dependency_overrides[get_current_user] = lambda: user
    with TestClient(app) as client:
        yield client, app, state, calls, user


def request(client, **overrides):
    return client.post("/api/v1/retrieval/search", json={"query": "service failure", "task_id": "task-test", **overrides})


def test_reuses_identity_release_and_manifest(isolated_api):
    client, _, state, calls, user = isolated_api
    response = request(client)
    assert response.status_code == 200
    body = response.json()
    assert body["knowledge_view"] == "release" and body["index_version"] == "release-test"
    assert body["items"][0]["source"] == "runbook.md"
    assert body["items"][0]["page"] is None
    assert "private/path" not in response.text
    assert calls[0] == ("resolve", "test-user", "task-test")
    assert calls[1][2]["user"] == user
    assert calls[1][2]["version_manifest"] == {"doc": "v1"}


def test_no_anonymous_retrieval(isolated_api):
    client, app, _, calls, _ = isolated_api
    app.dependency_overrides[get_current_user] = lambda: AuthUser(username="anonymous")
    response = request(client)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTH_REQUIRED"
    assert calls == []


@pytest.mark.parametrize("bad", [{"query": " "}, {"top_k": 11}, {"top_k": True},
                                  {"task_id": ""}, {"filter_dict": {}}, {"index_id": "candidate"}])
def test_invalid_or_privileged_inputs_rejected(isolated_api, bad):
    client, _, _, calls, _ = isolated_api
    response = request(client, **bad)
    assert response.status_code == 422
    assert calls == []


def test_no_release_fails_closed_and_no_fallback(isolated_api):
    client, _, state, calls, _ = isolated_api
    state["release"] = None
    response = request(client)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "KNOWLEDGE_RELEASE_UNAVAILABLE"
    assert len(calls) == 1


def test_empty_is_distinct_from_dependency_error(isolated_api):
    client, _, state, _, _ = isolated_api
    state["documents"] = []
    assert request(client).json()["items"] == []
    state["error"] = True
    response = request(client)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "KNOWLEDGE_RETRIEVAL_FAILED"
    assert "credentials" not in response.text


def test_truncation_budget_and_outside_manifest_drop(isolated_api):
    client, _, state, _, _ = isolated_api
    state["documents"] = [Document(page_content="x" * 7000, metadata={"document_id": "doc", "version_id": "v1"}) for _ in range(6)]
    state["documents"].insert(0, Document(page_content="unauthorized", metadata={"document_id": "secret", "version_id": "other"}))
    response = request(client, top_k=10)
    body = response.json()
    assert response.status_code == 200 and body["truncated"]
    assert sum(len(x["content"]) for x in body["items"]) == 24000
    assert all(x["truncated"] for x in body["items"])
    assert "unauthorized" not in response.text

