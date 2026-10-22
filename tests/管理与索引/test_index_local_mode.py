"""索引查看与构建的免登录验证，使用临时数据库，不连接模型服务。"""
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.routes.index import router
from config.settings import settings
from core import database
from core.auth import create_access_token
from core.db_models import BackgroundJob, Document, DocumentVersion, IndexVersion


LOCAL_OPERATIONS = [
    ("GET", "/index/versions", None),
    ("POST", "/index/versions/snapshot", {}),
]
PROTECTED_OPERATIONS = [
    ("POST", "/index/versions/candidate/review", {"notes": "checked"}),
    ("POST", "/index/versions/candidate/publish", {"eval_run_id": "run"}),
    ("POST", "/index/versions/candidate/rollback", None),
]


@pytest.fixture
def local_client(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'index-local.db'}", connect_args={"check_same_thread": False})
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(engine, expire_on_commit=False))
    monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
    monkeypatch.setattr(settings, "INDEX_SNAPSHOT_DIR", tmp_path / "snapshots")
    database.init_db()
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client
    engine.dispose()


def seed_document(document_id, tenant="default", scope=None):
    with database.session_scope() as db:
        db.add(Document(id=document_id, source_uri=document_id + ".txt", original_filename=document_id + ".txt",
            tenant_id=tenant, permission_scope=scope or {}, status="candidate"))
        db.flush()
        db.add(DocumentVersion(id=document_id + "-v1", document_id=document_id, version=1,
            content_hash="1" * 64, file_path=document_id + ".txt", status="candidate"))


def test_anonymous_build_preserves_document_scope_tenant_and_task_identity(local_client):
    seed_document("shared")
    seed_document("finance", scope={"departments": ["财务部"]})
    seed_document("other", tenant="other-company")
    response = local_client.post("/index/versions/snapshot", json={})
    assert response.status_code == 200
    candidate = response.json()
    assert candidate["status"] == "building" and candidate["tenant_id"] == "default"
    assert candidate["config"]["manifest"] == {"shared": "shared-v1"}
    with database.session_scope() as db:
        task = db.query(BackgroundJob).one()
        assert task.kind == "build_index" and task.payload["index_id"] == candidate["id"]
        assert task.payload["user"]["authenticated"] is False
        assert task.payload["user"]["role"] == "user"
        db.add(IndexVersion(version="foreign-index", tenant_id="other-company", embedding_model="model"))
    items = local_client.get("/index/versions").json()["items"]
    assert [item["id"] for item in items] == [candidate["id"]]


def test_anonymous_build_without_accessible_documents_reports_business_error(local_client):
    seed_document("finance", scope={"departments": ["财务部"]})
    response = local_client.post("/index/versions/snapshot", json={})
    assert response.status_code == 409
    assert "没有可构建的已入库文档" in response.json()["detail"]
    with database.session_scope() as db:
        assert db.query(BackgroundJob).count() == db.query(IndexVersion).count() == 0


@pytest.mark.parametrize("method,path,body", LOCAL_OPERATIONS)
def test_enabling_auth_requires_login_for_view_and_build(local_client, monkeypatch, method, path, body):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    assert local_client.request(method, path, json=body).status_code == 401


@pytest.mark.parametrize("method,path,body", LOCAL_OPERATIONS)
def test_invalid_token_cannot_fall_back_to_anonymous(local_client, method, path, body):
    response = local_client.request(method, path, json=body, headers={"Authorization": "Bearer invalid"})
    assert response.status_code == 401


@pytest.mark.parametrize("auth_required", [False, True])
def test_signed_regular_user_keeps_restricted_permissions(local_client, monkeypatch, auth_required):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", auth_required)
    headers = {"Authorization": "Bearer " + create_access_token("alice", role="user")}
    for method, path, body in LOCAL_OPERATIONS + PROTECTED_OPERATIONS:
        assert local_client.request(method, path, json=body, headers=headers).status_code == 403


@pytest.mark.parametrize("method,path,body", PROTECTED_OPERATIONS)
def test_review_publish_and_rollback_still_require_login(local_client, method, path, body):
    assert local_client.request(method, path, json=body).status_code == 401


@pytest.mark.parametrize("auth_required", [False, True])
@pytest.mark.parametrize("role", ["admin", "knowledge_admin"])
def test_signed_admin_can_view_and_build_own_company(local_client, monkeypatch, auth_required, role):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", auth_required)
    seed_document("finance", tenant="company-a", scope={"departments": ["财务部"]})
    headers = {"Authorization": "Bearer " + create_access_token("manager", role=role, tenant_id="company-a")}
    response = local_client.post("/index/versions/snapshot", json={}, headers=headers)
    assert response.status_code == 200
    assert response.json()["config"]["manifest"] == {"finance": "finance-v1"}
    items = local_client.get("/index/versions", headers=headers).json()["items"]
    assert [item["id"] for item in items] == [response.json()["id"]]
    assert items[0]["tenant_id"] == "company-a"


def test_index_page_build_and_refresh_work_without_token(local_client, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from frontend import api_client
    seed_document("shared")
    def request(method, url, **kwargs):
        path = urlsplit(url).path.removeprefix("/api/v1")
        assert not kwargs.get("headers", {}).get("Authorization")
        response = local_client.request(method, path, **kwargs)
        return SimpleNamespace(status_code=response.status_code, ok=response.is_success,
            json=response.json, text=response.text)
    monkeypatch.setattr(api_client.transport, "request", request)
    app = AppTest.from_string("from frontend.management import render_indexes\nrender_indexes()\n").run(timeout=20)
    assert not list(app.exception) and not list(app.error)
    next(button for button in app.button if button.label == "构建候选索引").click().run(timeout=20)
    assert not list(app.exception) and not list(app.error)
    assert any("构建任务已提交" in message.value for message in app.success)
    next(button for button in app.button if button.label == "刷新索引版本").click().run(timeout=20)
    assert not list(app.exception) and not list(app.error)
    assert next(field for field in app.selectbox if field.label == "索引版本").options
