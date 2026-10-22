"""免登录管理员上传的隔离验证，不改变全局登录和业务数据。"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.settings import settings
from core import task_queue, upload_service as uploads, database
from core.auth import AuthUser, can_access_document
from core.db_models import UploadTask
from tests.文档上传.test_upload_workflow import isolated, actor, client


@pytest.fixture
def local_client(isolated, monkeypatch):
    from api.routes.upload import router
    monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def submit_local(api, tenant="", department="财务部", content=b"company policy", **metadata):
    options = api.get("/upload/options", params={"tenant_id": tenant}).json()
    kb = next(item for item in options["knowledge_bases"] if item["owner_department"] is None)
    return api.post("/upload/tasks", params={"tenant_id": tenant},
        files={"file": ("policy.txt", content)},
        data={"knowledge_base_id": kb["id"], "metadata": json.dumps({
            "tenant_id": tenant, "owner_department": department, **metadata})})


def test_local_options_show_admin_fields_without_login(local_client):
    response = local_client.get("/upload/options")
    assert response.status_code == 200
    options = response.json()
    assert options["tenant_editable"] is True and options["tenant_id"] == "default"
    assert options["departments"] == ["财务部", "客服部"]
    assert len(options["knowledge_bases"]) == 3
    assert options["username"] == uploads.LOCAL_ADMIN_USERNAME


@pytest.mark.parametrize("tenant", ["", "   ", "company-a", " 公司A "])
def test_optional_company_is_normalized_and_preserved_through_processing(local_client, tenant):
    response = submit_local(local_client, tenant, "客服部")
    assert response.status_code == 202
    record = response.json()
    assert record["status"] == "queued"
    expected = tenant.strip() or "default"
    assert record["tenant_id"] == expected and record["metadata"]["tenant_id"] == expected
    assert record["metadata"]["owner_department"] == "客服部"
    with database.session_scope() as db:
        stored = db.get(UploadTask, record["id"])
        assert stored.actor["authenticated"] is False  # 管理模式不冒充真实登录。
    assert task_queue.run_once()
    restored = local_client.get("/upload/tasks/" + record["id"], params={"tenant_id": tenant}).json()
    assert restored["status"] == "ready" and restored["can_read_content"]


def test_local_company_switch_isolates_records_and_duplicate_checks(local_client):
    first = submit_local(local_client, "company-a").json()
    second = submit_local(local_client, "company-b").json()
    assert first["status"] == second["status"] == "queued"
    assert submit_local(local_client, "company-a").json()["status"] == "failed"
    assert local_client.get("/upload/tasks").json()["items"] == []
    records = local_client.get("/upload/tasks", params={"tenant_id": "company-a"}).json()["items"]
    assert {item["tenant_id"] for item in records} == {"company-a"}
    assert local_client.get("/upload/tasks/" + first["id"], params={"tenant_id": "company-b"}).status_code == 404


def test_department_ownership_and_reader_scope_are_independent_for_admin(local_client):
    record = submit_local(local_client, department="财务部", visibility="departments",
        allowed_departments=["客服部"]).json()
    assert record["status"] == "queued"
    assert record["permission_scope"]["departments"] == ["客服部"]
    assert task_queue.run_once()
    record = local_client.get("/upload/tasks/" + record["id"]).json()
    assert record["can_read_content"] and record["after_content"]
    assert not can_access_document(record["permission_scope"], actor(department="财务部", tenant="default"))
    assert can_access_document(record["permission_scope"], actor(department="客服部", tenant="default"))


def test_local_admin_can_preview_and_confirm_both_departments(local_client, monkeypatch):
    from core import ingestion_service
    for department, content in (("财务部", b"finance policy"), ("客服部", b"service policy")):
        task = submit_local(local_client, department=department, content=content).json()
        task_id = task["id"]
        assert task_queue.run_once()
        assert local_client.post(f"/upload/tasks/{task_id}/chunks",
            json={"strategy": "recursive", "params": {"chunk_size": 100, "chunk_overlap": 0}}).status_code == 202
        assert task_queue.run_once()
        result = local_client.get("/upload/tasks/" + task_id).json()
        assert result["status"] == "preview" and result["chunks"][0]["metadata"]["owner_department"] == department
        assert local_client.post(f"/upload/tasks/{task_id}/confirm").status_code == 200
        monkeypatch.setattr(ingestion_service, "get_vector_store_manager",
            lambda: SimpleNamespace(add_documents=lambda chunks: len(chunks)))
        assert task_queue.run_once()
        assert local_client.get("/upload/tasks/" + task_id).json()["status"] == "candidate"


def test_local_safety_failure_still_has_reason_and_recovery_action(local_client):
    result = submit_local(local_client, content=b"").json()
    assert result["status"] == "failed"
    failure = next(stage for stage in result["stages"] if stage["status"] == "failed")
    assert failure["key"] == "safety" and failure["detail"] and failure["next_action"]
    assert result["record_saved"] is False and result["id"] is None
    assert local_client.get("/upload/tasks").json()["items"] == []


def test_local_failed_processing_can_retry_and_resume(local_client, monkeypatch):
    from core import content_processing
    original = content_processing.clean_documents
    monkeypatch.setattr(content_processing, "clean_documents",
        lambda documents: (_ for _ in ()).throw(ValueError("清洗暂时失败")))
    task_id = submit_local(local_client, "company-a").json()["id"]
    assert task_queue.run_once()
    params = {"tenant_id": "company-a"}
    assert local_client.get("/upload/tasks/" + task_id, params=params).json()["can_retry"]
    monkeypatch.setattr(content_processing, "clean_documents", original)
    assert local_client.post(f"/upload/tasks/{task_id}/retry", params=params).status_code == 202
    assert task_queue.run_once()
    assert local_client.get("/upload/tasks/" + task_id, params=params).json()["status"] == "ready"


def test_switching_auth_on_disables_local_mode_and_queued_local_work(local_client, monkeypatch):
    task_id = submit_local(local_client).json()["id"]
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    for path in ("/upload/options", "/upload/tasks", "/upload/tasks/" + task_id):
        assert local_client.get(path).status_code == 401
    assert local_client.post(f"/upload/tasks/{task_id}/confirm").status_code == 401
    assert task_queue.run_once()
    with database.session_scope() as db:
        task = db.get(UploadTask, task_id)
        assert task.status == "failed" and task.stages[0]["status"] == "failed"


def test_authenticated_user_is_not_promoted_or_allowed_to_choose_another_company(local_client):
    api = client(actor())
    options = api.get("/upload/options").json()
    assert options["tenant_editable"] is False and options["departments"] == ["财务部"]
    assert api.get("/upload/options", params={"tenant_id": "company-b"}).status_code == 403
    assert api.get("/upload/tasks", params={"tenant_id": "company-b"}).status_code == 403
    admin = client(actor(role="admin"))
    assert admin.get("/upload/options", params={"tenant_id": "company-b"}).status_code == 403


def test_invalid_login_token_does_not_fall_back_to_local_admin(local_client):
    assert local_client.get("/upload/options", headers={"Authorization": "Bearer invalid"}).status_code == 401


def test_signed_admin_can_review_other_department_without_broadening_document_scope(local_client):
    user = actor(role="admin")
    api = client(user)
    options = api.get("/upload/options").json()
    kb = next(item for item in options["knowledge_bases"] if item["owner_department"] == "客服部")
    record = api.post("/upload/tasks", files={"file": ("policy.txt", b"service policy")},
        data={"knowledge_base_id": kb["id"], "metadata": json.dumps({"owner_department": "客服部"})}).json()
    assert task_queue.run_once()
    result = api.get("/upload/tasks/" + record["id"]).json()
    assert result["can_read_content"] and result["after_content"]
    assert result["permission_scope"]["departments"] == ["客服部"]
    assert api.post(f"/upload/tasks/{record['id']}/chunks", json={"strategy": "recursive"}).status_code == 202


def test_page_shows_optional_company_and_clears_old_company_state(local_client):
    from streamlit.testing.v1 import AppTest
    first = submit_local(local_client).json()
    assert task_queue.run_once()
    task_id = first["id"]
    def request(method, url, **kwargs):
        path = urlsplit(url).path.removeprefix("/api/v1")
        if path.startswith("/upload/"):
            return local_client.request(method, path, **kwargs)
        return SimpleNamespace(status_code=200, ok=True, json=lambda: {"items": []}, text="ok")
    page = next((Path(__file__).resolve().parents[2] / "frontend/pages").glob("2_*.py"))
    with patch("frontend.api_client.transport.request", side_effect=request):
        app = AppTest.from_file(str(page)).run(timeout=20)
        assert not list(app.exception)
        company = next(field for field in app.text_input if field.label == "公司 / 租户 ID（可选）")
        assert not company.disabled and company.value == ""
        department = next(field for field in app.selectbox if field.label == "归属部门")
        assert department.options == ["财务部", "客服部"]
        next(field for field in app.text_input if field.label == "文档标题（可选）").set_value("旧公司标题").run()
        next(field for field in app.checkbox if field.label == "我已检查处理后的文字，可以进入分块").check().run()
        assert app.session_state["upload_content_checked_" + task_id] is True
        next(field for field in app.text_input if field.label == "公司 / 租户 ID（可选）").set_value("company-b").run()
        assert not list(app.exception)
        assert next(field for field in app.text_input if field.label == "文档标题（可选）").value == ""
        assert "upload_content_checked_" + task_id not in app.session_state
        assert not any(field.label == "我已检查处理后的文字，可以进入分块" for field in app.checkbox)
        assert any("上传文件后" in info.value for info in app.info)
