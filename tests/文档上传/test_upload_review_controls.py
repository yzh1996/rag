"""分块对比、处理记录删除和上传失败不落库的隔离验证。"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
from sqlalchemy import text

from config.settings import settings
from core import database, task_queue, upload_service as uploads
from core.auth import actor_payload
from core.db_models import BackgroundJob, Document, DocumentVersion, UploadTask
from tests.文档上传.test_upload_workflow import isolated, actor, client, prepare, preview, submit


def forward(api):
    def request(method, url, **kwargs):
        return api.request(method, urlsplit(url).path.removeprefix("/api/v1"), **kwargs)
    return request


@pytest.mark.parametrize("status", ["checking", "queued", "processing", "chunking", "submitted"])
def test_executing_or_waiting_record_cannot_be_deleted(isolated, status):
    task_id = prepare()
    with database.session_scope() as db:
        db.get(UploadTask, task_id).status = status
    api = client(actor())
    assert not api.get("/upload/tasks/" + task_id).json()["can_delete"]
    response = api.delete("/upload/tasks/" + task_id)
    assert response.status_code == 409 and "仍在处理" in response.json()["detail"]
    with database.session_scope() as db:
        assert db.get(UploadTask, task_id).deleted_at is None


def test_delete_ready_record_persists_and_allows_new_upload(isolated):
    task_id = prepare()
    api = client(actor())
    assert api.get("/upload/tasks/" + task_id).json()["can_delete"]
    assert api.delete("/upload/tasks/" + task_id).status_code == 200
    assert api.get("/upload/tasks").json()["items"] == []
    assert api.get("/upload/tasks/" + task_id).status_code == 404
    assert api.post(f"/upload/tasks/{task_id}/chunks", json={"strategy": "recursive"}).status_code == 404
    with database.session_scope() as db:
        task = db.get(UploadTask, task_id)
        assert task.deleted_at is not None and task.reservation_key is None
        assert Path(task.file_path).exists()
    assert uploads.recover_upload_tasks() == 0
    assert submit() != task_id


def test_delete_unconfirmed_preview_releases_dedup_and_closes_confirmation(isolated):
    from api.dependencies import get_vector_store_dep
    task_id = preview()
    api = client(actor())
    result = api.get("/upload/tasks/" + task_id).json()
    assert api.delete("/upload/tasks/" + task_id).status_code == 200
    assert api.post(f"/upload/tasks/{task_id}/confirm").status_code == 404
    api.app.dependency_overrides[get_vector_store_dep] = lambda: object()
    assert api.post("/upload/file/confirm", data={"file_id": result["version_id"]}).status_code == 409
    with database.session_scope() as db:
        version = db.get(DocumentVersion, result["version_id"])
        assert version.status == "failed" and version.prepared_chunks
        assert db.get(Document, result["document_id"]).status == "failed"
    assert submit() != task_id


@pytest.mark.parametrize("published", [False, True])
def test_delete_completed_record_preserves_ingested_document(isolated, monkeypatch, published):
    from api.dependencies import get_vector_store_dep
    from core import ingestion_service
    task_id = preview()
    api = client(actor())
    assert api.post(f"/upload/tasks/{task_id}/confirm").status_code == 200
    monkeypatch.setattr(ingestion_service, "get_vector_store_manager",
        lambda: SimpleNamespace(add_documents=lambda chunks: len(chunks)))
    assert task_queue.run_once()
    result = api.get("/upload/tasks/" + task_id).json()
    if published:
        with database.session_scope() as db:
            db.get(DocumentVersion, result["version_id"]).status = "published"
            db.get(Document, result["document_id"]).status = "published"
    assert api.delete("/upload/tasks/" + task_id).status_code == 200
    api.app.dependency_overrides[get_vector_store_dep] = lambda: object()
    assert api.post("/upload/file/confirm", data={"file_id": result["version_id"]}).status_code == 404
    with database.session_scope() as db:
        version = db.get(DocumentVersion, result["version_id"])
        document = db.get(Document, result["document_id"])
        assert version.status == ("published" if published else "candidate")
        assert document.status == ("published" if published else "candidate")
        assert version.prepared_chunks and document.permission_scope == result["permission_scope"]
        assert Path(version.file_path).exists()
    assert api.get("/upload/tasks").json()["items"] == []
    assert uploads.recover_upload_tasks() == 0
    with pytest.raises(uploads.UploadAdmissionError) as rejection:
        submit()
    assert rejection.value.status_code == 409


@pytest.mark.parametrize("job_status", ["pending", "running"])
def test_ingestion_in_progress_cannot_be_deleted(isolated, job_status):
    from core.db_models import IngestionJob
    task_id = preview()
    api = client(actor())
    job_id = api.post(f"/upload/tasks/{task_id}/confirm").json()["job_id"]
    with database.session_scope() as db:
        db.get(IngestionJob, job_id).status = job_status
    assert api.delete("/upload/tasks/" + task_id).status_code == 409


def test_delete_is_company_and_uploader_scoped(isolated):
    task_id = prepare()
    assert client(actor(name="bob")).delete("/upload/tasks/" + task_id).status_code == 404
    assert client(actor(tenant="company-b")).delete("/upload/tasks/" + task_id).status_code == 404
    assert client(actor()).get("/upload/tasks/" + task_id).json()["can_delete"]


def test_old_rejected_record_can_be_removed(isolated):
    with database.session_scope() as db:
        task = UploadTask(tenant_id="company-a", username="alice", actor=actor_payload(actor()),
            filename="old.pdf", status="failed", stages=[{"key": "safety", "label": "基础文件安全校验",
                "status": "failed", "detail": "旧失败记录", "next_action": "请重新上传"}])
        db.add(task)
        db.flush()
        task_id = task.id
    api = client(actor())
    assert api.delete("/upload/tasks/" + task_id).status_code == 200
    assert api.get("/upload/tasks").json()["items"] == []


def test_schema_upgrade_preserves_old_upload_records(isolated):
    task_id = prepare()
    with database.engine.begin() as connection:
        connection.execute(text("ALTER TABLE upload_tasks DROP COLUMN deleted_at"))
    database.init_db()
    with database.session_scope() as db:
        task = db.get(UploadTask, task_id)
        assert task.deleted_at is None and task.status == "ready" and task.filename == "policy.txt"


@pytest.mark.parametrize("strategy", ["recursive", "parent_child"])
def test_page_shows_full_original_and_all_chunks_without_selector(isolated, strategy):
    from streamlit.testing.v1 import AppTest
    task_id = prepare(content=("年度休假制度与工资规定。" * 120).encode("utf-8"))
    params = {"chunk_size": 100, "chunk_overlap": 0} if strategy == "recursive" else {
        "parent_size": 300, "child_size": 100, "child_overlap": 10}
    uploads.request_chunks(task_id, strategy, params, actor())
    assert task_queue.run_once()
    record = uploads.task_payload(task_id, actor())
    with patch("frontend.api_client.transport.request", side_effect=forward(client(actor()))):
        app = AppTest.from_string("from frontend.upload_flow import render_upload_flow\nrender_upload_flow()").run(timeout=20)
    assert not list(app.exception)
    assert not any(item.label == "查看片段" for item in app.selectbox)
    rendered = app.get("iframe")[-1].proto.srcdoc
    assert "原文（清洗后）" in rendered and record["after_content"] in rendered
    assert rendered.count('<article class="chunk-card">') == len(record["chunks"])
    if strategy == "parent_child":
        parents = {chunk["metadata"]["parent_id"] for chunk in record["chunks"]}
        assert rendered.count('<section class="parent-group">') == len(parents)
        assert "父块 1" in rendered and "子块 1.1" in rendered
    assert any(item.label == "确认入库" and item.disabled for item in app.button)


def test_page_delete_updates_list_and_selected_detail(isolated):
    from streamlit.testing.v1 import AppTest
    task_id = prepare()
    api = client(actor())
    with patch("frontend.api_client.transport.request", side_effect=forward(api)):
        app = AppTest.from_string("from frontend.upload_flow import render_upload_flow\nrender_upload_flow()").run(timeout=20)
        next(item for item in app.button if item.label == "× 删除记录").click().run()
    assert not list(app.exception)
    assert api.get("/upload/tasks").json()["items"] == []
    assert not any(item.label == "查看文件处理详情" for item in app.selectbox)
    assert any("处理记录已删除" in item.value for item in app.success)


def test_page_upload_error_has_guidance_and_creates_no_record(isolated):
    from streamlit.testing.v1 import AppTest
    api = client(actor())
    bad_file = SimpleNamespace(name="bad.pdf", getvalue=lambda: b"not a PDF")
    with patch("frontend.api_client.transport.request", side_effect=forward(api)), patch(
        "streamlit.file_uploader", return_value=[bad_file]):
        app = AppTest.from_string("from frontend.upload_flow import render_upload_flow\nrender_upload_flow()").run(timeout=20)
        next(item for item in app.button if item.label == "上传并开始检查").click().run()
    assert not list(app.exception)
    assert any("bad.pdf" in item.value and "未通过" in item.value for item in app.error)
    assert any("检查文件格式" in item.value for item in app.info)
    with database.session_scope() as db:
        assert db.query(UploadTask).count() == db.query(BackgroundJob).count() == 0
    assert not list(settings.UPLOAD_DIR.glob("*"))


def test_confirmation_keeps_current_file_instead_of_old_failure(isolated):
    from streamlit.testing.v1 import AppTest
    task_id = preview()
    with database.session_scope() as db:
        failed = UploadTask(tenant_id="company-a", username="alice", actor=actor_payload(actor()),
            filename="old-duplicate.txt", status="failed", stages=[{"key": "dedup", "label": "完全相同文件去重",
                "status": "failed", "detail": "该文件与已有提交完全相同", "next_action": "请继续已有记录"}])
        db.add(failed)
        db.flush()
        failed_id = failed.id
    api = client(actor())
    with patch("frontend.api_client.transport.request", side_effect=forward(api)) as requests:
        app = AppTest.from_string("from frontend.upload_flow import _processing_records\n_processing_records()").run(timeout=20)
        assert app.selectbox(key="upload_selected_task").value == failed_id
        assert any("完全相同" in error.value for error in app.error)
        app.selectbox(key="upload_selected_task").select(task_id).run()
        app.checkbox(key="upload_chunks_checked_" + task_id).check().run()
        app.button(key="upload_confirm_" + task_id).click().run()
        assert not list(app.exception) and not list(app.error)
        assert app.selectbox(key="upload_selected_task").value == task_id
        assert api.get("/upload/tasks/" + task_id).json()["status"] == "submitted"
        assert any("policy.txt" in message.value and "已提交后台" in message.value for message in app.success)
        assert sum(call.args[0] == "POST" and call.args[1].endswith("/confirm") for call in requests.call_args_list) == 1


def test_polling_status_changes_keep_selected_file_and_allow_manual_switch(isolated, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from core import ingestion_service
    task_id = preview()
    newer_id = prepare(content=b"another policy")
    api = client(actor())
    monkeypatch.setattr(ingestion_service, "get_vector_store_manager",
        lambda: SimpleNamespace(add_documents=lambda chunks: len(chunks)))
    with patch("frontend.api_client.transport.request", side_effect=forward(api)):
        app = AppTest.from_string("from frontend.upload_flow import _processing_records\n_processing_records()").run(timeout=20)
        app.selectbox(key="upload_selected_task").select(task_id).run()
        assert api.post(f"/upload/tasks/{task_id}/confirm").status_code == 200
        app.run()
        assert app.selectbox(key="upload_selected_task").value == task_id
        assert task_queue.run_once()
        app.run()
        assert not list(app.exception) and not list(app.error)
        assert app.selectbox(key="upload_selected_task").value == task_id
        assert api.get("/upload/tasks/" + task_id).json()["status"] == "candidate"
        app.selectbox(key="upload_selected_task").select(newer_id).run()
        app.run()
        assert not list(app.exception)
        assert app.selectbox(key="upload_selected_task").value == newer_id


def test_record_list_changes_preserve_selection_and_delete_uses_valid_fallback(isolated):
    from streamlit.testing.v1 import AppTest
    first = prepare(content=b"first policy")
    second = prepare(content=b"second policy")
    api = client(actor())
    with patch("frontend.api_client.transport.request", side_effect=forward(api)):
        app = AppTest.from_string("from frontend.upload_flow import _processing_records\n_processing_records()").run(timeout=20)
        app.selectbox(key="upload_selected_task").select(first).run()
        third = prepare(content=b"third policy")
        app.run()
        assert app.selectbox(key="upload_selected_task").value == first
        assert api.delete("/upload/tasks/" + second).status_code == 200
        app.run()
        assert app.selectbox(key="upload_selected_task").value == first
        app.button(key="upload_delete_" + first).click().run()
        assert not list(app.exception) and not list(app.error)
        assert app.selectbox(key="upload_selected_task").value == third
        assert app.session_state["upload_current_task"] == third
        assert api.get("/upload/tasks/" + first).status_code == 404
