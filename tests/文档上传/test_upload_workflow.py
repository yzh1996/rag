"""上传流程隔离测试；不连接业务数据库或付费模型。"""
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document as Chunk
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from core import database, upload_service as uploads, task_queue
from core.auth import AuthUser, get_current_user
from core.db_models import UploadTask, Document, DocumentVersion, KnowledgeBase, BackgroundJob


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'upload.db'}", connect_args={"check_same_thread": False})
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(engine, expire_on_commit=False))
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "files")
    monkeypatch.setattr(settings, "KNOWLEDGE_REVIEW_REQUIRED", True)
    database.init_db()
    yield
    engine.dispose()


def actor(name="alice", tenant="company-a", department="财务部", role="user"):
    return AuthUser(username=name, tenant_id=tenant, department=department, role=role, authenticated=True)


def client(user):
    from api.routes.upload import router
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: user
    return TestClient(app)


def submit(user=None, content=b"policy  text\x00\n\n\nannual leave", **metadata):
    user = user or actor()
    options = uploads.upload_options(user)
    kb = next(item for item in options["knowledge_bases"] if item["owner_department"] == user.department)
    upload = SimpleNamespace(filename="policy.txt", file=io.BytesIO(content))
    values = {"owner_department": user.department, **metadata}
    return uploads.submit_upload(upload, kb["id"], values, user)


def prepare(user=None, **kwargs):
    user = user or actor()
    task_id = submit(user, **kwargs)
    assert task_queue.run_once()
    return task_id


def preview(user=None):
    user = user or actor()
    task_id = prepare(user)
    uploads.request_chunks(task_id, "recursive", {"chunk_size": 100, "chunk_overlap": 0}, user)
    assert task_queue.run_once()
    return task_id


def test_options_are_company_and_department_scoped(isolated):
    finance = uploads.upload_options(actor())
    service = uploads.upload_options(actor(department="客服部"))
    other_company = uploads.upload_options(actor(tenant="company-b"))
    assert finance["departments"] == ["财务部"]
    assert service["departments"] == ["客服部"]
    assert len(finance["knowledge_bases"]) == 1
    assert finance["knowledge_bases"][0]["id"] != other_company["knowledge_bases"][0]["id"]
    assert len(uploads.upload_options(actor(role="admin"))["knowledge_bases"]) == 3
    assert len(uploads.upload_options(actor(role="admin"))["departments"]) == 2


def test_upload_endpoints_require_identity_when_auth_is_enabled(isolated, monkeypatch):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    assert client(AuthUser(username="anonymous")).get("/upload/options").status_code == 401
    assert client(AuthUser(username="anonymous")).get("/upload/tasks").status_code == 401


@pytest.mark.parametrize("values", [
    {"tenant_id": "company-b"}, {"owner_department": "客服部"}, {"owner_department": "研发部"},
    {"business_owner": "unknown"}, {"visibility": "departments", "allowed_departments": ["客服部"]},
    {"visibility": "users", "allowed_users": ["unknown"]},
])
def test_invalid_ownership_stops_before_file_or_parser(isolated, monkeypatch, values):
    from core.document_loader import DocumentLoader
    monkeypatch.setattr(DocumentLoader, "load_raw_file", lambda *args: pytest.fail("must not parse"))
    with pytest.raises(uploads.UploadAdmissionError) as rejection:
        submit(**values)
    result = rejection.value.payload
    assert result["status"] == "failed"
    assert result["stages"][0]["status"] == "failed"
    assert result["stages"][0]["detail"] and result["stages"][0]["next_action"]
    assert all(stage["status"] == "pending" for stage in result["stages"][1:])
    assert not list(settings.UPLOAD_DIR.glob("*"))
    with database.session_scope() as db:
        assert db.query(UploadTask).count() == 0
    assert not task_queue.run_once()


def test_forged_company_target_is_rejected_without_leaking_details(isolated):
    other = actor(tenant="company-b")
    foreign = uploads.upload_options(other)["knowledge_bases"][0]["id"]
    response = client(actor()).post("/upload/tasks", files={"file": ("policy.txt", b"policy")},
        data={"knowledge_base_id": foreign, "metadata": "{}"})
    assert response.status_code == 403
    result = response.json()
    assert result["status"] == "failed"
    assert "company-b" not in json.dumps(result)
    assert "knowledge_bases" not in result


def test_safety_failure_has_reason_and_action_without_saving_record(isolated):
    user = actor()
    kb = uploads.upload_options(user)["knowledge_bases"][0]["id"]
    response = client(user).post("/upload/tasks", files={"file": ("forged.pdf", b"not pdf")},
        data={"knowledge_base_id": kb, "metadata": json.dumps({"owner_department": "财务部"})})
    result = response.json()
    assert result["status"] == "failed"
    failed = next(stage for stage in result["stages"] if stage["status"] == "failed")
    assert failed["key"] == "safety" and "PDF" in failed["detail"]
    assert failed["next_action"]
    assert response.status_code == 400 and result["id"] is None and result["record_saved"] is False
    assert client(user).get("/upload/tasks").json()["items"] == []
    with database.session_scope() as db:
        assert db.query(UploadTask).count() == 0


def test_duplicate_is_rejected_before_parse_and_is_company_local(isolated, monkeypatch):
    from core.document_loader import DocumentLoader
    first = submit(visibility="self")
    monkeypatch.setattr(DocumentLoader, "load_raw_file", lambda *args: pytest.fail("duplicate must not parse"))
    with pytest.raises(uploads.UploadAdmissionError) as rejection:
        submit(actor(name="bob"))
    result = rejection.value.payload
    assert result["status"] == "failed"
    assert next(stage for stage in result["stages"] if stage["status"] == "failed")["key"] == "dedup"
    assert first not in json.dumps(result)
    third = submit(actor(tenant="company-b"))
    assert uploads.task_payload(third, actor(tenant="company-b"))["status"] == "queued"


def test_concurrent_duplicate_uploads_reserve_only_one_task(isolated):
    from concurrent.futures import ThreadPoolExecutor
    def try_submit(name):
        try:
            return submit(actor(name=name))
        except uploads.UploadAdmissionError as exc:
            assert exc.payload["record_saved"] is False
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(try_submit, ["alice", "bob"]))
    with database.session_scope() as db:
        records = db.query(UploadTask).all()
    assert sum(task_id is not None for task_id in ids) == 1
    assert len(records) == 1 and records[0].status == "queued" and records[0].reservation_key


def test_processing_checkpoints_separate_ocr_cleaning_and_content_confirmation(isolated):
    task_id = prepare(title="财务制度", source_reference="制度系统/123")
    result = uploads.task_payload(task_id, actor())
    assert result["status"] == "ready"
    stages = {item["key"]: item for item in result["stages"]}
    assert stages["parse"]["status"] == "passed"
    assert stages["ocr"]["status"] == "skipped"
    assert stages["clean"]["status"] == "passed"
    assert stages["chunks"]["status"] == "pending"
    assert result["before_content"] != result["after_content"]
    assert "\x00" not in result["after_content"]
    assert result["metadata"]["department_id"] == "finance"
    assert result["metadata"]["tenant_id"] == "company-a"
    assert result["metadata"]["uploaded_by"] == "alice"
    assert result["metadata"]["title"] == "财务制度"
    assert client(actor()).post(f"/upload/tasks/{task_id}/confirm").status_code == 400
    assert not task_queue.run_once()


def test_failed_stage_can_retry_from_saved_checkpoint_after_restart(isolated, monkeypatch):
    from core import content_processing
    task_id = submit()
    original = content_processing.recognize_documents
    monkeypatch.setattr(content_processing, "recognize_documents",
        lambda *args: (_ for _ in ()).throw(ValueError("OCR不可用")))
    # TXT不需要OCR，这里使用清洗阶段故障验证检查点恢复。
    original_clean = content_processing.clean_documents
    monkeypatch.setattr(content_processing, "clean_documents",
        lambda *args: (_ for _ in ()).throw(ValueError("清洗暂时失败")))
    assert task_queue.run_once()
    result = uploads.task_payload(task_id, actor())
    assert result["status"] == "failed" and result["can_retry"]
    monkeypatch.setattr(content_processing, "recognize_documents", original)
    monkeypatch.setattr(content_processing, "clean_documents", original_clean)
    from core.document_loader import DocumentLoader
    monkeypatch.setattr(DocumentLoader, "load_raw_file", lambda *args: pytest.fail("parsed checkpoint must be reused"))
    assert client(actor()).post(f"/upload/tasks/{task_id}/retry").status_code == 202
    assert task_queue.run_once()
    assert uploads.task_payload(task_id, actor())["status"] == "ready"


def test_restart_recovers_registered_task_without_auto_confirming(isolated):
    task_id = submit()
    with database.session_scope() as db:
        db.query(BackgroundJob).delete()
    assert uploads.recover_upload_tasks() == 1
    assert task_queue.run_once()
    assert uploads.task_payload(task_id, actor())["status"] == "ready"
    assert uploads.recover_upload_tasks() == 0
    assert not task_queue.run_once()


def test_chunk_preview_persists_metadata_and_isolates_readers(isolated):
    task_id = preview()
    result = client(actor()).get("/upload/tasks/" + task_id).json()
    assert result["status"] == "preview"
    assert result["chunks"][0]["metadata"]["owner_department"] == "财务部"
    assert client(actor(name="bob")).get("/upload/tasks/" + task_id).status_code == 404
    assert client(actor(tenant="company-b")).get("/upload/tasks/" + task_id).status_code == 404
    assert client(actor(name="bob")).get("/upload/tasks").json()["items"] == []
    with database.session_scope() as db:
        doc = db.get(Document, result["document_id"])
        assert doc.upload_metadata["department_id"] == "finance"
        doc.permission_scope = {"tenant_ids": ["company-a"], "users": ["bob"]}
    hidden = uploads.task_payload(task_id, actor())
    assert hidden["can_read_content"] is False
    assert "before_content" not in hidden and "chunks" not in hidden


def test_approved_chunks_are_used_for_ingestion_without_reparse(isolated, monkeypatch):
    from core import ingestion_service
    task_id = preview()
    app = client(actor())
    first = app.post(f"/upload/tasks/{task_id}/confirm")
    second = client(actor()).post(f"/upload/tasks/{task_id}/confirm")
    assert first.status_code == second.status_code == 200
    assert first.json()["job_id"] == second.json()["job_id"]
    from core.document_loader import DocumentLoader
    monkeypatch.setattr(DocumentLoader, "load_file", lambda *args: pytest.fail("must ingest approved chunks"))
    received = []
    monkeypatch.setattr(ingestion_service, "get_vector_store_manager",
        lambda: SimpleNamespace(add_documents=lambda documents: received.extend(documents) or len(documents)))
    assert task_queue.run_once()
    result = uploads.task_payload(task_id, actor())
    assert result["status"] == "candidate"
    assert result["stages"][-1]["status"] == "passed"
    assert received[0].metadata["tenant_id"] == "company-a"
    assert received[0].metadata["owner_department"] == "财务部"
    assert json.loads(received[0].metadata["permission_scope"])["departments"] == ["财务部"]


def test_failed_file_does_not_block_another_file(isolated):
    with pytest.raises(uploads.UploadAdmissionError):
        submit(content=b"")
    valid = prepare(content=b"valid policy")
    assert uploads.task_payload(valid, actor())["status"] == "ready"


def test_real_pdf_ocr_is_a_separate_required_stage(isolated, tmp_path, monkeypatch):
    import fitz
    from PIL import Image
    from core import content_processing
    image = io.BytesIO()
    Image.new("RGB", (120, 80), "white").save(image, format="PNG")
    path = tmp_path / "scan.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_image(fitz.Rect(0, 0, 120, 80), stream=image.getvalue())
        pdf.save(path)
    user = actor()
    kb = uploads.upload_options(user)["knowledge_bases"][0]["id"]
    task_id = uploads.submit_upload(SimpleNamespace(filename="scan.pdf", file=io.BytesIO(path.read_bytes())),
        kb, {"owner_department": "财务部"}, user)
    monkeypatch.setattr(settings, "OCR_ENABLED", False)
    assert task_queue.run_once()
    result = uploads.task_payload(task_id, user)
    stages = {item["key"]: item for item in result["stages"]}
    assert stages["parse"]["status"] == "passed" and stages["ocr"]["status"] == "failed"
    assert "第1页" in stages["ocr"]["detail"] and stages["ocr"]["next_action"]
    monkeypatch.setattr(content_processing, "ocr_page", lambda page: "识别后的财务制度")
    uploads.retry_upload(task_id, user)
    assert task_queue.run_once()
    result = uploads.task_payload(task_id, user)
    assert result["status"] == "ready"
    assert "识别后的财务制度" in result["after_content"]


def test_old_database_can_add_upload_fields_without_changing_records(isolated):
    with database.engine.begin() as connection:
        connection.execute(text("DROP TABLE document_versions"))
        connection.execute(text("""CREATE TABLE document_versions (
            id VARCHAR(32) PRIMARY KEY, document_id VARCHAR(32), version INTEGER,
            content_hash VARCHAR(64), file_path VARCHAR(1024), chunking_strategy VARCHAR(64),
            chunking_params JSON, status VARCHAR(32), effective_at TIMESTAMP, expired_at TIMESTAMP,
            created_at TIMESTAMP)"""))
        connection.execute(text("INSERT INTO document_versions (id, document_id, status) VALUES ('old', 'doc', 'published')"))
    database.init_db()
    with database.session_scope() as db:
        version = db.get(DocumentVersion, "old")
        assert version.status == "published" and version.prepared_chunks is None


def test_revoked_read_permission_blocks_confirmation_and_legacy_entry(isolated):
    from api.dependencies import get_vector_store_dep
    task_id = preview()
    result = uploads.task_payload(task_id, actor())
    with database.session_scope() as db:
        db.get(Document, result["document_id"]).permission_scope = {"tenant_ids": ["company-a"], "users": ["bob"]}
    app = client(actor())
    app.app.dependency_overrides[get_vector_store_dep] = lambda: object()
    assert app.post(f"/upload/tasks/{task_id}/confirm").status_code == 403
    assert app.post(f"/upload/tasks/{task_id}/chunks", json={"strategy": "recursive"}).status_code == 403
    assert app.post("/upload/file/confirm", data={"file_id": result["version_id"]}).status_code == 403


def test_account_department_change_is_applied_before_processing(isolated):
    from core.db_models import User
    with database.session_scope() as db:
        account = User(username="alice", tenant_id="company-a", department="财务部")
        db.add(account)
        db.flush()
        user_id = account.id
    user = AuthUser(username="alice", tenant_id="company-a", department="财务部", user_id=user_id, authenticated=True)
    task_id = submit(user)
    with database.session_scope() as db:
        db.get(User, user_id).department = "客服部"
    assert task_queue.run_once()
    result = uploads.task_payload(task_id, user)
    assert result["status"] == "failed" and result["stages"][0]["status"] == "failed"
    assert uploads.upload_options(user)["departments"] == ["客服部"]


def test_exhausted_worker_recovery_is_visible_and_manually_retryable(isolated):
    task_id = submit()
    with database.session_scope() as db:
        db.query(BackgroundJob).one().status = "failed"
    result = uploads.task_payload(task_id, actor())
    assert result["status"] == "failed" and result["can_retry"]
    uploads.retry_upload(task_id, actor())
    assert task_queue.run_once()
    assert uploads.task_payload(task_id, actor())["status"] == "ready"


def test_changed_source_does_not_lock_original_file_out_of_reupload(isolated):
    task_id = submit()
    with database.session_scope() as db:
        path = db.get(UploadTask, task_id).file_path
    Path(path).write_text("changed file", encoding="utf-8")
    assert task_queue.run_once()
    assert uploads.task_payload(task_id, actor())["status"] == "failed"
    new_id = submit()
    assert uploads.task_payload(new_id, actor())["status"] == "queued"


def test_bad_chunk_parameters_have_plain_message_and_can_be_corrected(isolated):
    task_id = prepare()
    uploads.request_chunks(task_id, "recursive", {"chunk_size": 100, "chunk_overlap": 100}, actor())
    assert task_queue.run_once()
    result = uploads.task_payload(task_id, actor())
    assert result["status"] == "failed" and result["stages"][-2]["next_action"]
    uploads.request_chunks(task_id, "recursive", {"chunk_size": 100, "chunk_overlap": 0}, actor())
    assert task_queue.run_once()
    assert uploads.task_payload(task_id, actor())["status"] == "preview"


@pytest.mark.parametrize("state", ["failed", "ready", "preview"])
def test_upload_page_renders_real_stage_results_and_guidance(isolated, state):
    from streamlit.testing.v1 import AppTest
    user = actor()
    if state == "failed":
        with patch("core.content_processing.clean_documents", side_effect=ValueError("清洗暂时失败")):
            task_id = prepare()
    else:
        task_id = prepare() if state == "ready" else preview()
    options = uploads.upload_options(user)
    record = uploads.task_payload(task_id, user)
    def response(method, url, **kwargs):
        if url.endswith("/upload/options"):
            value = options
        elif url.endswith("/upload/tasks"):
            value = {"items": [record]}
        elif url.endswith("/upload/tasks/" + task_id):
            value = record
        else:
            value = {"items": []}
        return SimpleNamespace(status_code=200, ok=True, json=lambda: value, text="ok")
    page = next((Path(__file__).resolve().parents[2] / "frontend/pages").glob("2_*.py"))
    with patch("frontend.api_client.transport.request", side_effect=response):
        app = AppTest.from_file(str(page)).run(timeout=20)
    assert not list(app.exception), [item.message for item in app.exception]
    assert any(item.label == "所属公司（租户）" for item in app.text_input)
    if state == "failed":
        assert any("未通过" in item.value for item in app.error)
        assert any("重新上传" in item.value for item in app.info)
    else:
        assert any(item.label == "我已检查处理后的文字，可以进入分块" for item in app.checkbox)
        assert any(item.label == "确认内容并生成分块" and item.disabled for item in app.button)
    if state == "preview":
        assert any(item.label == "确认入库" and item.disabled for item in app.button)
