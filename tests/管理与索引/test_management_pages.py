"""页面渲染冒烟检查，接口使用固定响应，不连接运行中的服务。"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest


PAGES = sorted(path for path in (Path(__file__).resolve().parents[2] / "frontend" / "pages").glob("*.py")
    if path.name.startswith(("1_", "2_", "4_", "5_")))


def fake_response(method, url, **kwargs):
    if url.endswith("/knowledge/stats"):
        value = {"document_count": 2, "unique_sources": 1, "store_type": "chroma"}
    elif url.endswith("/documents"):
        value = {"items": [{"id": "doc", "original_filename": "policy.txt", "status": "published",
            "current_version": 1, "category": "policies", "tenant_id": "a", "permission_scope": {"tenant_ids": ["a"]}}]}
    elif url.endswith("/documents/doc"):
        value = {"versions": [{"version": 1, "status": "published", "chunking_strategy": "recursive"}]}
    elif url.endswith("/evaluation/runs"):
        value = {"items": [{"id": "run", "dataset_name": "default", "status": "completed", "total_cases": 1, "passed_cases": 1}]}
    elif url.endswith("/evaluation/runs/run"):
        value = {"id": "run", "status": "completed", "metrics": {"faithfulness": 0.9, "quality_gate_passed": True,
            "dataset_revision": "text-revision", "complete_suite": True}, "results": []}
    else:
        value = {"items": []}
    return SimpleNamespace(status_code=200, ok=True, json=lambda: value, text="ok")


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_management_pages_render_without_live_backend(page):
    with patch("frontend.api_client.transport.request", side_effect=fake_response):
        app = AppTest.from_file(str(page)).run(timeout=20)
    assert not list(app.exception), [exception.message for exception in app.exception]


@pytest.mark.parametrize("current_version", [0, 1])
def test_document_chunking_is_readonly_and_update_controls_are_collapsed(current_version):
    def response(method, url, **kwargs):
        if url.endswith("/documents"):
            value = {"items": [{"id": "doc", "original_filename": "policy.txt", "status": "candidate",
                "current_version": current_version, "category": "policies", "tenant_id": "a", "permission_scope": {}}]}
        elif url.endswith("/documents/doc"):
            value = {"versions": [{"version": 1, "status": "candidate", "chunking_strategy": "parent_child"}]}
        else:
            return fake_response(method, url, **kwargs)
        return SimpleNamespace(status_code=200, ok=True, json=lambda: value)

    with patch("frontend.api_client.transport.request", side_effect=response) as requests:
        app = AppTest.from_string("from frontend.management import render_management\nrender_management()").run(timeout=20)
        assert not list(app.exception)
        label = next(box for box in app.selectbox if box.label == "文档").options[0]
        assert ("尚未正式发布" if current_version == 0 else "当前生效第1版") in label
        assert "当前第0版" not in label and "candidate" not in label
        assert app.dataframe[0].value["分块方式"].tolist() == ["父子分块"]
        update = next(section for section in app.expander if section.label == "更新已有文件（可选）")
        assert update.proto.expanded is False
        assert any(box.label == "更新文件的分块方式" for box in update.selectbox)
        assert not any(box.label == "新版本分块方式" for box in app.selectbox)
        assert all(call.args[0] == "GET" for call in requests.call_args_list)
        app.radio[0].set_value("知识库审核与发布").run()
        assert not list(app.exception)
        assert not list(app.selectbox) and not list(app.get("file_uploader"))
        assert any(button.label == "构建待发布知识库版本" for button in app.button)


def test_update_file_still_sends_selected_strategy_only_when_submitted():
    upload = SimpleNamespace(name="updated.txt", type="text/plain", getvalue=lambda: b"updated policy")
    with patch("frontend.api_client.transport.request", side_effect=fake_response) as requests, patch(
        "streamlit.file_uploader", return_value=upload):
        app = AppTest.from_string("from frontend.management import render_documents\nrender_documents()").run(timeout=20)
        assert all(call.args[0] == "GET" for call in requests.call_args_list)
        next(box for box in app.selectbox if box.label == "更新文件的分块方式").select("parent_child")
        next(button for button in app.button if button.label == "提交更新文件").click().run()
    assert not list(app.exception)
    submitted = [call for call in requests.call_args_list if call.args[0] == "POST"]
    assert len(submitted) == 1 and submitted[0].args[1].endswith("/documents/doc/versions")
    assert submitted[0].kwargs["data"]["chunking_strategy"] == "parent_child"
    assert submitted[0].kwargs["files"]["file"] == ("updated.txt", b"updated policy", "text/plain")
