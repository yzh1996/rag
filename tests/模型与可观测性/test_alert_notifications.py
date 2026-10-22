"""Notification contracts without real mailboxes, robots or business services."""
import io
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

import pytest
import yaml

from scripts import alert_dingtalk as ding
from scripts.prepare_enterprise import render_configuration, write_configuration

TOKEN = "private-alert-test-token-123456789"
WEBHOOK = "https://oapi.dingtalk.com/robot/send?access_token=private-robot-token"


def values(**overrides):
    return {"ALERT_RECEIVER_TOKEN": TOKEN, "ALERT_EMAIL_ENABLED": "true",
        "ALERT_SMTP_HOST": "smtp.example.com", "ALERT_SMTP_PORT": "587",
        "ALERT_SMTP_FROM": "rag@example.com", "ALERT_SMTP_USERNAME": "rag@example.com",
        "ALERT_SMTP_PASSWORD": 'authorization-code:#"with-quotes',
        "ALERT_EMAIL_TO": "ops@example.com, backup@example.com", "ALERT_DINGTALK_ENABLED": "true",
        "ALERT_DINGTALK_WEBHOOK": WEBHOOK, "ALERT_DINGTALK_SECRET": "SEC-test-key", **overrides}


def batch(status="firing"):
    return {"status": status, "alerts": [{"status": status, "fingerprint": "test-alert",
        "labels": {"alertname": "RAGUnavailable", "severity": "critical", "instance": "rag-api"},
        "annotations": {"summary": "服务不可用", "description": "检查健康接口"},
        "startsAt": "2026-10-13T01:00:00Z", "endsAt": "2026-10-13T01:05:00Z"}]}


def test_existing_webhook_and_disabled_defaults_are_preserved():
    config = json.loads(render_configuration({"ALERT_RECEIVER_TOKEN": TOKEN,
        "ALERT_NOTIFICATION_URL": "https://gateway.example.com/alerts"}))
    receiver = config["receivers"][0]
    assert "email_configs" not in receiver
    assert [item["url"] for item in receiver["webhook_configs"]] == [
        "http://backend:8000/api/v1/operations/alerts", "https://gateway.example.com/alerts"]
    assert config["route"]["repeat_interval"] == "4h"


@pytest.mark.parametrize("mode,backend", [("docker", "backend:8000"), ("local", "host.docker.internal:8001")])
@pytest.mark.parametrize("port,require_tls", [("587", True), ("465", False)])
def test_both_channels_are_independent_and_tls_is_configured(mode, backend, port, require_tls):
    config = yaml.safe_load(render_configuration(values(ALERT_SMTP_PORT=port), mode))
    receiver = config["receivers"][0]
    email = receiver["email_configs"][0]
    assert email["smarthost"] == f"smtp.example.com:{port}"
    assert email["require_tls"] is require_tls
    assert email["auth_password"] == values()["ALERT_SMTP_PASSWORD"]
    assert email["send_resolved"] is True
    assert "StartsAt" in email["text"] and "EndsAt" in email["text"]
    webhooks = receiver["webhook_configs"]
    assert webhooks[0]["url"] == f"http://{backend}/api/v1/operations/alerts"
    assert webhooks[1]["url"] == "http://alert-notifier:8080/alerts"
    assert webhooks[1]["http_config"]["authorization"]["credentials"] == TOKEN
    assert webhooks[1]["send_resolved"] is True
    assert webhooks[1]["max_alerts"] == 100
    assert WEBHOOK not in render_configuration(values())


@pytest.mark.parametrize("overrides", [
    {"ALERT_EMAIL_TO": ""}, {"ALERT_SMTP_PASSWORD": ""}, {"ALERT_SMTP_PORT": "70000"},
    {"ALERT_SMTP_HOST": "smtp.example.com:587"}, {"ALERT_EMAIL_ENABLED": "maybe"},
    {"ALERT_DINGTALK_WEBHOOK": "http://oapi.dingtalk.com/robot/send?access_token=x"},
    {"ALERT_DINGTALK_WEBHOOK": "https://other.example.com/robot/send?access_token=x"},
    {"ALERT_NOTIFICATION_URL": "http://gateway.example.com/alerts"},
])
def test_incomplete_or_unsafe_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        render_configuration(values(**overrides))


def test_generated_config_requires_explicit_replace_and_keeps_env(tmp_path):
    env = tmp_path / ".env"
    env.write_text("original private configuration\n", encoding="utf-8")
    output = write_configuration(tmp_path, values(), "local")
    previous = output.read_bytes()
    with pytest.raises(ValueError, match="--force"):
        write_configuration(tmp_path, values(ALERT_EMAIL_ENABLED="false"), "local")
    assert output.read_bytes() == previous
    with pytest.raises(ValueError):
        write_configuration(tmp_path, values(ALERT_EMAIL_TO=""), "local", force=True)
    assert output.read_bytes() == previous
    write_configuration(tmp_path, values(ALERT_EMAIL_ENABLED="false"), "local", force=True)
    assert "email_configs" not in json.loads(output.read_text(encoding="utf-8"))["receivers"][0]
    assert env.read_text(encoding="utf-8") == "original private configuration\n"


def test_notifier_deployment_has_no_business_dependencies_or_public_port():
    compose = yaml.safe_load((Path(__file__).resolve().parents[2] / "deploy/enterprise/compose.yml").read_text(encoding="utf-8"))
    service = compose["services"]["alert-notifier"]
    assert "depends_on" not in service
    assert "ports" not in service
    assert len(service["volumes"]) == 1
    assert "alert_dingtalk.py" in service["volumes"][0]
    assert service["read_only"] is True


def test_signed_url_keeps_token_and_replaces_old_signature():
    config = ding.DingTalkConfig.from_environment(values(ALERT_DINGTALK_WEBHOOK=WEBHOOK + "&timestamp=old&sign=old"))
    query = parse_qs(urlsplit(ding.signed_url(config, 1600000000000)).query)
    assert query["access_token"] == ["private-robot-token"]
    assert query["timestamp"] == ["1600000000000"]
    assert query["sign"] == ["ub8K8700nNAdq2S3JNS2eyvA+jAdo63jSkzr+hFY5Z4="]
    assert ding.signed_url(ding.DingTalkConfig(True, TOKEN, WEBHOOK)) == WEBHOOK


@pytest.mark.parametrize("overrides", [
    {"ALERT_RECEIVER_TOKEN": "short"}, {"ALERT_DINGTALK_ENABLED": "yes"},
    {"ALERT_DINGTALK_TIMEOUT_SECONDS": "0"}, {"ALERT_DINGTALK_TIMEOUT_SECONDS": "15"},
    {"ALERT_DINGTALK_WEBHOOK": WEBHOOK + "&access_token=another"},
    {"ALERT_DINGTALK_WEBHOOK": WEBHOOK.replace("oapi.dingtalk.com", "oapi.dingtalk.com.evil.example")},
])
def test_notifier_configuration_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        ding.DingTalkConfig.from_environment(values(**overrides))


def test_messages_include_trigger_recovery_and_size_limit():
    firing = ding.message_text(batch())
    recovery = ding.message_text(batch("resolved"))
    assert all(item in firing for item in ["RAG 告警", "告警触发", "严重", "RAGUnavailable", "服务不可用", "2026-10-13T01:00:00Z"])
    assert "告警恢复" in recovery and "恢复时间：2026-10-13T01:05:00Z" in recovery
    payload = batch()
    payload["alerts"][0]["annotations"]["description"] = "长文本" * 10000
    text = ding.message_text(payload)
    assert len(text.encode("utf-8")) <= ding.MAX_MESSAGE_BYTES
    assert "内容已截断" in text
    payload = batch()
    payload["truncatedAlerts"] = 4
    assert "另有 4 条告警" in ding.message_text(payload)


def test_dingtalk_payload_and_response_are_checked(monkeypatch):
    calls = []

    def open_request(request, timeout):
        calls.append((request, timeout))
        return io.BytesIO(b'{"errcode":0,"errmsg":"ok"}')

    monkeypatch.setattr(ding.opener, "open", open_request)
    ding.send_dingtalk(ding.DingTalkConfig.from_environment(values()), batch())
    request, timeout = calls[0]
    assert request.full_url.startswith(WEBHOOK + "&timestamp=")
    assert timeout == 10
    payload = json.loads(request.data)
    assert payload["msgtype"] == "text"
    assert "服务不可用" in payload["text"]["content"]
    for response in [b'{"errcode":310000}', b'{"errmsg":"ok"}', b'{"errcode":false}', b'not json']:
        monkeypatch.setattr(ding.opener, "open", lambda *args, response=response, **kwargs: io.BytesIO(response))
        with pytest.raises((ValueError, RuntimeError)):
            ding.send_dingtalk(ding.DingTalkConfig.from_environment(values()), batch())


@pytest.fixture
def adapter():
    servers = []

    def start(config=None):
        server = ThreadingHTTPServer(("127.0.0.1", 0), ding.make_handler(config or ding.DingTalkConfig.from_environment(values())))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}"

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def post(url, payload, token=TOKEN):
    request = Request(url + "/alerts", data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as exc:
        response = exc
    with response:
        return response.code, json.loads(response.read())


def test_http_receiver_auth_validation_and_delivery(adapter, monkeypatch):
    deliveries = []
    monkeypatch.setattr(ding, "send_dingtalk", lambda config, payload: deliveries.append(payload))
    url = adapter()
    with urlopen(url + "/health", timeout=5) as response:
        assert json.loads(response.read())["status"] == "ok"
    assert post(url, batch(), token="wrong")[0] == 401
    assert post(url, batch(), token="é")[0] == 401
    for payload in [{}, {"alerts": []}, {"alerts": [{"status": "bad"}]},
        {"alerts": [{"status": "firing", "labels": {"name": 1}}]}]:
        assert post(url, payload)[0] == 400
    assert post(url, {"padding": "x" * ding.MAX_BODY_BYTES})[0] == 413
    assert not deliveries
    assert post(url, batch()) == (200, {"received": 1})
    assert post(url, batch("resolved"))[0] == 200
    assert len(deliveries) == 2


def test_failed_send_returns_retryable_status_without_leaking_secrets(adapter, monkeypatch, caplog):
    attempts = []

    def send(config, payload):
        attempts.append(payload)
        if len(attempts) == 1:
            raise URLError(WEBHOOK + config.secret + TOKEN)

    monkeypatch.setattr(ding, "send_dingtalk", send)
    url = adapter()
    status, body = post(url, batch())
    assert status == 502 and "retry" in body["error"]
    assert all(secret not in caplog.text + json.dumps(body) for secret in [WEBHOOK, TOKEN, "SEC-test-key"])
    assert post(url, batch())[0] == 200
    assert len(attempts) == 2


def test_disabled_channel_is_not_acknowledged(adapter):
    assert post(adapter(ding.DingTalkConfig(False)), batch())[0] == 503
