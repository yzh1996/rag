"""Independent, authenticated Alertmanager-to-DingTalk adapter (stdlib only)."""
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

logger = logging.getLogger("rag.alert_dingtalk")
MAX_BODY_BYTES = 262144
MAX_MESSAGE_BYTES = 18000


@dataclass(frozen=True)
class DingTalkConfig:
    enabled: bool
    token: str = ""
    webhook: str = ""
    secret: str = ""
    timeout: float = 10.0

    @classmethod
    def from_environment(cls, values):
        enabled = str(values.get("ALERT_DINGTALK_ENABLED") or "false").lower()
        if enabled not in {"true", "false"}:
            raise ValueError("ALERT_DINGTALK_ENABLED must be true or false")
        if enabled == "false":
            return cls(enabled=False)
        token = values.get("ALERT_RECEIVER_TOKEN") or ""
        if len(token) < 24 or token.startswith("replace"):
            raise ValueError("Set a unique ALERT_RECEIVER_TOKEN of at least 24 characters")
        webhook = values.get("ALERT_DINGTALK_WEBHOOK") or ""
        parsed = urlsplit(webhook)
        access_tokens = [value for key, value in parse_qsl(parsed.query) if key == "access_token"]
        if (parsed.scheme != "https" or parsed.netloc != "oapi.dingtalk.com"
                or parsed.path != "/robot/send" or parsed.fragment
                or len(access_tokens) != 1 or not access_tokens[0]):
            raise ValueError("Set a valid HTTPS custom robot ALERT_DINGTALK_WEBHOOK")
        timeout = float(values.get("ALERT_DINGTALK_TIMEOUT_SECONDS") or "10")
        if not 0 < timeout <= 10:
            raise ValueError("ALERT_DINGTALK_TIMEOUT_SECONDS must be greater than 0 and at most 10")
        return cls(True, token, webhook, values.get("ALERT_DINGTALK_SECRET") or "", timeout)


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


opener = build_opener(NoRedirects())


def signed_url(config, timestamp=None):
    if not config.secret:
        return config.webhook
    timestamp = str(timestamp if timestamp is not None else int(time.time() * 1000))
    digest = hmac.new(config.secret.encode("utf-8"),
        f"{timestamp}\n{config.secret}".encode("utf-8"), hashlib.sha256).digest()
    parsed = urlsplit(config.webhook)
    query = [(key, value) for key, value in parse_qsl(parsed.query) if key not in {"timestamp", "sign"}]
    query.extend([("timestamp", timestamp), ("sign", base64.b64encode(digest).decode("ascii"))])
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), ""))


def validate_batch(payload):
    alerts = payload.get("alerts") if isinstance(payload, dict) else None
    if not isinstance(alerts, list) or not 1 <= len(alerts) <= 100:
        raise ValueError("Expected 1 to 100 alerts")
    for alert in alerts:
        if not isinstance(alert, dict) or alert.get("status") not in {"firing", "resolved"}:
            raise ValueError("Invalid alert status")
        for key in ("labels", "annotations"):
            value = alert.get(key, {})
            if not isinstance(value, dict) or not all(isinstance(item, str) for item in value.values()):
                raise ValueError("Invalid alert labels or annotations")
        for key in ("startsAt", "endsAt"):
            if key in alert and not isinstance(alert[key], str):
                raise ValueError("Invalid alert time")
    truncated = payload.get("truncatedAlerts", 0)
    if type(truncated) is not int or truncated < 0:
        raise ValueError("Invalid truncated alert count")
    return payload


def message_text(payload):
    lines = ["RAG 告警通知"]
    for alert in payload["alerts"]:
        labels, annotations = alert.get("labels", {}), alert.get("annotations", {})
        severity = labels.get("severity", "warning")
        lines.extend(["", "状态：" + ("告警触发" if alert["status"] == "firing" else "告警恢复"),
            "名称：" + labels.get("alertname", "unknown"),
            "级别：" + {"critical": "严重", "warning": "警告", "info": "提示"}.get(severity, severity),
            "目标：" + labels.get("instance", "未提供"), "开始时间：" + alert.get("startsAt", "未提供"),
            "原因：" + annotations.get("summary", labels.get("alertname", "未提供"))])
        if alert["status"] == "resolved":
            lines.append("恢复时间：" + alert.get("endsAt", "未提供"))
        if annotations.get("description"):
            lines.append("说明：" + annotations["description"])
    if payload.get("truncatedAlerts"):
        lines.append(f"本次分组另有 {payload['truncatedAlerts']} 条告警，请查看 Alertmanager。")
    text = "\n".join(lines)
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_MESSAGE_BYTES:
        suffix = "\n内容已截断，请查看 Alertmanager 获取完整告警。"
        text = encoded[:MAX_MESSAGE_BYTES - len(suffix.encode("utf-8"))].decode("utf-8", errors="ignore") + suffix
    return text


def send_dingtalk(config, payload):
    data = json.dumps({"msgtype": "text", "text": {"content": message_text(payload)}}, ensure_ascii=False).encode("utf-8")
    request = Request(signed_url(config), data=data, headers={"Content-Type": "application/json"}, method="POST")
    with opener.open(request, timeout=config.timeout) as response:
        result = json.loads(response.read(65536))
    if not isinstance(result, dict) or type(result.get("errcode")) is not int:
        raise RuntimeError("DingTalk returned an invalid response")
    if result["errcode"] != 0:
        raise RuntimeError(f"DingTalk rejected notification (errcode={result['errcode']})")


def make_handler(config):
    class AlertHandler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def respond(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                return self.respond(404, {"error": "not found"})
            self.respond(200, {"status": "ok", "dingtalk_enabled": config.enabled})

        def do_POST(self):
            if self.path != "/alerts":
                return self.respond(404, {"error": "not found"})
            if not config.enabled:
                return self.respond(503, {"error": "DingTalk channel is disabled"})
            expected = "Bearer " + config.token
            if not hmac.compare_digest(self.headers.get("Authorization", "").encode("utf-8"), expected.encode("utf-8")):
                return self.respond(401, {"error": "invalid alert receiver credentials"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self.respond(400, {"error": "invalid content length"})
            if not 0 < length <= MAX_BODY_BYTES:
                return self.respond(413, {"error": "invalid request size"})
            try:
                payload = validate_batch(json.loads(self.rfile.read(length)))
            except (ValueError, TypeError):
                return self.respond(400, {"error": "invalid alert payload"})
            try:
                send_dingtalk(config, payload)
            except Exception as exc:
                # Never log exception text: it can contain webhook credentials.
                logger.warning("DingTalk delivery failed (%s); Alertmanager will retry", type(exc).__name__)
                return self.respond(502, {"error": "DingTalk delivery failed; retry required"})
            logger.info("DingTalk delivered %d alerts", len(payload["alerts"]))
            self.respond(200, {"received": len(payload["alerts"])})

        def log_message(self, format, *args):
            # Omit request paths and headers, which may contain credentials.
            pass

    return AlertHandler


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = DingTalkConfig.from_environment(os.environ)
    server = ThreadingHTTPServer(("0.0.0.0", 8080), make_handler(config))
    logger.info("Alert notification adapter listening on port 8080; DingTalk enabled=%s", config.enabled)
    server.serve_forever()


if __name__ == "__main__":
    main()
