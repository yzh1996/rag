"""Render Alertmanager credential configuration without modifying local .env."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from dotenv import dotenv_values


def enabled(values, key):
    value = str(values.get(key) or "false").lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{key} must be true or false")
    return value == "true"


def required(values, key):
    value = values.get(key) or ""
    if not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"Set {key} before enabling this notification channel")
    return value


def render_configuration(values, mode="docker"):
    if mode not in {"docker", "local"}:
        raise ValueError("Deployment mode must be docker or local")
    token = values.get("ALERT_RECEIVER_TOKEN") or ""
    if len(token) < 24 or token.startswith("replace"):
        raise ValueError("Set a unique ALERT_RECEIVER_TOKEN of at least 24 characters")
    backend = "host.docker.internal:8001" if mode == "local" else "backend:8000"
    webhooks = [{"url": f"http://{backend}/api/v1/operations/alerts", "send_resolved": True,
        "http_config": {"authorization": {"credentials": token}}}]
    receiver = {"name": "rag", "webhook_configs": webhooks}
    notification = values.get("ALERT_NOTIFICATION_URL", "")
    if notification:
        if not notification.startswith("https://"):
            raise ValueError("External alert notification gateway must use HTTPS")
        webhooks.append({"url": notification, "send_resolved": True})
    if enabled(values, "ALERT_EMAIL_ENABLED"):
        host = required(values, "ALERT_SMTP_HOST")
        port = int(values.get("ALERT_SMTP_PORT") or "587")
        if not 1 <= port <= 65535 or any(char in host for char in ":/ @"):
            raise ValueError("Set a valid ALERT_SMTP_HOST and ALERT_SMTP_PORT")
        receiver["email_configs"] = [{
            "to": required(values, "ALERT_EMAIL_TO"),
            "from": required(values, "ALERT_SMTP_FROM"),
            "smarthost": f"{host}:{port}",
            "auth_username": required(values, "ALERT_SMTP_USERNAME"),
            "auth_password": required(values, "ALERT_SMTP_PASSWORD"),
            # Port 465 uses implicit TLS; other ports must negotiate STARTTLS.
            "require_tls": port != 465,
            "send_resolved": True,
            "headers": {"Subject": '[RAG 告警] {{ if eq .Status "firing" }}异常{{ else }}恢复{{ end }} {{ .CommonLabels.alertname }}'},
            "html": "",
            "text": '{{ range .Alerts }}状态：{{ if eq .Status "firing" }}告警触发{{ else }}告警恢复{{ end }}\n'
                '名称：{{ .Labels.alertname }}\n级别：{{ .Labels.severity }}\n目标：{{ .Labels.instance }}\n'
                '开始时间：{{ .StartsAt }}\n{{ if eq .Status "resolved" }}恢复时间：{{ .EndsAt }}\n{{ end }}'
                '原因：{{ .Annotations.summary }}\n说明：{{ .Annotations.description }}\n\n{{ end }}',
        }]
    if enabled(values, "ALERT_DINGTALK_ENABLED"):
        webhook = urlsplit(required(values, "ALERT_DINGTALK_WEBHOOK"))
        access_tokens = parse_qs(webhook.query).get("access_token", [])
        if (webhook.scheme != "https" or webhook.netloc != "oapi.dingtalk.com"
                or webhook.path != "/robot/send" or webhook.fragment
                or len(access_tokens) != 1 or not access_tokens[0]):
            raise ValueError("Set a valid HTTPS custom robot ALERT_DINGTALK_WEBHOOK")
        webhooks.append({"url": "http://alert-notifier:8080/alerts", "send_resolved": True,
            "max_alerts": 100, "timeout": "15s",
            "http_config": {"authorization": {"credentials": token}}})
    # JSON is valid YAML; serializing scalars also safely quotes private values.
    return json.dumps({"route": {"receiver": "rag", "group_by": ["alertname"],
        "group_wait": "30s", "group_interval": "5m", "repeat_interval": "4h"},
        "receivers": [receiver]}, ensure_ascii=False, indent=2) + "\n"


def write_configuration(directory, values, mode="docker", force=False):
    name = "alertmanager.local.yml" if mode == "local" else "alertmanager.yml"
    output = directory / "runtime" / name
    if output.exists() and not force:
        raise ValueError("Alertmanager configuration already exists; use --force to explicitly replace it")
    configuration = render_configuration(values, mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".yml.tmp")
    temporary.write_text(configuration, encoding="utf-8")
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["docker", "local"], default="docker")
    parser.add_argument("--force", action="store_true", help="Explicitly replace the generated configuration")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / "deploy" / "enterprise"
    values = {**dotenv_values(directory / ".env"), **os.environ}
    write_configuration(directory, values, args.mode, args.force)
    print("Created private Alertmanager configuration")


if __name__ == "__main__":
    main()
