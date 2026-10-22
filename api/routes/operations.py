"""Authenticated Alertmanager receiver and persisted operational alerts."""
import hmac
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from config.settings import settings
from core.auth import AuthUser, require_admin
from core.database import get_db
from core.db_models import OperationalAlert

router = APIRouter(prefix="/operations", tags=["运行监控"])


class Alert(BaseModel):
    status: Literal["firing", "resolved"]
    fingerprint: str = Field(max_length=128)
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)


class AlertBatch(BaseModel):
    alerts: list[Alert] = Field(max_length=100)


@router.post("/alerts")
def receive_alerts(request: AlertBatch, authorization: str = Header(default=""), db: Session = Depends(get_db)):
    expected = "Bearer " + settings.ALERT_RECEIVER_TOKEN
    if not settings.ALERT_RECEIVER_TOKEN or not hmac.compare_digest(authorization, expected):
        raise HTTPException(401, "告警接收凭据无效")
    summaries = []
    for alert in request.alerts:
        name = alert.labels.get("alertname", "unknown")[:128]
        severity = alert.labels.get("severity", "warning")[:32]
        summary = alert.annotations.get("summary", name)[:4000]
        db.add(OperationalAlert(fingerprint=alert.fingerprint, name=name,
            severity=severity, status=alert.status, summary=summary))
        summaries.append({"name": name, "severity": severity, "status": alert.status, "summary": summary})
    db.commit()
    return {"received": len(request.alerts)}


@router.get("/alerts")
def list_alerts(db: Session = Depends(get_db), user: AuthUser = Depends(require_admin)):
    items = db.scalars(select(OperationalAlert).order_by(OperationalAlert.created_at.desc()).limit(100)).all()
    return {"items": [{"id": item.id, "name": item.name, "status": item.status,
        "severity": item.severity, "summary": item.summary, "created_at": item.created_at} for item in items]}
