"""阶段2/3核心控制逻辑的无外部服务测试。"""
import pytest

from config.settings import settings
from core.auth import AuthUser, can_access_document, create_access_token, decode_access_token
from core.evaluation_service import _case_passed


def test_access_token_round_trip_and_permission_scope():
    token = create_access_token(
        "alice",
        role="user",
        department="研发部",
        tenant_id="tenant-a",
        scopes=["knowledge:read"],
    )
    user = decode_access_token(token)
    assert user.authenticated is True
    assert user.username == "alice"
    assert user.tenant_id == "tenant-a"
    assert can_access_document({"tenant_ids": ["tenant-a"], "roles": ["user"]}, user)
    assert not can_access_document({"tenant_ids": ["tenant-b"]}, user)
    assert not can_access_document({"departments": ["财务部"]}, user)


def test_expired_token_is_rejected():
    token = create_access_token("alice", expires_in=-1)
    with pytest.raises(Exception):
        decode_access_token(token)


def test_quality_gate_uses_configured_thresholds():
    passing = {
        "faithfulness": settings.EVAL_MIN_FAITHFULNESS,
        "answer_relevancy": settings.EVAL_MIN_ANSWER_RELEVANCY,
        "context_precision": settings.EVAL_MIN_CONTEXT_PRECISION,
        "context_recall": settings.EVAL_MIN_CONTEXT_RECALL,
    }
    assert _case_passed(passing, has_reference=True)
    failing = dict(passing, faithfulness=settings.EVAL_MIN_FAITHFULNESS - 0.01)
    assert not _case_passed(failing, has_reference=True)
