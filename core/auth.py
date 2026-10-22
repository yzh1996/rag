"""轻量 Bearer Token 认证和知识权限判断。

生产环境可以由 OIDC 网关签发同样格式的 HS256 token；本地开发通过开发令牌接口验证流程。
"""
import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import settings

_bearer = HTTPBearer(auto_error=False)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


@dataclass(frozen=True)
class AuthUser:
    username: str
    role: str = "user"
    department: str | None = None
    tenant_id: str = "default"
    user_id: str | None = None
    authenticated: bool = False
    scopes: set[str] = field(default_factory=set)


ANONYMOUS_USER = AuthUser(username="anonymous")


def actor_payload(user: AuthUser) -> dict:
    """后台任务保存发起人的身份，重启后仍按相同权限执行。"""
    return {"username": user.username, "role": user.role, "department": user.department,
        "tenant_id": user.tenant_id, "user_id": user.user_id, "authenticated": user.authenticated,
        "scopes": sorted(user.scopes)}


def create_access_token(
    username: str,
    *,
    role: str = "user",
    department: str | None = None,
    tenant_id: str = "default",
    user_id: str | None = None,
    scopes: list[str] | None = None,
    expires_in: int | None = None,
) -> str:
    """创建不依赖第三方 JWT 包的 HS256 Bearer Token。"""
    now = int(time.time())
    payload = {
        "sub": username,
        "role": role,
        "department": department,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "scopes": scopes or [],
        "iat": now,
        "exp": now + (expires_in or settings.AUTH_TOKEN_EXPIRE_MINUTES * 60),
    }
    header = {"alg": "HS256", "typ": "JWT"}
    encoded_header = _b64encode(json.dumps(header, separators=(",", ":")).encode())
    encoded_payload = _b64encode(json.dumps(payload, separators=(",", ":")).encode())
    unsigned = f"{encoded_header}.{encoded_payload}".encode()
    signature = hmac.new(settings.AUTH_SECRET_KEY.encode(), unsigned, hashlib.sha256).digest()
    return f"{encoded_header}.{encoded_payload}.{_b64encode(signature)}"


def decode_access_token(token: str) -> AuthUser:
    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".")
        unsigned = f"{encoded_header}.{encoded_payload}".encode()
        expected = hmac.new(
            settings.AUTH_SECRET_KEY.encode(), unsigned, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(expected, _b64decode(encoded_signature)):
            raise ValueError("签名无效")
        header = json.loads(_b64decode(encoded_header))
        payload = json.loads(_b64decode(encoded_payload))
        if header.get("alg") != "HS256" or int(payload.get("exp", 0)) <= int(time.time()):
            raise ValueError("令牌已过期或算法不支持")
        username = str(payload.get("sub", "")).strip()
        if not username:
            raise ValueError("令牌缺少用户")
        return AuthUser(
            username=username,
            role=str(payload.get("role", "user")),
            department=payload.get("department"),
            tenant_id=str(payload.get("tenant_id", "default")),
            user_id=payload.get("user_id"),
            authenticated=True,
            scopes=set(payload.get("scopes") or []),
        )
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"认证令牌无效：{exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AuthUser:
    if not credentials:
        if settings.AUTH_REQUIRED:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="请先登录",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return ANONYMOUS_USER
    return decode_access_token(credentials.credentials)


def require_authenticated(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    if not user.authenticated:
        raise HTTPException(status_code=401, detail="该操作需要登录")
    return user


def require_admin(user: AuthUser = Depends(require_authenticated)) -> AuthUser:
    if user.role not in {"admin", "knowledge_admin"}:
        raise HTTPException(status_code=403, detail="没有管理员权限")
    return user


def require_admin_or_local(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    """本地免认证模式允许匿名使用指定管理页面，认证开启时仍要求管理员。"""
    if not user.authenticated and not settings.AUTH_REQUIRED:
        return user
    if not user.authenticated:
        raise HTTPException(status_code=401, detail="该操作需要登录")
    if user.role not in {"admin", "knowledge_admin"}:
        raise HTTPException(status_code=403, detail="没有管理员权限")
    return user


def can_access_document(permission_scope: dict[str, Any] | None, user: AuthUser) -> bool:
    """在检索前判断文档权限；没有范围声明的旧文档视为公开知识。"""
    scope = permission_scope or {}
    if not scope:
        return True
    if scope.get("public") is True:
        return True
    tenant_ids = set(scope.get("tenant_ids") or [])
    if tenant_ids and user.tenant_id not in tenant_ids:
        return False
    users = set(scope.get("users") or [])
    if users and user.username not in users and user.user_id not in users:
        return False
    roles = set(scope.get("roles") or [])
    if roles and user.role not in roles:
        return False
    departments = set(scope.get("departments") or [])
    if departments and user.department not in departments:
        return False
    return True
