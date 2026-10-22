"""认证接口。"""
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from config.settings import settings
from core.auth import create_access_token
from core.database import get_db
from core.db_models import User

router = APIRouter(prefix="/auth", tags=["认证"])


class DevTokenRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    role: str = Field(default="user", min_length=1, max_length=64)
    department: str | None = Field(default=None, max_length=128)
    tenant_id: str = Field(default="default", min_length=1, max_length=128)
    user_id: str | None = Field(default=None, max_length=64)
    scopes: list[str] = Field(default_factory=list)


@router.post("/dev-token")
async def issue_dev_token(request: DevTokenRequest, db: Session = Depends(get_db)):
    """仅供本地联调；生产环境应由 OIDC 或网关签发令牌。"""
    if not settings.AUTH_DEV_MODE:
        raise HTTPException(404, "开发令牌接口未启用")
    username = request.username.strip()
    user = db.query(User).filter(User.username == username).first()
    if not user:
        user = User(
            username=username,
            role=request.role.strip(),
            department=request.department,
            tenant_id=request.tenant_id.strip(),
        )
        db.add(user)
    else:
        user.role = request.role.strip()
        user.department = request.department
        user.tenant_id = request.tenant_id.strip()
        user.is_active = True
    db.commit()
    db.refresh(user)
    token = create_access_token(
        username,
        role=request.role.strip(),
        department=request.department,
        tenant_id=request.tenant_id.strip(),
        user_id=request.user_id or user.id,
        scopes=request.scopes,
    )
    return {"access_token": token, "token_type": "bearer", "expires_in": settings.AUTH_TOKEN_EXPIRE_MINUTES * 60}
