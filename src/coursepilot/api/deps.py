"""FastAPI 依赖项：获取当前用户、获取 session 等"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.auth.jwt import decode_token
from coursepilot.db import get_session
from coursepilot.models import User

_bearer = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    session: AsyncSession = Depends(get_session),
) -> User:
    """从 JWT token 解析当前用户"""
    payload = decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 token",
        )

    user_id = payload.get("sub")
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
        )
    return user

async def require_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    """SuperUser 权限校验依赖"""
    if current_user.role != "super":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权限",
        )
    return current_user

