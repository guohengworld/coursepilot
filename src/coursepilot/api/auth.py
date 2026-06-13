"""认证 API：register/login/me"""
from fastapi import Depends, HTTPException, APIRouter
from pydantic import BaseModel, Field
from typing import Dict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from streamlit import status

from coursepilot.api.deps import get_current_user
from coursepilot.auth.jwt import create_token
from coursepilot.auth.password import hash_password, verify_password
from coursepilot.db import get_session
from coursepilot.models import User

router = APIRouter(prefix="/auth", tags=["auth"])


# ====== request/response model =======
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=8, max_length=128)

class RegisterResponse(BaseModel):
    user_id: str = Field(description="uuid 的用户id")
    username: str
    role: str

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    token: str = Field(description="JWT token")
    expires_in: int = Field(description="token 过期时间，秒数")
    user: dict = Field(description="用户信息")

class UserInfo(BaseModel):
    id: str = Field(description="uuid 的用户id")
    username: str
    role: str
    created_at: str = Field(description="创建时间")

# ====== API =======

@router.post("/register", status_code=201)
async def register(
        request: RegisterRequest,
        session: AsyncSession = Depends(get_session),
):
    """注册新用户（默认 role = student）"""
    # 1. 检查数据库中是否已存在同名用户
    result = await session.execute(select(User).where(User.username == request.username))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户已存在")

    # 2. 创建用户
    user = User(
        username=request.username,
        password_hash=hash_password(request.password),
        role="student",
    )

    session.add(user)
    # 需要 flush 才能拿到 user.id
    await session.flush()

    # 3. 返回结果
    return RegisterResponse(
        user_id=str(user.id),
        username=user.username,
        role=user.role,
    )

@router.post("/login")
async def login(
    request: LoginRequest,
    session: AsyncSession = Depends(get_session),
):
    """登录，返回 JWT token"""
    # 1. 检查用户是否存在，密码是否正确
    result = await session.execute(select(User).where(User.username == request.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在或密码错误",
        )

    # 2. 创建 JWT token
    token, expires_in = create_token(str(user.id), user.role)

    # 3. 返回结果
    return LoginResponse(
        token=token,
        expires_in=expires_in,
        user={
            "id": str(user.id),
            "username": user.username,
            "role": user.role,
        },
    )

@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    """获取当前登录用户信息"""
    return UserInfo(
        id=str(current_user.id),
        username=current_user.username,
        role=current_user.role,
        created_at=current_user.created_at.isoformat(),
    )

