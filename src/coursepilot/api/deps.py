"""FastAPI 依赖项：获取当前用户、获取 session 等"""
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.auth.jwt import decode_token
from coursepilot.db import get_session
from coursepilot.models import Course, Enrollment, User

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

async def require_teacher(
    current_user: User = Depends(get_current_user),
) -> User:
    """Teacher 及以上权限校验（teacher / super）。"""
    if current_user.role not in ("teacher", "super"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权限",
        )
    return current_user


# ── 课程归属校验（②：enrollments 权威判据 + created_by 兜底） ─────────
# 语义（与 api/tasks.py 的课程归属双判据一致）：
#   - super 恒为 teacher 级
#   - teacher 级 = enrollments role=teacher 或 course.created_by == 当前用户
#   - 成员级（student）= enrollments role=student/teacher 任一生效，或 created_by
# 用途：POST /agent/chat 等带 course_id 的端点，防"传别课 id 读别课资料库 / 写别课学情"。

async def get_course_membership(
    session: AsyncSession, user: User, course_id: UUID
) -> str | None:
    """返回用户在课程内的角色（"teacher"/"student"）；非成员返回 None。

    super 恒返回 "teacher"。判据：enrollments 行（任意 role）→ created_by 兜底
    （兼容 enrollments 未回填的旧库，课程创建者天然是教师）。
    """
    if user.role == "super":
        return "teacher"
    enr = await session.execute(
        select(Enrollment).where(
            Enrollment.user_id == user.id,
            Enrollment.course_id == course_id,
        )
    )
    enrollment = enr.scalar_one_or_none()
    role = enrollment.role if enrollment is not None else None
    if role == "teacher":
        return "teacher"
    owner = await session.execute(
        select(Course.id).where(
            Course.id == course_id,
            Course.created_by == user.id,
        )
    )
    if owner.scalar_one_or_none() is not None:
        return "teacher"
    if role == "student":
        return "student"
    return None


async def require_course_member(
    session: AsyncSession, user: User, course_id: UUID
) -> str:
    """校验当前用户是该课程成员（学生或教师级），否则 403。返回角色。"""
    role = await get_course_membership(session, user, course_id)
    if role is None:
        raise HTTPException(status_code=403, detail="您不属于该课程")
    return role


async def require_course_teacher(
    session: AsyncSession, user: User, course_id: UUID
) -> str:
    """校验当前用户是该课程教师（teacher 级），否则 403。恒返回 "teacher"。"""
    role = await get_course_membership(session, user, course_id)
    if role != "teacher":
        raise HTTPException(status_code=403, detail="您不是该课程的教师")
    return role
