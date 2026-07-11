"""查询学生知识点掌握度 Skill

从 user_profiles 获取该学生在当前课程的各 KP 掌握度数据
被 get_mastery_node 调用，结果存入 state["mastery"]
"""
import logging
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from coursepilot.models import UserProfile

logger = logging.getLogger(__name__)


async def get_mastery(
    session: AsyncSession,
    user_id: str,
    course_id: str,
) -> dict:
    """查询该学生的知识点掌握度

    Returns:
        无 profile 时返回 {"mastery_level": {}, "weak_kps": [], "avg_correct_rate": None}
    """
    result = await session.execute(
        select(UserProfile).where(
            UserProfile.user_id == UUID(user_id),
            UserProfile.course_id == UUID(course_id)
        )
    )
    profiles = result.scalars().all()
    if not profiles:
        return {"mastery_level": {}, "weak_kps": [], "avg_correct_rate": None}
    if len(profiles) > 1:
        logger.warning(
            "get_mastery: %d 条 UserProfile 重复 user=%s course=%s，取第一条",
            len(profiles), user_id, course_id,
        )
    p = profiles[0]
    return {
        "mastery_level": p.mastery_level or {},
        "weak_kps": p.weak_kps or [],
        "avg_correct_rate": (
            float(p.avg_correct_rate) if p.avg_correct_rate else None
        ),
    }

