"""查询学生知识点掌握度 Skill

从 user_profiles 获取该学生在当前课程的各 KP 掌握度数据
被 get_mastery_node 调用，结果存入 state["mastery"]
"""
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from coursepilot.models import UserProfile

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
    profile = result.scalar_one_or_none()
    if not profile:
        return {"mastery_level": {}, "weak_kps": [], "avg_correct_rate": None}
    return {
        "mastery_level": profile.mastery_level or {},
        "weak_kps": profile.weak_kps or [],
        "avg_correct_rate": (
            float(profile.avg_correct_rate) if profile.avg_correct_rate else None
        ),
    }

