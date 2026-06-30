"""build_context 节点逻辑：课程上下文 + 学生画像 + 最近问答

被 nodes.py 中的 build_context_node 调用，可在单元测试中独立测试
"""
import logging
from uuid import UUID

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.models import QARecord, UserProfile
from coursepilot.rag.generator import build_course_context

logger = logging.getLogger(__name__)

async def build_context(
    session: AsyncSession,
    user_id: str,
    course_id: str,
) -> tuple[dict, dict | None, list[dict]]:
    """构建上下文三元组

    :returns: (course_context, user_profile_summary, recent_qa_list)
    """
    # 1. 课程上下文（KP树大纲 + 教材名）
    course_ctx = await build_course_context(session, UUID(course_id))

    # 2. 学生画像（若有与计算结果）
    profile = None
    result = await session.execute(
        select(UserProfile).where(
            UserProfile.user_id == UUID(user_id),
            UserProfile.course_id == UUID(course_id)
        )
    )
    up = result.scalar_one_or_none()
    if up:
        profile = {
            "mastery_level": up.mastery_level,
            "weak_kps": up.weak_kps or [],
            "avg_correct_rate": (
                float(up.avg_correct_rate) if up.avg_correct_rate else None
            ),
        }

    # 3. 最近 5 条问答（截断答案避免上下文膨胀）
    result = await session.execute(
        select(QARecord).where(
            QARecord.user_id == UUID(user_id),
            QARecord.course_id == UUID(course_id)
        )
        .order_by(desc(QARecord.created_at))
        .limit(5)
    )
    recent_qa = [
        {"query": q.query, "answer": q.answer[:200]}
        for q in result.scalars().all()
    ]

    return course_ctx, profile, recent_qa

