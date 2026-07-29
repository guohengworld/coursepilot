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


def _first_profile(result, uid: str, cid: str) -> dict | None:
    """安全获取 UserProfile 第一条，避免 duplicates 导致 MultipleResultsFound。"""
    rows = result.scalars().all()
    if not rows:
        return None
    if len(rows) > 1:
        logger.warning(
            "发现 %d 条 UserProfile 重复记录 user=%s course=%s，取第一条",
            len(rows), uid, cid,
        )
    p = rows[0]
    return {
        "mastery_level": p.mastery_level,
        "weak_kps": p.weak_kps or [],
        "avg_correct_rate": (
            float(p.avg_correct_rate) if p.avg_correct_rate else None
        ),
    }

async def build_context(
    session: AsyncSession,
    user_id: str,
    course_id: str,
) -> tuple[dict, dict | None, list[dict]]:
    """构建上下文三元组

    :returns: (course_context, user_profile_summary, recent_qa_list)
    """
    # 尝试解析 UUID，无效时返回空默认值（eval/test 场景）
    try:
        uid = UUID(user_id)
        cid = UUID(course_id)
    except (ValueError, TypeError):
        logger.warning("build_context: 无效的 UUID user_id=%s course_id=%s", user_id, course_id)
        if _is_valid_uuid(course_id):
            course_ctx = await build_course_context(session, UUID(course_id))
        else:
            course_ctx = {}
        return course_ctx, None, []

    # 1. 课程上下文（KP树大纲 + 教材名）
    course_ctx = await build_course_context(session, cid)

    # 2. 学生画像（处理 duplicates 避免 MultipleResultsFound）
    profile = None
    result = await session.execute(
        select(UserProfile).where(
            UserProfile.user_id == uid,
            UserProfile.course_id == cid
        )
    )
    profile = _first_profile(result, user_id, course_id)

    # 3. 最近 5 条问答（截断答案避免上下文膨胀）
    result = await session.execute(
        select(QARecord).where(
            QARecord.user_id == uid,
            QARecord.course_id == cid
        )
        .order_by(desc(QARecord.created_at))
        .limit(5)
    )
    recent_qa = [
        {"query": q.query, "answer": q.answer[:200]}
        for q in result.scalars().all()
    ]

    return course_ctx, profile, recent_qa


def _is_valid_uuid(s: str) -> bool:
    try:
        UUID(s)
        return True
    except (ValueError, TypeError):
        return False

