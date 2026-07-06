"""学生画像预计算

finalize_node 末尾异步触发，聚合 PracticeRecord → 按 KP 算掌握度 → upsert user_profiles。
使用独立 DB session 执行，不阻塞主流程。
"""
import logging
from uuid import UUID
from sqlalchemy import Integer, select, func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession
from coursepilot.db import async_session_factory
from coursepilot.models import PracticeRecord, Question, KnowledgePoint, UserProfile, QARecord

logger = logging.getLogger(__name__)

async def update_profile(
    user_id: str,
    course_id: str,
) -> None:
    """聚合做题记录 → 更新 user_profile 表（独立 session，不抛异常）"""
    try:
        async with async_session_factory() as session:
            await _do_update(session, user_id, course_id)
            await session.commit()
    except Exception:
        logger.exception("update_profile 失败 (不影响主线程)")

async def _do_update(
    session: AsyncSession,
    user_id: str,
    course_id: str,
) -> None:
    """实际的 profile 更新逻辑"""

    # 1. 按 KP 聚合正确率
    result = await session.execute(
        select(
            KnowledgePoint.kp_path,
            sa_func.count(PracticeRecord.id),
            sa_func.sum(
                sa_func.cast(PracticeRecord.correct_flag, Integer)
            ),
        )
        .select_from(PracticeRecord)
        .join(Question, PracticeRecord.question_id == Question.id)
        .join(KnowledgePoint, Question.kp_id == KnowledgePoint.id)
        .where(
            PracticeRecord.user_id == UUID(user_id),
            KnowledgePoint.course_id == UUID(course_id),
            PracticeRecord.correct_flag.isnot(None),
        )
        .group_by(KnowledgePoint.kp_path)
    )
    rows = result.all()

    mastery_level = {}
    weak_kps = []
    total_correct = 0
    total_answered = 0

    for kp_path, total, correct_sum in rows:
        t = int(total)
        c = int(correct_sum or 0)
        rate = c / t if t > 0 else 0.0
        mastery_level[kp_path] = round(rate, 2)
        total_answered += t
        total_correct += c
        if rate < 0.6:
            weak_kps.append(kp_path)

    avg_rate = total_correct / total_answered if total_answered > 0 else None

    # 2. QA 计算
    qa_count = await session.scalar(
        select(sa_func.count(QARecord.id)).where(
            QARecord.user_id == UUID(user_id),
            QARecord.course_id == UUID(course_id),
        )
    )

    # 3. Upsert
    existing = await session.execute(
        select(UserProfile).where(
            UserProfile.user_id == UUID(user_id),
            UserProfile.course_id == UUID(course_id),
        )
    )
    profile = existing.scalar_one_or_none()

    if profile:
        profile.mastery_level = mastery_level
        profile.weak_kps = weak_kps
        profile.total_practice_count = total_answered
        profile.total_qa_count = qa_count or 0
        profile.avg_correct_rate = avg_rate
    else:
        session.add(UserProfile(
            user_id=UUID(user_id),
            course_id=UUID(course_id),
            mastery_level=mastery_level,
            weak_kps=weak_kps,
            total_practice_count=total_answered,
            total_qa_count=qa_count or 0,
            avg_correct_rate=avg_rate,
        ))
    await session.flush()
    logger.info("Profile updated for user=%s course=%s", user_id, course_id)









