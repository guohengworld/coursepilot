"""学情诊断 Skill

聚合 PracticeRecord → 关联 Question → 获取 KP → 按 KP 路径计算正确率 → 识别薄弱点
"""
from uuid import UUID
from sqlalchemy import select, func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession
from coursepilot.models import PracticeRecord, Question, KnowledgePoint

async def diagnose(
    session: AsyncSession,
    user_id: str,
    course_id: str
) -> dict:
    """执行学情诊断

    Returns:
        {"weak_kps": [...], "kp_stats": {...}, "summary": "...",
         "total_practiced": int, "overall_rate": float}
    """
    result = await session.execute(
        select(
            KnowledgePoint.kp_path,
            sa_func.count(PracticeRecord.id),
            sa_func.sum(
                sa_func.cast(PracticeRecord.correct_flag, sa_func.Integer())
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

    kp_stats = {}
    weak_kps = []
    total_answered = 0
    total_correct = 0

    for kp_path, count, correct_sum in rows:
        total = int(count)
        correct = int(correct_sum or 0)
        rate = correct / total if total > 0 else 0.0
        kp_stats[kp_path] = {"total": total, "correct": correct, "rate": round(rate, 2)}
        total_answered += total
        total_correct += correct
        if rate < 0.6:
            weak_kps.append(kp_path)

    overall_rate = total_correct / total_answered if total_answered > 0 else 0.0

    summary = f"共练习 {total_answered} 题，正确率 {overall_rate:.0%}。"
    if weak_kps:
        summary += f"薄弱知识点（正确率<60%）：{'、'.join(weak_kps[:5])}"
        if len(weak_kps) > 5:
            summary += f" 等 {len(weak_kps)} 个"

    return {
        "weak_kps": weak_kps,
        "kp_stats": kp_stats,
        "summary": summary,
        "total_practiced": total_answered,
        "overall_rate": round(overall_rate, 2)
    }


