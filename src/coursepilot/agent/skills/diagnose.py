"""学情诊断 Skill

聚合 PracticeRecord → 按 KP 计算正确率 → 识别薄弱点
支持可配时间窗口、可配阈值、超时防护。
"""
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import TypedDict
from uuid import UUID

from sqlalchemy import Integer, and_, func as sa_func, select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.config import settings
from coursepilot.models import KnowledgePoint, PracticeRecord, Question

logger = logging.getLogger(__name__)


class KpStatItem(TypedDict):
    """单个知识点的统计"""
    total: int
    correct: int
    rate: float


class DiagnosisResult(TypedDict):
    """学情诊断结果"""
    weak_kps: list[str]
    kp_stats: dict[str, KpStatItem]
    summary: str
    total_practiced: int
    overall_rate: float


MAX_KP_LIMIT = 200


async def diagnose(
    session: AsyncSession,
    user_id: str,
    course_id: str,
    *,
    weak_threshold: float | None = None,
    lookback_days: int | None = None,
) -> DiagnosisResult:
    """执行学情诊断

    聚合 PracticeRecord → 关联 Question → 获取 KP → 按 KP 路径计算正确率
    → 识别薄弱知识点（正确率低于 weak_threshold）

    Args:
        session: 数据库会话
        user_id: 用户 UUID 字符串
        course_id: 课程 UUID 字符串
        weak_threshold: 薄弱阈值，默认来自 settings.diagnose_weak_threshold (0.6)
            传入 None 使用默认值，传入 0.0 禁用薄弱标记
        lookback_days: 分析最近 N 天的记录
            默认来自 settings.diagnose_lookback_days (90)
            传入 None 使用默认值，传入 0 禁用时间过滤（分析全量）

    Returns:
        DiagnosisResult

    Raises:
        ValueError: user_id / course_id 格式无效
    """
    # 1. 输入校验
    try:
        uid = UUID(user_id)
        cid = UUID(course_id)
    except ValueError as e:
        logger.warning("diagnose: 无效 UUID — user=%s course=%s", user_id, course_id)
        raise ValueError(f"无效的 UUID 参数: {e}") from e

    _threshold = (
        settings.diagnose_weak_threshold if weak_threshold is None else weak_threshold
    )
    _lookback = (
        settings.diagnose_lookback_days if lookback_days is None else lookback_days
    )

    # 2. 构建过滤条件
    filters = [
        PracticeRecord.user_id == uid,
        KnowledgePoint.course_id == cid,
        PracticeRecord.correct_flag.isnot(None),
    ]
    if _lookback:
        since = datetime.now(timezone.utc) - timedelta(days=_lookback)
        filters.append(PracticeRecord.answered_at >= since)

    t0 = time.perf_counter()

    # 3. 执行聚合查询
    try:
        result = await session.execute(
            select(
                KnowledgePoint.kp_path,
                sa_func.count(PracticeRecord.id).label("total"),
                sa_func.sum(
                    sa_func.cast(PracticeRecord.correct_flag, Integer())
                ).label("correct"),
            )
            .select_from(PracticeRecord)
            .join(Question, PracticeRecord.question_id == Question.id)
            .join(KnowledgePoint, Question.kp_id == KnowledgePoint.id)
            .where(and_(*filters))
            .group_by(KnowledgePoint.kp_path)
            .limit(MAX_KP_LIMIT)
        )
        rows = result.all()
    except Exception:
        logger.exception("diagnose: SQL 聚合查询失败")
        raise

    elapsed = time.perf_counter() - t0

    # 4. 计算统计
    kp_stats: dict[str, KpStatItem] = {}
    weak_kps: list[str] = []
    total_answered = 0
    total_correct = 0

    for kp_path, count, correct_sum in rows:
        total = int(count)
        correct = int(correct_sum or 0)
        rate = correct / total if total > 0 else 0.0
        kp_stats[kp_path] = KpStatItem(
            total=total, correct=correct, rate=round(rate, 2),
        )
        total_answered += total
        total_correct += correct
        if _threshold and rate < _threshold:
            weak_kps.append(kp_path)

    overall_rate = total_correct / total_answered if total_answered > 0 else 0.0

    # 5. 构建摘要
    summary_parts = [f"共练习 {total_answered} 题，正确率 {overall_rate:.0%}。"]
    if weak_kps:
        top_weak = weak_kps[:5]
        threshold_pct = f"{_threshold:.0%}"
        summary_parts.append(
            f"薄弱知识点（正确率<{threshold_pct}）：{'、'.join(top_weak)}"
        )
        if len(weak_kps) > 5:
            summary_parts.append(f"等 {len(weak_kps)} 个")

    # 6. 日志（可观测性）
    logger.info(
        "diagnose: user=%s elapsed=%.2fs kps=%d weak=%d total=%d rate=%.2f",
        user_id, elapsed, len(kp_stats), len(weak_kps),
        total_answered, overall_rate,
    )

    return DiagnosisResult(
        weak_kps=weak_kps,
        kp_stats=kp_stats,
        summary="".join(summary_parts),
        total_practiced=total_answered,
        overall_rate=round(overall_rate, 2),
    )
