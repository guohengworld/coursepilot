"""指标聚合Agent 运营数据统计

所有函数使用独立只读 session，不参与业务事务
返回的数据结构可直接序列化为 JSON 供前端展示

Usage:
    from coursepilot.observability.metrrics import get_daily_stats

    stats = await get_daily_stats(course_id="...")
    # {"total_sessions": 120, "total_tokens": 50000, ...}
"""
from datetime import UTC, date, datetime, timedelta
from uuid import UUID

from sqlalchemy import select, func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.db import async_session_factory
from coursepilot.models import AgentSession

async def get_course_stats(
    course_id: str,
    days: int = 30,
) -> dict:
    """获取最近 N 天某课程的 Agent 使用统计

    Returns:
        {
            "total_sessions": 整体会话数,
            "total_tokens": 总 token 消耗,
            "total_cost": 总预估成本（元）,
            "daily_avg_sessions": 日均会话数,
            "intent_distribution": {"question": 80, "practice": 30, ...},
            "status_distribution": {"completed": 100, "failed": 5, ...},
        }
    """
    since = datetime.now(UTC) - timedelta(days=days)

    async with async_session_factory() as session:
        # 基础统计
        total = await session.scalar(
            select(sa_func.count(AgentSession.id))
            .where(
                AgentSession.course_id == UUID(course_id),
                AgentSession.created_at >= since,
            )
        )

        token_sum = await session.scalar(
            select(sa_func.coalesce(sa_func.sum(AgentSession.token_count), 0))
            .where(
                AgentSession.course_id == UUID(course_id),
                AgentSession.created_at >= since,
            )
        )

        cost_sum = await session.scalar(
            select(sa_func.coalesce(sa_func.sum(AgentSession.estimated_cost), 0))
            .where(
                AgentSession.course_id == UUID(course_id),
                AgentSession.created_at >= since,
            )
        )

        # Intent 分布
        intent_rows = await session.execute(
            select(AgentSession.intent, sa_func.count(AgentSession.id))
            .where(
                AgentSession.course_id == UUID(course_id),
                AgentSession.created_at >= since,
            )
            .group_by(AgentSession.intent)
        )
        intent_dist = {row[0]: row[1] for row in intent_rows}

        # Status 分布
        status_rows = await session.execute(
            select(AgentSession.status, sa_func.count(AgentSession.id))
            .where(
                AgentSession.course_id == UUID(course_id),
                AgentSession.created_at >= since,
            )
            .group_by(AgentSession.status)
        )
        status_dist = {row[0]: row[1] for row in status_rows}

    total_count = total or 0
    return {
        "course_id": course_id,
        "period_days": days,
        "total_sessions": total_count,
        "total_tokens": float(token_sum or 0),
        "total_cost": float(cost_sum or 0),
        "daily_avg_sessions": round(total_count / days, 1) if days > 0 else 0,
        "intent_distribution": intent_dist,
        "status_distribution": status_dist,
    }

async def get_week_kp_summary(
    course_id: str,
    days: int = 30,
) -> list[dict]:
    """聚合薄弱知识点排名（从 user_profiles 表）

    Returns:
        [{"kp_path": "OS/进程同步", "student_count": 15, "avg_rate": 0.45}, ...]
    """
    from coursepilot.models import UserProfile

    since = datetime.now(UTC) - timedelta(days=days)

    async with async_session_factory() as session:
        rows = await session.execute(
            select(UserProfile.weak_kps, UserProfile.avg_correct_rate)
            .where(
                UserProfile.course_id == UUID(course_id),
                UserProfile.computed_at >= since,
            )
        )

    # 聚合所有学生的薄弱知识点
    kp_counter: dict[str, dict] = {}
    for row in rows:
        weak_list = row[0] or []
        for kp in weak_list:
            if kp not in kp_counter:
                kp_counter[kp] = {"count": 0, "rates": []}
            kp_counter[kp]["count"] += 1
            if row[1] is not None:
                kp_counter[kp]["rates"].append(float(row[1]))

    result = []
    for kp_path, data in sorted(kp_counter.items(), key=lambda x: -x[1]["count"]):
        avg = sum(data["rates"]) / len(data["rates"]) if data["rates"] else 0
        result.append({
            "kp_path": kp_path,
            "student_count": data["count"],
            "avg_rate": round(avg, 2),
        })

    return result

async def get_today_token_usage(user_id: str) -> int:
    """查询用户今日已用 token 数（用于 guardrails 限流）"""
    today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    async with async_session_factory() as session:
        total = await session.scalar(
            select(sa_func.coalesce(sa_func.sum(AgentSession.token_count), 0))
            .where(
                AgentSession.user_id == UUID(user_id),
                AgentSession.created_at >= today_start,
            )
        )
    return int(total or 0)


async def get_daily_counts(
        course_id: str,
        days: int = 7,
) -> list[dict]:
    """获取每日会话数趋势

    Returns:
        [{"date": "2026-07-01", "count": 15}, ...]
    """
    since = datetime.now(UTC) - timedelta(days=days)

    async with async_session_factory() as session:
        rows = await session.execute(
            select(
                sa_func.cast(AgentSession.created_at, sa_func.Date),
                sa_func.count(AgentSession.id),
            )
            .where(
                AgentSession.course_id == UUID(course_id),
                AgentSession.created_at >= since,
            )
            .group_by(sa_func.cast(AgentSession.created_at, sa_func.Date))
            .order_by(sa_func.cast(AgentSession.created_at, sa_func.Date))
        )

    return [{"date": str(row[0]), "count": row[1]} for row in rows]


async def get_context_metrics(
    session_id: str,
) -> dict:
    """获取单次会话的上下文预算与记忆层指标（P5 可观测）。

    从 agent_sessions.conversation / rolling_summary 反推 L1/L2，
    并汇总 llm_calls 中记录的 context_budget / layer_tokens / cache_hit_estimated。
    """
    from coursepilot.agent.memory.context_manager import estimate_tokens

    async with async_session_factory() as session:
        agent_session = await session.get(AgentSession, UUID(session_id))
        if not agent_session:
            return {"error": "会话不存在"}

        conversation = agent_session.conversation or []
        rolling_summary = agent_session.rolling_summary or ""
        llm_calls = agent_session.sources or []  # sources 被复用存放 llm_calls？不，llm_calls 没存。这里只展示当前能拿到的。

        # 实际上 llm_calls 只存在于运行时 state，未持久化到 DB。
        # 为了可观测，把它放到 agent_session.sources 里不合适；先保留结构，后续可改 schema。
        l1_tokens = sum(
            estimate_tokens(t.get("content", ""))
            for t in conversation
        )
        l2_tokens = estimate_tokens(rolling_summary)

        return {
            "session_id": session_id,
            "l1_turns": len(conversation),
            "l1_tokens": l1_tokens,
            "l2_tokens": l2_tokens,
            "total_history_tokens": l1_tokens + l2_tokens,
            "rolling_summary_present": bool(rolling_summary),
            "token_count": agent_session.token_count,
            "estimated_cost": float(agent_session.estimated_cost),
            "note": "llm_calls 详细指标需在 AgentState 中持久化后扩展",
        }


async def get_memory_layer_stats(
    course_id: str,
    days: int = 30,
) -> dict:
    """统计课程层面的记忆层健康度（P5）。"""
    from coursepilot.models import QARecord, UserProfile

    since = datetime.now(UTC) - timedelta(days=days)

    async with async_session_factory() as session:
        qa_total = await session.scalar(
            select(sa_func.count(QARecord.id)).where(
                QARecord.course_id == UUID(course_id),
                QARecord.created_at >= since,
            )
        )
        qa_with_embedding = await session.scalar(
            select(sa_func.count(QARecord.id)).where(
                QARecord.course_id == UUID(course_id),
                QARecord.created_at >= since,
                QARecord.embedding.isnot(None),
            )
        )
        profile_count = await session.scalar(
            select(sa_func.count(UserProfile.id)).where(
                UserProfile.course_id == UUID(course_id),
            )
        )
        profile_with_facts = await session.scalar(
            select(sa_func.count(UserProfile.id)).where(
                UserProfile.course_id == UUID(course_id),
                UserProfile.memory_facts.isnot(None),
            )
        )

    return {
        "course_id": course_id,
        "period_days": days,
        "qa_total": qa_total or 0,
        "qa_with_embedding": qa_with_embedding or 0,
        "embedding_coverage": round((qa_with_embedding or 0) / max(1, qa_total or 0), 2),
        "profile_count": profile_count or 0,
        "profile_with_facts": profile_with_facts or 0,
        "facts_coverage": round((profile_with_facts or 0) / max(1, profile_count or 0), 2),
    }
