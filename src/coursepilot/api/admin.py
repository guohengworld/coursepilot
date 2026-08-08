"""Admin 控制台 API：上下文窗口/记忆层可观测，仅限 super 角色。

提供端点：
- GET /admin/memory/dashboard?course_id=...&days=...
- GET /admin/memory/session/{session_id}
- GET /admin/memory/recall?user_id=...&course_id=...&query=...
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.agent.memory import recall_memory_turns
from coursepilot.api.deps import get_current_user, require_superuser
from coursepilot.db import get_session
from coursepilot.models import AgentSession, AuditLog, QARecord, User, UserProfile
from coursepilot.observability.metrics import (
    get_context_metrics,
    get_course_stats,
    get_memory_layer_stats,
)

router = APIRouter(prefix="/admin", tags=["admin"])


class MemoryDashboardResponse(BaseModel):
    course_id: str
    course_stats: dict
    memory_layer_stats: dict
    recent_sessions: list[dict]


class SessionMemoryResponse(BaseModel):
    session_id: str
    user_id: str
    course_id: str
    intent: str
    status: str
    context_metrics: dict
    conversation: list[dict] | None
    rolling_summary: str | None
    memory_facts: dict | None
    agent_steps: dict | None = None  # P3.1: Agentic RAG 决策轨迹（来自审计日志）


class MemoryRecallResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]


@router.get("/memory/dashboard", response_model=MemoryDashboardResponse)
async def memory_dashboard(
    course_id: str,
    days: int = 7,
    current_user: User = Depends(require_superuser),
    session: AsyncSession = Depends(get_session),
):
    """课程级记忆层仪表盘。"""
    course_stats = await get_course_stats(course_id, days=days)
    memory_stats = await get_memory_layer_stats(course_id, days=days)

    recent_result = await session.execute(
        select(AgentSession)
        .where(AgentSession.course_id == UUID(course_id))
        .order_by(AgentSession.updated_at.desc())
        .limit(20)
    )
    recent_sessions = []
    for s in recent_result.scalars().all():
        recent_sessions.append({
            "session_id": str(s.id),
            "user_id": str(s.user_id),
            "intent": s.intent,
            "status": s.status,
            "token_count": s.token_count,
            "estimated_cost": float(s.estimated_cost),
            "conversation_turns": len(s.conversation or []),
            "has_rolling_summary": bool(s.rolling_summary),
            "created_at": s.created_at.isoformat(),
        })

    return MemoryDashboardResponse(
        course_id=course_id,
        course_stats=course_stats,
        memory_layer_stats=memory_stats,
        recent_sessions=recent_sessions,
    )


@router.get("/memory/session/{session_id}", response_model=SessionMemoryResponse)
async def session_memory_detail(
    session_id: str,
    current_user: User = Depends(require_superuser),
    session: AsyncSession = Depends(get_session),
):
    """单次会话的完整记忆层详情。"""
    agent_session = await session.get(AgentSession, UUID(session_id))
    if not agent_session:
        raise HTTPException(status_code=404, detail="会话不存在")

    context_metrics = await get_context_metrics(session_id)

    # P3.1: 读取该会话最新的 Agentic RAG 决策轨迹（审计日志 agent.rag_steps）
    audit_result = await session.execute(
        select(AuditLog)
        .where(
            AuditLog.action == "agent.rag_steps",
            AuditLog.resource_id == session_id,
        )
        .order_by(AuditLog.created_at.desc())
        .limit(1)
    )
    agent_steps_log = audit_result.scalar_one_or_none()

    # 查询该用户课程的 L3 记忆
    profile_result = await session.execute(
        select(UserProfile).where(
            UserProfile.user_id == agent_session.user_id,
            UserProfile.course_id == agent_session.course_id,
        )
    )
    profile = profile_result.scalar_one_or_none()

    return SessionMemoryResponse(
        session_id=session_id,
        user_id=str(agent_session.user_id),
        course_id=str(agent_session.course_id),
        intent=agent_session.intent,
        status=agent_session.status,
        context_metrics=context_metrics,
        conversation=agent_session.conversation,
        rolling_summary=agent_session.rolling_summary,
        memory_facts=profile.memory_facts if profile else None,
        agent_steps=agent_steps_log.details if agent_steps_log else None,
    )


@router.get("/memory/recall", response_model=MemoryRecallResponse)
async def memory_recall(
    user_id: str,
    course_id: str,
    query: str,
    top_k: int = 5,
    current_user: User = Depends(require_superuser),
    session: AsyncSession = Depends(get_session),
):
    """测试 L4 归档记忆召回（recency × relevance × importance）。"""
    results = await recall_memory_turns(
        session=session,
        user_id=user_id,
        course_id=course_id,
        query=query,
        top_k=top_k,
    )
    return MemoryRecallResponse(query=query, results=results)
