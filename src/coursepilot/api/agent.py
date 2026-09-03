"""Agent API：聊天入口、会话查询与删除

遵循与 api/auth.py、api/courses.py 相同的模式：
APIRouter + Depends(get_session) + Depends(get_current_user)。
HITL 审批（approve 端点 / waiting_human 状态）已随 commit ③ 移除。
"""
import asyncio
import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.agent.graph import build_agent_graph
from coursepilot.api.deps import get_current_user, require_course_member
from coursepilot.db import async_session_factory, get_session
from coursepilot.governance.guardrails import guard_token_limit
from coursepilot.models import AgentSession, DiagnosisReport, User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])

# Request / Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    course_id: str = Field(..., description="课程 ID（UUID）")
    session_id: str | None = Field(None, description="已有会话 ID，用于继续多轮对话")

class ChatAcceptedResponse(BaseModel):
    session_id: str
    status: str = "processing"

class SessionListItem(BaseModel):
    session_id: str
    course_id: str
    intent: str
    status: str
    query: str | None = None
    token_count: int = 0
    estimated_cost: float = 0
    created_at: str
    updated_at: str

class SessionPollResponse(BaseModel):
    session_id: str
    course_id: str
    status: str
    intent: str
    query: str | None = None
    answer: str | None = None
    sources: list[dict] | None = None
    questions: list[dict] | None = None
    diagnosis_data: dict | None = None  # 学情诊断结构化数据
    token_count: int = 0
    estimated_cost: float = 0
    conversation: list[dict] | None = None
    created_at: str
    updated_at: str

# Graph 实例（模块级缓存）
_graph_app = None

async def _get_graph():
    global _graph_app
    if _graph_app is None:
        _graph_app = await build_agent_graph()
    return _graph_app


async def _run_graph_background(
    graph, initial_state: dict, config: dict, session_id: str
) -> None:
    """后台执行 LangGraph，异常时将会话标记为 failed"""
    try:
        await graph.ainvoke(initial_state, config)
        # finalize_node 已负责写入所有数据与状态
    except Exception:
        logger.exception("后台图执行失败 session_id=%s", session_id)
        async with async_session_factory() as bg_session:
            result = await bg_session.execute(
                select(AgentSession).where(AgentSession.id == UUID(session_id))
            )
            agent_session = result.scalar_one_or_none()
            if agent_session:
                agent_session.status = "failed"
                agent_session.intent = "error"
            await bg_session.commit()


# API 端点
@router.post("/chat", status_code=202)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """主入口：创建或继续会话，后台执行 LangGraph workflow"""
    # ── RBAC 检查 ──
    from coursepilot.governance.rbac import has_permission
    if not has_permission(current_user.role, "agent:chat"):
        raise HTTPException(status_code=403, detail="无权使用 Agent")

    # ── Daily limit guard ──
    from coursepilot.observability.metrics import get_today_token_usage
    daily_tokens = await get_today_token_usage(str(current_user.id))
    limit_msg = guard_token_limit(
        session_token=0, daily_token=daily_tokens
    )
    if limit_msg:
        raise HTTPException(status_code=429, detail=limit_msg)

    # ── 继续已有会话 ──
    if request.session_id:
        result = await session.execute(
            select(AgentSession).where(AgentSession.id == UUID(request.session_id))
        )
        agent_session = result.scalar_one_or_none()
        if not agent_session:
            raise HTTPException(status_code=404, detail="会话不存在")
        if agent_session.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="无权访问此会话")
        if agent_session.status not in ("completed", "failed"):
            raise HTTPException(status_code=400, detail="会话当前不可继续，请稍后或新建会话")

        # 更新 session 准备新一轮
        agent_session.query = request.message
        agent_session.status = "processing"
        agent_session.answer = ""
        await session.commit()

        initial_state = {
            "query": request.message,
            "course_id": str(agent_session.course_id),
            "user_id": str(current_user.id),
            "user_role": current_user.role,
            "session_id": str(agent_session.id),
            "messages": agent_session.conversation or [],
            "conversation": agent_session.conversation or [],
            "rolling_summary": agent_session.rolling_summary or "",
            "course_context": {},
            "user_profile": None,
            "recent_qa": [],
            "intent": "",
            "context": "",
            "retrieved_metadata": {},
            "answer": "",
            "sources": [],
            "token_count": 0,
            "llm_calls": [],
            "context_budget": None,
            "layer_tokens": None,
            "cache_hit_estimated": None,
            "compaction_count": 0,
            "error": None,
        }

        thread_id = agent_session.langgraph_thread_id or str(agent_session.id)
        config = {"configurable": {"thread_id": thread_id}}
    else:
        # ── 创建新会话 ──
        # ② 课程归属校验：course_id 来自请求体，必须确认用户属于该课程，
        # 否则可传别课 id 读别课资料库 / 在别课名下写自己的学情数据。
        try:
            cid = UUID(request.course_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="course_id 不是合法的 UUID") from None
        await require_course_member(session, current_user, cid)

        agent_session = AgentSession(
            user_id=current_user.id,
            course_id=cid,
            query=request.message,
            intent="pending",
            status="processing",
        )
        session.add(agent_session)
        await session.commit()

        initial_state = {
            "query": request.message,
            "course_id": str(cid),
            "user_id": str(current_user.id),
            "user_role": current_user.role,
            "session_id": str(agent_session.id),
            "messages": [],
            "conversation": [],
            "rolling_summary": "",
            "course_context": {},
            "user_profile": None,
            "recent_qa": [],
            "intent": "",
            "context": "",
            "retrieved_metadata": {},
            "answer": "",
            "sources": [],
            "token_count": 0,
            "llm_calls": [],
            "context_budget": None,
            "layer_tokens": None,
            "cache_hit_estimated": None,
            "compaction_count": 0,
            "error": None,
        }

        config = {"configurable": {"thread_id": str(agent_session.id)}}

    # 后台执行（不阻塞请求）
    graph = await _get_graph()
    asyncio.create_task(_run_graph_background(graph, initial_state, config, str(agent_session.id)))

    return ChatAcceptedResponse(
        session_id=str(agent_session.id),
        status="processing",
    )


@router.get("/sessions")
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db_session: AsyncSession = Depends(get_session),
    status_filter: str | None = None,
):
    """查询会话列表

    - 学生：仅查自己的会话
    - 教师/管理员（agent:session:list_all）：查所有会话
    - 可选 status 参数过滤
    """
    from coursepilot.governance.rbac import has_permission

    query = select(AgentSession)

    if has_permission(current_user.role, "agent:session:list_all"):
        pass  # 查所有
    else:
        query = query.where(AgentSession.user_id == current_user.id)

    if status_filter:
        query = query.where(AgentSession.status == status_filter)

    query = query.order_by(AgentSession.updated_at.desc()).limit(50)
    result = await db_session.execute(query)
    sessions = result.scalars().all()
    return [
        SessionListItem(
            session_id=str(s.id),
            course_id=str(s.course_id),
            intent=s.intent,
            status=s.status,
            query=s.query,
            token_count=s.token_count,
            estimated_cost=float(s.estimated_cost),
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}")
async def get_session_status(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db_session: AsyncSession = Depends(get_session)
):
    """查询会话状态（轮询端点，返回完整结果）"""
    result = await db_session.execute(
        select(AgentSession).where(AgentSession.id == UUID(session_id))
    )
    agent_session = result.scalar_one_or_none()
    if not agent_session:
        raise HTTPException(status_code=404, detail="会话不存在")
    if agent_session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权访问此会话")

    # 检测处理超时：若会话 processing 超过 5 分钟，自动标记为 failed
    if agent_session.status == "processing":
        stale_threshold = datetime.now(UTC) - timedelta(seconds=300)
        updated_at = agent_session.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)
        if updated_at < stale_threshold:
            agent_session.status = "failed"
            agent_session.intent = "error"
            await db_session.commit()

    # 从 quiz_data 中剥离答案，只返回题目供前端展示
    questions = None
    if agent_session.quiz_data:
        raw_questions = agent_session.quiz_data.get("questions", [])
        questions = [
            {
                "question_text": q.get("question_text", ""),
                "options": q.get("options", {}),
                "kp_path": q.get("kp_path", ""),
            }
            for q in raw_questions
        ]

    # 查询学情诊断报告
    diagnosis_data = None
    if agent_session.intent == "diagnose":
        diag_result = await db_session.execute(
            select(DiagnosisReport).where(
                DiagnosisReport.session_id == UUID(session_id)
            ).order_by(DiagnosisReport.created_at.desc()).limit(1)
        )
        diag = diag_result.scalar_one_or_none()
        if diag:
            diagnosis_data = {
                "overall_rate": diag.overall_rate,
                "total_practiced": diag.total_practiced,
                "kp_stats": diag.kp_stats,
                "weak_kps": diag.weak_kps,
                "llm_analysis": diag.llm_analysis,
                "recommendations": diag.recommendations,
            }

    return SessionPollResponse(
        session_id=str(agent_session.id),
        course_id=str(agent_session.course_id),
        intent=agent_session.intent,
        status=agent_session.status,
        query=agent_session.query,
        answer=agent_session.answer or None,
        sources=agent_session.sources or None,
        questions=questions,
        diagnosis_data=diagnosis_data,
        conversation=agent_session.conversation or None,
        token_count=agent_session.token_count,
        estimated_cost=float(agent_session.estimated_cost),
        created_at=agent_session.created_at.isoformat(),
        updated_at=agent_session.updated_at.isoformat(),
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db_session: AsyncSession = Depends(get_session),
):
    """删除指定会话"""
    result = await db_session.execute(
        select(AgentSession).where(AgentSession.id == UUID(session_id))
    )
    agent_session = result.scalar_one_or_none()
    if not agent_session:
        raise HTTPException(status_code=404, detail="会话不存在")
    if agent_session.user_id != current_user.id:
        from coursepilot.governance.rbac import has_permission
        if not has_permission(current_user.role, "agent:session:list_all"):
            raise HTTPException(status_code=403, detail="无权删除此会话")

    # 若会话正在处理中，先标记为 failed 以终止后台任务的状态更新
    if agent_session.status == "processing":
        agent_session.status = "failed"
        agent_session.intent = "error"

    await db_session.delete(agent_session)
    await db_session.commit()
