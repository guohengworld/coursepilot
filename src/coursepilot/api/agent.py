"""Agent API：聊天入口、会话查询、人口审批占位

遵循与 api/auth.py、api/courses.py 相同的模式：
APIRouter + Depends(get_session) + Depends(get_current_user)。
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.agent.graph import build_agent_graph
from coursepilot.api.deps import get_current_user
from coursepilot.db import get_session
from coursepilot.models import AgentSession, User

router = APIRouter(prefix="/agent", tags=["agent"])

# Request / Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    course_id: str = Field(..., description="课程 ID（UUID）")

class ChatResponse(BaseModel):
    session_id: str
    intent: str
    answer: str
    sources: list[dict]
    token_count: int

class SessionStatusResponse(BaseModel):
    session_id: str
    intent: str
    status: str
    token_count: int
    estimated_cost: float
    created_at: str
    updated_at: str

# Graph 实例（模块级缓存）
_graph_app = None

def _get_graph():
    global _graph_app
    if _graph_app is None:
        _graph_app = build_agent_graph()
    return _graph_app

# API 端点
@router.post("/chat", status_code=201)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """主入口：发送消息给 Agent，执行完整 workflow 并返回回答"""
    # 1. 创建会话记录
    agent_session = AgentSession(
        user_id=current_user.id,
        course_id=UUID(request.course_id),
        intent="pending",
        status="running"
    )
    session.add(agent_session)
    await session.flush()

    # 2. 构建初始状态
    initial_state = {
        "query": request.message,
        "course_id": request.course_id,
        "user_id": str(current_user.id),
        "session_id": str(agent_session.id),
        "messages": [],
        "course_context": {},
        "user_profile": None,
        "recent_qa": [],
        "intent": "",
        "context": "",
        "retrieved_metadata": {},
        "answer": "",
        "sources": [],
        "token_count": 0,
        "error": None,
    }

    # 3. 执行 LangGraph
    graph = _get_graph()
    config = {"configurable": {"thread_id": str(agent_session.id)}}
    result = await graph.ainvoke(initial_state, config)

    # 4. 写回 classify 结果
    if result.get("intent"):
        agent_session.intent = result["intent"]

    return ChatResponse(
        session_id=str(agent_session.id),
        intent=result.get("intent", "question"),
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        token_count=result.get("token_count", 0),
    )

@router.get("/sessions/{session_id}")
async def get_session_status(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db_session: AsyncSession = Depends(get_session)
):
    """查询会话状态"""
    result = await db_session.execute(
        select(AgentSession).where(AgentSession.id == UUID(session_id))
    )
    agent_session = result.scalar_one_or_none()
    if not agent_session:
        raise HTTPException(status_code=404, detail="会话不存在")
    if agent_session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权访问此会话")

    return SessionStatusResponse(
        session_id=str(agent_session.id),
        intent=agent_session.intent,
        status=agent_session.status,
        token_count=agent_session.token_count,
        estimated_cost=float(agent_session.estimated_cost),
        created_at=agent_session.created_at.isoformat(),
        updated_at=agent_session.updated_at.isoformat(),
    )

@router.post("/sessions/{session_id}/approve")
async def approve_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db_session: AsyncSession = Depends(get_session),
):
    """人工审批（Phase 1 占位，Phase 3 实现 interrupt 恢复）。"""
    result = await db_session.execute(
        select(AgentSession).where(AgentSession.id == UUID(session_id))
    )
    agent_session = result.scalar_one_or_none()
    if not agent_session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"status": "approved", "session_id": session_id}




















