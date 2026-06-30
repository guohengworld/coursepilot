"""写入问答记录 + 更新会话状态"""
import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.models import AgentSession, QARecord

logger = logging.getLogger(__name__)

async def update_qa_record(
    session: AsyncSession,
    user_id: str,
    course_id: str,
    query: str,
    answer: str,
    kp_path: str | None,
    retrieved_units: list,
    citations: list,
    session_id: str,
) -> int:
    """持久化 QA 记录并更新会话 token 计数

    :returns: 当前 token_count（Phase 1 返回 0，Phase 2 实现真实计数）。
    """
    # 1. 写入 QA 记录
    qa = QARecord(
        user_id=UUID(user_id),
        course_id=UUID(course_id),
        query=query,
        answer=answer,
        kp_path=kp_path,
        retrieved_units=retrieved_units,
        citations=citations,
    )
    session.add(qa)

    # 2. 更新会话状态
    result = await session.execute(
        select(AgentSession).where(AgentSession.id == UUID(session_id))
    )
    agent_session = result.scalar_one_or_none()
    if agent_session:
        agent_session.status = "completed"
        if kp_path:
            agent_session.intent = "question"   # classify 产出更精确的 intent

    await session.flush()
    return 0    # Phase 2：从 LLM response 提取真实 token 计数


