"""写入问答记录 + 更新会话状态 + Token 计数"""
import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.config import settings
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
    token_count: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> int:
    """持久化 QA 记录并更新会话 token 计数和成本估算

    :returns: token_count
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

    # 2. 更新会话 token 计数和成本
    result = await session.execute(
        select(AgentSession).where(AgentSession.id == UUID(session_id))
    )
    agent_session = result.scalar_one_or_none()
    if agent_session:
        agent_session.status = "completed"
        agent_session.token_count = token_count
        input_cost = (prompt_tokens / 1000) * settings.token_cost_per_1k_input
        output_cost = (completion_tokens / 1000) * settings.token_cost_per_1k_output
        agent_session.estimated_cost = input_cost + output_cost

    await session.flush()
    return token_count
