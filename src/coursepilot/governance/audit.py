"""审计日志：记录关键操作和权限检查结果

写入 audit_logs 表 (AuditLog model)，异步不阻塞
所有函数都是“发后即忘”模式，不阻塞主流程

Usage:
    from coursepilot.governance.audit import log_action

    await log_action(
        user_id="...",
        action="agent.chat",
        resource_type="agent_session",
        resource_id=session_id,
        details={"intent": "question", "token_count": 150},
        ip_address="192.168.1.1",
    )
"""
import logging
from uuid import UUID

from coursepilot.db import async_session_factory
from coursepilot.models import AuditLog

logger = logging.getLogger(__name__)

async def log_action(
    user_id: str,
    action: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    details: dict | None = None,
    ip_address: str | None = None,
) -> None:
    """写一条审计日志（独立 session，异常不影响主流程）

    Args:
        user_id: 操作用户 ID (UUID 字符串)
        action: 操作名称。命名约定："{领域}.{操作}",如"course.delete"
        resource_type: 资源类型，如 "course" "agent_session"
        details: 操作详情 (JSON 序列化)
        ip_address: 客户端 IP (可选)
    """
    try:
        async with async_session_factory() as session:
            session.add(AuditLog(
                user_id=UUID(user_id),
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details,
                ip_address=ip_address,
            ))
            await session.flush()
    except Exception:
        logger.exception("审计日志写入失败（不影响主流程）")

# 便携封装
async def log_agent_chat(
    user_id: str, session_id: str, intent: str, query: str
) -> None:
    """记录 Agent 对话"""
    await log_action(
        user_id=user_id,
        action="agent.chat",
        resource_type="agent_session",
        resource_id=session_id,
        details={"intent": intent, "query_preview": query[:100]},
    )

async def log_permission_denied(
    user_id: str, permission: str, resource_type: str | None = None
) -> None:
    """记录权限拒绝"""
    await log_action(
        user_id=user_id,
        action="permission.denied",
        resource_type=resource_type,
        details={"permission": permission}
    )

async def log_quiz_generated(
    user_id: str, session_id: str, question_count: int
) -> None:
    """记录试题生成"""
    await log_action(
        user_id=user_id,
        action="quiz.generated",
        resource_type="agent_session",
        resource_id=session_id,
        details={"question_count": question_count},
    )

async def log_guardrail_violation(
    user_id: str, session_id: str, issues: list[str]
) -> None:
    """记录护栏违规"""
    await log_action(
        user_id=user_id,
        action="guardrail.violation",
        resource_type="agent_session",
        resource_id=session_id,
        details={"issues": issues},
    )
