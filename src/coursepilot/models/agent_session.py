import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from coursepilot.db import Base


class AgentSession(Base):
    """Agent 会话表：业务元数据，供查询会话列表、成本统计。

    langgraph_thread_id 关联 LangGraph PostgresSaver checkpoint，
    执行细节由 checkpoint 管理，本表只存业务元数据。
    """
    __tablename__ = "agent_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True, comment="用户 ID",
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("courses.id"),
        nullable=False, index=True, comment="课程 ID",
    )
    intent: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="question/practice/diagnose/review/code_help",
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending",
        comment="pending/running/waiting_human/completed/failed",
    )
    token_count: Mapped[int] = mapped_column(
        Integer, default=0, comment="Token 消耗量",
    )
    estimated_cost: Mapped[float] = mapped_column(
        Numeric(10, 4), default=0, comment="预估成本（元）",
    )
    langgraph_thread_id: Mapped[str | None] = mapped_column(
        String(100), comment="LangGraph checkpoint thread_id",
    )
    quiz_data: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="练习题数据（含问题、选项、答案）",
    )
    query: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="用户原始提问",
    )
    answer: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Agent 回答文本",
    )
    sources: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="引用来源列表",
    )
    conversation: Mapped[list | None] = mapped_column(
        JSONB, nullable=True, default=list, comment="多轮对话历史 [{'role', 'content', 'intent'}]",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False,
    )

    def __repr__(self) -> str:
        return f"<AgentSession {self.intent} {self.status}>"
