import uuid
from datetime import datetime

from sqlalchemy import UUID, ForeignKey, Text, String, SmallInteger, ARRAY, DateTime, Float, func, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from coursepilot.db import Base


class QARecord(Base):
    """问答历史记录。

    retrieved_units: 检索到的知识单元 UUID 列表
    citations: [{"source": "OSTEP p42", "kp_path": "OS/process/scheduling"}]
    feedback: 1(有用) / 0(无用) / NULL(未反馈)
    """
    __tablename__ = "qa_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True, comment="用户 ID",
    )
    course_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("courses.id"), index=True,)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_sessions.id", ondelete="CASCADE"),
        index=True, comment="所属会话 ID（L3 记忆 provenance 用）",
    )
    query: Mapped[str] = mapped_column(Text, nullable=False, comment="问题",)
    kp_path: Mapped[str | None] = mapped_column(String(512), comment="知识点路径",)
    retrieved_units: Mapped[list | None] = mapped_column(
        ARRAY(UUID(as_uuid=True)), default=list, comment="检索到的单元 ID",
    )
    answer: Mapped[str] = mapped_column(Text, nullable=False, comment="答案",)
    citations: Mapped[list] = mapped_column(JSONB, default=list, comment="参考来源",)
    feedback: Mapped[int | None] = mapped_column(SmallInteger, default=None, comment="用户反馈",)
    latency_ms: Mapped[int | None] = mapped_column(Integer, default=None, comment="处理耗时（毫秒）",)
    embedding: Mapped[list | None] = mapped_column(
        ARRAY(Float), default=None, comment="query+answer 的 BGE-M3 dense 向量",
    )
    importance: Mapped[float | None] = mapped_column(
        Float, default=None, comment="记忆重要性评分 0.0-1.0",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment="创建时间",
    )

    def __repr__(self) -> str:
        return f"<QA {self.query[:30]}...>"
