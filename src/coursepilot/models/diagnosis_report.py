"""学情诊断报告模型（聚合级别）

存储每次学情诊断的完整结果，包括：
- 整体统计数据（总题数、正确率）
- 各知识点细粒度统计
- LLM 生成的深度分析和学习建议
"""
import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import DateTime, Float, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from coursepilot.db import Base


class DiagnosisReport(Base):
    """学情诊断报告（聚合级别）。

    系统根据学生练习记录进行聚合分析，识别薄弱知识点，
    并由 LLM 生成个性化学习建议。
    """
    __tablename__ = "diagnosis_reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("courses.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("agent_sessions.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    overall_rate: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, comment="总体正确率",
    )
    total_practiced: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="总练习题目数",
    )
    kp_stats: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="各知识点统计 {kp_path: {total, correct, rate}}",
    )
    weak_kps: Mapped[list | None] = mapped_column(
        JSONB, nullable=True, comment="薄弱知识点路径列表",
    )
    llm_analysis: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="LLM 生成的深度分析",
    )
    recommendations: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="LLM 生成的学习建议",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    def __repr__(self) -> str:
        return f"<DiagnosisReport user={self.user_id} rate={self.overall_rate:.0%}>"
