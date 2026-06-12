import uuid
from datetime import datetime

from sqlalchemy import String, Text, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from coursepilot.db import Base


class DiagnosisReport(Base):
    """错题诊断报告。

    系统对学生的错误答案进行自动分析，定位原因并推荐复习知识点。
    """
    __tablename__ = "diagnosis_reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    practice_record_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("practice_records.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    error_reason: Mapped[str] = mapped_column(Text, nullable=False, comment="错误原因",)
    error_category: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="错误类别",
    )
    remedy_kp_ids: Mapped[list | None] = mapped_column(
        ARRAY(UUID(as_uuid=True)), default=list, comment="建议知识点 ID",
    )
    report_content: Mapped[str] = mapped_column(Text, nullable=False, comment="报告内容",)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment="创建时间",
    )

    def __repr__(self) -> str:
        return f"<Diagnosis {self.error_category}>"
