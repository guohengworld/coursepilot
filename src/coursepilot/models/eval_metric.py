import uuid
from datetime import datetime

from sqlalchemy import String, Integer, Float, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from coursepilot.db import Base


class EvalMetric(Base):
    """RAG 评估指标记录"""
    __tablename__ = "eval_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    eval_type: Mapped[str] = mapped_column(String(32), nullable=False, comment="评估类型")
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False, comment="评估指标名称")
    metric_value: Mapped[float] = mapped_column(Float, nullable=False, comment="评估指标值")
    sample_size: Mapped[int | None] = mapped_column(Integer, default=None, comment="样本数量")
    meta_data: Mapped[dict] = mapped_column(JSONB, default=dict, comment="附加信息")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment="创建时间",
    )

    def __repr__(self) -> str:
        return f"<Eval {self.eval_type}/{self.metric_name}={self.metric_value:.3f}>"
