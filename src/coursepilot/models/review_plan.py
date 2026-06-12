import uuid
from datetime import datetime

from sqlalchemy import SmallInteger, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from coursepilot.db import Base


class ReviewPlan(Base):
    """复习计划。

    items JSONB 结构：
    [{"kp_id": "...", "kp_path": "OS/...", "priority": 1,
      "reason": "错题率 60%", "status": "pending"}]
    priority: 1(最薄弱) ~ 5(已掌握)
    status: 'pending' | 'reviewed'
    """
    __tablename__ = "review_plans"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("courses.id"),
        nullable=False, index=True,
    )
    items: Mapped[dict] = mapped_column(JSONB, nullable=False, default=list, comment="复习计划项")
    reviewed_count: Mapped[int] = mapped_column(SmallInteger, default=0, comment="已复习数量")
    total_count: Mapped[int] = mapped_column(SmallInteger, default=0, comment="总数量")
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment="生成时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False, comment="更新时间",
    )

    def __repr__(self) -> str:
        return f"<ReviewPlan user={self.user_id} course={self.course_id}>"
