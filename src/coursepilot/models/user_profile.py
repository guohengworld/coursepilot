import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from coursepilot.db import Base


class UserProfile(Base):
    """学生画像表：持久化长期记忆，异步预计算，非实时聚合。

    mastery_level: {"数据结构/树/二叉树": 0.7, ...}
    weak_kps: 薄弱知识点 kp_path 数组
    common_mistakes: [{"category": "概念混淆", "pattern": "...", "count": 5}]
    """
    __tablename__ = "user_profiles"
    __table_args__ = (
        UniqueConstraint("user_id", "course_id", name="uq_user_profiles_user_course"),
    )

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
    mastery_level: Mapped[dict] = mapped_column(
        JSONB, default=dict, comment='{"kp_path": mastery_rate, ...}',
    )
    weak_kps: Mapped[list | None] = mapped_column(
        ARRAY(Text), default=list, comment="薄弱知识点 kp_path 数组",
    )
    common_mistakes: Mapped[list | None] = mapped_column(
        JSONB, default=list, comment="常见错误模式",
    )
    learning_style: Mapped[str | None] = mapped_column(
        String(50), comment="visual/textual/practice",
    )
    total_qa_count: Mapped[int] = mapped_column(
        Integer, default=0, comment="累计问答次数",
    )
    total_practice_count: Mapped[int] = mapped_column(
        Integer, default=0, comment="累计练习次数",
    )
    avg_correct_rate: Mapped[float | None] = mapped_column(
        Numeric(5, 2), comment="平均正确率",
    )
    last_diagnosis_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), comment="最近一次诊断时间",
    )
    last_review_plan_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), comment="最近复习计划 ID",
    )
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        nullable=False, comment="最后一次预计算时间",
    )

    def __repr__(self) -> str:
        return f"<UserProfile user={self.user_id} course={self.course_id}>"
