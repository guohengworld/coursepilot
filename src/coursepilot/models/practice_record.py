import uuid
from datetime import datetime

from sqlalchemy import (
    String, Boolean, DateTime, ForeignKey, func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from coursepilot.db import Base


class PracticeRecord(Base):
    """做题记录。

    correct_flag: True(对) / False(错) / NULL(未作答/跳过)
    answered_at: 学生提交答案的时间（留空表示未作答）
    """
    __tablename__ = "practice_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("questions.id"),
        nullable=False, index=True,
    )
    user_answer: Mapped[str | None] = mapped_column(String(4), default=None, comment="用户答案")
    correct_flag: Mapped[bool | None] = mapped_column(Boolean, default=None, comment="对错标记")
    answered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None, comment="提交答案时间",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment="创建时间",
    )

    def __repr__(self) -> str:
        return f"<Practice user={self.user_id} q={self.question_id}>"
