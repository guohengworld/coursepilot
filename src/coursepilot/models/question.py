import uuid
from datetime import datetime

from sqlalchemy import (
    String, Text, SmallInteger, Boolean, DateTime, ForeignKey, func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from coursepilot.db import Base


class Question(Base):
    """题目，由 LLM 自动生成或人工录入。

    question_type: choice_4 (四选一) / true_false (判断)
    options: {"A": "text", "B": "text", "C": "text", "D": "text"}
    verified: 是否通过三重自验证
    """
    __tablename__ = "questions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    kp_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_points.id"),
        nullable=False, index=True,
    )
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    question_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default="choice_4",
    )
    options: Mapped[dict] = mapped_column(JSONB, nullable=False)
    correct_answer: Mapped[str] = mapped_column(String(4), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty: Mapped[int] = mapped_column(SmallInteger, default=1)
    source: Mapped[str] = mapped_column(
        String(32), nullable=False, default="system", index=True,
    )
    verified: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Question {self.question_type} kp={self.kp_id}>"
