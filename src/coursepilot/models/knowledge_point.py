import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, String, Text, SmallInteger, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import mapped_column, Mapped

from coursepilot.db import Base

class KnowledgePoint(Base):
    """知识点树（adjacency list + 递归 CTE）。

    例如 kp_path = "OS/process/scheduling/rr"，深度不超过 4 层。
    parent_id 指向父节点，根节点的 parent_id 为 NULL。
    """
    __tablename__ = "knowledge_points"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="CASCADE"),
        nullable=False, index=True, comment="课程 ID",
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_points.id", ondelete="CASCADE"),
        default=None, index=True, comment="父节点 ID",
    )
    kp_path: Mapped[str] = mapped_column(
        String(512), nullable=False, comment="知识点路径",
    )
    title: Mapped[str] = mapped_column(String(256), nullable=False, comment="知识点标题")
    summary: Mapped[str | None] = mapped_column(Text, default=None, comment="知识点摘要")
    difficulty: Mapped[int] = mapped_column(SmallInteger, default=1, comment="知识点难度")
    source: Mapped[str] = mapped_column(
        String(32), nullable=False, default="course", comment="知识点来源",
    )
    sort_order: Mapped[int] = mapped_column(default=0, comment="排序顺序")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment="创建时间",
    )

    def __repr__(self) -> str:
        return f"<KP {self.kp_path}>"

