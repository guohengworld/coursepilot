import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Text, DateTime, func, String
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import mapped_column, Mapped

from coursepilot.db import Base

class KnowledgeUnit(Base):
    """知识单元 — 文档切分后的检索基本单位。

    Milvus 中存一份向量，PostgreSQL 中存一份元数据。
    metadata 字段存扩展属性（页码、章节号等），用于 Milvus filter。
    """
    __tablename__ = "knowledge_units"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    kp_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_points.id", ondelete="CASCADE"), nullable=False, index=True, comment="知识点 ID",
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"), index=True, comment="文档 ID",
    )
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="内容")
    summary: Mapped[str | None] = mapped_column(Text, default=None, comment="摘要")
    seq_order: Mapped[int] = mapped_column(default=0, comment="在同一知识点内的顺序")
    page_ref: Mapped[str | None] = mapped_column(String(64), default=None, comment="页码引用") # p45-47
    meta_data: Mapped[dict] = mapped_column(
        JSONB, default=dict, server_default="{}", comment="元数据"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment="创建时间",
    )

    def __repr__(self):
        return f"<KU {self.id}>"

