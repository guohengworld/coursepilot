import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from coursepilot.db import Base


class AuditLog(Base):
    """审计日志表：记录关键操作和权限检查。
    """
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False, index=True, comment="操作用户 ID",
    )
    action: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="操作名称",
    )
    resource_type: Mapped[str | None] = mapped_column(
        String(50), comment="资源类型",
    )
    resource_id: Mapped[str | None] = mapped_column(
        String(100), comment="资源 ID",
    )
    details: Mapped[dict | None] = mapped_column(
        JSONB, comment="操作详情",
    )
    ip_address: Mapped[str | None] = mapped_column(
        INET, comment="客户端 IP",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    def __repr__(self) -> str:
        return f"<AuditLog {self.action}>"
