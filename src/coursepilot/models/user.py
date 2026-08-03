import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from coursepilot.db import Base

class User(Base):
    __tablename__ = "users"

    # Mapped[uuid.UUID]：告诉 Python 这个变量是 UUID 类型
    # mapped_column：告诉数据库这个列的具体规则、属性
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    username: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True, comment="用户名",
    )
    # passlib bcrypt hash，长度 256 留足空间
    password_hash: Mapped[str] = mapped_column(
        String(256), nullable=False, comment="密码哈希值",
    )
    role: Mapped[str] = mapped_column(
        String(16), nullable=False, default="student", index=True, comment="用户角色",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment="创建时间",
    )

    def __repr__(self):
        return f"<User {self.username} {self.role}>"
