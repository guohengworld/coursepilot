import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from coursepilot.db import Base


class Enrollment(Base):
    """选课表：课程成员关系，课程归属判据（谁属于这门课、以什么身份）。

    role 取值：student（选了这门课的学生）/ teacher（课程创建者）。
    (user_id, course_id) 联合唯一——同一用户在门课内只有一条成员记录，
    教师身份优先于学生身份（回填时 teacher 先插、student 用 ON CONFLICT 跳过）。
    """
    __tablename__ = "enrollments"
    __table_args__ = (
        UniqueConstraint("user_id", "course_id", name="uq_enrollments_user_course"),
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
        ForeignKey("courses.id", ondelete="CASCADE"),
        nullable=False, index=True, comment="课程 ID",
    )
    role: Mapped[str] = mapped_column(
        String(16), nullable=False, default="student",
        index=True, comment="课程内角色：student/teacher",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        nullable=False, comment="加入课程时间",
    )

    def __repr__(self) -> str:
        return f"<Enrollment user={self.user_id} course={self.course_id} role={self.role}>"
