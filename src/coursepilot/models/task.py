import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from coursepilot.db import Base


class Task(Base):
    """教师发布的 AI 任务（草稿 / 已发布）。

    生成场景（⑤）：教师选课程 → 选学生（enrollments role=student 候选）→
    AI 依据该生学情生成四层结构化任务 → 教师审核/编辑 → 发布 → 学生可见。

    四层结构落为独立 JSONB 列（对齐"PUT 字段级编辑"需求）：
      - diagnosis  诊断依据：{"mastery_level": {...}, "weak_kps": [...],
                     "common_mistakes": [...], "avg_correct_rate": float|null,
                     "class_rank": str|null}
      - goal       任务目标：{"metric": str, "description": str}
      - groups     任务本体题组：[{"kp_path": str, "kp_name": str,
                     "question_count": int, "difficulty": int(1-5),
                     "source": str|null, "reason": str|null}]
      - acceptance 验收标准：{"pass_condition": str, "fallback_action": str}
    total_count / time_limit_minutes 为任务本体的顶层冗余，方便列表展示。

    status: draft（教师审核中）/ published（学生端可见）。
    created_by 是布置任务的教师；归属校验（谁能编辑/发布）以 enrollments
    role=teacher 判据为准（C3 选课表）。
    """
    __tablename__ = "tasks"
    __table_args__ = (
        # 教师看"某课程自己布置的任务"、学生看"发布给自己的任务"
        # 两查询热路径分别命中下述复合索引
        Index("ix_tasks_course_created_status", "course_id", "created_by", "status"),
        Index("ix_tasks_student_status", "student_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("courses.id"),
        nullable=False, comment="课程 ID",
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, comment="目标学生 ID",
    )
    created_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, comment="布置任务的教师 ID",
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="draft", index=True,
        comment="draft/published",
    )
    diagnosis: Mapped[dict] = mapped_column(
        JSONB, default=dict, nullable=False, comment="四层-1 诊断依据",
    )
    goal: Mapped[dict] = mapped_column(
        JSONB, default=dict, nullable=False, comment="四层-2 任务目标",
    )
    groups: Mapped[list] = mapped_column(
        JSONB, default=list, nullable=False, comment="四层-3 任务本体题组",
    )
    total_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="题组总量（=Σ question_count）",
    )
    time_limit_minutes: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="建议完成时限（分钟）",
    )
    acceptance: Mapped[dict] = mapped_column(
        JSONB, default=dict, nullable=False, comment="四层-4 验收标准",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False, comment="更新时间",
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="发布时间",
    )

    def __repr__(self) -> str:
        return f"<Task {self.status} course={self.course_id} student={self.student_id}>"
