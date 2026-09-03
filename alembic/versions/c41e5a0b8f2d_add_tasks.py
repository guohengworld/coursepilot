"""add tasks (teacher-published AI tasks, draft/published)

Revision ID: c41e5a0b8f2d
Revises: a137d53503d7
Create Date: 2026-09-03 17:20:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c41e5a0b8f2d'
down_revision: Union[str, None] = 'a137d53503d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 教师给指定学生发布的 AI 任务（四层结构化：诊断依据/目标/题组/验收）。
    # student_id = 目标学生，created_by = 布置教师，status = draft/published。
    op.create_table('tasks',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('course_id', sa.UUID(), nullable=False, comment='课程 ID'),
    sa.Column('student_id', sa.UUID(), nullable=False, comment='目标学生 ID'),
    sa.Column('created_by', sa.UUID(), nullable=False, comment='布置任务的教师 ID'),
    sa.Column('status', sa.String(length=16), nullable=False, comment='draft/published'),
    sa.Column('diagnosis', sa.dialects.postgresql.JSONB(), nullable=False, comment='四层-1 诊断依据'),
    sa.Column('goal', sa.dialects.postgresql.JSONB(), nullable=False, comment='四层-2 任务目标'),
    sa.Column('groups', sa.dialects.postgresql.JSONB(), nullable=False, comment='四层-3 任务本体题组'),
    sa.Column('total_count', sa.Integer(), nullable=False, comment='题组总量（=Σ question_count）'),
    sa.Column('time_limit_minutes', sa.Integer(), nullable=True, comment='建议完成时限（分钟）'),
    sa.Column('acceptance', sa.dialects.postgresql.JSONB(), nullable=False, comment='四层-4 验收标准'),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='创建时间'),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='更新时间'),
    sa.Column('published_at', sa.DateTime(timezone=True), nullable=True, comment='发布时间'),
    sa.ForeignKeyConstraint(['course_id'], ['courses.id']),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['student_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_tasks_status'), 'tasks', ['status'], unique=False)
    op.create_index('ix_tasks_course_created_status', 'tasks', ['course_id', 'created_by', 'status'], unique=False)
    op.create_index('ix_tasks_student_status', 'tasks', ['student_id', 'status'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_tasks_student_status', table_name='tasks')
    op.drop_index('ix_tasks_course_created_status', table_name='tasks')
    op.drop_index(op.f('ix_tasks_status'), table_name='tasks')
    op.drop_table('tasks')
