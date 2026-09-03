"""add enrollments (course membership)

Revision ID: a137d53503d7
Revises: 38335823d24c
Create Date: 2026-09-03 15:40:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a137d53503d7'
down_revision: Union[str, None] = '38335823d24c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 选课表：课程成员关系，(user_id, course_id) 联合唯一，role=student/teacher。
    # 回填来源：user_profiles 去重(student) + courses.created_by(teacher)。
    op.create_table('enrollments',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False, comment='用户 ID'),
    sa.Column('course_id', sa.UUID(), nullable=False, comment='课程 ID'),
    sa.Column('role', sa.String(length=16), nullable=False, comment='课程内角色：student/teacher'),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='加入课程时间'),
    sa.ForeignKeyConstraint(['course_id'], ['courses.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'course_id', name='uq_enrollments_user_course')
    )
    op.create_index(op.f('ix_enrollments_course_id'), 'enrollments', ['course_id'], unique=False)
    op.create_index(op.f('ix_enrollments_role'), 'enrollments', ['role'], unique=False)
    op.create_index(op.f('ix_enrollments_user_id'), 'enrollments', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_enrollments_user_id'), table_name='enrollments')
    op.drop_index(op.f('ix_enrollments_role'), table_name='enrollments')
    op.drop_index(op.f('ix_enrollments_course_id'), table_name='enrollments')
    op.drop_table('enrollments')
