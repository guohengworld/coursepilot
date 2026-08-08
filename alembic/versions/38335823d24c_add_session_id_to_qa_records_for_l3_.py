"""add session_id to qa_records for L3 memory provenance

Revision ID: 38335823d24c
Revises: 94e81d0d679b
Create Date: 2026-08-09 00:07:02.346753
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '38335823d24c'
down_revision: Union[str, None] = '94e81d0d679b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 为 qa_records 增加 session_id（可空：历史记录没有归属会话），
    # 供 L3 记忆抽取按会话过滤（extract_facts_for_session）。
    op.add_column(
        'qa_records',
        sa.Column(
            'session_id',
            sa.UUID(),
            nullable=True,
            comment='所属会话 ID（L3 记忆 provenance 用）',
        ),
    )
    op.create_index(
        op.f('ix_qa_records_session_id'), 'qa_records', ['session_id'], unique=False,
    )
    op.create_foreign_key(
        op.f('qa_records_session_id_fkey'),
        'qa_records', 'agent_sessions', ['session_id'], ['id'],
        ondelete='CASCADE',
    )


def downgrade() -> None:
    op.drop_constraint(op.f('qa_records_session_id_fkey'), 'qa_records', type_='foreignkey')
    op.drop_index(op.f('ix_qa_records_session_id'), table_name='qa_records')
    op.drop_column('qa_records', 'session_id')
