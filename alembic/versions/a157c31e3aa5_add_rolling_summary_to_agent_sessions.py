"""add rolling_summary to agent_sessions

Revision ID: a157c31e3aa5
Revises: a1b2c3d4e5f6
Create Date: 2026-07-23 20:50:14.778665
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a157c31e3aa5'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 仅新增 rolling_summary 列；checkpoint 相关表由 LangGraph 自行管理，不在这里处理
    op.add_column(
        'agent_sessions',
        sa.Column(
            'rolling_summary',
            sa.Text(),
            nullable=True,
            comment='L2 滚动摘要（老轮次压缩后）',
        ),
    )


def downgrade() -> None:
    op.drop_column('agent_sessions', 'rolling_summary')
