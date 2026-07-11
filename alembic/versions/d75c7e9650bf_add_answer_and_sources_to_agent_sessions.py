"""add answer and sources to agent_sessions

Revision ID: d75c7e9650bf
Revises: 6c28e2b9cefc
Create Date: 2026-07-11 16:46:12.809379
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd75c7e9650bf'
down_revision: Union[str, None] = '6c28e2b9cefc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('agent_sessions', sa.Column('answer', sa.Text(), nullable=True, comment='Agent 回答文本'))
    op.add_column('agent_sessions', sa.Column('sources', postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment='引用来源列表'))


def downgrade() -> None:
    op.drop_column('agent_sessions', 'sources')
    op.drop_column('agent_sessions', 'answer')
