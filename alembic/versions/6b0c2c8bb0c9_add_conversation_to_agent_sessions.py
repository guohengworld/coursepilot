"""add conversation to agent_sessions

Revision ID: 6b0c2c8bb0c9
Revises: 09d68a630367
Create Date: 2026-07-11 17:30:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = '6b0c2c8bb0c9'
down_revision: Union[str, None] = '09d68a630367'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('agent_sessions', sa.Column('conversation', JSONB(), nullable=True, comment='多轮对话历史'))


def downgrade() -> None:
    op.drop_column('agent_sessions', 'conversation')
