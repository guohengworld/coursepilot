"""add quiz_data to agent_sessions

Revision ID: 6c28e2b9cefc
Revises: 0cb0ee18f542
Create Date: 2026-07-11 14:45:59.372679
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '6c28e2b9cefc'
down_revision: Union[str, None] = 'e7df0f7e4afb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('agent_sessions', sa.Column('quiz_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment='练习题数据（含问题、选项、答案）'))


def downgrade() -> None:
    op.drop_column('agent_sessions', 'quiz_data')
