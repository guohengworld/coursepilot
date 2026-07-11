"""add query to agent_sessions

Revision ID: 09d68a630367
Revises: d75c7e9650bf
Create Date: 2026-07-11 17:02:46.851723
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '09d68a630367'
down_revision: Union[str, None] = 'd75c7e9650bf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('agent_sessions', sa.Column('query', sa.Text(), nullable=True, comment='用户原始提问'))


def downgrade() -> None:
    op.drop_column('agent_sessions', 'query')
