"""add document_id to knowledge_points

Revision ID: e7df0f7e4afb
Revises: 0cb0ee18f542
Create Date: 2026-07-09 21:15:01.165287
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e7df0f7e4afb'
down_revision: Union[str, None] = '0cb0ee18f542'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('knowledge_points', sa.Column('document_id', sa.UUID(), nullable=True, comment='来源文档 ID'))
    op.create_index(op.f('ix_knowledge_points_document_id'), 'knowledge_points', ['document_id'], unique=False)
    op.create_foreign_key(None, 'knowledge_points', 'documents', ['document_id'], ['id'], ondelete='CASCADE')


def downgrade() -> None:
    op.drop_constraint(None, 'knowledge_points', type_='foreignkey')
    op.drop_index(op.f('ix_knowledge_points_document_id'), table_name='knowledge_points')
    op.drop_column('knowledge_points', 'document_id')
