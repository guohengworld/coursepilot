"""repurpose diagnosis_reports for aggregate diagnosis

Replaces per-question error columns with aggregate diagnosis report fields.
Old table has no data, so a full drop/recreate is safe.

Revision ID: a1b2c3d4e5f6
Revises: 6b0c2c8bb0c9
Create Date: 2026-07-11 20:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "6b0c2c8bb0c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("diagnosis_reports")
    op.create_table(
        "diagnosis_reports",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", UUID(as_uuid=True),
                  sa.ForeignKey("users.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("course_id", UUID(as_uuid=True),
                  sa.ForeignKey("courses.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("session_id", UUID(as_uuid=True),
                  sa.ForeignKey("agent_sessions.id", ondelete="SET NULL"),
                  nullable=True, index=True),
        sa.Column("overall_rate", sa.Float(), nullable=False,
                  server_default=sa.text("0.0")),
        sa.Column("total_practiced", sa.Integer(), nullable=False,
                  server_default=sa.text("0")),
        sa.Column("kp_stats", JSONB, nullable=True,
                  comment="各知识点统计 {kp_path: {total, correct, rate}}"),
        sa.Column("weak_kps", JSONB, nullable=True,
                  comment="薄弱知识点列表"),
        sa.Column("llm_analysis", sa.Text(), nullable=True,
                  comment="LLM 深度分析"),
        sa.Column("recommendations", sa.Text(), nullable=True,
                  comment="LLM 学习建议"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("diagnosis_reports")
    op.create_table(
        "diagnosis_reports",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", UUID(as_uuid=True),
                  sa.ForeignKey("users.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("practice_record_id", UUID(as_uuid=True),
                  sa.ForeignKey("practice_records.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("error_reason", sa.Text(), nullable=False),
        sa.Column("error_category", sa.String(32), nullable=False),
        sa.Column("remedy_kp_ids", sa.ARRAY(UUID(as_uuid=True)), nullable=True),
        sa.Column("report_content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
    )
