"""SQLAlchemy ORM 模型，与 System_Design 中的 DDL 一一对应。

11 张业务表 + 2 张辅助表（eval_metrics, review_plans）。

关系概览：
    User ──1:N──→ PracticeRecord ──N:1──→ Question
      │
      ├──1:N──→ DiagnosisReport ──N:1──→ Course
      ├──1:N──→ ReviewPlan
      ├──1:N──→ QARecord
      ├──1:N──→ AgentSession ────N:1──→ DiagnosisReport
      └──1:N──→ Document ──N:1──→ Course ➝ KnowledgePoint ➝ KnowledgeUnit
"""

from coursepilot.models.agent_session import AgentSession
from coursepilot.models.audit_log import AuditLog
from coursepilot.models.course import Course
from coursepilot.models.diagnosis_report import DiagnosisReport
from coursepilot.models.document import Document
from coursepilot.models.enrollment import Enrollment
from coursepilot.models.eval_metric import EvalMetric
from coursepilot.models.knowledge_point import KnowledgePoint
from coursepilot.models.knowledge_unit import KnowledgeUnit
from coursepilot.models.practice_record import PracticeRecord
from coursepilot.models.qa_record import QARecord
from coursepilot.models.question import Question
from coursepilot.models.review_plan import ReviewPlan
from coursepilot.models.user import User
from coursepilot.models.user_profile import UserProfile

__all__ = [
    "User",
    "Course",
    "KnowledgePoint",
    "KnowledgeUnit",
    "Document",
    "Question",
    "QARecord",
    "PracticeRecord",
    "DiagnosisReport",
    "ReviewPlan",
    "Enrollment",
    "EvalMetric",
    "AgentSession",
    "UserProfile",
    "AuditLog",
]
