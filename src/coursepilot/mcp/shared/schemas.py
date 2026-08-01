"""MCP 工具共享 Pydantic Schema。

所有工具参数、响应模型统一在此定义，确保 Server 与 Gateway 校验一致。
"""

import re
import unicodedata
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── 通用校验函数 ─────────────────────────────────────

def normalize_text(value: str) -> str:
    """对字符串做 Unicode NFC 规范化。"""
    return unicodedata.normalize("NFC", value)


def validate_kp_path(value: str) -> str:
    """校验 kp_path 只允许中文、字母、数字、/、-、_。"""
    if not value:
        return value
    if len(value) > 512:
        raise ValueError("kp_path 长度不能超过 512 字符")
    if not re.fullmatch(r"[\u4e00-\u9fa5a-zA-Z0-9\/_\-]+", value):
        raise ValueError("kp_path 包含非法字符，只允许中文、字母、数字、/、-、_")
    return value


# ── 工具参数基类 ─────────────────────────────────────

class BaseToolParams(BaseModel):
    """所有工具参数的基类，开启严格模式并允许丢弃未声明字段。"""

    model_config = ConfigDict(
        extra="ignore",       # 丢弃 schema 未声明参数（幻觉参数过滤）
        str_strip_whitespace=True,
    )

    idempotency_key: str | None = Field(
        default=None,
        description="幂等键，重复调用返回相同结果",
        max_length=128,
    )


# ── 教学问答工具 ─────────────────────────────────────

class QueryKnowledgeParams(BaseToolParams):
    """query_knowledge 工具参数。"""

    query: str = Field(
        ...,
        description="学生的问题",
        min_length=1,
        max_length=2000,
    )
    course_id: UUID = Field(..., description="课程 UUID")
    kp_path: str = Field(default="", description="可选的知识点范围")

    _normalize_query = field_validator("query", mode="before")(normalize_text)
    _validate_kp_path = field_validator("kp_path", mode="before")(validate_kp_path)


class QueryKnowledgeResult(BaseModel):
    """query_knowledge 工具响应。"""

    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: int | None = None


# ── 练习工具 ─────────────────────────────────────────

class GeneratePracticeParams(BaseToolParams):
    """generate_practice 工具参数。"""

    course_id: UUID = Field(..., description="课程 UUID")
    kp_path: str = Field(default="", description="目标知识点路径")
    count: int = Field(default=3, ge=1, le=10, description="题目数量")
    difficulty: int = Field(default=3, ge=1, le=5, description="难度 1-5")

    _validate_kp_path = field_validator("kp_path", mode="before")(validate_kp_path)


class GradeAnswersParams(BaseToolParams):
    """grade_answers 工具参数。"""

    user_id: UUID = Field(..., description="学生 UUID")
    question_id: UUID = Field(..., description="题目 UUID")
    answer: str = Field(..., min_length=1, max_length=16, description="学生答案")


# ── 诊断与复习工具 ───────────────────────────────────

class DiagnoseParams(BaseToolParams):
    """diagnose 工具参数。"""

    user_id: UUID = Field(..., description="学生 UUID")
    course_id: UUID = Field(..., description="课程 UUID")


class GetReviewPlanParams(BaseToolParams):
    """get_review_plan 工具参数。"""

    user_id: UUID = Field(..., description="学生 UUID")
    course_id: UUID = Field(..., description="课程 UUID")


# ── 知识库工具 ───────────────────────────────────────

class SearchKnowledgeUnitsParams(BaseToolParams):
    """search_knowledge_units 工具参数。"""

    query: str = Field(..., min_length=1, max_length=2000, description="检索查询")
    course_id: UUID = Field(..., description="课程 UUID")
    top_k: int = Field(default=10, ge=1, le=50, description="返回条数")


class GetKPTreeParams(BaseToolParams):
    """get_kp_tree 工具参数。"""

    course_id: UUID = Field(..., description="课程 UUID")


# ── Gateway 校验辅助 ─────────────────────────────────

class MCPRequestMeta(BaseModel):
    """MCP 请求 _meta 字段。"""

    protocol_version: str = Field(default="2025-06-18", description="MCP 协议版本")
