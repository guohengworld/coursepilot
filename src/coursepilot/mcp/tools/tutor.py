"""MCP Tutor 工具实现。

包含：query_knowledge（P1 核心）
后续会加入：diagnose, get_review_plan, get_mastery, get_student_report
"""

from __future__ import annotations

import json
import logging
from uuid import UUID

from mcp.types import CallToolResult, TextContent
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.db import async_session_factory
from coursepilot.rag.generator import Generator, build_course_context
from coursepilot.rag.retriever import Retriever

from coursepilot.mcp.shared.schemas import QueryKnowledgeParams

_LOGGER = logging.getLogger(__name__)


async def _get_session() -> AsyncSession:
    """获取一个异步 DB 会话。"""
    return async_session_factory()


async def query_knowledge(params: QueryKnowledgeParams) -> CallToolResult:
    """基于课程教材内容回答学生问题。

    [1-用途] 根据教材知识库回答课程相关问题，返回带引用来源的答案。
    [2-限制] query 长度不超过 2000 字符；仅查询指定 course_id 下的内容。
    [3-成本] 中，需要调用一次 LLM 生成答案。
    [4-副作用] 无，只读工具。
    [5-输入格式] query: 问题文本；course_id: 课程 UUID；kp_path: 可选知识点范围。
    [6-输出格式] 返回答案文本、涉及知识点和 Token 用量。
    """
    session = await _get_session()
    try:
        course_context = await build_course_context(session, str(params.course_id))
        if not course_context:
            return _make_error(f"课程 {params.course_id} 不存在")

        retriever = Retriever()
        context, metadata = await retriever.retrieve(
            session,
            params.query,
            str(params.course_id),
        )

        generator = Generator()
        answer, token_info = await generator.generate(
            params.query, context, course_context
        )

        source_kps = metadata.get("source_kp_paths", [])
        result = {
            "answer": answer,
            "source_kps": source_kps[:5],
            "tokens": token_info,
        }
        return _make_success(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        _LOGGER.exception("query_knowledge failed: %s", e)
        return _make_error(f"查询失败: {e}")
    finally:
        await session.close()


def _make_success(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=False)


def _make_error(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)
