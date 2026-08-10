"""MCP Knowledge 工具实现。

包含：search_knowledge_units, get_kp_tree
"""

from __future__ import annotations

import json
import logging
from uuid import UUID

from mcp.types import CallToolResult, TextContent
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.db import async_session_factory
from coursepilot.mcp.shared.schemas import GetKPTreeParams, SearchKnowledgeUnitsParams
from coursepilot.models import KnowledgePoint, KnowledgeUnit
from coursepilot.rag.retriever import Retriever

_LOGGER = logging.getLogger(__name__)


async def _get_session() -> AsyncSession:
    """获取一个异步 DB 会话。"""
    return async_session_factory()


async def search_knowledge_units(params: SearchKnowledgeUnitsParams) -> CallToolResult:
    """检索课程知识单元。

    [1-用途] 根据学生查询从课程教材中检索最相关的知识单元。
    [2-限制] query 长度不超过 2000 字符；仅检索指定 course_id 下的内容。
    [3-成本] 低，走检索管线（向量化 + RRF + 重排序），不调用 LLM。
    [4-副作用] 无，只读工具。
    [5-输入格式] query: 检索文本；course_id: 课程 UUID；top_k: 返回条数（1-50）。
    [6-输出格式] 返回知识单元列表，每单元包含 uuid、content、summary、kp_path、score。
    """
    session = await _get_session()
    try:
        course_id = str(params.course_id)
        retriever = Retriever()
        _, metadata = await retriever.retrieve(
            session,
            params.query,
            course_id,
            enable_rewrite=False,
        )

        top_uuids = metadata.get("top_uuids", [])
        if not top_uuids:
            return _make_success(json.dumps({"units": []}, ensure_ascii=False))

        stmt = (
            select(KnowledgeUnit, KnowledgePoint.kp_path)
            .join(KnowledgePoint, KnowledgeUnit.kp_id == KnowledgePoint.id)
            .where(KnowledgeUnit.id.in_([UUID(uid) for uid in top_uuids]))
        )
        result = await session.execute(stmt)
        rows = result.all()

        uuid_to_unit = {}
        for unit, kp_path in rows:
            uuid_to_unit[str(unit.id)] = {
                "uuid": str(unit.id),
                "content": unit.content,
                "summary": unit.summary or "",
                "kp_path": kp_path or "",
                "page_ref": unit.page_ref or "",
            }

        # 保持重排序后的顺序
        units = []
        for uid in top_uuids[: params.top_k]:
            if uid in uuid_to_unit:
                units.append(uuid_to_unit[uid])

        return _make_success(json.dumps({"units": units}, ensure_ascii=False))
    except Exception as e:
        _LOGGER.exception("search_knowledge_units failed: %s", e)
        return _make_error(f"检索失败: {e}")
    finally:
        await session.close()


async def get_kp_tree(params: GetKPTreeParams) -> CallToolResult:
    """获取课程知识点树。

    [1-用途] 列出某门课程的全部知识点路径，帮助学生定位学习范围。
    [2-限制] 仅返回指定 course_id 下的知识点。
    [3-成本] 低，仅查数据库。
    [4-副作用] 无，只读工具。
    [5-输入格式] course_id: 课程 UUID。
    [6-输出格式] 返回知识点路径列表。
    """
    session = await _get_session()
    try:
        course_id = str(params.course_id)
        result = await session.execute(
            select(KnowledgePoint.kp_path)
            .where(KnowledgePoint.course_id == course_id)
            .order_by(KnowledgePoint.kp_path)
        )
        kp_paths = [row[0] for row in result.all() if row[0]]

        return _make_success(
            json.dumps(
                {"course_id": course_id, "kp_paths": kp_paths},
                ensure_ascii=False,
            )
        )
    except Exception as e:
        _LOGGER.exception("get_kp_tree failed: %s", e)
        return _make_error(f"获取知识点树失败: {e}")
    finally:
        await session.close()


def _make_success(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=False)


def _make_error(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)
