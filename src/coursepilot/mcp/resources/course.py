"""MCP Course 资源实现。

包含：
    course://{course_id}/kp-tree
    course://{course_id}/documents
    course://{course_id}/stats
    student://{user_id}/{course_id}/report
    student://{user_id}/{course_id}/mastery
"""

from __future__ import annotations

import json
import logging
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.db import async_session_factory
from coursepilot.models import Document, KnowledgePoint, KnowledgeUnit

_LOGGER = logging.getLogger(__name__)


async def _get_session() -> AsyncSession:
    """获取一个异步 DB 会话。"""
    return async_session_factory()


async def _course_exists(session: AsyncSession, course_id: str) -> bool:
    """检查课程是否有知识点，作为课程存在的代理判断。"""
    result = await session.execute(
        select(KnowledgePoint.id).where(KnowledgePoint.course_id == course_id).limit(1)
    )
    return result.scalar_one_or_none() is not None


async def read_kp_tree(course_id: str) -> str:
    """返回课程知识点树。"""
    session = await _get_session()
    try:
        if not await _course_exists(session, course_id):
            return json.dumps({"error": "课程不存在"}, ensure_ascii=False)

        result = await session.execute(
            select(KnowledgePoint.kp_path)
            .where(KnowledgePoint.course_id == course_id)
            .order_by(KnowledgePoint.kp_path)
        )
        kp_paths = [row[0] for row in result.all() if row[0]]
        return json.dumps(
            {"course_id": course_id, "kp_paths": kp_paths},
            ensure_ascii=False,
        )
    except Exception as e:
        _LOGGER.exception("read_kp_tree failed: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    finally:
        await session.close()


async def read_documents(course_id: str) -> str:
    """返回课程文档清单及状态。"""
    session = await _get_session()
    try:
        result = await session.execute(
            select(Document.id, Document.filename, Document.status, Document.created_at)
            .where(Document.course_id == course_id)
            .order_by(Document.created_at.desc())
        )
        documents = [
            {
                "id": str(row[0]),
                "filename": row[1],
                "status": row[2],
                "created_at": row[3].isoformat() if row[3] else None,
            }
            for row in result.all()
        ]
        return json.dumps(
            {"course_id": course_id, "documents": documents},
            ensure_ascii=False,
        )
    except Exception as e:
        _LOGGER.exception("read_documents failed: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    finally:
        await session.close()


async def read_stats(course_id: str) -> str:
    """返回课程统计信息。"""
    session = await _get_session()
    try:
        kp_count = await session.scalar(
            select(func.count(KnowledgePoint.id)).where(
                KnowledgePoint.course_id == course_id
            )
        )
        unit_count = await session.scalar(
            select(func.count(KnowledgeUnit.id)).where(
                KnowledgeUnit.course_id == course_id
            )
        )
        doc_count = await session.scalar(
            select(func.count(Document.id)).where(Document.course_id == course_id)
        )
        return json.dumps(
            {
                "course_id": course_id,
                "kp_count": kp_count or 0,
                "unit_count": unit_count or 0,
                "document_count": doc_count or 0,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        _LOGGER.exception("read_stats failed: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    finally:
        await session.close()


async def read_report(user_id: str, course_id: str) -> str:
    """返回学生综合学情报告（MVP 阶段为简化版）。"""
    session = await _get_session()
    try:
        # MVP 阶段返回知识点覆盖和文档状态
        result = await session.execute(
            select(KnowledgePoint.kp_path)
            .where(KnowledgePoint.course_id == course_id)
            .order_by(KnowledgePoint.kp_path)
        )
        kp_paths = [row[0] for row in result.all() if row[0]]
        return json.dumps(
            {
                "user_id": user_id,
                "course_id": course_id,
                "summary": "MVP 阶段简化报告",
                "total_kps": len(kp_paths),
                "kp_paths": kp_paths[:20],
            },
            ensure_ascii=False,
        )
    except Exception as e:
        _LOGGER.exception("read_report failed: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    finally:
        await session.close()


async def read_mastery(user_id: str, course_id: str) -> str:
    """返回学生掌握度画像（MVP 阶段为简化版）。"""
    session = await _get_session()
    try:
        return json.dumps(
            {
                "user_id": user_id,
                "course_id": course_id,
                "summary": "MVP 阶段简化掌握度画像",
                "overall_rate": None,
                "weak_kps": [],
            },
            ensure_ascii=False,
        )
    except Exception as e:
        _LOGGER.exception("read_mastery failed: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    finally:
        await session.close()
