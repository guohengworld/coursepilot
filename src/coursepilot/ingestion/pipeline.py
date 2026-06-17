"""Ingestion 管线：解析文件 → 切分 → KP 分配 → 入库

流程：
  Document(status=pending)
    → 解析文件（pdf_parser / docx_parser）
    → extract_knowledge_units 切分
    → KPSplitter 分配到知识点
    → Document（status=ready）
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.models import Document, KnowledgePoint, KnowledgeUnit

logger = logging.getLogger(__name__)

async def run_ingestion(
    session: AsyncSession,
    document_id: str,
    start_page: int = 0,
    end_page: int | None = None,
    *,
    preparsed_content_list: list[dict] | None = None,
) -> None:
    """执行单个 Document 的 ingestion 管线。

    调用时机：POST /courses/upload 上传完成后。
    若传入 preparsed_content_list 则跳过文件解析，直接复用已有结果，
    避免同一个 PDF 被 MinerU 重复 OCR（一次解析耗时 5~100 分钟）。
    Week 3 会在本函数末尾追加 Milvus 编码 + BM25 索引步骤。
    """
    # 1. 获取 Document 记录
    doc = await session.get(Document, UUID(document_id))
    if not doc:
        logger.error(f"Document {document_id} not found")
        return

    doc.status = "processing"
    await session.flush()

    try:
        # 2. 解析文件 → content_list（优先使用预解析结果）
        if preparsed_content_list is not None:
            content_list = preparsed_content_list
        else:
            file_path = doc.file_path
            ext = doc.file_type

            if ext == "pdf":
                from coursepilot.ingestion.pdf_parser import parse_pdf
                result = await parse_pdf(
                    file_path,
                    start_page=start_page,
                    end_page=end_page,
                )
            elif ext == "docx":
                from coursepilot.ingestion.docx_parser import parse_docx
                result = parse_docx(file_path)
            elif ext == "md":
                from coursepilot.ingestion.markdown_parser import parse_markdown
                result = parse_markdown(file_path)
            else:
                raise ValueError(f"不支持的文件格式: .{ext}")

            content_list = result.get("content_list", [])

        if not content_list:
            raise ValueError("解析结果为空")

        # 3. 切分为 KnowledgeUnit
        from coursepilot.ingestion.parser_utils import extract_knowledge_units
        units = extract_knowledge_units(
            content_list,
            document_id=str(doc.id),
            kp_id="",   # 暂时为空，下面由 KPSplitter 分配
        )

        # 4. 用知识点树分配 kp_id
        kp_result = await session.execute(
            select(KnowledgePoint)
            .where(KnowledgePoint.course_id == doc.course_id)
            .order_by(KnowledgePoint.sort_order)
        )
        kp_nodes = [
            {
                "id": str(kp.id), "title": kp.title,
                "kp_path": kp.kp_path, "level": _kp_level(kp.kp_path),
            }
            for kp in kp_result.scalars()
        ]

        if kp_nodes:
            from coursepilot.knowledge.kp_splitter import KPSplitter
            splitter = KPSplitter(kp_nodes, str(doc.course_id))
            units = splitter.assign(units)

        # 5. 批量插入 knowledge_units
        for u in units:
            ku = KnowledgeUnit(
                kp_id=UUID(u["kp_id"]) if u.get("kp_id") else None,
                document_id=doc.id,
                content=u["content"],
                summary=u.get("summary"),
                seq_order=u.get("seq_order", 0),
                page_ref=u.get("page_ref", ""),
                meta_data=u.get("meta_data", {}),
            )
            session.add(ku)

        doc.status = "ready"
        doc.page_count = len(units)
        await session.flush()
        logger.info(f"Document {doc.filename} ingestion complete: {len(units)} units")

    except Exception as exc:
        doc.status = "failed"
        doc.error_message = str(exc)
        await session.flush()
        logger.error(f"Document {doc.filename} ingestion failed: {exc}")
        raise


def _kp_level(kp_path: str) -> int:
    """从 kp_path 推断层级深度，如 'OS/process/scheduling' → 3"""
    return len(kp_path.split("/"))
