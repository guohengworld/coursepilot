"""Ingestion 管线：解析文件 → 自动构建 KP 树 → 切分 → KP 分配 → 摘要 → 入库

流程：
  Document(status=pending)
    → 解析文件（pdf_parser / docx_parser）
    → _ensure_kp_tree 从标题自动构建/合并知识点树（新增：无需手动预建 KP）
    → extract_knowledge_units 切分（垃圾过滤 + heading 追踪 + 数学块感知）
    → KPSplitter 分配到知识点
    → SummaryBridge 生成摘要
    → encode_units → Milvus
    → Document（status=ready）
"""

from __future__ import annotations

import logging
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.models import Course, Document, KnowledgePoint, KnowledgeUnit

logger = logging.getLogger(__name__)


async def _ensure_kp_tree(
    session: AsyncSession,
    course_id: UUID,
    content_list: list[dict],
    document_id: str | None = None,
) -> list[dict]:
    """从 content_list 提取标题，自动构建/合并知识点树。

    幂等：已存在的 kp_path 不重复创建，支持多文档逐本上传。
    当传入 document_id 时，KP 树按文档隔离——每份文档拥有独立的 KP 子树。
    返回: KP 节点列表 [{id, title, kp_path, level}, ...]，供 KPSplitter 使用。
    """
    from coursepilot.knowledge.syllabus_parser import extract_headings, headings_to_syllabus

    headings = extract_headings(content_list)
    if not headings:
        logger.info("B0: 未提取到标题（text_level ≤ 4），跳过 KP 树构建")
        return []

    course = await session.get(Course, course_id)
    course_name = course.name if course else "未命名课程"

    nodes = headings_to_syllabus(headings, course_name)
    logger.info(
        "B0: 从 %d 个标题构建 %d 个 KP 节点（课程: %s, document: %s）",
        len(headings),
        len(nodes),
        course_name,
        document_id,
    )

    # 按 (course_id, document_id) 确定已有 KPs —— per-document 隔离
    existing_query = select(KnowledgePoint.kp_path).where(
        KnowledgePoint.course_id == course_id,
    )
    if document_id is not None:
        doc_uuid = UUID(document_id)
        existing_query = existing_query.where(KnowledgePoint.document_id == doc_uuid)
    else:
        existing_query = existing_query.where(KnowledgePoint.document_id.is_(None))
    existing_result = await session.execute(existing_query)
    existing_paths = {row[0] for row in existing_result.fetchall()}

    new_nodes = [n for n in nodes if n["kp_path"] not in existing_paths]
    if not new_nodes:
        logger.info("B0: 所有 KP 节点已存在，跳过插入")
    else:
        title_to_id: dict[str, str] = {}
        for node in new_nodes:
            kp = KnowledgePoint(
                course_id=course_id,
                document_id=UUID(document_id) if document_id else None,
                kp_path=node["kp_path"],
                title=node["title"],
                summary=node.get("summary", ""),
                difficulty=node.get("difficulty", 1),
                sort_order=node.get("sort_order", 0),
                source=node.get("source", "textbook"),
            )
            session.add(kp)
            await session.flush()
            title_to_id[node["title"]] = str(kp.id)
            node["id"] = str(kp.id)

        # 回填 parent_id（限同一文档内的 KPs）
        for node in new_nodes:
            if node.get("parent_title") and node["parent_title"] in title_to_id:
                kp = await session.get(KnowledgePoint, UUID(title_to_id[node["title"]]))
                if kp:
                    kp.parent_id = UUID(title_to_id[node["parent_title"]])
        await session.flush()
        logger.info("B0: 新增 %d 个 KP 节点（已存在 %d 个）", len(new_nodes), len(existing_paths))

    # 返回当前文档范围下的完整 KP 列表
    all_kp_query = (
        select(KnowledgePoint)
        .where(
            KnowledgePoint.course_id == course_id,
        )
        .order_by(KnowledgePoint.sort_order)
    )
    if document_id is not None:
        all_kp_query = all_kp_query.where(KnowledgePoint.document_id == UUID(document_id))
    else:
        all_kp_query = all_kp_query.where(KnowledgePoint.document_id.is_(None))
    all_kp_result = await session.execute(all_kp_query)
    return [
        {
            "id": str(kp.id),
            "title": kp.title,
            "kp_path": kp.kp_path,
            "level": _kp_level(kp.kp_path),
        }
        for kp in all_kp_result.scalars()
    ]


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
    """
    # 1. 获取 Document 记录
    doc = await session.get(Document, UUID(document_id))
    if not doc:
        logger.error(f"Document {document_id} not found")
        return

    doc.status = "processing"
    await session.flush()

    try:
        # ── B0: 解析文件 → content_list ──────────────────
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

        # ── B1: 自动构建/合并知识点树（新增） ──────────
        kp_nodes = await _ensure_kp_tree(
            session,
            doc.course_id,
            content_list,
            document_id=str(doc.id),
        )

        # ── B2: 切分为 KnowledgeUnit（阶段 A 改造） ──────
        #   A1: _split_by_headings 追踪 heading 写入 meta_data
        #   A2: _filter_garbage 垃圾过滤
        #   A3: _split_text_v2 数学块感知 + 段落边界优先
        from coursepilot.ingestion.parser_utils import extract_knowledge_units

        logger.info("B2: 切分知识单元（%d 行 content_list）...", len(content_list))
        units = extract_knowledge_units(
            content_list,
            document_id=str(doc.id),
            kp_id="",  # 暂时为空，下面由 KPSplitter 分配
        )
        logger.info("B2: 切分完成 → %d 个知识单元", len(units))

        # ── B3: 用知识点树分配 kp_id ────────────────────
        logger.info("B3: KP 分配（%d 个 KP 节点）...", len(kp_nodes))
        if kp_nodes:
            from coursepilot.knowledge.kp_splitter import KPSplitter

            splitter = KPSplitter(kp_nodes, str(doc.course_id))
            units = splitter.assign(units)
        logger.info("B3: KP 分配完成")

        # ── B4: SummaryBridge 生成摘要（阶段 A 新增） ────
        from coursepilot.rag.summary_bridge import SummaryBridge

        logger.info("B6: 开始生成摘要（%d 个 unit）...", len(units))
        bridge = SummaryBridge()
        units = await bridge.run(units)
        logger.info("B6: 摘要生成完成")

        # 回填 kp_path 到 units（Milvus 入库需要）
        kp_map = {n["id"]: n["kp_path"] for n in kp_nodes}
        for u in units:
            u["kp_path"] = kp_map.get(u.get("kp_id", ""), "")

        # ── B5: encode + Milvus insert（阶段 B 实施） ────
        await _encode_units(units, str(doc.course_id))

        # ── B6: 批量插入 knowledge_units ─────────────────
        for u in units:
            ku = KnowledgeUnit(
                id=UUID(u["_unit_id"]) if u.get("_unit_id") else None,
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


async def _encode_units(units: list[dict], course_id: str) -> None:
    """B7: BGE-M3 编码 + Milvus insert"""
    if not units:
        return

    from coursepilot.rag.encoder import Encoder
    from coursepilot.rag.vector_store import VectorStore

    encoder = Encoder()
    store = VectorStore()
    store.create_collection()

    # 为每个 unit 预生成 UUID（同时用作 KnowledgeUnit.id 和 Milvus uuid）
    for u in units:
        u["_unit_id"] = str(uuid4())

    # 批量编码：summary + content 混合，兼顾语义概括与细节
    texts = [(u.get("summary") or "") + "\n" + u["content"] for u in units]
    vecs = encoder.encode(texts)

    # 构建 Milvus 插入数据
    payloads = []
    for u, vec in zip(units, vecs):
        payloads.append(
            {
                "uuid": u["_unit_id"],
                "dense_vec": vec["dense"],
                "sparse_vec": vec["sparse"],
                "kp_id": u.get("kp_id", ""),
                "course_id": course_id,
                "kp_path": u.get("kp_path", ""),
                "content": u["content"][:8192],
            }
        )

    store.insert(payloads)
    logger.info("B7 encode_units: %d units encoded + inserted to Milvus", len(units))


def _kp_level(kp_path: str) -> int:
    """从 kp_path 推断层级深度，如 'OS/process/scheduling' → 3"""
    return len(kp_path.split("/"))
