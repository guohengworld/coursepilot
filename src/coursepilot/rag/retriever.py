"""
五阶段检索编排器 + KP 文档金字塔扩展

用法：
    retriever = Retriever()
    context, metadata = await retriever.retrieve(
        session, query="什么是定积分的集合意义", course_id="..."
    )
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.models import KnowledgeUnit, KnowledgePoint, Document
from coursepilot.rag.config import config
from coursepilot.rag.encoder import Encoder
from coursepilot.rag.query_rewriter import QueryRewriter
from coursepilot.rag.reranker import Reranker
from coursepilot.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)




class Retriever:
    """
    五阶段检索编排：

    阶段0：查询改写（DeepSeek）
    阶段1：BGE-M3 编码（dense + sparse）
    阶段2：Milvus 混合检索 + RRF
    阶段3：bge-reranker 重排序 + 层级惩罚
    阶段4：KP 文档金字塔扩展（拉取同 KP 全部 unit）
    """

    def __init__(
        self,
        rewriter: QueryRewriter | None = None,
        encoder: Encoder | None = None,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
    ):
        self.rewriter = rewriter or QueryRewriter()
        self.encoder = encoder or Encoder()
        self.vector_store = vector_store or VectorStore()
        self.reranker = reranker or Reranker()

    async def retrieve(
        self,
        session: AsyncSession,
        query: str,
        course_id: str,
        *,
        enable_rewrite: bool = True,
    ) -> tuple[str, dict]:
        """
        执行完整检索管线

        :return (formatted_context, metadata)
          - formatted_context: XML 格式的教材内容，直接送入 LLM
          - metadata: {query_raw, query_rewritten, top_rerank_scores, source_kp_paths, candidate_count}
        """
        # 阶段0：查询改写
        rewritten = query
        if enable_rewrite and config.enable_rewrite:
            rewritten = await self.rewriter.rewrite(query)

        # 阶段1：BGE-M3 编码
        vecs = self.encoder.encode_query(rewritten)

        # 阶段2：Milvus 混合检索 + RRF → top-20 候选
        candidates = self.vector_store.hybrid_search(
            vecs["dense"], vecs["sparse"], course_id, top_k=20
        )

        # 阶段3：重排序 → top-5
        if config.enable_rerank:
            top_units = self.reranker.rerank(rewritten, candidates, top_k=5)
        else:
            # 不用重排序时，直接用 RRF score 排序取 top-5
            candidates.sort(key=lambda x : x.get("score", 0), reverse=True)
            for c in candidates:
                c["rerank_score"] = c.get("score", 0)
            top_units = candidates[:5]

        # 阶段4：KP 扩展（从 PG(PostgreSQL) 拉取同 KP 全部 unit，组装为 LLM 上下文）
        if config.enable_kp_expand:
            context = await _kp_expand(session, top_units, config.context_max_chars)
        else:
            context = _format_units(top_units, config.context_max_chars)

        metadata = {
            "query_raw": query,
            "query_rewritten": rewritten,
            "top_rerank_scores": [u.get("rerank_score", 0) for u in top_units],
            "source_kp_paths": [u.get("kp_path", "") for u in top_units],
            "candidate_count": len(candidates),
        }

        return context, metadata


# == KP 文档金字塔扩展

async def _kp_expand(
    session: AsyncSession,
    top_units: list[dict],
    max_chars: int = 8000
) -> str:
    """拉取 top-5 unit 所在的 KP 的全部unit，组装为结构化上下文"""
    kp_ids = list({u["kp_id"] for u in top_units})

    if not kp_ids:
        return ""

    # 从 PG 拉取这些 KP 下的所有unit（按 seq_order 排序）
    from uuid import UUID

    stmt = (
        select(
            KnowledgeUnit.content,
            KnowledgeUnit.summary,
            KnowledgeUnit.page_ref,
            KnowledgeUnit.seq_order,
            KnowledgePoint.kp_path,
            Document.filename,
        )
        .join(KnowledgePoint, KnowledgeUnit.kp_id == KnowledgePoint.id)
        .outerjoin(Document, KnowledgeUnit.document_id == Document.id)
        .where(KnowledgeUnit.kp_id.in_([UUID(k) for k in kp_ids]))
        .order_by(KnowledgeUnit.kp_id, KnowledgeUnit.seq_order)
    )
    result = await session.execute(stmt)
    rows = result.all()

    # 按 kp_id 分组，reranker 得分高的 KP 排前面
    kp_order = {kp_id: i for i, kp_id in enumerate(kp_ids)}
    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(str(row.kp_path), []).append(row)

    # 按 kp_order 排序列出所有的 kp_path
    kp_paths = sorted(grouped.keys(), key=lambda p: kp_order.get(p, 99))

    # 组装 context，超上限截断
    parts: list[str] = []
    total_chars = 0
    ref_id = 0

    for kp_path in kp_paths:
        units = grouped[kp_path]
        if not units:
            continue

        parts.append(f"## {kp_path}\n")
        total_chars += len(parts[-1])

        for unit in units:
            ref_id += 1
            if total_chars > max_chars:
                break

            page_ref = unit.page_ref or ""
            book = unit.filename or "未知教材"

            source_header = (
                f'<source id="{ref_id}" path="{kp_path}" '
                f'pages="{page_ref}" book="{book}">\n'
            )

            summary_line = f"{unit.summary}\n" if unit.summary else ""
            body = f"{summary_line}{unit.content}\n"
            footer = "</source>\n"

            block = f"{source_header}{body}{footer}"
            parts.append(block)
            total_chars += len(block)

    return "\n".join(parts)


def _format_units(
    top_units: list[dict],
    max_chars: int = 8000,
) -> str:
    """不使用 KP 扩展时，直接用检索到的 unit 原文组装上下文。"""
    parts: list[str] = []
    total_chars = 0

    for i, u in enumerate(top_units):
        if total_chars > max_chars:
            break
        block = (
            f'<source id="{i + 1}" path="{u.get("kp_path", "")}" '
            f'pages="" book="">\n{u["content"]}\n</source>\n'
        )
        parts.append(block)
        total_chars += len(block)

    return "\n".join(parts)
