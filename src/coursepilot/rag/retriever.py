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
        import time as _time
        t_total = _time.monotonic()

        # 阶段0：查询改写
        rewritten = query
        if enable_rewrite and config.enable_rewrite:
            t0 = _time.monotonic()
            rewritten = await self.rewriter.rewrite(query)
            print(f"[retriever] 阶段0-改写 耗时={(_time.monotonic()-t0)*1000:.0f}ms")

        # 阶段1：BGE-M3 编码
        t0 = _time.monotonic()
        print("[retriever] 阶段1-编码 开始...")
        vecs = self.encoder.encode_query(rewritten)
        print(f"[retriever] 阶段1-编码 耗时={(_time.monotonic()-t0)*1000:.0f}ms")

        # 阶段2：Milvus 混合检索 + RRF → top-20 候选
        t0 = _time.monotonic()
        print("[retriever] 阶段2-检索 开始...")
        candidates = self.vector_store.hybrid_search(
            vecs["dense"], vecs["sparse"], course_id, top_k=20
        )
        print(f"[retriever] 阶段2-检索 耗时={(_time.monotonic()-t0)*1000:.0f}ms, 候选数={len(candidates)}")

        # 阶段3：重排序 → top-5
        t0 = _time.monotonic()
        if config.enable_rerank:
            print("[retriever] 阶段3-重排序 开始...")
            top_units = self.reranker.rerank(rewritten, candidates, top_k=5)
            print(f"[retriever] 阶段3-重排序 耗时={(_time.monotonic()-t0)*1000:.0f}ms")
        else:
            # 不用重排序时，直接用 RRF score 排序取 top-5
            candidates.sort(key=lambda x : x.get("score", 0), reverse=True)
            for c in candidates:
                c["rerank_score"] = c.get("score", 0)
            top_units = candidates[:5]
            print(f"[retriever] 阶段3-跳过重排序, top_k=5")

        # 阶段4：KP 扩展（滑动窗口取 unit + query-unit 重排序）
        t0 = _time.monotonic()
        if config.enable_kp_expand:
            print("[retriever] 阶段4-KP扩展 开始...")
            context = await _kp_expand(
                session, top_units, config.context_max_chars,
                query=rewritten, reranker=self.reranker,
            )
            print(f"[retriever] 阶段4-KP扩展 耗时={(_time.monotonic()-t0)*1000:.0f}ms, context_chars={len(context)}")
        else:
            context = _format_units(top_units, config.context_max_chars)
            print(f"[retriever] 阶段4-格式化 耗时={(_time.monotonic()-t0)*1000:.0f}ms, context_chars={len(context)}")

        print(f"[retriever] 总耗时={(_time.monotonic()-t_total)*1000:.0f}ms")

        metadata = {
            "query_raw": query,
            "query_rewritten": rewritten,
            "top_rerank_scores": [u.get("rerank_score", 0) for u in top_units],
            "source_kp_paths": [u.get("kp_path", "") for u in top_units],
            "top_uuids": [u.get("uuid", "") for u in top_units],
            "top_kp_ids": [u.get("kp_id", "") for u in top_units],
            "candidate_count": len(candidates),
        }

        return context, metadata


# == KP 文档金字塔扩展

def _fast_rank(query: str, units: list[dict], top_k: int = 30) -> list[dict]:
    """基于 token 重叠率（简化 Jaccard）快速粗排，用于削减重排序规模。

    query 与每个 unit 的 content+summary 做中/英文 token 提取，
    计算 Jaccard 系数 = |A ∩ B| / |A ∪ B|，纯 CPU 字符串操作。
    200 个 unit 耗时 < 1ms。
    """
    import re as _re

    _tokenize = lambda s: set(
        _re.findall(r"[一-鿿]+|[a-zA-Z0-9]+", s.lower())
    )
    q_tokens = _tokenize(query)
    if not q_tokens:
        return units[:top_k]

    scored: list[tuple[dict, float]] = []
    for u in units:
        text = f"{u.get('summary', '')} {u['content']}"
        u_tokens = _tokenize(text)
        if not u_tokens:
            scored.append((u, 0.0))
        else:
            overlap = q_tokens & u_tokens
            score = len(overlap) / len(q_tokens | u_tokens)
            scored.append((u, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [u for u, _ in scored[:top_k]]


async def _kp_expand(
    session: AsyncSession,
    top_units: list[dict],
    max_chars: int = 8000,
    *,
    query: str = "",
    reranker=None,
) -> str:
    """KP 全量扩展 + query-unit 重排序

    1. 拉取 top-5 KP 下的全部 unit
    2. query-unit 重排序（复用 bge-reranker），按相关性降序
    3. 按得分组装结构化上下文，超上限截断
    """
    from uuid import UUID

    kp_ids = list({u["kp_id"] for u in top_units})
    if not kp_ids:
        return ""

    kp_order = {kp_id: i for i, kp_id in enumerate(kp_ids)}

    # 一次性拉取全部 KP 下的 unit（join kp_path + filename）
    stmt = (
        select(
            KnowledgeUnit.id,
            KnowledgeUnit.content,
            KnowledgeUnit.summary,
            KnowledgeUnit.page_ref,
            KnowledgeUnit.kp_id,
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

    if not rows:
        return ""

    all_units: list[dict] = []
    for row in rows:
        all_units.append({
            "uuid": str(row[0]),
            "content": row[1] or "",
            "summary": row[2] or "",
            "page_ref": row[3] or "",
            "kp_id": str(row[4]),
            "kp_path": row[5] or "",
            "filename": row[6] or "未知教材",
        })

    # 两阶段过滤：粗排(关键词) → 精排(cross-encoder)
    N_COARSE = 30  # 粗排保留数，≤30 直接精排
    if query and reranker is not None and len(all_units) > N_COARSE:
        import time as _time
        print(f"[kp_expand] 两阶段过滤: 全量={len(all_units)} → 粗排top-{N_COARSE} → 精排")
        t0 = _time.monotonic()
        coarse = _fast_rank(query, all_units, top_k=N_COARSE)
        print(f"[kp_expand] 粗排完成, 耗时={(_time.monotonic()-t0)*1000:.0f}ms")
        try:
            t1 = _time.monotonic()
            all_units = reranker.rerank(query, coarse, top_k=len(coarse))
            print(f"[kp_expand] 精排完成, 耗时={(_time.monotonic()-t1)*1000:.0f}ms")
        except Exception:
            all_units = coarse
    elif query and reranker is not None and len(all_units) > 10:
        print(f"[kp_expand] 全量精排, unit数={len(all_units)}")
        try:
            all_units = reranker.rerank(
                query, all_units, top_k=len(all_units)
            )
        except Exception:
            pass

    # 组装上下文，重排序后高相关度 unit 优先
    grouped: dict[str, list[dict]] = {}
    for u in all_units:
        grouped.setdefault(u["kp_path"], []).append(u)

    path_rank: dict[str, int] = {}
    for kp_id, rank in kp_order.items():
        for u in all_units:
            if u["kp_id"] == kp_id:
                path_rank[u["kp_path"]] = rank
                break
    kp_paths = sorted(grouped.keys(), key=lambda p: path_rank.get(p, 99))

    parts: list[str] = []
    total_chars = 0
    ref_id = 0

    for kp_path in kp_paths:
        units = grouped[kp_path]
        parts.append(f"## {kp_path}\n")
        total_chars += len(parts[-1])

        for u in units:
            if total_chars > max_chars:
                break
            ref_id += 1
            source_header = (
                f'<source id="{ref_id}" path="{kp_path}" '
                f'pages="{u["page_ref"]}" book="{u["filename"]}">\n'
            )
            summary_line = f"{u['summary']}\n" if u["summary"] else ""
            body = f"{summary_line}{u['content']}\n"
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
        header = (
            f'<source id="{i + 1}" path="{u.get("kp_path", "")}" '
            f'pages="" book="">\n'
        )
        footer = '\n</source>\n'
        # 计算可用空间
        block_overhead = len(header) + len(footer)
        remaining = max_chars - total_chars - block_overhead
        if remaining <= 0:
            break
        content = u["content"][:remaining]
        block = header + content + footer
        parts.append(block)
        total_chars += len(block)

    return "\n".join(parts)
