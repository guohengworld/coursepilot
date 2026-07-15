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

from coursepilot.models import Document, KnowledgePoint, KnowledgeUnit
from coursepilot.rag.bm25 import BM25Indexer, rrf_fuse
from coursepilot.rag.config import config
from coursepilot.rag.encoder import Encoder
from coursepilot.rag.query_rewriter import QueryRewriter
from coursepilot.rag.reranker import Reranker
from coursepilot.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)




class Retriever:
    """
    六阶段检索编排：

    阶段0：查询改写（DeepSeek）
    阶段1：BGE-M3 编码（dense + sparse）
    阶段2a：Milvus 混合检索 + RRF
    阶段2b：BM25 检索
    阶段2c：RRF 融合（Milvus + BM25）
    阶段3：bge-reranker 重排序 + 层级惩罚
    阶段4：KP 文档金字塔扩展（拉取同 KP 全部 unit）
    """

    def __init__(
        self,
        rewriter: QueryRewriter | None = None,
        encoder: Encoder | None = None,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
        bm25_indexer: BM25Indexer | None = None,
    ):
        self.rewriter = rewriter or QueryRewriter()
        self.encoder = encoder or Encoder()
        self.vector_store = vector_store or VectorStore()
        self.reranker = reranker or Reranker()
        self.bm25_indexer = bm25_indexer or BM25Indexer()

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

        # 阶段2a：Milvus 混合检索 + RRF
        t0 = _time.monotonic()
        print("[retriever] 阶段2-检索 开始...")
        milvus_candidates = self.vector_store.hybrid_search(
            vecs["dense"], vecs["sparse"], course_id, top_k=20
        )
        print(f"[retriever] 阶段2-检索 耗时={(_time.monotonic()-t0)*1000:.0f}ms, Milvus候选数={len(milvus_candidates)}")

        # 阶段2b：BM25 检索
        bm25_candidates: list[dict] = []
        if config.enable_bm25:
            t1 = _time.monotonic()
            bm25_candidates = await self.bm25_indexer.search(
                session, rewritten, course_id, top_k=config.bm25_top_k,
            )
            print(f"[retriever] 阶段2b-BM25 耗时={(_time.monotonic()-t1)*1000:.0f}ms, BM25候选数={len(bm25_candidates)}")

        # 阶段2c：RRF 融合 Milvus + BM25
        if config.enable_bm25 and bm25_candidates:
            candidates = rrf_fuse(
                [milvus_candidates, bm25_candidates],
                k=config.rrf_k,
                top_k=20,
                weights=list(config.rrf_weights),
            )
            print(f"[retriever] 阶段2c-RRF融合 候选数={len(candidates)}")
        else:
            candidates = milvus_candidates

        # 阶段3：重排序 → top-k
        t0 = _time.monotonic()
        if config.enable_rerank:
            print("[retriever] 阶段3-重排序 开始...")
            top_units = self.reranker.rerank(rewritten, candidates, top_k=config.rerank_top_k)
            print(f"[retriever] 阶段3-重排序 耗时={(_time.monotonic()-t0)*1000:.0f}ms")
        else:
            # 不用重排序时，直接用 RRF score 排序取 top-k
            candidates.sort(key=lambda x : x.get("score", 0), reverse=True)
            for c in candidates:
                c["rerank_score"] = c.get("score", 0)
            top_units = candidates[:config.rerank_top_k]
            print(f"[retriever] 阶段3-跳过重排序, top_k={config.rerank_top_k}")

        # 阶段4：KP 扩展（滑动窗口取 unit + query-unit 重排序）
        t0 = _time.monotonic()
        if config.enable_kp_expand:
            print("[retriever] 阶段4-KP扩展 开始...")
            context, context_uuids = await _kp_expand(
                session, top_units, config.context_max_chars,
                query=rewritten, reranker=self.reranker,
                encoder=self.encoder, query_dense=vecs["dense"],
            )
            print(f"[retriever] 阶段4-KP扩展 耗时={(_time.monotonic()-t0)*1000:.0f}ms, context_chars={len(context)}")
        else:
            context, context_uuids = _format_units(top_units, config.context_max_chars)
            print(f"[retriever] 阶段4-格式化 耗时={(_time.monotonic()-t0)*1000:.0f}ms, context_chars={len(context)}")

        print(f"[retriever] 总耗时={(_time.monotonic()-t_total)*1000:.0f}ms")

        metadata = {
            "query_raw": query,
            "query_rewritten": rewritten,
            "top_rerank_scores": [u.get("rerank_score", 0) for u in top_units],
            "source_kp_paths": [u.get("kp_path", "") for u in top_units],
            "top_uuids": [u.get("uuid", "") for u in top_units],
            "top_kp_ids": [u.get("kp_id", "") for u in top_units],
            "context_uuids": context_uuids,
            "candidate_count": len(candidates),
        }

        return context, metadata


# == KP 文档金字塔扩展

def _cosine(a: list[float], b: list[float]) -> float:
    """两个稠密向量的余弦相似度"""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _dense_rank(
    query_dense: list[float],
    units: list[dict],
    encoder,
    top_k: int = 30,
) -> list[dict]:
    """BGE-M3 dense embedding 语义粗排

    全部 unit 批量编码后与 query 做余弦相似度排序，取 top_k。
    200 个 unit 编码 ~1-2s（CPU），语义匹配远优于关键词重叠。
    """
    if not units or encoder is None:
        return units[:top_k]

    import time as _time
    t0 = _time.monotonic()
    contents = [u["content"] for u in units]
    vecs = encoder.encode(contents)

    scored = []
    for i, u in enumerate(units):
        sim = _cosine(query_dense, vecs[i]["dense"])
        scored.append((u, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    print(f"[dense_rank] {len(units)} unit 编码+排序, 耗时={(_time.monotonic()-t0)*1000:.0f}ms")
    return [u for u, _ in scored[:top_k]]


def _fast_rank(query: str, units: list[dict], top_k: int = 30) -> list[dict]:
    """关键词 Jaccard 粗排（encoder 不可用时的 fallback）"""
    import re as _re
    _tokenize = lambda s: set(_re.findall(r"[一-鿿]+|[a-zA-Z0-9]+", s.lower()))
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
            scored.append((u, len(overlap) / len(q_tokens | u_tokens)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [u for u, _ in scored[:top_k]]


async def _kp_expand(
    session: AsyncSession,
    top_units: list[dict],
    max_chars: int = 8000,
    *,
    query: str = "",
    reranker=None,
    encoder=None,
    query_dense: list[float] | None = None,
) -> tuple[str, list[str]]:
    """KP 扩展 + 语义粗排 + cross-encoder 精排

    支持两种模式（由 config.kp_expand_mode 控制）：
    - "full"（默认）: 拉取 top KP 下的全部 unit
    - "neighbor": 只取命中 unit 前后各 N 个相邻 unit（N = config.kp_neighbor_window）
    返回 (context_string, context_uuids)
    """
    from uuid import UUID

    kp_ids = [u["kp_id"] for u in top_units if u.get("kp_id")]
    if not kp_ids:
        return "", []

    kp_order = {kp_id: i for i, kp_id in enumerate(kp_ids)}

    # ── 根据扩展模式构建查询 ──────────────────────────────────
    base_select = (
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
    )

    if config.kp_expand_mode == "neighbor":
        from sqlalchemy import or_

        WINDOW = config.kp_neighbor_window
        # 先查 top unit 的 seq_order
        top_uuids = [u["uuid"] for u in top_units if u.get("uuid")]
        pos_result = await session.execute(
            select(KnowledgeUnit.id, KnowledgeUnit.kp_id, KnowledgeUnit.seq_order)
            .where(KnowledgeUnit.id.in_([UUID(uid) for uid in top_uuids]))
        )
        pos_map: dict[str, tuple[str, int]] = {}
        for row in pos_result.all():
            pos_map[str(row[0])] = (str(row[1]), row[2])  # (kp_id, seq_order)

        conditions = []
        for u in top_units:
            uid = u.get("uuid", "")
            if uid in pos_map:
                kp_id_val, seq = pos_map[uid]
                conditions.append(
                    (KnowledgeUnit.kp_id == UUID(kp_id_val))
                    & (KnowledgeUnit.seq_order.between(seq - WINDOW, seq + WINDOW))
                )

        if conditions:
            stmt = base_select.where(or_(*conditions)).order_by(
                KnowledgeUnit.kp_id, KnowledgeUnit.seq_order
            )
        else:
            return "", []
    else:
        # Full mode：拉取 KP 下全部 unit
        stmt = base_select.where(
            KnowledgeUnit.kp_id.in_([UUID(k) for k in kp_ids])
        ).order_by(KnowledgeUnit.kp_id, KnowledgeUnit.seq_order)

    result = await session.execute(stmt)
    rows = result.all()

    if not rows:
        return "", []

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

    # Pin top units: 阶段3重排序已验证的 unit 必须保留在最终上下文中，
    # 避免 KP 扩展内的二次精排把它们误伤或截断。
    top_uuids = {u["uuid"] for u in top_units if u.get("uuid")}
    must_include = [u for u in all_units if u["uuid"] in top_uuids]
    must_order = {u["uuid"]: i for i, u in enumerate(top_units) if u.get("uuid")}
    must_include.sort(key=lambda u: must_order.get(u["uuid"], 999))

    # 其余 unit 再做精排过滤噪声
    other_units = [u for u in all_units if u["uuid"] not in top_uuids]

    # 两阶段过滤：语义粗排(BGE-M3 dense) → 精排(cross-encoder)
    N_COARSE = 30  # 粗排保留数，≤30 直接精排
    reranker_ok = reranker is not None and reranker.available
    if query and reranker_ok and len(other_units) > N_COARSE:
        import time as _time
        print(f"[kp_expand] 两阶段过滤: 非top_unit={len(other_units)} → 粗排top-{N_COARSE} → 精排")
        if encoder is not None and query_dense is not None:
            coarse = _dense_rank(query_dense, other_units, encoder, top_k=N_COARSE)
        else:
            t0 = _time.monotonic()
            coarse = _fast_rank(query, other_units, top_k=N_COARSE)
            print(f"[kp_expand] 关键词粗排完成, 耗时={(_time.monotonic()-t0)*1000:.0f}ms")
        try:
            t1 = _time.monotonic()
            other_units = reranker.rerank(query, coarse, top_k=len(coarse))
            print(f"[kp_expand] 精排完成, 耗时={(_time.monotonic()-t1)*1000:.0f}ms")
        except Exception:
            other_units = coarse
    elif query and reranker_ok and len(other_units) > 10:
        print(f"[kp_expand] 全量精排(非top_unit), unit数={len(other_units)}")
        try:
            other_units = reranker.rerank(
                query, other_units, top_k=len(other_units)
            )
        except Exception:
            pass

    # 组装上下文：top_units 优先，再按 KP 分组补全
    ordered_units = must_include + other_units
    grouped: dict[str, list[dict]] = {}
    for u in ordered_units:
        grouped.setdefault(u["kp_path"], []).append(u)

    path_rank: dict[str, int] = {}
    for kp_id, rank in kp_order.items():
        for u in all_units:
            if u["kp_id"] == kp_id:
                path_rank[u["kp_path"]] = rank
                break
    kp_paths = sorted(grouped.keys(), key=lambda p: path_rank.get(p, 99))

    parts: list[str] = []
    context_uuids: list[str] = []
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
            context_uuids.append(u["uuid"])
            total_chars += len(block)

    return "\n".join(parts), context_uuids


def _format_units(
    top_units: list[dict],
    max_chars: int = 8000,
) -> tuple[str, list[str]]:
    """不使用 KP 扩展时，直接用检索到的 unit 原文组装上下文。
    返回 (context_string, context_uuids)
    """
    parts: list[str] = []
    total_chars = 0
    context_uuids: list[str] = []

    for i, u in enumerate(top_units):
        header = (
            f'<source id="{i + 1}" path="{u.get("kp_path", "")}" '
            f'pages="" book="">\n'
        )
        footer = '\n</source>\n'
        block_overhead = len(header) + len(footer)
        remaining = max_chars - total_chars - block_overhead
        if remaining <= 0:
            break
        context_uuids.append(u.get("uuid", ""))
        content = u["content"][:remaining]
        block = header + content + footer
        parts.append(block)
        total_chars += len(block)

    return "\n".join(parts), context_uuids
