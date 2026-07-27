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
import re

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

# ── 对比类问题匹配 ──────────────────────────────────────────
# 匹配 "A和B有什么区别"、"A与B的联系"、"A与B相比较" 等模式
# 概念名提取后用 _trim_concept 去除尾部噪声
_COMPARISON_PAT = re.compile(
    r"(.{2,30}?)(?:与|和|跟|同)(.{2,30}?)(?:(?:有什么)?(?:区别|联系|不同|相同|差异|比较|对比))",
    re.UNICODE,
)

# 概念名尾部噪声清除
# 只去掉尾部"在…中/上/下/时""对于…""关于…"等结构噪声
# 注意: 不要用"的"做切分点，因为"的"可能是概念内部结构（如"函数的单调性"）
_CONCEPT_TRIM_PAT = re.compile(
    r"(?:对于|关于|从).*$|在[^，。；]*$",
    re.UNICODE,
)


def _trim_concept(s: str) -> str:
    """去除概念名尾部附着的结构噪声。

    例:
      "函数的单调性在区间I上" → "函数的单调性"（保留"的"）
      "Fermat定理" → "Fermat定理"
      "对于函数f(x)" → "对于函数f(x)"（正则匹配的是"关于…"）
    """
    m = _CONCEPT_TRIM_PAT.search(s)
    if m:
        return s[:m.start()].strip()
    return s.strip()


def _extract_comparison_concepts(query: str) -> list[str] | None:
    """提取对比类问题中的两个概念名。

    例如 "Fermat定理和Lagrange中值定理有什么区别" → ["Fermat定理", "Lagrange中值定理"]
    返回 None 表示不是对比类问题。
    """
    m = _COMPARISON_PAT.search(query)
    if m:
        a, b = _trim_concept(m.group(1)), _trim_concept(m.group(2))
        # 过滤明显不是概念对的匹配（如"题"和"区别"这种）
        if len(a) >= 2 and len(b) >= 2:
            return [a, b]
    return None


def _ensure_both_concepts(
    units: list[dict],
    candidates: list[dict],
    concepts: list[str],
    max_units: int | None = None,
) -> list[dict]:
    """确保双概念对比问题的两个概念至少各有一个 unit 在 top 结果中。

    如果 rerank 后某个概念的 unit 缺失，从 candidates 中补回。
    units: rerank 后的 top units
    candidates: rerank 前的候选池（RRF 排序）
    concepts: [概念A, 概念B]
    max_units: 最终最大数量（默认=rerank_top_k），最多允许超出 2 个
    """
    if not concepts or len(concepts) != 2:
        return units

    from coursepilot.rag.config import config as _cfg
    max_units = max_units or _cfg.rerank_top_k
    max_allowed = max_units + 2  # 最多多留 2 个补充 unit

    def _matches_concept(text: str, concept: str) -> bool:
        """判断内容是否匹配概念（支持部分匹配，避免过度严格要求全等）。"""
        if not text or not concept:
            return False
        # 全等、包含、关键词命中
        return (concept in text or text in concept
                or any(kw in text for kw in concept.split()
                       if len(kw) >= 2))

    # 检查每个概念在 top units 中是否有代表
    unit_contents = [(u.get("content", "") + u.get("summary", "") + u.get("kp_path", ""))
                     for u in units]
    missing_idx = []
    for i, concept in enumerate(concepts):
        has_rep = any(_matches_concept(content, concept) for content in unit_contents)
        if not has_rep:
            missing_idx.append(i)

    if not missing_idx:
        return units  # 两个概念都已覆盖

    # 从 candidates 中为缺失的概念补充最佳 unit（rerank 后）
    # 用候选池中 rank 最高的、匹配该概念的 unit
    added: list[dict] = []
    existing_uuids = {u.get("uuid", "") for u in units}
    for idx in missing_idx:
        for c in candidates:
            uid = c.get("uuid", "")
            if uid and uid not in existing_uuids:
                content = c.get("content", "") + c.get("summary", "") + c.get("kp_path", "")
                if _matches_concept(content, concepts[idx]):
                    c = dict(c)
                    c["rerank_score"] = 0.0  # 保底分
                    added.append(c)
                    existing_uuids.add(uid)
                    break

    result = units + added
    return result[:max_allowed]



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

    async def _dual_concept_search(
        self,
        session: AsyncSession,
        rewritten_query: str,
        course_id: str,
        concepts: list[str],
    ) -> list[dict]:
        """对比类问题：对两个概念分别检索后合并候选集。

        分别编码+检索概念 A 和概念 B，合并去重后保留 top_k 候选。
        """
        import time as _time

        all_candidates: dict[str, float] = {}  # uuid -> max_score
        for concept in concepts:
            t0 = _time.monotonic()
            concept_query = f"{rewritten_query} {concept}"
            print(f"[dual_concept] 检索词: '{concept_query[:80]}'")
            vecs = self.encoder.encode_query(concept_query)
            milvus = self.vector_store.hybrid_search(
                vecs["dense"], vecs["sparse"], course_id, top_k=20,
            )
            bm25: list[dict] = []
            if config.enable_bm25:
                bm25 = await self.bm25_indexer.search(
                    session, concept_query, course_id, top_k=config.bm25_top_k,
                )
            merged = rrf_fuse(
                [milvus, bm25] if bm25 else [milvus],
                k=config.rrf_k,
                top_k=20,
                weights=[config.dense_weight, 1.0 - config.dense_weight],
            )
            for c in merged:
                uid = c.get("uuid", "")
                if uid:
                    score = c.get("rrf_score", c.get("score", 0))
                    if uid not in all_candidates or score > all_candidates[uid]:
                        all_candidates[uid] = score
            print(f"[dual_concept] 概念'{concept}'检索完成, "
                  f"候选={len(merged)}, 耗时={(_time.monotonic()-t0)*1000:.0f}ms")

        # 合并排序，取 top N
        sorted_uuids = sorted(all_candidates, key=lambda u: all_candidates[u], reverse=True)
        # 回查原始数据（需要从结果中获取原始 dict）
        # 简单做法：用最后一个 merged 中的完整数据
        merged_pool: dict[str, dict] = {}
        for c in concepts:
            concept_query = f"{rewritten_query} {c}"
            vecs = self.encoder.encode_query(concept_query)
            milvus = self.vector_store.hybrid_search(
                vecs["dense"], vecs["sparse"], course_id, top_k=20,
            )
            for item in milvus:
                uid = item.get("uuid", "")
                if uid:
                    merged_pool[uid] = item
            if config.enable_bm25:
                bm25 = await self.bm25_indexer.search(
                    session, concept_query, course_id, top_k=config.bm25_top_k,
                )
                for item in bm25:
                    uid = item.get("uuid", "")
                    if uid:
                        merged_pool[uid] = item

        result = []
        for uid in sorted_uuids[:20]:
            if uid in merged_pool:
                item = dict(merged_pool[uid])
                item["score"] = all_candidates[uid]
                result.append(item)

        print(f"[dual_concept] 双概念检索合并完成, 总候选={len(result)}")
        return result

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

        # 对比类问题强制双概念检索
        concepts = _extract_comparison_concepts(query)
        if concepts:
            print(f"[retriever] 检测到对比类问题，概念A={concepts[0]}, 概念B={concepts[1]}")
            candidates = await self._dual_concept_search(session, rewritten, course_id, concepts)
            # 编码改写后的查询，供阶段4 KP 扩展精排使用
            t0 = _time.monotonic()
            vecs = self.encoder.encode_query(rewritten)
            print(f"[retriever] 阶段1-编码(对比类) 耗时={(_time.monotonic()-t0)*1000:.0f}ms")
        else:
            # 阶段1：BGE-M3 编码
            t0 = _time.monotonic()
            print("[retriever] 阶段1-编码 开始...")
            search_query = rewritten
            vecs = self.encoder.encode_query(search_query)
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
                    session, search_query, course_id, top_k=config.bm25_top_k,
                )
                print(f"[retriever] 阶段2b-BM25 耗时={(_time.monotonic()-t1)*1000:.0f}ms, BM25候选数={len(bm25_candidates)}")

            # 阶段2c：RRF 融合 Milvus + BM25
            rrf_top_k = 30
            if config.enable_bm25 and bm25_candidates:
                candidates = rrf_fuse(
                    [milvus_candidates, bm25_candidates],
                    k=config.rrf_k,
                    top_k=rrf_top_k,
                    weights=[config.dense_weight, 1.0 - config.dense_weight],
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

        # 对比类问题：rerank 后确保两个概念至少各有一个 unit
        if concepts:
            top_units = _ensure_both_concepts(top_units, candidates, concepts)

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

    # 第一遍：must_include 无条件保留（不占 max_chars 预算）
    for u in must_include:
        ref_id += 1
        source_header = (
            f'<source id="{ref_id}" path="{u.get("kp_path", "")}" '
            f'pages="{u["page_ref"]}" book="{u["filename"]}">\n'
        )
        summary_line = f"{u['summary']}\n" if u["summary"] else ""
        body = f"{summary_line}{u['content']}\n"
        footer = "</source>\n"
        block = f"{source_header}{body}{footer}"
        parts.append(block)
        context_uuids.append(u["uuid"])
        total_chars += len(block)

    # 第二遍：逐 KP 路径补全其他 unit（受 max_chars 限制）
    for kp_path in kp_paths:
        units = grouped[kp_path]
        # 排除已处理的 must_include
        remaining = [u for u in units if u["uuid"] not in context_uuids]
        if not remaining:
            continue
        section_header = f"## {kp_path}\n"
        parts.append(section_header)
        total_chars += len(section_header)
        for u in remaining:
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
