"""
Per-course BM25 索引 + RRF 融合

用法：
    indexer = BM25Indexer()
    results = await indexer.search(session, query, course_id, top_k=20)
    indexer.invalidate(course_id)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from uuid import UUID

import jieba
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.rag.config import config as rag_config

logger = logging.getLogger(__name__)

# ── 模块级缓存（所有 BM25Indexer 实例共享） ──────────────────────
_caches: dict[str, _CacheEntry] = {}
_lock = asyncio.Lock()


@dataclass
class _CacheEntry:
    index: BM25Okapi
    metas: list[dict]       # 与 corpus 同顺序的元数据
    built_at: float         # time.monotonic()


def rrf_fuse(
    ranked_lists: list[list[dict]],
    k: int = 60,
    top_k: int = 20,
    weights: list[float] | None = None,
) -> list[dict]:
    """RRF 融合多条排序列表为一条 top-k。

    RRF(d) = Σ w_l / (k + rank(d, l))  for each list l that contains d

    :param ranked_lists: 多个按 score 降序排列的结果列表，
                         每个 item 必须有 "uuid" 字段。
    :param k: RRF 参数（默认 60，与 Milvus RRF 一致）
    :param top_k: 最终返回条数
    :param weights: 每个 ranked_list 的权重，长度不足时补 1.0
    :return: 按 RRF score 降序排列的列表，每项包含 score=RRF 融合分
    """
    # 过滤空列表
    ranked_lists = [lst for lst in ranked_lists if lst]
    if not ranked_lists:
        return []
    if len(ranked_lists) == 1:
        return ranked_lists[0][:top_k]

    # 权重对齐：缺省补 1.0
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    elif len(weights) < len(ranked_lists):
        weights = list(weights) + [1.0] * (len(ranked_lists) - len(weights))

    fused: dict[str, dict] = {}
    for lst_idx, lst in enumerate(ranked_lists):
        w = weights[lst_idx]
        for rank, item in enumerate(lst):
            uid = item.get("uuid")
            if not uid:
                continue
            if uid not in fused:
                fused[uid] = dict(item)
                fused[uid]["score"] = 0.0
            fused[uid]["score"] += w / (k + rank + 1)

    return sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_k]


class BM25Indexer:
    """Per-course 内存 BM25 索引，带 TTL 缓存。

    索引内容 = KnowledgeUnit.summary + KnowledgeUnit.content（jieba 分词）。
    缓存粒度：一个 course_id 一个 BM25Okapi 实例。
    所有实例共享模块级 _caches，保证 pipeline 失效与 Retriever 搜索协同。
    """

    def __init__(self, ttl: int | None = None):
        self._ttl = ttl if ttl is not None else rag_config.bm25_cache_ttl

    async def search(
        self,
        session: AsyncSession,
        query: str,
        course_id: str,
        top_k: int = 20,
    ) -> list[dict]:
        """BM25 检索。

        :return: [{uuid, kp_id, kp_path, content, summary, score}, ...]
                 按 BM25 score 降序，score=0 的 item 已过滤。
        """
        await self._ensure_index(session, course_id)
        entry = _caches.get(course_id)
        if not entry or not entry.metas or entry.index is None:
            return []

        tokenized_query = list(jieba.cut(query))
        scores = entry.index.get_scores(tokenized_query)

        results = [
            {**entry.metas[i], "score": float(score)}
            for i, score in enumerate(scores)
            if score != 0
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def _ensure_index(self, session: AsyncSession, course_id: str) -> None:
        """缓存缺失或 TTL 过期时重建索引（带 double-check lock）。"""
        entry = _caches.get(course_id)
        if entry and (time.monotonic() - entry.built_at) < self._ttl:
            return

        async with _lock:
            entry = _caches.get(course_id)
            if entry and (time.monotonic() - entry.built_at) < self._ttl:
                return
            await self._build(session, course_id)

    async def _build(self, session: AsyncSession, course_id: str) -> None:
        """从 PostgreSQL 拉取全部 KU 并构建 BM25Okapi。"""
        from coursepilot.models import KnowledgePoint, KnowledgeUnit

        stmt = (
            select(
                KnowledgeUnit.id,
                KnowledgeUnit.content,
                KnowledgeUnit.summary,
                KnowledgeUnit.kp_id,
                KnowledgePoint.kp_path,
            )
            .join(KnowledgePoint, KnowledgeUnit.kp_id == KnowledgePoint.id)
            .where(KnowledgePoint.course_id == UUID(course_id))
            .order_by(KnowledgeUnit.kp_id, KnowledgeUnit.seq_order)
        )
        result = await session.execute(stmt)
        rows = result.all()

        if not rows:
            _caches[course_id] = _CacheEntry(
                index=None,  # type: ignore[arg-type]
                metas=[],
                built_at=time.monotonic(),
            )
            logger.info("BM25 索引构建完成（空）: course=%s", course_id)
            return

        texts: list[str] = []
        metas: list[dict] = []
        for row in rows:
            # row[0]=KU.id, row[1]=KU.content, row[2]=KU.summary,
            # row[3]=KU.kp_id, row[4]=KP.kp_path
            content = row[1] or ""
            summary = row[2] or ""
            texts.append(f"{summary}\n{content}")
            metas.append({
                "uuid": str(row[0]),
                "content": content,
                "summary": summary,
                "kp_id": str(row[3]),
                "kp_path": row[4] or "",
            })

        tokenized_corpus = [list(jieba.cut(t)) for t in texts]
        index = BM25Okapi(tokenized_corpus)

        _caches[course_id] = _CacheEntry(
            index=index,
            metas=metas,
            built_at=time.monotonic(),
        )
        logger.info("BM25 索引构建完成: course=%s, KUs=%d", course_id, len(rows))

    @staticmethod
    def invalidate(course_id: str) -> None:
        """清除某课程的 BM25 缓存（在 ingestion 成功后调用）。"""
        removed = _caches.pop(course_id, None)
        if removed is not None:
            logger.info("BM25 缓存已清除: course=%s", course_id)
