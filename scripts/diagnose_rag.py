"""RAG 检索诊断脚本。

用法：
    PYTHONPATH=src .venv/Scripts/python -m scripts.diagnose_rag

输出：逐题展示每个阶段的候选是否包含 ground truth，用于定位召回失败根因。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.db import get_session_etx
from coursepilot.models import KnowledgeUnit
from coursepilot.rag.bm25 import BM25Indexer
from coursepilot.rag.config import config as rag_config
from coursepilot.rag.encoder import Encoder
from coursepilot.rag.query_rewriter import QueryRewriter
from coursepilot.rag.reranker import Reranker
from coursepilot.rag.retriever import Retriever
from coursepilot.rag.vector_store import VectorStore

DATASET = Path("eval/questions/eval_questions.json")


async def check_ground_truth_in_candidates(
    session: AsyncSession,
    question: dict,
    course_id: str,
    gt_uuids: set[str],
) -> dict:
    """对单题运行多种配置，检查 ground truth 在哪些阶段出现。"""
    query = question["question"]

    # 1. 查询改写
    rewriter = QueryRewriter()
    rewritten = await rewriter.rewrite(query) if rag_config.enable_rewrite else query

    # 2. 编码
    encoder = Encoder()
    vecs = encoder.encode_query(rewritten)

    # 3. Milvus 候选
    vector_store = VectorStore()
    milvus_candidates = vector_store.hybrid_search(
        vecs["dense"], vecs["sparse"], course_id, top_k=rag_config.dense_top_k
    )
    milvus_uuids = {c["uuid"] for c in milvus_candidates}

    # 4. BM25 候选
    bm25 = BM25Indexer()
    bm25_candidates = await bm25.search(session, rewritten, course_id, top_k=rag_config.bm25_top_k)
    bm25_uuids = {c["uuid"] for c in bm25_candidates}

    # 5. RRF 融合后候选
    from coursepilot.rag.bm25 import rrf_fuse
    fused_candidates = rrf_fuse([milvus_candidates, bm25_candidates], k=rag_config.rrf_k, top_k=20)
    fused_uuids = {c["uuid"] for c in fused_candidates}

    # 6. Reranker 后 top-k
    reranker = Reranker()
    reranked = reranker.rerank(rewritten, fused_candidates, top_k=rag_config.rerank_top_k)
    reranked_uuids = {c["uuid"] for c in reranked}

    # 7. 真实 Retriever 最终上下文（含 KP 扩展）
    retriever = Retriever()
    _, metadata = await retriever.retrieve(session, query, course_id)
    final_context_uuids = set(metadata.get("context_uuids", []))

    # 8. 查询 ground truth 是否仍在数据库中
    missing_in_db = []
    for uid in gt_uuids:
        ku = await session.get(KnowledgeUnit, UUID(uid))
        if ku is None:
            missing_in_db.append(uid)

    return {
        "question": query[:80],
        "rewritten": rewritten[:80],
        "gt_uuids": list(gt_uuids),
        "missing_in_db": missing_in_db,
        "milvus_hits": list(gt_uuids & milvus_uuids),
        "bm25_hits": list(gt_uuids & bm25_uuids),
        "fused_hits": list(gt_uuids & fused_uuids),
        "reranked_hits": list(gt_uuids & reranked_uuids),
        "final_context_hits": list(gt_uuids & final_context_uuids),
        "milvus_top_kp_paths": [c.get("kp_path", "") for c in milvus_candidates[:5]],
        "reranked_top_kp_paths": [c.get("kp_path", "") for c in reranked[:5]],
    }


async def main():
    if not DATASET.exists():
        print(f"[ERROR] 数据集不存在: {DATASET}")
        return

    questions = json.loads(DATASET.read_text(encoding="utf-8"))
    course_id = questions[0]["course_id"]

    async with get_session_etx() as session:
        for i, q in enumerate(questions, 1):
            gt = set(q.get("ground_truth_contexts", []))
            diag = await check_ground_truth_in_candidates(session, q, course_id, gt)
            print(f"\n{'='*60}")
            print(f"Q{i}: {diag['question']}")
            print(f"改写: {diag['rewritten']}")
            print(f"GT UUIDs: {diag['gt_uuids']}")
            if diag["missing_in_db"]:
                print(f"[WARN] 以下 ground truth 不在数据库中: {diag['missing_in_db']}")
            print(f"Milvus命中: {diag['milvus_hits']} | BM25命中: {diag['bm25_hits']}")
            print(f"RRF融合命中: {diag['fused_hits']} | Reranker后命中: {diag['reranked_hits']}")
            print(f"最终上下文命中: {diag['final_context_hits']}")
            print(f"Milvus top KP: {diag['milvus_top_kp_paths']}")
            print(f"Rerank top KP: {diag['reranked_top_kp_paths']}")


if __name__ == "__main__":
    asyncio.run(main())
