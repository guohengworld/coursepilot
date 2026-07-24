"""L4 归档记忆召回：基于 QARecord 的历史问答语义检索。

评分公式：
    score = α · exp(-Δt/τ) + β · cosine(query, turn_embedding) + γ · importance

其中 importance 在入库时由轻量 LLM 判断，复用现有的 BGE-M3 编码器。
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.config import settings
from coursepilot.models import QARecord
from coursepilot.rag.encoder import Encoder

logger = logging.getLogger(__name__)

# 默认权重：recency、relevance、importance
DEFAULT_ALPHA = 0.25
DEFAULT_BETA = 0.55
DEFAULT_GAMMA = 0.20
DEFAULT_TAU_DAYS = 30.0


def _cosine(a: list[float], b: list[float]) -> float:
    """两个稠密向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _recency_score(created_at: datetime | None, tau_days: float = DEFAULT_TAU_DAYS) -> float:
    """基于指数衰减的时效分。"""
    if created_at is None:
        return 0.5
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    delta_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    return math.exp(-delta_days / tau_days)


def score_memory_turn(
    query_embedding: list[float],
    turn: QARecord,
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    tau_days: float = DEFAULT_TAU_DAYS,
) -> dict[str, float]:
    """计算单条 QARecord 的记忆召回分。"""
    recency = _recency_score(turn.created_at, tau_days)
    relevance = 0.0
    if turn.embedding and query_embedding:
        relevance = _cosine(query_embedding, turn.embedding)
    importance = turn.importance if turn.importance is not None else 0.5

    score = alpha * recency + beta * relevance + gamma * importance
    return {
        "score": round(score, 4),
        "recency": round(recency, 4),
        "relevance": round(relevance, 4),
        "importance": round(importance, 4),
    }


async def estimate_importance(query: str, answer: str) -> float:
    """用轻量规则估算 QA 记忆重要性。

    规则：
    - 包含薄弱信号（如“不懂”、“错了”、“不明白”）+0.3
    - 答案较长且含公式 +0.2
    - 用户明确反馈有用 +0.1
    - 基础问候类 -0.2
    """
    text = f"{query}\n{answer}".lower()
    score = 0.5
    weak_signals = ["不懂", "错了", "不明白", "不会", "困惑", "难点"]
    if any(s in text for s in weak_signals):
        score += 0.3
    if "$" in answer and len(answer) > 200:
        score += 0.2
    greeting_signals = ["你好", "谢谢", "再见", "哈喽"]
    if sum(1 for s in greeting_signals if s in query) >= 1 and len(query) < 15:
        score -= 0.2
    return round(max(0.0, min(1.0, score)), 2)


async def recall_memory_turns(
    session: AsyncSession,
    user_id: str,
    course_id: str,
    query: str,
    encoder: Encoder | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """召回与当前 query 最相关的历史 QA 记录。

    返回带 score  breakdown 的 QARecord 列表，按 score 降序。
    """
    encoder = encoder or Encoder()
    query_vec = encoder.encode_query(query)["dense"]

    result = await session.execute(
        select(QARecord).where(
            QARecord.user_id == user_id,
            QARecord.course_id == course_id,
        ).order_by(QARecord.created_at.desc()).limit(200)
    )
    records = result.scalars().all()

    scored = []
    for r in records:
        # 没有 embedding 时即时编码（冷启动）
        if r.embedding is None and encoder._model is not None:
            try:
                vec = encoder.encode_qa_records([{"query": r.query, "answer": r.answer}])[0]
                r.embedding = vec["dense"]
            except Exception:
                logger.warning("QARecord %s embedding 失败", r.id)
                continue

        scores = score_memory_turn(query_vec, r)
        scored.append({
            "qa_id": str(r.id),
            "query": r.query,
            "answer": r.answer[:300],
            "kp_path": r.kp_path,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "scores": scores,
        })

    scored.sort(key=lambda x: x["scores"]["score"], reverse=True)
    return scored[:top_k]


async def ensure_qa_embeddings(session: AsyncSession, encoder: Encoder | None = None) -> int:
    """后台任务：为所有缺失 embedding 的 QARecord 补全向量与 importance。

    返回补全条数。
    """
    encoder = encoder or Encoder()
    result = await session.execute(
        select(QARecord).where(QARecord.embedding.is_(None)).limit(100)
    )
    records = result.scalars().all()
    if not records:
        return 0

    vectors = encoder.encode_qa_records([{"query": r.query, "answer": r.answer} for r in records])
    updated = 0
    for r, vec in zip(records, vectors):
        r.embedding = vec["dense"]
        r.importance = await estimate_importance(r.query, r.answer)
        updated += 1

    await session.commit()
    logger.info("QARecord embedding 补全 %d 条", updated)
    return updated
