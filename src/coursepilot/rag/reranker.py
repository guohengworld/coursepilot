"""
重排序 —— bge-reranker-v2-m3 cross-encoder 对候选逐对打分

用法：
    reranker = Reranker()
    top5 = reranker.rerank(query_str, candidates_list, top_k=5)
"""

from __future__ import annotations

import logging

from coursepilot.config import settings
from coursepilot.rag.config import config

logger = logging.getLogger(__name__)

_reranker_instance = None


def _load_reranker():
    """惰性加载 bge-reranker-v2-m3（~1.5 GB，CPU 推理）"""
    from FlagEmbedding import FlagReranker
    logger.info("加载 bge-reranker-v2-m3: %s", settings.reranker_model_path)
    return FlagReranker(
        settings.reranker_model_path,
        use_fp16=False,
        device="cpu",
    )

class Reranker:
    """Cross-encoder 重排序器，全局单例（惰性加载）"""

    def __init__(self):
        global _reranker_instance
        if _reranker_instance is None:
            _reranker_instance = _load_reranker()
        self.model = _reranker_instance

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        对候选列表逐对打分，返回 top-k

        :param query: 查询文本
        :param candidates: 候选人列表  [{"content": str, "kp_path": str, ...}, ...]
        :param top_k: 重排序后最相关的k条数据
        :return: 带"rerank_score" 字段的排序后列表
        """
        top_k = top_k or config.rerank_top_k
        if not candidates:
            return []

        # 构造 query-doc 对
        pairs = [[query, c["content"]] for c in candidates]
        scores = self.model.compute_score(pairs, normalize=True)

        # 层级惩罚：更深的 kp_path 给轻微加分
        for i, c in enumerate(candidates):
            depth = c.get("kp_path", "").count("/") + 1
            penalty = min((depth - 1) * 0.02, config.level_penalty)
            scores[i] -= penalty

        # 合并得分
        for i, c in enumerate(candidates):
            c["rerank_score"] = scores[i]

        # 过滤低于阈值的
        filtered = [
            c for c in candidates
            if c["rerank_score"] >= config.reranker_min_score
        ]

        filtered.sort(key=lambda x : x["rerank_score"], reverse=True)
        return filtered[:top_k]




