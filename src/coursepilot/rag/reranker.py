"""
重排序 —— bge-reranker-v2-m3 cross-encoder 对候选逐对打分

用法：
    reranker = Reranker()
    top5 = reranker.rerank(query_str, candidates_list, top_k=5)
"""

from __future__ import annotations

import logging

import torch

from coursepilot.config import settings
from coursepilot.rag.config import config

logger = logging.getLogger(__name__)


def _select_device() -> str:
    """选择重排序设备：优先 CUDA，否则 CPU。"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_reranker():
    """加载 bge-reranker-v2-m3，优先 GPU，失败则降级 CPU。"""
    try:
        from FlagEmbedding import FlagReranker

        device = _select_device()
        use_fp16 = device.startswith("cuda")

        logger.info(
            "加载 bge-reranker-v2-m3: %s, device=%s, fp16=%s",
            settings.reranker_model_path,
            device,
            use_fp16,
        )
        return FlagReranker(
            settings.reranker_model_path,
            use_fp16=use_fp16,
            devices=device,
        )
    except OSError as e:
        logger.warning("reranker 加载失败 (OSError: %s)，将跳过重排序步骤", e)
        return None
    except Exception as exc:
        logger.warning("reranker 加载到 GPU 失败 (%s)，尝试降级到 CPU", exc)
        try:
            from FlagEmbedding import FlagReranker

            logger.info("降级加载 bge-reranker-v2-m3 到 CPU")
            return FlagReranker(
                settings.reranker_model_path,
                use_fp16=False,
                devices="cpu",
            )
        except Exception as exc2:
            logger.warning("reranker CPU 加载也失败 (%s)，将跳过重排序步骤", exc2)
            return None


class Reranker:
    """Cross-encoder 重排序器（单例）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = _load_reranker()
        return cls._instance

    @property
    def available(self) -> bool:
        return self.model is not None

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
        :param top_k: 重排序后最终送入 LLM 的条数
        :return: 带"rerank_score" 字段的排序后列表
        """
        top_k = top_k or config.rerank_top_k
        if not candidates:
            return []
        if self.model is None:
            logger.warning("reranker 不可用，跳过重排序，直接返回前 %d 个候选", top_k)
            for i, c in enumerate(candidates[:top_k]):
                c["rerank_score"] = 1.0 - i * 0.01
            return candidates[:top_k]

        logger.info("开始重排序: 候选数=%d, top_k=%d", len(candidates), top_k)
        t0 = __import__("time").monotonic()

        try:
            # 构造 query-doc 对
            pairs = [[query, c["content"]] for c in candidates]
            scores = self.model.compute_score(pairs, normalize=True)
        except Exception as exc:
            logger.warning("reranker compute_score 失败 (%s)，跳过重排序", exc)
            for i, c in enumerate(candidates[:top_k]):
                c["rerank_score"] = 1.0 - i * 0.01
            return candidates[:top_k]

        elapsed = (__import__("time").monotonic() - t0) * 1000
        logger.info("reranker compute_score 完成, 耗时=%.0fms", elapsed)

        # 合并得分
        for i, c in enumerate(candidates):
            c["rerank_score"] = scores[i]

        # 过滤低于阈值的
        filtered = [
            c for c in candidates
            if c["rerank_score"] >= config.reranker_min_score
        ]

        filtered.sort(key=lambda x: x["rerank_score"], reverse=True)
        # 兜底：阈值过滤后为空时降级为不过滤阈值，
        # 避免空 context 进入 LLM（完整检索链路白白消耗后才被兜底拦截）
        if not filtered:
            logger.warning(
                "reranker 阈值 %.2f 过滤后为空，降级返回未过滤的 top-%d（需检查阈值配置或候选质量）",
                config.reranker_min_score,
                top_k,
            )
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            return candidates[:top_k]
        return filtered[:top_k]
