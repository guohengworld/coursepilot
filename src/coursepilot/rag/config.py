"""RAG 降级开关、阈值、检索参数集中管理"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    # == 功能开关
    enable_rewrite: bool = True  # 关闭 → 直接用原始 query
    enable_sparse: bool = True  # 关闭 → 只用 dense 检索
    enable_rerank: bool = True  # 关闭 → RRF 后直接取 top-5
    enable_kp_expand: bool = True  # 关闭 → 只用检索到的 unit 本身（不拉取同 KP 全部文本）

    # == 阈值
    reranker_min_score: float = 0.3  # 低于此分的 source 直接丢弃
    context_max_chars: int = 8000  # 送入 LLM 的上下文软上限

    # == 检索参数
    dense_top_k: int = 20  # dense 检索返回条数
    sparse_top_k: int = 20  # sparse 检索返回条数
    rrf_k: int = 60  # RRF 融合参数
    rerank_top_k: int = 5  # 重排序后最终送入 LLM 的条数
    level_penalty: float = 0.1  # 层级不匹配惩罚系数（设计中 0.02/级，这里统一为 max 0.1）

    # == 编码参数
    batch_size: int = 32  # BGE-M3 编码 batch size
    dim: int = 1024  # BGE-M3 dense 维度


config = RAGConfig()
