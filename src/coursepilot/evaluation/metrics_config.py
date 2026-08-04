"""RAGAS 评估指标与阈值配置。

参考：docs/rag/RAG评估体系构建.md
"""

from __future__ import annotations

# RAGAS 8 大核心指标名称（与 EvalReport 字段保持一致）
METRIC_NAMES = {
    "context_recall": "Context Recall",
    "context_precision": "Context Precision",
    "context_entity_recall": "Context Entity Recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "answer_correctness": "Answer Correctness",
    "answer_similarity": "Answer Similarity",
    "aspect_critique": "Aspect Critique",
}

# 指标分类
RETRIEVAL_METRICS = [
    "context_recall",
    "context_precision",
    "context_entity_recall",
]

GENERATION_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "answer_similarity",
    "aspect_critique",
]

# 是否需要参考答案（Ground Truth）
NEEDS_GROUND_TRUTH = {
    "context_recall": True,
    "context_precision": True,  # 使用 LLMContextPrecisionWithReference，需要参考答案
    "context_entity_recall": True,
    "faithfulness": False,
    "answer_relevancy": False,
    "answer_correctness": True,
    "answer_similarity": True,
    "aspect_critique": False,
}

# CI/CD 质量门禁阈值
# 策略：略低于预期基线，避免正常波动阻塞发布
THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_recall": 0.85,
    "context_precision": 0.75,
    "answer_correctness": 0.80,
}

# 网格搜索三轮策略（参考文档 5.2）
GRID_SEARCH_PLAN = {
    1: {"rrf_k": [20, 40, 60, 100], "dense_weight": [0.3, 0.5, 0.7]},
    2: {"rerank_top_k": [3, 5, 8, 10], "context_max_chars": [4000, 6000, 8000, 10000]},
}

# RAG 全部参数默认值（与 RAGConfig 对齐）
# 网格搜索时，未被搜索的参数以此处为准固定下来。
# 完整参数说明见 docs/rag/RAG评估体系构建.md §5.1。
ALL_PARAM_DEFAULTS = {
    # 功能开关
    "enable_rewrite": True,
    "enable_sparse": True,
    "enable_bm25": True,
    "enable_rerank": True,
    "enable_kp_expand": True,
    "kp_expand_mode": "full",
    "kp_neighbor_window": 2,
    # 阈值
    "reranker_min_score": 0.3,
    "context_max_chars": 5000,
    # 检索参数
    "dense_top_k": 20,
    "sparse_top_k": 20,
    "rrf_k": 60,
    "dense_weight": 0.5,
    "rerank_top_k": 5,
    # BM25 参数
    "bm25_top_k": 20,
    "bm25_cache_ttl": 600,
    # 编码参数
    "batch_size": 32,
    "dim": 1024,
}

# 网格搜索每轮固定参数（未被搜索的参数保持默认值）
GRID_FIXED_DEFAULTS = {
    k: v for k, v in ALL_PARAM_DEFAULTS.items()
}
