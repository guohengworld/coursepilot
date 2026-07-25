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
    3: {"level_penalty": [0.0, 0.1, 0.2]},
}

# 网格搜索每轮固定参数（未被搜索的参数保持当前配置）
GRID_FIXED_DEFAULTS = {
    "rrf_k": 60,
    "dense_weight": 0.5,
    "rerank_top_k": 5,
    "context_max_chars": 6000,
    "level_penalty": 0.0,
}
