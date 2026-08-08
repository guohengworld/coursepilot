"""RAG 管线全部可配置参数集中管理。

使用时直接导入模块级实例 `config`：
    from coursepilot.rag.config import config
    print(config.rrf_k)
    config.rrf_k = 40  # 临时覆盖

网格搜索通过 RAGEvaluator(config_overrides=...) 覆盖，不会改动原始值。
完整参数说明见 docs/rag/RAG评估体系构建.md §5.1。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RAGConfig:
    """RAG 检索与生成管线的全部可调参数。

    共 25 项，按功能分为 6 类：
    功能开关、智能路由、阈值、检索参数、BM25 参数、编码参数。
    """

    # ── 功能开关 ────────────────────────────────────────────────
    enable_rewrite: bool = True       # 查询改写：True→LLM 改写 query；False→直接用原始 query
    enable_sparse: bool = True        # 稀疏检索：True→Milvus 同时查 dense+sparse；False→只查 dense
    enable_bm25: bool = True          # BM25 检索：True→额外走 BM25 关键词检索；False→只用 Milvus
    enable_rerank: bool = True        # 重排序：True→cross-encoder 精排；False→RRF score 直接排序
    enable_kp_expand: bool = True     # KP 扩展：True→拉取同 KP 全部 unit 丰富上下文；False→仅用命中 unit

    # ── 智能路由 ────────────────────────────────────────────────
    enable_routing: bool = True       # 智能路由：True→按复杂度走快慢通道；False→一律走全量流程
    simple_top_k: int = 3             # 简单通道最终送入 LLM 的 chunk 数（轻量检索）
    complex_max_rounds: int = 3       # 复杂问题最多补搜轮数（用于多轮质检+补搜）
    context_sufficiency_threshold: float = 0.7  # 质检通过阈值（用于上下文充足性判断）
    agent_max_steps: int = 8          # Agent 循环最大步数（超过强制结束并降级）
    agent_max_web_searches: int = 2   # Agent 循环中 web_search 工具最多调用次数
    agent_token_budget: int = 8000    # Agent 循环累计 token 预算（含 messages 全量重发）

    kp_expand_mode: str = "full"      # KP 扩展模式："full"（同 KP 全量）| "neighbor"（相邻滑动窗口）
    kp_neighbor_window: int = 2       # neighbor 模式下前后各取 N 个相邻 unit（仅 mode="neighbor" 生效）

    # ── 阈值 ────────────────────────────────────────────────────
    reranker_min_score: float = 0.3   # 重排序最低分数阈值，低于此值的 source 直接丢弃
    context_max_chars: int = 5000     # 送入 LLM 的上下文软上限（字符数），超长截断

    # ── 检索参数 ────────────────────────────────────────────────
    dense_top_k: int = 20             # Milvus dense 向量检索返回候选条数
    sparse_top_k: int = 20            # Milvus sparse 向量检索返回候选条数（enable_sparse=True 时生效）
    rrf_k: int = 60                   # RRF 融合参数 k，控制稀疏/稠密候选的排序平衡
    dense_weight: float = 0.5         # dense 在 RRF 融合中的权重（0~1），sparse 权重 = 1 - dense_weight
    rerank_top_k: int = 5             # 重排序后最终送入 LLM 生成器的 chunk 数量

    # ── BM25 参数 ───────────────────────────────────────────────
    bm25_top_k: int = 20              # BM25 检索返回候选条数（enable_bm25=True 时生效）
    bm25_cache_ttl: int = 600         # BM25 索引缓存有效期（秒）

    # ── 编码参数 ────────────────────────────────────────────────
    batch_size: int = 32              # BGE-M3 编码器 batch size，影响编码速度和显存占用
    dim: int = 1024                   # BGE-M3 dense 向量维度，需与 Milvus collection schema 一致


config = RAGConfig()
