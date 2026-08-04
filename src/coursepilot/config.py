"""应用配置，所有可调参数集中管理。

环境变量覆盖：在项目根目录创建 .env 文件，例如：
    DATABASE_URL=postgresql+asyncpg://postgres:xxx@localhost:5432/coursepilot
    LLM_API_KEY=sk-api-key
"""
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（config.py 位于 src/coursepilot/，上溯三级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # ── 数据库 ──────────────────────────────────────────
    database_url: str = ""
    database_url_sync: str = ""   # PostgresSaver 同步连接（langgraph checkpoint）
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # ── 模型路径（本地部署的 embedding / reranker） ──────
    embedding_model_path: str = "F:/all-projs/models/bge-m3"
    reranker_model_path: str = "F:/all-projs/models/bge-reranker-v2-m3"

    # ── LLM ────────────────────────────────────────────
    llm_api_key: str = ""
    llm_model: str = "deepseek-v4-flash"
    llm_base_url: str = "https://api.deepseek.com"
    llm_temperature: float = 0.3

    # ── MiMO Judge ─────────────────────────────────────
    mimo_api_key: str = ""
    mimo_model: str = "mimo-v2.5-pro"
    mimo_base_url: str = "https://api.xiaomimimo.com/v1"
    mimo_embedding_model: str = "text-embedding-ada-002"

    # ── 上下文窗口预算（ContextManager 使用）───────────────
    llm_context_budget: dict = {
        "total_tokens": 64_000,
        "reserved_output": 4_096,
        "safety_margin": 1_024,
        "max_recent_turns": 6,
        "rolling_summary_max_tokens": 1_500,
        "user_profile_max_tokens": 400,
        "rag_default_max_tokens": 8_000,
    }

    # ── RAG 检索参数 ────────────────────────────────────
    dense_top_k: int = 20       # 稠密检索返回条数
    sparse_top_k: int = 20      # BM25 稀疏检索返回条数
    rrf_k: int = 60             # 融合参数（RRF 公式中的 k）
    rerank_top_n: int = 5       # 重排序后最终送入 LLM 的条数

    # ── Milvus ─────────────────────────────────────────
    # Milvus Lite 存储路径（基于项目根目录的相对路径，也支持绝对路径）
    milvus_uri: str = "data/milvus/milvus.db"
    milvus_collection: str = "knowledge_units"

    # ── Ingestion ───────────────────────────────────────
    pdf_heading_font_min: int = 14   # 识别为标题的最小字号
    kp_max_tokens: int = 512         # 知识单元最大 token 数
    chunk_overlap: int = 50          # 切分重叠字数
    pdf_enable_text_fast_path: bool = True  # 文字版 PDF 是否走 PyMuPDF 快速通道
    pdf_text_min_chars_per_page: int = 200  # 采样页平均字符数超过此值视为文字版

    # ── MinerU ────────────────────────────────────────
    mineru_backend: str = "pipeline"       # pipeline / hybrid-engine
    mineru_method: str = "auto"            # auto / txt / ocr
    mineru_lang: str = "ch"               # 文档主要语言
    mineru_output_dir: str = "F:/all-projs/coursepilot/parsed"     # MinerU 输出根目录
    mineru_model_source: str = "local"     # local / huggingface / modelscope
    mineru_formula_enable: bool = True     # 公式识别
    mineru_table_enable: bool = True       # 表格识别

    # ── JWT ────────────────────────────────────────────
    jwt_secret_key: str = ""  # 从 .env 加载
    jwt_algorithm: str = "HS256"
    jwt_expire_seconds: int = 86400  # 24 小时

    # ── MCP ────────────────────────────────────────────
    mcp_transport: str = "stdio"  # stdio / http
    mcp_gateway: str = "https://mcp.coursepilot.example.com/mcp"
    mcp_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            # 与 cli 端环境变量命名一致（优先）；兼容旧名 MCP_API_KEY
            "COURSEPILOT_MCP_API_KEY",
            "MCP_API_KEY",
        ),
    )  # MVP 阶段轻量认证
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8080
    mcp_protocol_version: str = "2025-06-18"

    # ── 学情诊断 ───────────────────────────────────────
    diagnose_weak_threshold: float = 0.6   # 正确率低于此值视为薄弱
    diagnose_lookback_days: int = 90       # 分析最近 N 天的做题记录

    # ── Token 计价（仅内部成本估算，非用户定价） ────────
    token_cost_per_1k_input: float = 0.0005   # 每千输入 token 成本（元）
    token_cost_per_1k_output: float = 0.0015  # 每千输出 token 成本（元）

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        extra="ignore",
    )


settings = Settings()
