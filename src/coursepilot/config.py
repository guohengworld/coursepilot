"""应用配置，所有可调参数集中管理。

环境变量覆盖：在项目根目录创建 .env 文件，例如：
    DATABASE_URL=postgresql+asyncpg://postgres:cp_dev@localhost:5432/coursepilot
    LLM_API_KEY=sk-api-key
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── 数据库 ──────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://postgres:cp_dev@localhost:5432/coursepilot"
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

    # ── RAG 检索参数 ────────────────────────────────────
    dense_top_k: int = 20       # 稠密检索返回条数
    sparse_top_k: int = 20      # BM25 稀疏检索返回条数
    rrf_k: int = 60             # 融合参数（RRF 公式中的 k）
    rerank_top_n: int = 5       # 重排序后最终送入 LLM 的条数
    level_penalty: float = 0.1  # 层级不匹配惩罚系数

    # ── Milvus ─────────────────────────────────────────
    milvus_uri: str = "./milvus.db"          # Milvus Lite 存储路径
    milvus_collection: str = "coursepilot_knowledge"

    # ── Ingestion ───────────────────────────────────────
    pdf_heading_font_min: int = 14   # 识别为标题的最小字号
    kp_max_tokens: int = 512         # 知识单元最大 token 数
    chunk_overlap: int = 50          # 切分重叠字数

    # ── JWT ────────────────────────────────────────────
    jwt_secret_key: str = "1MWdnNG-fSp8hcp0dHQlga7rFAkNaN261p2FFwefgN0"  # 生产环境修改
    jwt_algorithm: str = "HS256"
    jwt_expire_seconds: int = 86400  # 24 小时

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
