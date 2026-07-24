"""RAG 检索 + LLM 生成 Skill。

零改动调用现有的 Retriever 和 Generator。
Retriever / Generator / Encoder / Reranker 在模块内部有单例缓存，
每次构造不重加载模型，放在函数内安全且避开了 import-time 副作用（如 Milvus 连接）。
"""
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.rag.generator import Generator
from coursepilot.rag.retriever import Retriever


async def query_rag(
    session: AsyncSession,
    query: str,
    course_id: str,
    course_context: dict,
    conversation: list[dict[str, Any]] | None = None,
    rolling_summary: str = "",
    user_profile: dict[str, Any] | None = None,
) -> tuple[str, str, dict, list[dict], dict]:
    """执行完整 RAG 检索 + LLM 生成，支持多轮上下文。

    Args:
        conversation: 完整多轮对话（L1），包含 role/content/intent 等字段
        rolling_summary: L2 滚动摘要
        user_profile: L3 学生画像

    Returns:
        (answer, raw_context, metadata, sources, token_info)
    """
    # 1. 五阶段检索
    retriever = Retriever()
    context, metadata = await retriever.retrieve(session, query, course_id)

    # 2. 组装引用来源
    source_kp_paths = metadata.get("source_kp_paths", [])
    sources = [{"kp_path": p} for p in source_kp_paths]

    # 3. LLM 生成（携带分层记忆）
    generator = Generator()
    answer, token_info = await generator.generate(
        query=query,
        context=context,
        course_context=course_context,
        conversation=conversation,
        rolling_summary=rolling_summary,
        user_profile=user_profile,
    )

    return answer, context, metadata, sources, token_info











