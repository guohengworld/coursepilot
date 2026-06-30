"""LangGraph 节点函数

每个节点接收 AgentState，返回状态更新字典（只写自己负责的字段）
"""
import logging

from coursepilot.agent.context import build_context as build_context_logic
from coursepilot.agent.skills.classify_intent import classify_intent
from coursepilot.agent.skills.query_rag import query_rag
from coursepilot.agent.skills.update_qa_record import update_qa_record
from coursepilot.db import async_session_factory

logger = logging.getLogger(__name__)

async def build_context_node(state: dict) -> dict:
    """构建上下文：课程信息 + 学生画像 + 最近问答"""
    try:
        async with async_session_factory() as session:
            course_ctx, profile, recent_qa = await build_context_logic(
                session,
                user_id=state["user_id"],
                course_id=state["course_id"]
            )
        return {
            "course_context": course_ctx,
            "user_profile": profile,
            "recent_qa": recent_qa,
            "error": None,
        }
    except Exception as e:
        logger.exception("build_context 节点异常")
        return {"course_context": {}, "user_profile": None, "recent_qa": [], "error": str(e)}

async def classify_node(state: dict) -> dict:
    """意图分类"""
    try:
        intent = await classify_intent(
            query=state["query"],
            course_context=state.get("course_context", {}),
            recent_qa=state.get("recent_qa", []),
        )
        return {"intent": intent, "error": None}
    except Exception as e:
        logger.exception("classify 节点异常")
        return {"intent": "question", "error": str(e)}

async def query_rag_node(state: dict) -> dict:
    """RAG 检索 + LLM 生成"""
    try:
        async with async_session_factory() as session:
            answer, context, metadata, sources = await query_rag(
                session=session,
                query=state["query"],
                course_id=state["course_id"],
                course_context=state.get("course_context", {}),
            )
        return {
            "answer": answer,
            "context": context,
            "retrieved_metadata": metadata,
            "sources": sources,
            "error": None,
        }
    except Exception as e:
        logger.exception("query_rag 节点异常")
        return {
            "answer": f"抱歉，检索知识库时出错了：{e}",
            "error": str(e),
        }

async def finalize_node(state: dict) -> dict:
    """持久化问答记录 + 更新会话状态。"""
    try:
        async with async_session_factory() as session:
            token_count = await update_qa_record(
                session=session,
                user_id=state["user_id"],
                course_id=state["course_id"],
                query=state["query"],
                answer=state["answer"],
                kp_path=_first_kp_path(state.get("retrieved_metadata", {})),
                retrieved_units=state.get("retrieved_metadata", {}).get("top_uuids", []),
                citations=state.get("sources", []),
                session_id=state["session_id"],
            )
        return {"token_count": token_count, "error": None}
    except Exception as e:
        logger.exception("finalize 节点异常")
        return {"error": str(e)}

def _first_kp_path(metadata: dict) -> str | None:
    paths = metadata.get("source_kp_paths", [])
    return paths[0] if paths else None
