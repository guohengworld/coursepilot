"""LangGraph 状态机构建

Phase 1 线性路径：
    build_context → classify → query_rag → finalize → END

checkpointer 使用 PostgresSaver（同步 psycopg 连接），
支持断点恢复和人类接管（Phase 3 启用）。
"""
import logging

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START, END

from coursepilot.agent.nodes import (
    build_context_node,
    classify_node,
    finalize_node,
    query_rag_node,
)
from coursepilot.agent.state import AgentState
from coursepilot.config import settings

logger = logging.getLogger(__name__)

# 模块级 Saver 缓存（build_agent_graph 首次调用时初始化）
_saver: PostgresSaver | None = None

def _get_saver() -> PostgresSaver:
    """延迟初始化 PostgresSaver 单例"""
    global _saver
    if _saver is None:
        conn = psycopg.connect(settings.database_url_sync, autocommit=True)
        _saver = PostgresSaver(conn)
        _saver.setup()
        logger.info("PostgresSaver 已初始化")
    return _saver

def build_agent_graph():
    """构建并编译 Agent 状态图

    :returns: CompiledStateGraph: 可直接调用 .ainvoke() 的编译图
    """
    builder = StateGraph(AgentState)

    # 注册节点
    builder.add_node("build_context", build_context_node)
    builder.add_node("classify", classify_node)
    builder.add_node("query_rag", query_rag_node)
    builder.add_node("finalize", finalize_node)

    # Phase 1: 线性路径（无条件边）
    builder.add_edge(START, "build_context")
    builder.add_edge("build_context", "classify")
    builder.add_edge("classify", "query_rag")
    builder.add_edge("query_rag", "finalize")
    builder.add_edge("finalize", END)

    # checkpointer：PostgresSaver
    saver = _get_saver()
    graph = builder.compile(checkpointer=saver)
    return graph
