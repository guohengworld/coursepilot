"""LangGraph 状态机构建

Phase 2 拓扑（条件边 + 重试循环）：
    START → build_context → classify → [route_by_intent]
      ├→ query_rag → [route_after_rag] → finalize → END
      │   └─ (practice/review) → generate_quiz → evaluate_quiz
      │        └─ [route_after_evaluate]:
      │            FAIL+retry<2 → generate_quiz (重试)
      │            PASS+practice → create_plan → finalize
      │            PASS+review → review_plan → finalize
      ├→ get_mastery → query_rag → (同上)
      └→ diagnose → finalize → END
"""
import logging

import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph, START, END

from coursepilot.agent.nodes import (
    build_context_node, classify_node, finalize_node, query_rag_node,
    get_mastery_node, generate_quiz_node, evaluate_quiz_node,
    create_plan_node, diagnose_node, review_plan_node, human_review_node
)
from coursepilot.agent.routing import (
    route_by_intent, route_after_rag, route_after_evaluate,
)
from coursepilot.agent.state import AgentState
from coursepilot.config import settings

logger = logging.getLogger(__name__)

# 模块级 Saver 缓存（build_agent_graph 首次调用时初始化）
_saver: AsyncPostgresSaver | None = None


async def _get_saver() -> AsyncPostgresSaver:
    """延迟初始化 AsyncPostgresSaver 单例"""
    global _saver
    if _saver is None:
        # 手动创建连接（绕过 from_conn_string 的 context manager 限制）
        conn = await psycopg.AsyncConnection.connect(
            settings.database_url_sync,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        )
        _saver = AsyncPostgresSaver(conn=conn)
        # Windows + psycopg 3.3: pipeline 模式会导致 put() 卡死
        _saver.supports_pipeline = False
        await _saver.setup()
        logger.info("✅ AsyncPostgresSaver 初始化完成（pipeline 已禁用）")
    return _saver


async def build_agent_graph():
    """构建并编译 Agent 状态图

    :returns: CompiledStateGraph: 可直接调用 .ainvoke() 的编译图
    """
    builder = StateGraph(AgentState)

    # 注册节点
    builder.add_node("build_context", build_context_node)
    builder.add_node("classify", classify_node)
    builder.add_node("query_rag", query_rag_node)
    builder.add_node("get_mastery", get_mastery_node)
    builder.add_node("generate_quiz", generate_quiz_node)
    builder.add_node("evaluate_quiz", evaluate_quiz_node)
    builder.add_node("create_plan", create_plan_node)
    builder.add_node("diagnose", diagnose_node)
    builder.add_node("review_plan", review_plan_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("human_review", human_review_node)

    builder.add_edge(START, "build_context")
    builder.add_edge("build_context", "classify")

    # 条件边
    # classify → intent 分发
    builder.add_conditional_edges("classify", route_by_intent, {
        "query_rag": "query_rag",
        "get_mastery": "get_mastery",
        "diagnose": "diagnose",
        "human_review": "human_review"
    })

    # human_review 批准后继续
    builder.add_conditional_edges("human_review", route_after_review, {
        "get_mastery": "get_mastery",  # 已批准 → 继续
        "finalize": "finalize",  # 已拒绝 → 结束
    })

    # get_mastery → 始终进 query_rag 获取教材内容
    builder.add_edge("get_mastery", "query_rag")

    # query_rag → 按 intent 决定是否继续出题
    builder.add_conditional_edges("query_rag", route_after_rag, {
        "generate_quiz": "generate_quiz",
        "finalize": "finalize",
    })

    # generate_quiz → evaluate_quiz（固定边）
    builder.add_edge("generate_quiz", "evaluate_quiz")

    # evaluate_quiz → 重试 / 继续
    builder.add_conditional_edges("evaluate_quiz", route_after_evaluate, {
        "generate_quiz": "generate_quiz",
        "review_plan": "review_plan",
        "create_plan": "create_plan",
        "finalize": "finalize",
    })

    # create_plan → finalize（固定边）
    builder.add_edge("create_plan", "finalize")
    # review_plan → finalize（固定边）
    builder.add_edge("review_plan", "finalize")
    # diagnose → finalize（固定边）
    builder.add_edge("diagnose", "finalize")
    # finalize → END
    builder.add_edge("finalize", END)

    # checkpointer：AsyncPostgresSaver
    saver = await _get_saver()
    graph = builder.compile(checkpointer=saver)
    return graph
