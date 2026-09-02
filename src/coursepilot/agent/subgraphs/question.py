"""问答子图：query_rag（快速通道）+ agentic_rag（LLM 自主 ReAct 循环）双通道。

拓扑：
    START → [route_question] ─┬─ query_rag ──→ END
                              └─ agentic_rag → END

双通道互斥（由 complexity 二选一），不存在并行写冲突。route_question
是 question 内部的 complexity 分发，与父图 route_by_intent 的 intent 分发
分层独立：父图只负责「question 意图 → question 子图」，子图内部再按
complexity 决定走快通道还是 Agentic RAG。

复用现有节点函数（签名 (state: dict) -> dict 不变），通过
QuestionInput / QuestionOutput 裁剪与父图的边界。本子图无私有中间态。

注意：compile() 不传 checkpointer——checkpointer 由父图统一注入。
"""
from langgraph.graph import END, START, StateGraph

from coursepilot.agent.nodes import query_rag_node
from coursepilot.agent.rag_agent import agentic_rag_node
from coursepilot.agent.sub_state import (
    QuestionInput,
    QuestionOutput,
    QuestionState,
)
from coursepilot.rag.config import config as rag_config


def route_question(state: dict) -> str:
    """子图内部复杂度分发：complex → agentic_rag，simple → query_rag。

    与父图 route_by_intent 中 question 分支的判定逻辑一致（flag 关闭时
    该判定仍在父图生效，flag 开启后移入子图）。
    """
    complexity = state.get("complexity", "simple")
    if complexity == "complex" and rag_config.enable_routing:
        return "agentic_rag"
    return "query_rag"


def build_question_subgraph():
    """构建问答子图（CompiledStateGraph）。"""
    builder = StateGraph(
        QuestionState,
        input_schema=QuestionInput,
        output_schema=QuestionOutput,
    )
    builder.add_node("query_rag", query_rag_node)
    builder.add_node("agentic_rag", agentic_rag_node)
    builder.add_conditional_edges(START, route_question, {
        "query_rag": "query_rag",
        "agentic_rag": "agentic_rag",
    })
    builder.add_edge("query_rag", END)
    builder.add_edge("agentic_rag", END)
    return builder.compile()
