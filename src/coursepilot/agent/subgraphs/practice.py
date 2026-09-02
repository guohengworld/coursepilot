"""练习子图：get_mastery → query_rag → generate_quiz → evaluate_quiz → create_plan。

拓扑（含重试循环）：
    START → get_mastery → query_rag → generate_quiz → evaluate_quiz
      ── [route_practice_evaluate] ─┬─ generate_quiz（FAIL 且 retry<3 重试）
                                     └─ create_plan → END

复用现有节点函数（签名 (state: dict) -> dict 不变），通过 PracticeInput /
PracticeOutput 裁剪与父图的边界。私有字段 mastery / eval_result / retry_count
只在子图内流转，不出边界（quiz_data 交回父图供 finalize 持久化）。

注意：compile() 不传 checkpointer——checkpointer 由父图统一注入。
"""
from langgraph.graph import END, START, StateGraph

from coursepilot.agent.nodes import (
    create_plan_node,
    evaluate_quiz_node,
    generate_quiz_node,
    get_mastery_node,
    query_rag_node,
)
from coursepilot.agent.sub_state import (
    PracticeInput,
    PracticeOutput,
    PracticeState,
)


def route_practice_evaluate(state: dict) -> str:
    """练习子图内部：evaluate 后重试或收口 create_plan。

    与父图 route_after_evaluate 的 practice 分支等价（FAIL 且 retry<3 回
    generate_quiz 重试，否则进 create_plan）。子图内 intent 恒为 practice，
    故只保留两条分支，不再按 intent 分发。阈值与 route_after_evaluate 一致。
    """
    eval_result = state.get("eval_result", {})
    retry_count = state.get("retry_count", 0)
    if eval_result.get("status") == "FAIL" and retry_count < 3:
        return "generate_quiz"
    return "create_plan"


def build_practice_subgraph():
    """构建练习子图（CompiledStateGraph）。"""
    builder = StateGraph(
        PracticeState,
        input_schema=PracticeInput,
        output_schema=PracticeOutput,
    )
    builder.add_node("get_mastery", get_mastery_node)
    builder.add_node("query_rag", query_rag_node)
    builder.add_node("generate_quiz", generate_quiz_node)
    builder.add_node("evaluate_quiz", evaluate_quiz_node)
    builder.add_node("create_plan", create_plan_node)

    builder.add_edge(START, "get_mastery")
    builder.add_edge("get_mastery", "query_rag")
    builder.add_edge("query_rag", "generate_quiz")
    builder.add_edge("generate_quiz", "evaluate_quiz")
    builder.add_conditional_edges("evaluate_quiz", route_practice_evaluate, {
        "generate_quiz": "generate_quiz",
        "create_plan": "create_plan",
    })
    builder.add_edge("create_plan", END)
    return builder.compile()
