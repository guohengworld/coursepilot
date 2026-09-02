"""复习子图：get_mastery → query_rag → generate_quiz → evaluate_quiz → review_plan。

拓扑（含重试循环）：
    START → get_mastery → query_rag → generate_quiz → evaluate_quiz
      ── [route_review_evaluate] ─┬─ generate_quiz（FAIL 且 retry<3 重试）
                                   └─ review_plan → END

与 practice 子图共享 get_mastery / query_rag / generate_quiz / evaluate_quiz
前缀，末步由 create_plan 换成 review_plan（同函数注册多子图，非复制）。
review_plan_node 读 state["diagnosis"]——review 路径不经 diagnose，恒为空，
节点内部实时重查，与原父图行为一致。

注意：compile() 不传 checkpointer——checkpointer 由父图统一注入。
"""
from langgraph.graph import END, START, StateGraph

from coursepilot.agent.nodes import (
    evaluate_quiz_node,
    generate_quiz_node,
    get_mastery_node,
    query_rag_node,
    review_plan_node,
)
from coursepilot.agent.sub_state import (
    ReviewInput,
    ReviewOutput,
    ReviewState,
)


def route_review_evaluate(state: dict) -> str:
    """复习子图内部：evaluate 后重试或收口 review_plan。

    与父图 route_after_evaluate 的 review 分支等价。阈值与 route_after_evaluate
    一致（FAIL 且 retry<3 回 generate_quiz 重试，否则进 review_plan）。
    """
    eval_result = state.get("eval_result", {})
    retry_count = state.get("retry_count", 0)
    if eval_result.get("status") == "FAIL" and retry_count < 3:
        return "generate_quiz"
    return "review_plan"


def build_review_subgraph():
    """构建复习子图（CompiledStateGraph）。"""
    builder = StateGraph(
        ReviewState,
        input_schema=ReviewInput,
        output_schema=ReviewOutput,
    )
    builder.add_node("get_mastery", get_mastery_node)
    builder.add_node("query_rag", query_rag_node)
    builder.add_node("generate_quiz", generate_quiz_node)
    builder.add_node("evaluate_quiz", evaluate_quiz_node)
    builder.add_node("review_plan", review_plan_node)

    builder.add_edge(START, "get_mastery")
    builder.add_edge("get_mastery", "query_rag")
    builder.add_edge("query_rag", "generate_quiz")
    builder.add_edge("generate_quiz", "evaluate_quiz")
    builder.add_conditional_edges("evaluate_quiz", route_review_evaluate, {
        "generate_quiz": "generate_quiz",
        "review_plan": "review_plan",
    })
    builder.add_edge("review_plan", END)
    return builder.compile()
