"""条件路由函数（Phase 2 启用）。

Phase 1 为线性路径，无条件边。此文件占位，
Phase 2 实现 route_by_intent() 分发到 5 种 workflow。
"""

def route_by_intent(state: dict) -> str:
    """根据 classify 节点的 intent 输出路由到对应路径

    Phase 2 实现：
        question  → "query_rag"
        practice  → "get_mastery"
        diagnose  → "diagnose"
        review    → "review_plan"
        code_help → "code_help"

    Phase 1 占位：始终返回 "query_rag"。
    """
    return "query_rag"
