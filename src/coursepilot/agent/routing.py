"""条件路由函数（Phase 2 启用）。

提供 3 个路由函数供 graph.py 的条件边使用：
  - route_by_intent: classify 后分发
  - route_after_rag: query_rag 后分发
  - route_after_evaluate: evaluate_quiz 后分发（含重试逻辑）
"""

def route_by_intent(state: dict) -> str:
    """classify 节点后根据 intent 路由

    Returns:
        "get_mastery" — practice / review（需要掌握度）
        "diagnose"    — diagnose
        "query_rag"   — question / code_help（默认 RAG 问答）
    """
    intent = state.get("intent", "question")
    routing = {
        "practice": "get_mastery",
        "review": "get_mastery",
        "diagnose": "diagnose",
        "code_help": "query_rag",
        "question": "query_rag",
    }
    return routing.get(intent, "query_rag")

def route_after_rag(state: dict) -> str:
    """query_rag 节点判断是否继续出题

    Return:
        "generate_quiz" — practice / review（需要生成练习题）
        "finalize"      — question / code_help / diagnose（直接结束）
    """
    intent = state.get("intent", "question")
    return "generate_quiz" if intent in ("practice", "review") else "finalize"

def route_after_evaluate(state: dict) -> str:
    """evaluate_quiz 节点后判断重试或下一步

    逻辑：
    - FAIL 且 retry_count < 2 → 回到 generate_quiz 重试
    - PASS 或 retry_count >= 2：
        - review → review_plan
        - practice → finalize
    """
    eval_result = state.get("eval_result", {})
    retry_count = state.get("retry_count", 0)

    if eval_result.get("status") == "FAIL" and retry_count < 2:
        return "generate_quiz"  # 回 generate_quiz 重试

    if state.get("intent") == "review":
        return "review_plan"
    if state.get("intent") == "practice":
        return "create_plan"
    return "finalize"
