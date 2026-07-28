"""条件路由函数（Phase 2 启用）。

提供 5 个路由函数供 graph.py 的条件边使用：
  - route_by_intent: classify 后分发（P1 增强：按复杂度路由）
  - route_after_check: check_sufficiency 后分发（P1 新增）
  - route_after_rag: query_rag / synthesize 后分发
  - route_after_evaluate: evaluate_quiz 后分发（含重试逻辑）
  - route_after_review: human_review 后分发
"""
import logging

from coursepilot.rag.config import config as rag_config

logger = logging.getLogger(__name__)


def route_by_intent(state: dict) -> str:
    """classify 节点后根据 intent + complexity 路由

    P1 增强：复杂问题走 Agentic RAG 循环（retrieve → check → synthesize），
    简单问题走快速通道（query_rag）。

    Returns:
        "human_review" — 需要审批的路径
        "get_mastery"  — practice / review（需要掌握度）
        "diagnose"     — diagnose
        "retrieve"     — complex question / code_help（进入质检循环）
        "query_rag"    — simple question / code_help（快速通道）
    """
    intent = state.get("intent", "")
    complexity = state.get("complexity", "simple")

    if intent in ("practice", "review"):
        return "human_review"      # 需要教师审批
    elif intent == "diagnose":
        return "diagnose"          # 诊断只读，无需审批
    elif intent in ("question", "code_help"):
        if complexity == "complex" and rag_config.enable_routing:
            return "retrieve"      # 复杂 → 进入质检循环
        return "query_rag"         # 简单 → 快速通道
    return "query_rag"


def route_after_review(state: dict) -> str:
    """human_review 节点后的路由"""
    if state.get("human_review_result") == "rejected":
        return "finalize"
    intent = state.get("intent", "")
    if intent in ("practice", "review"):
        return "get_mastery"
    return "finalize"


def route_after_rag(state: dict) -> str:
    """query_rag / synthesize 节点后判断是否继续出题

    Return:
        "generate_quiz" — practice / review（需要生成练习题）
        "finalize"      — question / code_help / diagnose（直接结束）
    """
    intent = state.get("intent", "question")
    return "generate_quiz" if intent in ("practice", "review") else "finalize"


def route_after_check(state: dict) -> str:
    """check_sufficiency 节点后判断：补搜 or 生成

    逻辑：
    - 质检不足 且 retry_count < complex_max_rounds → "retrieve"（继续补搜）
    - 质检通过 或 已达最大轮数 → "synthesize"（生成答案）

    Returns:
        "retrieve"   — 回到 retrieve_node 继续补搜
        "synthesize" — 进入 synthesize_node 生成答案
    """
    sufficiency = state.get("sufficiency", {})
    retry_count = state.get("retrieval_retry_count", 0)
    max_rounds = rag_config.complex_max_rounds

    if sufficiency.get("sufficient", True):
        return "synthesize"

    if retry_count < max_rounds:
        logger.debug("质检不足 (retry=%d/%d)，继续补搜", retry_count, max_rounds)
        return "retrieve"

    logger.debug("已达最大补搜轮数 %d，强制生成", max_rounds)
    return "synthesize"


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

    if eval_result.get("status") == "FAIL" and retry_count < 3:
        return "generate_quiz"  # 回 generate_quiz 重试

    if state.get("intent") == "review":
        return "review_plan"
    if state.get("intent") == "practice":
        return "create_plan"
    return "finalize"
