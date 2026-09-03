"""条件路由函数。

提供 4 个路由函数供 graph.py 的条件边使用：
  - route_by_intent: classify 后分发
  - route_after_rag: query_rag / agentic_rag 后分发
  - route_after_evaluate: evaluate_quiz 后分发（含重试逻辑）
  - route_after_review: human_review 后分发
"""
import logging

from coursepilot.agent.state_models import VALID_INTENTS
from coursepilot.config import settings
from coursepilot.rag.config import config as rag_config

logger = logging.getLogger(__name__)


def route_by_intent(state: dict) -> str:
    """classify 节点后根据 intent + complexity 路由

    复杂问题（complex question）走 Agentic RAG 节点（LLM 自主决策），
    简单问题走快速通道（query_rag）。

    orch_route_fallback=True 时，四类「无法进入业务分支」的输入统一收口
    "fallback"（区别于现状的四合一静默兜底）：
      - classify 节点异常降级（classify_degraded=True）
      - intent == "none"（寒暄 / 离题 / 欠指定，classify 正常输出）
      - intent 缺失 / ""（UNCLASSIFIED）
      - intent ∉ VALID_INTENTS（未知值）
    none 单独显式分支，不依赖 VALID_INTENTS 是否含 none：白名单只筛未知值，
    两者互不耦合。flag 关闭时保持原行为（末行四合一兜底）。

    Returns:
        "fallback"     — 兜底收口（flag 开启时的 none/未知/缺失/降级）
        "human_review" — 需要审批的路径
        "get_mastery"  — practice / review（需要掌握度）
        "diagnose"     — diagnose
        "question"     — question（子图隔离开启后，复杂度分发移入子图）
        "agentic_rag"  — complex question（LLM 自主 ReAct 循环）
        "query_rag"    — simple question（快速通道）
    """
    intent = state.get("intent", "")
    complexity = state.get("complexity", "simple")

    if settings.orch_route_fallback:
        # 收口判断先于一切业务分支：classify 异常时 intent 被写成 "question"，
        # classify_degraded 必须最先读，否则降级请求会误入问答路径。
        if state.get("classify_degraded") or intent == "none" or intent not in VALID_INTENTS:
            return "fallback"

    if intent in ("practice", "review"):
        return "human_review"      # 需要教师审批
    elif intent == "diagnose":
        return "diagnose"          # 诊断只读，无需审批
    elif intent == "question":
        if settings.orch_subgraph_question:
            return "question"      # 子图接管，复杂度分发移入子图
        if complexity == "complex" and rag_config.enable_routing:
            return "agentic_rag"   # 复杂 → LLM 自主决策
        return "query_rag"         # 简单 → 快速通道
    # intent 未识别：按简单问答兜底
    return "question" if settings.orch_subgraph_question else "query_rag"


def route_after_review(state: dict) -> str:
    """human_review 节点后的路由

    审批通过后按 intent 分发：
      - practice → "practice" 子图（flag 开启）或 "get_mastery" 旧链路
      - review   → "review" 子图（flag 开启）或 "get_mastery" 旧链路
      - rejected / 其他 → 收口 finalize
    """
    if state.get("human_review_result") == "rejected":
        return "finalize"
    intent = state.get("intent", "")
    if intent == "practice":
        return "practice" if settings.orch_subgraph_practice else "get_mastery"
    if intent == "review":
        return "review" if settings.orch_subgraph_review else "get_mastery"
    return "finalize"


def route_after_rag(state: dict) -> str:
    """query_rag / agentic_rag 节点后判断是否继续出题

    Return:
        "generate_quiz" — practice / review（需要生成练习题）
        "finalize"      — question / diagnose（直接结束）
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

    if eval_result.get("status") == "FAIL" and retry_count < 3:
        return "generate_quiz"  # 回 generate_quiz 重试

    if state.get("intent") == "review":
        return "review_plan"
    if state.get("intent") == "practice":
        return "create_plan"
    return "finalize"
