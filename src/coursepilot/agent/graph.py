"""LangGraph 状态机构建

P1 拓扑（复杂问题走 Agentic RAG，CRAG 补搜循环已删除；HITL 审批已移除，
practice/review 无审批直通，classify 后直接分发到 get_mastery 旧链路前缀）：
    START → build_context → classify → [route_by_intent]
      ├─ query_rag → [route_after_rag] ─┬─ question 收口 → finalize → END
      │                                 └─ practice/review → generate_quiz
      │                                      → evaluate_quiz → [route_after_evaluate]:
      │                                          FAIL+retry<2 → generate_quiz (重试)
      │                                          PASS+practice → create_plan → finalize
      │                                          PASS+review → review_plan → finalize
      ├─ agentic_rag → [route_after_rag] → (同上)   # complex question：LLM 自主 ReAct 循环
      ├─ get_mastery → query_rag → (同上)           # practice/review 旧链路前缀
      ├─ diagnose → finalize → END
      └─ fallback → finalize → END                  # 兜底收口（flag 开启时可达）

机制 3 子图隔离（Strangler Fig 渐进切换，flag 默认 False 行为零变化）：
    - orch_subgraph_diagnose=True  → diagnose 抽为子图
    - orch_subgraph_question=True → query_rag/agentic_rag 抽为 question 子图
      （classify 分发到 "question"，复杂度分发移入子图内部）
    - orch_subgraph_practice=True → get_mastery→query_rag→generate_quiz→evaluate_quiz→
      create_plan 抽为 practice 子图（重试循环移入子图内部）
    - orch_subgraph_review=True   → 同 practice，末步换成 review_plan

四个 flag 相互独立：任一相关 flag 关闭，对应旧节点/旧边即保留。旧节点是否保留
由 need_* 布尔推导（见 build_agent_graph 开头），保证单开任一 flag 时拓扑仍连通。
"""
import logging

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg.rows import dict_row

from coursepilot.agent.nodes import (
    build_context_node,
    classify_node,
    create_plan_node,
    diagnose_node,
    evaluate_quiz_node,
    fallback_node,
    finalize_node,
    generate_quiz_node,
    get_mastery_node,
    query_rag_node,
    review_plan_node,
)
from coursepilot.agent.rag_agent import agentic_rag_node
from coursepilot.agent.routing import (
    route_after_evaluate,
    route_after_rag,
    route_by_intent,
)
from coursepilot.agent.state import AgentState, InputState, OutputState
from coursepilot.agent.subgraphs import (
    build_diagnose_subgraph,
    build_practice_subgraph,
    build_question_subgraph,
    build_review_subgraph,
)
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
        logger.info("AsyncPostgresSaver 初始化完成（pipeline 已禁用）")
    return _saver


async def build_agent_graph():
    """构建并编译 Agent 状态图

    :returns: CompiledStateGraph: 可直接调用 .ainvoke() 的编译图
    """
    # 三 schema：AgentState 为节点可见的完整数据面，
    # InputState / OutputState 仅在图的入口与出口生效（过滤输入、约束输出）。
    builder = StateGraph(
        AgentState,
        input_schema=InputState,
        output_schema=OutputState,
    )

    # 机制 3 flag 本地快照（避免多次读 settings，且便于推导旧节点去留）
    sg_diagnose = settings.orch_subgraph_diagnose
    sg_question = settings.orch_subgraph_question
    sg_practice = settings.orch_subgraph_practice
    sg_review = settings.orch_subgraph_review

    # 旧节点是否仍需保留（任一依赖它的 flag 关闭即保留）
    # query_rag 供 question（simple）与 practice/review（get_mastery 后检索）共用
    need_query_rag = (not sg_question) or (not sg_practice) or (not sg_review)
    # get_mastery/generate_quiz/evaluate_quiz 是 practice/review 旧链路的共享前缀
    need_practice_chain = (not sg_practice) or (not sg_review)
    need_create_plan = not sg_practice
    need_review_plan = not sg_review

    # ── 注册节点 ──────────────────────────────────────
    builder.add_node("build_context", build_context_node)
    builder.add_node("classify", classify_node)
    # 路由兜底收口节点：常驻注册。flag orch_route_fallback 关闭时
    # route_by_intent 不返 "fallback"，该节点无入边、不可达（编译不报错）。
    builder.add_node("fallback", fallback_node)
    builder.add_node("finalize", finalize_node)
    # 诊断：子图 or 节点函数
    builder.add_node(
        "diagnose",
        build_diagnose_subgraph() if sg_diagnose else diagnose_node,
    )
    # 问答：question 子图 or agentic_rag 节点（query_rag 单独按需注册）
    if sg_question:
        builder.add_node("question", build_question_subgraph())
    else:
        builder.add_node("agentic_rag", agentic_rag_node)
    # 练习 / 复习：子图 or 旧链路节点
    if sg_practice:
        builder.add_node("practice", build_practice_subgraph())
    if sg_review:
        builder.add_node("review", build_review_subgraph())
    if need_query_rag:
        builder.add_node("query_rag", query_rag_node)
    if need_practice_chain:
        builder.add_node("get_mastery", get_mastery_node)
        builder.add_node("generate_quiz", generate_quiz_node)
        builder.add_node("evaluate_quiz", evaluate_quiz_node)
    if need_create_plan:
        builder.add_node("create_plan", create_plan_node)
    if need_review_plan:
        builder.add_node("review_plan", review_plan_node)

    # ── 连接边 ────────────────────────────────────────
    builder.add_edge(START, "build_context")
    builder.add_edge("build_context", "classify")

    # classify → intent + complexity 分发
    classify_targets = {
        "diagnose": "diagnose",
        "fallback": "fallback",
    }
    if sg_question:
        classify_targets["question"] = "question"
    else:
        classify_targets["query_rag"] = "query_rag"
        classify_targets["agentic_rag"] = "agentic_rag"
    # practice/review 直通（HITL 审批已移除）：子图 flag 开启直达子图，
    # 否则经旧链路前缀 get_mastery（need_practice_chain 时节点在册）
    if sg_practice:
        classify_targets["practice"] = "practice"
    if sg_review:
        classify_targets["review"] = "review"
    if need_practice_chain:
        classify_targets["get_mastery"] = "get_mastery"
    builder.add_conditional_edges("classify", route_by_intent, classify_targets)

    # 路由兜底 → finalize（flag 关闭时该边不可达）
    builder.add_edge("fallback", "finalize")

    # diagnose → finalize
    builder.add_edge("diagnose", "finalize")

    # 问答出口：question 子图 → finalize；agentic_rag → 按是否仍有旧链路决定出口
    if sg_question:
        builder.add_edge("question", "finalize")
    elif need_practice_chain:
        builder.add_conditional_edges("agentic_rag", route_after_rag, {
            "generate_quiz": "generate_quiz",
            "finalize": "finalize",
        })
    else:
        # practice/review 已抽子图，agentic_rag 只服务 complex question
        builder.add_edge("agentic_rag", "finalize")

    # 旧 practice/review 链路（任一 flag 关闭时）
    if need_practice_chain:
        builder.add_edge("get_mastery", "query_rag")
        builder.add_edge("generate_quiz", "evaluate_quiz")
        eval_targets = {"generate_quiz": "generate_quiz", "finalize": "finalize"}
        if need_create_plan:
            eval_targets["create_plan"] = "create_plan"
        if need_review_plan:
            eval_targets["review_plan"] = "review_plan"
        builder.add_conditional_edges("evaluate_quiz", route_after_evaluate, eval_targets)
    if need_create_plan:
        builder.add_edge("create_plan", "finalize")
    if need_review_plan:
        builder.add_edge("review_plan", "finalize")

    # query_rag 出口：question（收口）或旧 practice/review（继续出题）
    if need_query_rag:
        if need_practice_chain:
            builder.add_conditional_edges("query_rag", route_after_rag, {
                "generate_quiz": "generate_quiz",
                "finalize": "finalize",
            })
        else:
            # practice/review 已抽子图，query_rag 只服务 simple question
            builder.add_edge("query_rag", "finalize")

    # practice / review 子图 → finalize
    if sg_practice:
        builder.add_edge("practice", "finalize")
    if sg_review:
        builder.add_edge("review", "finalize")

    builder.add_edge("finalize", END)

    # checkpointer：AsyncPostgresSaver
    saver = await _get_saver()
    graph = builder.compile(checkpointer=saver)
    return graph
