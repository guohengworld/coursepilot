"""LangGraph Agent 状态定义

所有节点共享的 TypeDict 状态
Phase 1仅使用线性路径的字段，Phase 2+ 在此基础上扩展
"""
from typing import TypedDict


class AgentState(TypedDict):
    """Agent 执行过程中所有节点共享的状态"""
    # 输入
    query: str              # 用户原始消息
    course_id: str          # 课程 ID(UUID 字符串)
    user_id: str            # 用户 ID(UUID 字符串)

    # 会话元数据
    session_id: str         # agent_sessions.id
    messages: list[dict]    # 历史消息 [{role, content}]
    conversation: list[dict]  # 完整多轮对话（L1 + 已压缩指针）
    rolling_summary: str    # L2 滚动摘要

    # build_context 节点输出
    course_context: dict    # {name, textbook, chapters} 供 System Prompt
    user_profile: dict | None   # 学生画像概要（若有）
    recent_qa: list[dict]       # 最近 5 条问答

    # classify 节点输出
    intent: str                 # question / practice / diagnose / review / code_help
    complexity: str             # simple / complex（P0: Agentic RAG 智能路由）

    # query_rag 节点输出
    context: str                # RAG 检索到的教材上下文（XML 格式）
    retrieved_metadata: dict    # {query_raw, query_rewritten, source_kp_paths, ...}

    # finalize 节点输出
    answer: str                 # 最终回答
    sources: list[dict]         # [{source, kp_path, page_ref}]
    token_count: int            # LLM token 消耗

    # 控制字段
    error: str | None           # 节点执行错误信息

    # P1: Agentic RAG 质检与多轮检索
    retrieval_retry_count: int  # 当前补搜轮数（0~complex_max_rounds）
    sufficiency: dict | None    # check_sufficiency 输出:
                                #   {"sufficient": bool, "missing_info": str, ...}

    # P2: 查询分解与降级生成
    sub_queries: list[dict]     # decompose 输出: [{"id":1,"query":"...","target_concept":"..."}]
    degraded_mode: bool         # True=质检全失败，降级生成（含免责声明）

    # 掌握度
    mastery: dict               # get_mastery 输出: {"mastery_level": {...}, "weak_kps": [...]}

    # 练习
    quiz_data: dict             # generate_quiz 输出: {"questions": [...]}
    eval_result: dict           # evaluate_quiz 输出
    retry_count: int            # evaluate 重试计数器（0/1/2）

    # 诊断
    diagnosis: dict             # diagnose 输出

    # 复习计划
    review_plan: dict           # review_plan 输出

    # Token 用量追踪（P5 可观测：含 context_budget / layer_tokens / cache_hit_estimated）
    llm_calls: list[dict]

    # 上下文预算与可观测快照（供 admin 控制台展示）
    context_budget: dict | None
    layer_tokens: dict | None
    cache_hit_estimated: dict | None
    compaction_count: int

    # 人工审核结果
    human_review_result: str | None  # None / "approved" / "rejected"
