"""LangGraph Agent 状态定义

状态按「生命周期 + 消费边界」拆成三层，对应 LangGraph 的三种 schema：

    InputState  —— 图的输入契约：调用方（api/agent.py 的 ainvoke/astream）
                   允许传入的字段集合。
    OutputState —— 图的输出契约：图正常结束后 ainvoke() 返回的字段集合。
    AgentState  —— 父图运行期间的完整内部状态：继承 Input + Output，
                   再补充只在节点之间流转的中间字段。

三者不是三个独立对象，而是同一份 state 的三个视角：节点永远只面对
AgentState（完整数据面），Input/Output 只在图的入口与出口生效。

运行时语义（LangGraph 实际行为，已实测确认）：
    - input_schema 中未声明的 key 会在入口被静默丢弃，不报错；
    - output_schema 会在出口过滤返回值，只保留声明的 key；
    - checkpoint 持久化的始终是 AgentState 全量，不受 output_schema 影响。

因此 InputState 必须与调用方实际传入的字段集严格对齐，否则会造成
「传了但没进图」的静默丢字段。当前 InputState 与 api/agent.py 的
initial_state 完全对齐，其中的节点产物类字段属于历史遗留的启动占位，
后续应作为独立的契约收窄提交处理，不混进本次重构。

嵌套字段的结构契约见 state_models.py：那里为 LLM 产出的字段
（quiz_data / eval_result / diagnosis / review_plan）定义了 Pydantic 模型，
仅作结构与类型提示用，本模块不承担运行时校验（在写入点接入校验
属后续独立提交，当前尚未发生）。这里的注解保持 dict——因为运行时值
必须是 dict，否则下游 70 余处 state.get(...) 会失效、checkpoint
序列化体积也会显著变大。注解与运行时不一致比没有注解更危险，
故不把字段类型写成模型类。
"""
from typing import TypedDict


class InputState(TypedDict):
    """图的输入契约：调用方在图外装配完成、允许进入图的数据

    判断归属的标准不是「是否来自 HTTP 请求体」，而是：
    进入 START 之前，这个值是否已经由图外准备完成。
    """
    # 请求要素（api/agent.py 由 ChatRequest + 认证用户装配）
    query: str                      # 用户原始消息（ChatRequest.message）
    course_id: str                  # 课程 ID(UUID 字符串)
    user_id: str                    # 用户 ID(UUID 字符串)
    user_role: str                  # 认证用户角色（student/teacher/super），图外鉴权层确定
    session_id: str                 # agent_sessions.id

    # 会话记忆恢复（api/agent.py 从 DB 读取后装配）
    messages: list[dict]            # 历史消息 [{role, content}]
    conversation: list[dict]        # 完整多轮对话（L1 + 已压缩指针）
    rolling_summary: str            # L2 滚动摘要

    # 以下为调用方的启动占位字段（节点产物的空初始值）。
    # 保留原因：input_schema 会静默丢弃未声明 key，删减会改变 checkpoint
    # 首个快照的 key 集合，属于行为变更。收窄需独立提交 + 测试验证。
    course_context: dict            # {name, textbook, chapters} 供 System Prompt
    user_profile: dict | None       # 学生画像概要（若有）
    recent_qa: list[dict]           # 最近 5 条问答
    # 取值集合与 state_models.Intent / Complexity 同源，写入点已做白名单校验；
    # 未分类时为 ""（UNCLASSIFIED），不计入 Literal。
    intent: str                     # question / practice / diagnose / review / none
    complexity: str                 # simple / complex（Agentic RAG 智能路由）
    context: str                    # RAG 检索到的教材上下文（XML 格式）
    retrieved_metadata: dict        # {query_raw, query_rewritten, source_kp_paths, ...}
    answer: str                     # 最终回答
    sources: list[dict]             # [{source, kp_path, page_ref}]
    token_count: int                # LLM token 消耗
    llm_calls: list[dict]           # Token 用量追踪
    context_budget: dict | None     # 上下文预算快照（admin 控制台）
    layer_tokens: dict | None       # 分层 token 用量
    cache_hit_estimated: dict | None
    compaction_count: int
    error: str | None               # 节点执行错误信息


class OutputState(TypedDict):
    """图的输出契约：ainvoke() 返回给图调用方的字段

    字段选择：覆盖现有测试依赖的最终可观察字段（intent/complexity/
    answer/sources/token_count/llm_calls）+ 运行态标记（degraded_mode/
    error）+ 路由兜底标记（fallback_reason/routing_notes）。
    context / retrieved_metadata / quiz_data
    等中间与观测字段留在内部状态，不对外公开。
    """
    # 最终产物
    answer: str
    sources: list[dict]
    token_count: int

    # 路由结论（外部需观察 classify 的判定结果）
    intent: str
    complexity: str

    # 路由兜底（fallback_node 写入；非兜底路径恒为缺省值）
    fallback_reason: str | None     # none / unclassified / classify_degraded
    routing_notes: str | None       # 兜底来源的补充说明（原始 intent / 异常信息，观测用）

    # 运行态标记
    degraded_mode: bool             # True=guardrail 触发，降级生成（含免责声明）
    error: str | None

    # 可观测
    llm_calls: list[dict]


class AgentState(InputState, OutputState):
    """父图运行期间的完整内部状态（对应官方 OverallState 角色）

    节点签名、路由函数、checkpoint 持久化的都是这一份状态。
    注意：本类不描述图拓扑、节点连接关系或子图划分——那些由 graph.py
    的 add_node / add_edge / add_conditional_edges 定义。
    """
    # 掌握度（DB 聚合，结构由代码固定）
    mastery: dict                   # get_mastery 输出: {"mastery_level": {...}, "weak_kps": [...]}

    # 路由兜底（classify_node 异常分支写入；route_by_intent 据此收口 fallback）
    classify_degraded: bool

    # 练习（LLM 产出，结构契约见 state_models.QuizData）
    quiz_data: dict                 # {"questions": [QuizQuestion]}
    eval_result: dict               # EvalResult: {"status", "score", "feedback"}
    retry_count: int                # evaluate 重试计数器

    # 诊断（DB 聚合 + LLM 分析，结构契约见 state_models.DiagnosisReport）
    diagnosis: dict

    # 复习计划（LLM 产出 + skill 回填 plan_id，见 state_models.ReviewPlanData）
    review_plan: dict

    # Agentic RAG（LLM 自主决策循环的决策轨迹）
    agent_steps: list[dict]         # Agent 决策轨迹: [{tool, args}]
    tool_history: list[dict]        # 工具执行历史: [{tool, args, result_summary}]
    evidence: list[str]             # 多轮检索累积的证据块（供 finalize 合并与调试）
