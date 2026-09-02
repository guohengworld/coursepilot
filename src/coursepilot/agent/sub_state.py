"""子图边界定义

按技能粒度把编排层拆成四个子图，每个子图用「三件套」描述边界：

    XxxInput   —— 只声明「从父图读什么」，是子图只见自己需要字段的载体。
    XxxOutput  —— 只声明「交回父图什么」，不含共享读字段与私有字段。
    XxxState   —— 子图内部完整状态 = Input ∪ Output ∪ 私有中间态。
"""
from typing import TypedDict


class DiagnoseInput(TypedDict):
    """诊断子图入接口：从父图读哪些字段。

    字段逐项对照 diagnose_node 的读取点（nodes.py:476）：
        user_id / course_id —— 聚合统计与知识点过滤的定位键；
        query —— _find_topic_kp 用它做特定知识点子树过滤；
        llm_calls —— 读-改-写累积 token 用量。
    """
    user_id: str
    course_id: str
    query: str
    llm_calls: list[dict]


class DiagnoseOutput(TypedDict):
    """诊断子图出接口：交回父图哪些字段。

    对照 diagnose_node 的返回值（正常/异常分支的并集）。
    """
    diagnosis: dict
    answer: str
    llm_calls: list[dict]
    error: str | None


class DiagnoseState(TypedDict):
    """诊断子图内部完整状态 = Input ∪ Output（无私有中间态）。"""
    user_id: str
    course_id: str
    query: str
    llm_calls: list[dict]
    diagnosis: dict
    answer: str
    error: str | None


class QuestionInput(TypedDict):
    """问答子图入接口：从父图读哪些字段。

    字段逐项对照两个通道的读取点：
      - query_rag_node（nodes.py:79）读 query / course_id / course_context /
        conversation / rolling_summary / user_profile / complexity / llm_calls；
      - agentic_rag_node（rag_agent.py:841）经 build_agent_messages 读
        course_context / user_profile / rolling_summary / conversation(recent_qa)
        / query，循环里读-改-写 llm_calls / agent_steps / tool_history。
    complexity 是子图内部双通道路由的判据（simple→query_rag / complex→agentic_rag）。
    """
    query: str
    course_id: str
    course_context: dict
    conversation: list[dict]
    rolling_summary: str
    user_profile: dict | None
    recent_qa: list[dict]          # build_agent_messages 的 conversation 兜底
    complexity: str                # simple / complex，路由判据
    llm_calls: list[dict]          # 读-改-写累积
    agent_steps: list[dict]        # agentic_rag 读-改-写
    tool_history: list[dict]       # agentic_rag 读-改-写


class QuestionOutput(TypedDict):
    """问答子图出接口：交回父图哪些字段。

    对照两个通道返回值的并集（query_rag_node 正常/异常分支 +
    agentic_rag_node 经 finalize_answer 的闭环契约）。
    """
    answer: str
    context: str
    retrieved_metadata: dict
    sources: list[dict]
    degraded_mode: bool            # agentic_rag guardrail 强制停止标记
    error: str | None
    llm_calls: list[dict]
    agent_steps: list[dict]
    tool_history: list[dict]
    evidence: list[str]


class QuestionState(TypedDict):
    """问答子图内部完整状态 = Input ∪ Output（无私有中间态）。

    双通道互斥（complexity 二选一），不存在并行写同一字段的冲突，
    故全部用默认覆盖（LastValue）reducer，与父图一致。
    """
    query: str
    course_id: str
    course_context: dict
    conversation: list[dict]
    rolling_summary: str
    user_profile: dict | None
    recent_qa: list[dict]
    complexity: str
    llm_calls: list[dict]
    agent_steps: list[dict]
    tool_history: list[dict]
    answer: str
    context: str
    retrieved_metadata: dict
    sources: list[dict]
    degraded_mode: bool
    error: str | None
    evidence: list[str]


class PracticeInput(TypedDict):
    """练习子图入接口：从父图读哪些字段。

    字段逐项对照内部五个节点的读取点：
      - get_mastery_node（nodes.py:405）读 user_id / course_id；
      - query_rag_node（nodes.py:79）读 query / course_id / complexity /
        course_context / conversation / rolling_summary / user_profile / llm_calls；
      - generate_quiz / evaluate_quiz / create_plan 读的 context / mastery /
        quiz_data / retry_count 均为本子图内部产物，不入 Input。
    """
    user_id: str
    course_id: str
    query: str
    complexity: str                # simple / complex，query_rag 快慢通道判据
    course_context: dict
    conversation: list[dict]
    rolling_summary: str
    user_profile: dict | None
    llm_calls: list[dict]          # 读-改-写累积


class PracticeOutput(TypedDict):
    """练习子图出接口：交回父图哪些字段。

    对照五个节点返回值的并集。quiz_data 必须交回——finalize_node（nodes.py:192）
    读它持久化到 agent_session.quiz_data，供 api/practice.py 后续出题/判题消费，
    否则练习流程断链。
    """
    answer: str
    context: str
    retrieved_metadata: dict
    sources: list[dict]
    quiz_data: dict
    llm_calls: list[dict]
    error: str | None


class PracticeState(TypedDict):
    """练习子图内部完整状态 = Input ∪ Output ∪ 私有中间态。

    私有：mastery（get_mastery→generate_quiz 传递）、eval_result / retry_count
    （evaluate_quiz→内部重试路由）。quiz_data 跨节点传递且需交回父图，故入 Output。
    """
    # Input
    user_id: str
    course_id: str
    query: str
    complexity: str
    course_context: dict
    conversation: list[dict]
    rolling_summary: str
    user_profile: dict | None
    llm_calls: list[dict]
    # Output
    answer: str
    context: str
    retrieved_metadata: dict
    sources: list[dict]
    quiz_data: dict
    error: str | None
    # 私有中间态
    mastery: dict
    eval_result: dict
    retry_count: int


class ReviewInput(TypedDict):
    """复习子图入接口：从父图读哪些字段 = PracticeInput + diagnosis。

    diagnosis 供 review_plan_node（nodes.py:601）读取；review 路径不经 diagnose_node，
    父图该字段恒为空 → 节点内部实时重查（与原父图行为一致）。列为 Input 是为
    未来「基于诊断结果发起复习」预留投影通道。
    """
    user_id: str
    course_id: str
    query: str
    complexity: str
    course_context: dict
    conversation: list[dict]
    rolling_summary: str
    user_profile: dict | None
    llm_calls: list[dict]
    diagnosis: dict


class ReviewOutput(TypedDict):
    """复习子图出接口 = PracticeOutput + review_plan。

    review_plan 是本子图的产物记录（写入 review_plans 表），交回父图保留在
    checkpoint；finalize_node 不读它，但保留与旧父图「review_plan 进状态」的一致性。
    """
    answer: str
    context: str
    retrieved_metadata: dict
    sources: list[dict]
    quiz_data: dict
    llm_calls: list[dict]
    error: str | None
    review_plan: dict


class ReviewState(TypedDict):
    """复习子图内部完整状态 = Input ∪ Output ∪ 私有中间态（mastery/eval_result/retry_count）。"""
    # Input
    user_id: str
    course_id: str
    query: str
    complexity: str
    course_context: dict
    conversation: list[dict]
    rolling_summary: str
    user_profile: dict | None
    llm_calls: list[dict]
    diagnosis: dict
    # Output
    answer: str
    context: str
    retrieved_metadata: dict
    sources: list[dict]
    quiz_data: dict
    error: str | None
    review_plan: dict
    # 私有中间态
    mastery: dict
    eval_result: dict
    retry_count: int
