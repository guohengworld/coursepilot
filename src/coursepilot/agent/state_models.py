"""Agent 状态中「LLM 产出」字段的结构契约。

## 定位

本模块只定义「结构契约」：用 Pydantic 描述 LLM 产出字段该长什么样，
供类型提示、结构文档化与后续写入点校验复用。**它不是运行时数据结构**——
state 运行时值仍保持 dict（见 state.py 中字段注解为 dict 的原因），
LangGraph 不会对 TypedDict 字段做运行时校验，模型实例会让下游
state.get(...) 失效并放大 checkpoint 序列化体积。

在 LLM 解析点显式接入 model_validate（提交 2b）时，由各 skill 在 json.loads
之后调用对应模型的 model_validate + model_dump；本模块提供
validation_error_brief 用于日志摘要，不直接参与 state 写入。

## 划分判据

一个状态字段该用 Pydantic 建模，还是保持原生类型注解，按四问判定：

  1. 生产者是否在进程外？（LLM / 用户输入 / 第三方接口 / 历史 checkpoint）
  2. 结构是否复杂到容易写错？（嵌套 ≥2 层、key 多、含枚举或数值约束）
  3. 是否有 ≥2 个跨节点消费者？
  4. 校验失败的代价是什么？（静默降级 / 崩溃 / 走错分支）

四问全中的字段在此建模：quiz_data / eval_result / diagnosis / review_plan。
程序内构造的固定结构（llm_calls / sources / agent_steps / tool_history /
evidence）保持 list[dict] 注解——它们不跨越信任边界，静态检查足够。
"""
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError


# ── 受控枚举：LLM 产出但直接决定控制流 ──────────────────────────────
# intent 决定路由分支，complexity 决定是否走 Agentic RAG，
# 取值必须在写入点收敛（classify_intent 已做白名单校验，此处提供共享常量）。

Intent = Literal["question", "practice", "diagnose", "review", "none"]
Complexity = Literal["simple", "complex"]
# 未分类/异常时的占位值，不包含在上面两个 Literal 里（state 初始化为 ""）
UNCLASSIFIED = ""

VALID_INTENTS: frozenset[str] = frozenset(get_args(Intent))
VALID_COMPLEXITIES: frozenset[str] = frozenset(get_args(Complexity))


class QuizQuestion(BaseModel):
    """单道练习题（generate_quiz 的 questions[] 元素）"""

    model_config = ConfigDict(extra="allow")

    question_text: str = ""
    question_type: str = ""
    options: dict[str, Any] = Field(default_factory=dict)
    correct_answer: str = ""
    explanation: str = ""
    kp_path: str = ""


class QuizData(BaseModel):
    """generate_quiz 产出 → state["quiz_data"]

    下游 create_plan_node 用下标 q["question_text"] 读取，结构错误会直接崩溃。
    """

    model_config = ConfigDict(extra="allow")

    questions: list[QuizQuestion] = Field(default_factory=list)


class EvalFeedback(BaseModel):
    """evaluate_quiz 的 feedback 子结构"""

    model_config = ConfigDict(extra="allow")

    correctness_issues: list[str] = Field(default_factory=list)
    coverage_issues: list[str] = Field(default_factory=list)
    hallucination_issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    """evaluate_quiz 产出 → state["eval_result"]

    status 决定 route_after_evaluate 是否重试，是控制流字段。
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["PASS", "FAIL"] = "PASS"
    score: float = 0.0
    feedback: EvalFeedback = Field(default_factory=EvalFeedback)


class KpStat(BaseModel):
    """单个知识点的练习统计（diagnosis.kp_stats 的 value）"""

    total: int = 0
    correct: int = 0
    rate: float = 0.0


class DiagnosisReport(BaseModel):
    """diagnose 产出 → state["diagnosis"]

    结构由 DB 聚合固定（weak_kps / kp_stats / summary / total_practiced /
    overall_rate），llm_analysis 与 recommendations 由 LLM 追加。
    跨 diagnose_node → review_plan_node → finalize 三个节点消费。
    """

    model_config = ConfigDict(extra="allow")

    weak_kps: list[str] = Field(default_factory=list)
    kp_stats: dict[str, KpStat] = Field(default_factory=dict)
    summary: str = ""
    total_practiced: int = 0
    overall_rate: float = 0.0
    llm_analysis: str = ""
    recommendations: str = ""


class ReviewPlanItem(BaseModel):
    """复习计划条目（review_plan.items[] 元素）"""

    model_config = ConfigDict(extra="allow")

    kp_path: str = ""
    priority: int = 0
    reason: str = ""
    status: str = "pending"


class ReviewPlanData(BaseModel):
    """review_plan 产出 → state["review_plan"]

    plan_id 由 skill 在持久化后回填，不来自 LLM。
    """

    model_config = ConfigDict(extra="allow")

    items: list[ReviewPlanItem] = Field(default_factory=list)
    total_count: int = 0
    plan_summary: str = ""
    plan_id: str = ""


class DiagnosisAnalysis(BaseModel):
    """generate_llm_analysis 的 LLM 输出（两个自由文本字段）"""

    model_config = ConfigDict(extra="allow")

    analysis: str = ""
    recommendations: str = ""


# ── 降级值：与既有降级行为逐字一致，不得随意改动 ─────────────────────
# 这些常量是「行为基线」，重构期必须保持与现状完全一致，包括其中的历史拼写。

QUIZ_FALLBACK: dict[str, Any] = {"questions": []}

# evaluate_quiz 解析失败时的既有降级值。
# 注意 feedback 里的 "suggestion" 是历史单数拼写（有效路径为 suggestions），
# 此处原样保留以保持行为不变，不做「顺手修正」。
EVAL_FALLBACK: dict[str, Any] = {
    "status": "PASS",
    "score": 0.8,
    "feedback": {"suggestion": ["审核结果解析失败"]},
}


# ── 校验辅助 ──────────────────────────────────────────────────────
# 供 skill 层在 LLM 解析点接入 model_validate 时打印日志摘要，
# 避免 pydantic 的多行错误对象刷屏。

def validation_error_brief(e: Exception) -> str:
    """把校验/解析异常压缩为单行日志摘要。

    对 ValidationError 取首个错误的 loc+msg；其他异常截断 str。
    返回值只用于日志，不参与控制流。
    """
    if isinstance(e, ValidationError):
        errs = e.errors()
        if errs:
            first = errs[0]
            loc = ".".join(str(x) for x in first.get("loc", ()))
            return f"{loc}: {first.get('msg', '')}"
    return str(e)[:200]
