> 状态：设计 v4.0，LangGraph 多 Agent 架构已实现。
> 代码对应：`src/coursepilot/agent/`、`src/coursepilot/api/agent.py`。
> 最后更新：2026-08-01

# CoursePilot Agent 架构设计文档

> **文档版本**：v4.0（LangGraph 有向状态图）  
> **设计日期**：2026年6月28日  
> **适用项目**：CoursePilot AI 教学助手  
> **技术栈**：FastAPI + SQLAlchemy(async) + PostgreSQL + Milvus + DeepSeek + BGE-M3 + **LangGraph**  
> **设计原则**：Workflow 为骨架（确定性流程编排），Harness 思想为肌肉（验证、护栏、追踪、记忆）

---

## 设计理念：Workflow + Harness

**Workflow 是骨架，Harness 是肌肉，两者不矛盾。**

- **Workflow**（LangGraph 有向状态图）：教学场景意图有限可枚举（5 种），执行路径确定，用声明式状态图而非通用 Planner 动态规划
- **Harness 思想**：体现在 workflow 的节点设计中——关键步骤内嵌独立验证（不护短）、确定性护栏、每步追踪、人类接管点

```
LangGraph StateGraph（确定性流程编排）
    │
    ├── 节点: 查掌握度  ──┐
    ├── 节点: RAG检索   ──┤
    ├── 节点: 生成练习   ──┼── Harness 思想体现在这里：
    ├── 节点: 验证答案   ──┤    ├── 生成-验证分离（条件边 + 重试循环）
    ├── 节点: 生成计划   ──┤    ├── 确定性护栏（不靠LLM判断）
    └── 节点: 更新画像   ──┘    ├── 每步自动追踪（LangGraph native tracing）
                                  ├── 断点恢复（PostgresSaver checkpoint）
                                  └── 人类接管（interrupt()）
```

### 为什么选 LangGraph

| 维度 | 手写编排 | LangGraph |
|------|---------|-----------|
| 状态管理 | 手动传递 `step_results` + 模板变量解析 | TypedDict 状态对象，节点自动读写 |
| 条件分支 | `if-else` 手写 | `add_conditional_edges` 声明式 |
| 重试循环 | `while retry < 2` 手写计数器 | 图的边天然支持循环 |
| 持久化 | 手写 JSONB + 恢复逻辑 | `PostgresSaver` 自动 checkpoint，断点恢复 |
| 可观测性 | 手写 Tracer | 内置 tracing + LangSmith 可视化 |
| 人工介入 | 手写暂停 + 状态存 DB | `interrupt()` 原生支持 |
| 行业背书 | 无 | Deep Agents 教程明确用 LangGraph 作为 Runtime 层（AI HOT 2026.6.20 精选） |

---

## 一、Agent 是干什么的

### 1.1 业务定位

CoursePilot Agent 是一个**能自主执行教学任务的 AI 教学系统**。它不是在 RAG 上加个聊天框，而是用 LangGraph 有向状态图编排完整的教学工作流——模型是 CPU，Harness 是操作系统。

**当前状态**：RAG 问答系统（学生提问 → 检索 → 生成回答）

**目标状态**：教学 Agent（理解学生意图 → 路由到对应 workflow → 节点逐步执行 → 关键步骤验证 → 持续改进）

| 学生需求 | 现在 | Agent 化后 |
|---------|------|-----------|
| "帮我复习二叉树" | 返回文本 | 查掌握度 → 检索薄弱点 → 生成针对性练习 → 验证答案 → 更新画像 → 生成复习计划 |
| "出一套树结构练习" | 不支持 | 基于知识点树出题 → 独立验证答案 key → 学生作答 → 批改 → 归因错误知识点 |
| "我上次错的那题再讲讲" | 不记得 | 从长期记忆中检索历史 → 定位错误 → 关联知识点 → 重新讲解 |
| "这学期哪些没掌握" | 不支持 | 聚合全学期练习记录 → 按知识点树统计正确率 → 生成诊断报告 |
| "这段代码为什么报错" | 不支持 | 代码沙箱运行 → 错误分析 → 关联课程知识点 → 给出解释 |
| "制定下周复习计划" | 不支持 | 学情诊断 → 知识点优先级排序 → 生成分天计划 → 存入复习计划表 |

### 1.2 LangGraph 有向状态图架构

用一张有向状态图编排所有教学 workflow。意图分类是一个节点，分类后通过条件边路由到不同的子图路径。

```
                    ┌─────────────┐
                    │  classify   │ 意图分类节点
                    │  (意图路由)  │
                    └──────┬──────┘
                           │ 条件边
        ┌──────────┬───────┼───────┬──────────┐
        ↓          ↓       ↓       ↓          ↓
   ┌────────┐ ┌────────┐ ┌────┐ ┌──────┐ ┌────────┐
   │question│ │practice│ │diag│ │review│ │code_help│
   │ 子图   │ │ 子图   │ │子图│ │ 子图 │ │ 子图   │
   └────┬───┘ └────┬───┘ └─┬──┘ └──┬───┘ └────┬───┘
        │          │       │       │          │
        └──────────┴───────┴───────┴──────────┘
                           │
                    ┌──────┴──────┐
                    │  finalize   │ 收尾节点
                    │  (更新记忆)  │
                    └─────────────┘
```

**review 子图（含验证循环）**：

```
get_mastery → query_rag → generate_quiz → evaluate_quiz
                                           │ 条件边
                                    ┌──────┴──────┐
                                    ↓             ↓
                              (FAIL, retry<2)  (PASS)
                                    │             │
                            generate_quiz    create_plan
                                            (带反馈)        │
                                            └──────→ END
```

**为什么用 workflow 而非通用 Planner**：
- 教学意图有限可枚举（5 种），执行路径确定，不需要 Planner 动态规划
- Workflow 延迟低（3-5 秒 vs 三 Agent 串行 6-12 秒）
- Token 成本低（1 倍 vs 3 倍 LLM 调用）
- 结果可预测，便于调试

**Harness 思想如何保留**：
- **生成-验证分离**：不是独立 Agent，而是图里的验证节点 + 条件边 + 重试循环
- **确定性护栏**：不靠 LLM 判断，用规则
- **每步追踪**：LangGraph native tracing
- **人类接管**：`interrupt()` 原生支持

### 1.3 核心能力矩阵

| 能力 | Workflow 路径 | 关键验证节点 | 复用现有资产 |
|------|-------------|------------|-------------|
| **智能问答** | classify → query_rag → finalize | 验证引用来源准确性 | Retriever/Generator/Reranker |
| **练习生成** | classify → get_mastery → generate_quiz → evaluate_quiz → finalize | 独立验证答案key正确性 | 知识点树+知识单元 |
| **学情诊断** | classify → aggregate_history → analyze_weakness → finalize | 验证诊断结论与数据一致性 | PracticeRecord表 |
| **复习规划** | classify → get_mastery → query_rag → generate_quiz → evaluate_quiz → create_plan → finalize | 验证计划覆盖所有薄弱点 | ReviewPlan表 |
| **代码辅导** | classify → extract_code → run_sandbox → analyze_error → finalize | 验证解释与课程知识点一致 | 无（新增） |

---

## 二、七层架构映射

```
┌────────────────────────────────────────────────────────────────┐
│ 1. 用户接入层                                                   │
│    Streamlit UI / REST API / 飞书钉钉 / LMS 集成 / MCP Client   │
├────────────────────────────────────────────────────────────────┤
│ 2. 编排与控制层                                                 │
│    LangGraph StateGraph / 条件路由 / interrupt(5类) / 持久化    │
├────────────────────────────────────────────────────────────────┤
│ 3. Agent 核心层                                                 │
│    ┌─────────────────────────────────────────────┐             │
│    │  LangGraph 节点（每个节点调用一个 Skill）     │             │
│    │  classify / get_mastery / query_rag /        │             │
│    │  generate_quiz / evaluate_quiz / create_plan │             │
│    │  / diagnose / code_help / finalize           │             │
│    └─────────────────────────────────────────────┘             │
│    ┌─────────────────────────────────────────────┐             │
│    │  AGENTS.md / 上下文工程 / 条件边+重试循环     │             │
│    └─────────────────────────────────────────────┘             │
├────────────────────────────────────────────────────────────────┤
│ 4. 工具与环境层                                                 │
│    RAG引擎(现有) / 代码沙箱(Docker) / MCP Server                │
├────────────────────────────────────────────────────────────────┤
│ 5. 数据与记忆层                                                 │
│    PostgreSQL(12张表) / Milvus / agent_sessions / user_profiles │
│    LangGraph checkpoint(PostgresSaver) / 长期记忆               │
├────────────────────────────────────────────────────────────────┤
│ 6. 安全与治理层（贯穿所有层级）                                   │
│    RBAC(学生/教师/管理员) / 审计日志 / 确定性护栏 / FERPA合规   │
├────────────────────────────────────────────────────────────────┤
│ 7. 可观测性层（贯穿所有层级）                                    │
│    LangGraph tracing / LangSmith / RAGAS / Token成本 / 异常告警 │
└────────────────────────────────────────────────────────────────┘
```

---

## 三、各层详细设计

### 3.1 用户接入层

| 接入方式 | 场景 | 实现 |
|---------|------|------|
| **前端** | 学生端问答、练习、报告 | 现有 `ui/app.py` 增强 |
| **REST API** | 外部系统集成、教师管理端 | 现有 FastAPI 路由扩展 |
| **飞书/钉钉** | 教师在 IM 中 @Agent 查看班级学情 | 新增 webhook 接入 |
| **LMS 集成** | 学校 LMS 调用 CoursePilot 检索能力 | MCP Server 暴露 |
| **MCP Client** | Claude Code/Cursor 等工具直接调用 | MCP Server 暴露 |

**新增 API 路由**：

```python
# src/coursepilot/api/agent.py

@router.post("/api/v1/agent/chat")
async def agent_chat(request: AgentChatRequest, 
                     user: User = Depends(get_current_user),
                     session: AsyncSession = Depends(get_session)):
    """Agent 对话入口 — 调用 LangGraph"""
    # 1. 创建/恢复 Agent 会话
    agent_session = await create_or_resume_session(user.id, request.course_id, session)
    
    # 2. 调用 LangGraph 编译后的图
    config = {"configurable": {"thread_id": str(agent_session.id)}}
    result = await graph_app.ainvoke(
        {
            "user_query": request.message,
            "user_id": user.id,
            "course_id": request.course_id,
        },
        config=config,
    )
    
    # 3. 返回（支持 SSE 流式）
    return AgentChatResponse(
        session_id=agent_session.id,
        intent=result.get("intent"),
        answer=result.get("final_output"),
        sources=result.get("sources", []),
        token_count=result.get("token_count", 0),
        status=result.get("status", "completed"),
    )

@router.get("/api/v1/agent/sessions/{session_id}")
async def get_session_status(session_id: str, 
                             user: User = Depends(get_current_user)):
    """查询 Agent 会话执行状态（从 LangGraph checkpoint 读取）"""

@router.post("/api/v1/agent/sessions/{session_id}/approve")
async def approve_action(session_id: str, 
                         user: User = Depends(get_current_user)):
    """人类确认操作（恢复 interrupt 的图执行）"""
    config = {"configurable": {"thread_id": session_id}}
    result = await graph_app.ainvoke(None, config=config)  # 从断点恢复
    return result

@router.get("/api/v1/agent/sessions/{session_id}/trace")
async def get_session_trace(session_id: str,
                            user: User = Depends(require_teacher)):
    """获取 Agent 执行追踪（LangSmith / checkpoint 读取）"""
```

### 3.2 编排与控制层：LangGraph StateGraph

```python
# src/coursepilot/agent/graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from typing import TypedDict, Literal, Optional
import operator

class AgentState(TypedDict):
    # 输入
    user_query: str
    user_id: int
    course_id: int
    
    # 上下文（build_context 节点填充）
    context: dict                # 学生画像 + 课程信息 + 历史
    
    # 意图分类
    intent: str                  # question/practice/diagnose/review/code_help
    
    # 各节点输出（按需填充）
    mastery: Optional[dict]      # get_mastery 输出
    materials: Optional[list]    # query_rag 输出
    questions: Optional[list]    # generate_quiz 输出
    answer_key: Optional[list]
    evaluation: Optional[dict]   # evaluate_quiz 输出
    diagnosis: Optional[dict]    # diagnose 输出
    code_result: Optional[dict]  # code_help 输出
    review_plan: Optional[dict]  # create_plan 输出
    final_output: Optional[str]  # 最终输出
    
    # 追踪
    retry_count: int
    token_count: int
    status: str


# ========== 节点定义 ==========

async def build_context(state: AgentState) -> dict:
    """构建会话上下文：学生画像 + 课程信息 + 历史记录"""
    context = await context_builder.build(
        state["user_id"], state["course_id"]
    )
    return {"context": context, "token_count": 0}

async def classify_node(state: AgentState) -> dict:
    """意图分类节点：5 种核心意图 + 通用回退"""
    intent = await classify_intent(state["user_query"], state["context"])
    return {"intent": intent}

async def get_mastery_node(state: AgentState) -> dict:
    """查询学生掌握度"""
    kp_path = await extract_kp(state["user_query"], state["context"]["kp_tree"])
    mastery = await get_mastery_skill(
        state["user_id"], state["course_id"], kp_path
    )
    return {"mastery": mastery}

async def query_rag_node(state: AgentState) -> dict:
    """RAG 检索节点：复用现有 Retriever + Generator"""
    query = state["user_query"]
    # 如果有掌握度信息，按难度过滤
    difficulty = state.get("mastery", {}).get("level")
    result = await query_rag_skill(
        query=query,
        course_id=state["course_id"],
        difficulty_filter=difficulty,
    )
    return {
        "materials": result["sources"],
        "final_output": result["answer"],  # question 意图直接作为输出
        "token_count": state["token_count"] + result["token_count"],
    }

async def generate_quiz_node(state: AgentState) -> dict:
    """生成练习题节点"""
    kp_path = await extract_kp(state["user_query"], state["context"]["kp_tree"])
    difficulty = state.get("mastery", {}).get("level", "中等")
    materials = state.get("materials", [])
    
    result = await generate_quiz_skill(
        kp_path=kp_path,
        difficulty=difficulty,
        count=3,
        materials=materials,
        feedback=state.get("evaluation", {}).get("feedback"),  # 重试时带反馈
    )
    return {
        "questions": result["questions"],
        "answer_key": result["answer_key"],
        "token_count": state["token_count"] + result["token_count"],
    }

async def evaluate_quiz_node(state: AgentState) -> dict:
    """验证节点：独立验证答案 key 正确性（Harness：生成-验证分离）"""
    eval_result = await evaluate_quiz_skill(
        questions=state["questions"],
        answer_key=state["answer_key"],
        context=state["context"],
    )
    return {
        "evaluation": eval_result,
        "retry_count": state["retry_count"] + 1,
    }

async def create_plan_node(state: AgentState) -> dict:
    """生成复习计划节点"""
    weak_kps = state.get("mastery", {}).get("weak_subtopics", [])
    materials = state.get("materials", [])
    plan = await create_review_plan_skill(
        user_id=state["user_id"],
        course_id=state["course_id"],
        weak_kps=weak_kps,
        materials=materials,
    )
    return {"review_plan": plan, "token_count": state["token_count"] + 200}

async def diagnose_node(state: AgentState) -> dict:
    """学情诊断节点"""
    diagnosis = await diagnose_skill(
        user_id=state["user_id"],
        course_id=state["course_id"],
    )
    return {"diagnosis": diagnosis, "final_output": format_diagnosis(diagnosis)}

async def code_help_node(state: AgentState) -> dict:
    """代码辅导节点"""
    code = extract_code_from_query(state["user_query"])
    run_result = await code_sandbox.execute(code)
    analysis = await analyze_error_skill(code, run_result, state["context"])
    return {"code_result": analysis, "final_output": format_code_help(analysis)}

async def finalize_node(state: AgentState) -> dict:
    """收尾节点：更新长期记忆 + 格式化输出"""
    # 更新 QA 记录
    await update_qa_record_skill(
        user_id=state["user_id"],
        course_id=state["course_id"],
        query=state["user_query"],
        answer=state.get("final_output", ""),
    )
    # 更新学生画像（异步触发）
    await profile_updater.trigger_update(state["user_id"], state["course_id"])
    
    return {"status": "completed"}


# ========== 条件路由函数 ==========

def route_by_intent(state: AgentState) -> str:
    """意图分类后的条件路由"""
    intent = state["intent"]
    routes = {
        "question": "query_rag",
        "practice": "get_mastery",
        "diagnose": "diagnose",
        "review": "get_mastery",
        "code_help": "code_help",
    }
    return routes.get(intent, "query_rag")  # 通用回退

def route_after_eval(state: AgentState) -> str:
    """练习验证后的条件路由：不通过且重试<2 则重做"""
    if not state["evaluation"]["passed"] and state["retry_count"] < 2:
        return "generate_quiz"  # 重试
    return "create_plan"  # 通过或达到重试上限


# ========== 构建图 ==========

def build_agent_graph(checkpointer: PostgresSaver):
    graph = StateGraph(AgentState)
    
    # 添加节点
    graph.add_node("build_context", build_context)
    graph.add_node("classify", classify_node)
    graph.add_node("get_mastery", get_mastery_node)
    graph.add_node("query_rag", query_rag_node)
    graph.add_node("generate_quiz", generate_quiz_node)
    graph.add_node("evaluate_quiz", evaluate_quiz_node)
    graph.add_node("create_plan", create_plan_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("code_help", code_help_node)
    graph.add_node("finalize", finalize_node)
    
    # 入口
    graph.set_entry_point("build_context")
    
    # 固定边
    graph.add_edge("build_context", "classify")
    
    # 条件边：意图路由
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "query_rag": "query_rag",
            "get_mastery": "get_mastery",
            "diagnose": "diagnose",
            "code_help": "code_help",
        },
    )
    
    # question 路径
    graph.add_edge("query_rag", "finalize")
    
    # practice/review 路径
    graph.add_edge("get_mastery", "query_rag")  # 先查掌握度
    graph.add_edge("query_rag", "generate_quiz")  # 再检索素材
    graph.add_edge("generate_quiz", "evaluate_quiz")
    
    # 验证后的条件边（重试循环）
    graph.add_conditional_edges(
        "evaluate_quiz",
        route_after_eval,
        {
            "generate_quiz": "generate_quiz",  # 重试
            "create_plan": "create_plan",      # 通过
        },
    )
    
    # review 路径继续
    graph.add_edge("create_plan", "finalize")
    
    # diagnose 路径
    graph.add_edge("diagnose", "finalize")
    
    # code_help 路径
    graph.add_edge("code_help", "finalize")
    
    # 终点
    graph.add_edge("finalize", END)
    
    # 编译（带持久化）
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["create_plan"],  # 可选：生成复习计划前可暂停
    )


# ========== 初始化 ==========

# src/coursepilot/agent/__init__.py

from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(settings.DATABASE_URL_SYNC)
graph_app = build_agent_graph(checkpointer)
```

#### 人类接管点（5 类，通过 interrupt 实现）

```python
# 在节点中调用 interrupt 触发人类接管

from langgraph.types import interrupt

async def create_plan_node(state: AgentState) -> dict:
    """生成复习计划 — 如果是高风险内容，暂停等教师确认"""
    plan = await create_review_plan_skill(...)
    
    # 高风险内容需教师审核
    if is_high_risk_content(plan):
        approval = interrupt({
            "type": "high_risk_content_delivery",
            "description": "诊断报告含负面评价，需教师审核后发送",
            "plan": plan,
        })
        if not approval["approved"]:
            return {"status": "rejected"}
    
    return {"review_plan": plan}

# 5 类需人类接管的操作
HUMAN_APPROVAL_REQUIRED = {
    "generate_exam": "生成正式考试卷需要教师确认",
    "modify_grade": "修改学生成绩需要教师确认",
    "delete_course": "删除课程需要管理员确认",
    "access_other_user_data": "访问其他学生数据需要教师授权",
    "high_risk_content_delivery": "高风险内容发送前需教师审核",
}
```

### 3.3 Agent 核心层

#### AGENTS.md（项目宪法）

```markdown
# CoursePilot Agent 指令

## 角色定位
你是计算机科学课程的 AI 教学助手，服务学生、教师和管理员。

## 核心原则
1. 准确性优先：所有知识性回答必须标注来源（文档名+页码+知识点路径）
2. 引导而非替代：不得直接给出考试答案，必须引导思考
3. 个性化：每次回答前先查询学生画像，根据掌握度调整深度
4. 可追溯：每个生成的内容必须可追溯到知识库中的原始知识单元

## 工作模式
- 收到学生消息后，LangGraph 图从 build_context 开始执行
- classify 节点分类意图，条件边路由到对应 workflow
- 关键步骤（练习生成）后接 evaluate_quiz 验证节点，不通过则重试
- 关键操作（出考试卷、改成绩、删课程）通过 interrupt 暂停等待人类确认
- finalize 节点更新长期记忆后结束

## 编码规范
- 所有数据库操作使用 AsyncSession
- 知识点引用使用 kp_path 格式（如 "数据结构/树/二叉树/AVL旋转"）
- content_list 是通用中间格式，三种解析器均生成该格式
- text_level ≤ 4 表示标题，99 表示正文
- 严禁对同一文件重复解析（MinerU 解析必须检查 GPU）

## 禁止事项
- 不得生成超出课程知识点范围的内容
- 不得访问其他学生的数据（除非教师授权）
- 不得在未经教师确认的情况下发送诊断报告给学生
- 不得跳过 evaluate_quiz 验证节点直接返回练习题
```

### 3.4 Skill 体系

每个 LangGraph 节点调用一个 Skill。Skill 是纯函数，不依赖图的状态管理。

| Skill | 对应节点 | 复用/新增 | 调用的现有模块 | 输出 |
|-------|---------|----------|--------------|------|
| **query_rag** | query_rag | 封装现有 | Retriever + Generator + Reranker | 回答 + 来源 + Token |
| **generate_quiz** | generate_quiz | 新增 | 知识单元（Milvus 检索）+ DeepSeek | 题目 + 答案key |
| **evaluate_quiz** | evaluate_quiz | 新增 | DeepSeek（严格审查 prompt） | 验证结果 + 反馈 |
| **grade_answers** | （practice后） | 新增 | DeepSeek | 批改结果 + 错误归因 |
| **diagnose** | diagnose | 新增 | PracticeRecord 表聚合 | 薄弱知识点 + 报告 |
| **create_review_plan** | create_plan | 新增 | ReviewPlan 表 + DeepSeek | 分天复习计划 |
| **code_sandbox** | code_help | 新增 | Docker 沙箱 + DeepSeek | 代码运行结果 + 错误分析 |
| **get_mastery** | get_mastery | 新增 | user_profiles 表 | 掌握度 + 薄弱子主题 |
| **update_qa_record** | finalize | 新增 | QARecord 表 | 无（副作用操作） |
| **classify_intent** | classify | 新增 | DeepSeek 轻量分类 | 意图标签 |

**query_rag Skill（零改动复用现有 RAG）**：

```python
# src/coursepilot/agent/skills/query_rag.py

class QueryRagSkill:
    """封装现有 Retriever + Generator，核心逻辑零改动"""
    
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever     # 现有五阶段检索
        self.generator = generator     # 现有 DeepSeek 生成
    
    async def execute(self, query: str, course_id: int, 
                      difficulty_filter: str = None) -> dict:
        # 1. 检索（完全复用现有五阶段：改写→编码→检索→重排序→KP扩展）
        results = await self.retriever.retrieve(query, course_id)
        
        # 2. 可选过滤
        if difficulty_filter:
            results = [r for r in results if r.metadata.get("difficulty") == difficulty_filter]
        
        # 3. 生成（完全复用现有 Generator）
        answer = await self.generator.generate(query, results)
        
        return {
            "answer": answer,
            "sources": [
                {"doc": r.doc_name, "page": r.page_idx, "kp_path": r.kp_path}
                for r in results
            ],
            "token_count": self.generator.last_token_count,
        }
```

**evaluate_quiz Skill（独立验证，不护短）**：

```python
# src/coursepilot/agent/skills/evaluate_quiz.py

class EvaluateQuizSkill:
    """独立验证练习题答案 key — Harness 生成-验证分离"""
    
    EVALUATOR_SYSTEM_PROMPT = """
    你是一个严格的审查员。你的职责是找出问题，而不是表扬。
    审查标准：
    1. 知识准确性：答案是否与课程知识库一致？
    2. 引用完整性：是否标注了来源？
    3. 难度匹配：难度是否与学生掌握度匹配？
    4. 知识点覆盖：是否覆盖了所有应覆盖的知识点？
    5. 幻觉检测：题目中是否有知识库中不存在的信息？
    
    只回答 PASS 或 FAIL + 具体问题。
    """
    
    async def execute(self, questions: list, answer_key: list, 
                      context: dict) -> dict:
        issues = []
        for i, (q, a) in enumerate(zip(questions, answer_key)):
            # 1. 用不同 prompt 验证每个答案
            verification = await self._verify_answer(q, a, context)
            if not verification.is_correct:
                issues.append(f"Q{i+1} 答案可能有误：{verification.reason}")
            
            # 2. 检查题目是否在知识点范围内
            kp_coverage = self._check_kp_coverage(q, context["kp_tree"])
            if not kp_coverage.within_scope:
                issues.append(f"Q{i+1} 超出课程范围：{kp_coverage.detail}")
        
        if issues:
            return {"passed": False, "feedback": "; ".join(issues)}
        return {"passed": True, "feedback": None}
```

### 3.5 工具与环境层

#### 代码沙箱（CS 课程核心工具）

```python
# src/coursepilot/tools/code_sandbox.py

class CodeSandbox:
    """Docker 沙箱：安全执行学生代码"""
    
    SANDBOX_CONFIG = {
        "image": "python:3.12-slim",
        "timeout": 5,           # 5 秒超时
        "memory": "128m",        # 128MB 内存
        "cpus": "0.5",           # 半核 CPU
        "network": "none",       # 禁止网络
        "read_only": True,       # 只读根文件系统
        "tmpfs": {"/tmp": "size=64m"},  # 临时目录
    }
    
    async def execute(self, code: str, language: str = "python") -> dict:
        return {
            "output": stdout,
            "error": stderr,
            "returncode": returncode,
            "timeout": timed_out,
        }
```

#### MCP Server（对外暴露能力）

```python
# src/coursepilot/mcp/server.py

@app.tool()
async def query_knowledge(course_id: int, query: str) -> str:
    """检索课程知识库"""

@app.tool()
async def get_kp_tree(course_id: int) -> dict:
    """获取课程知识点树"""

@app.tool()
async def generate_practice(kp_path: str, difficulty: str) -> dict:
    """针对知识点生成练习题"""

@app.tool()
async def get_student_report(user_id: int, course_id: int) -> dict:
    """获取学生学情报告（需教师权限）"""
```

### 3.6 数据与记忆层

#### 数据持久化双轨制

```
                ┌──────────────────────────────────┐
                │  agent_sessions 表（业务元数据）   │
                │  谁/什么时候/什么意图/状态/成本     │
                │  供业务查询（会话列表、成本统计）   │
                └──────────────────────────────────┘
                              ↓ 不同关注点
                ┌──────────────────────────────────┐
                │  LangGraph checkpoint（执行细节）  │
                │  每步的输入/输出/重试次数/状态     │
                │  供断点恢复和执行追踪              │
                │  PostgresSaver 自动管理            │
                └──────────────────────────────────┘
```

**方案 B：业务元数据存 agent_sessions，执行细节存 LangGraph checkpoint，各管各的。**

#### 新增 3 张表

```sql
-- Agent 会话表：业务元数据（供查询会话列表、成本统计）
CREATE TABLE agent_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES users(id),
    course_id INTEGER NOT NULL REFERENCES courses(id),
    intent VARCHAR(20) NOT NULL,        -- question/practice/diagnose/review/code_help
    status VARCHAR(20) DEFAULT 'pending', -- pending/running/waiting_human/completed/failed
    token_count INTEGER DEFAULT 0,
    estimated_cost DECIMAL(10,4) DEFAULT 0,
    langgraph_thread_id VARCHAR(100),   -- 关联 LangGraph checkpoint
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 学生画像表：持久化长期记忆（异步预计算，非实时聚合）
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES users(id),
    course_id INTEGER NOT NULL REFERENCES courses(id),
    mastery_level JSONB DEFAULT '{}',   -- {"数据结构/树/二叉树": 0.7, ...}
    weak_kps TEXT[] DEFAULT '{}',       -- 薄弱知识点路径数组
    common_mistakes JSONB DEFAULT '[]', -- 常见错误模式
    learning_style VARCHAR(50),         -- visual/textual/practice
    total_qa_count INTEGER DEFAULT 0,
    total_practice_count INTEGER DEFAULT 0,
    avg_correct_rate DECIMAL(5,2) DEFAULT 0,
    last_diagnosis_at TIMESTAMPTZ,
    last_review_plan_id UUID,
    computed_at TIMESTAMPTZ DEFAULT NOW(),  -- 最后一次预计算时间
    UNIQUE(user_id, course_id)
);

-- 审计日志表
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    details JSONB,
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_agent_sessions_user ON agent_sessions(user_id);
CREATE INDEX idx_agent_sessions_course ON agent_sessions(course_id);
CREATE INDEX idx_agent_sessions_status ON agent_sessions(status);
CREATE INDEX idx_user_profiles_user_course ON user_profiles(user_id, course_id);
CREATE INDEX idx_audit_logs_user ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_created ON audit_logs(created_at);
```

#### 记忆系统架构

```
                ┌─────────────────────────────────┐
                │    会话记忆（短期）               │
                │  LangGraph checkpoint            │
                │  PostgresSaver 自动持久化         │
                │  每步状态、可断点恢复             │
                └───────────────┬─────────────────┘
                                │ 会话结束后
                                ↓
                ┌─────────────────────────────────┐
                │    画像记忆（长期）               │
                │  user_profiles 表                │
                │  掌握度/薄弱点/学习风格            │
                │  异步预计算（非实时聚合）          │
                └───────────────┬─────────────────┘
                                │
                                ↓
                ┌─────────────────────────────────┐
                │    知识记忆（永久）               │
                │  knowledge_points 表             │
                │  knowledge_units 表              │
                │  Milvus 向量存储                  │
                │  课程知识本体                     │
                └─────────────────────────────────┘
```

**异步预计算机制**（避免实时聚合性能问题）：

```python
# src/coursepilot/agent/profile_updater.py

class ProfileUpdater:
    """定时任务：从 PracticeRecord + DiagnosisReport 聚合到 user_profiles"""
    
    async def trigger_update(self, user_id: int, course_id: int):
        """Agent 会话结束后异步触发"""
        asyncio.create_task(self.update_profile(user_id, course_id))
    
    async def update_profile(self, user_id: int, course_id: int):
        # 1. 聚合练习记录
        # 2. 按知识点计算正确率
        # 3. 识别薄弱点（正确率 < 60%）
        # 4. 写入 user_profiles（upsert）
    
    # 每小时全量刷新兜底
```

### 3.7 安全与治理层

```python
# src/coursepilot/governance/rbac.py

class RBACManager:
    ROLES = {
        "student": [
            "agent_chat", "query_own_course", "submit_practice",
            "view_own_report", "view_own_review_plan",
        ],
        "teacher": [
            "agent_chat", "query_course", "create_quiz", "view_all_reports",
            "approve_exam", "modify_grade", "view_student_profiles",
            "access_other_user_data", "view_audit_logs",
        ],
        "admin": ["*"],
    }

# src/coursepilot/governance/guardrails.py

TEACHING_GUARDRAILS = {
    "content": {
        "no_direct_exam_answer": "不得直接给出考试答案，必须引导思考",
        "within_course_scope": "生成内容必须在课程知识点范围内",
        "cite_sources": "所有知识性回答必须标注来源（文档名+页码+kp_path）",
        "no_hallucination": "生成内容不得超出知识库范围（evaluate_quiz 检测）",
    },
    "permission": {
        "student_only_own_data": "学生只能查看自己的数据",
        "teacher_can_view_all": "教师可以查看所有学生的数据",
        "grade_modification_requires_teacher": "修改成绩需要教师确认",
        "exam_generation_requires_teacher": "生成考试卷需要教师确认",
    },
    "resource": {
        "max_questions_per_quiz": 10,
        "max_session_duration_seconds": 1800,
        "max_token_per_session": 50000,
        "max_token_per_day": 500000,
    },
}

# src/coursepilot/governance/audit.py

class AuditLogger:
    async def log(self, user_id: int, action: str, resource: str, 
                  details: dict, ip: str, db: AsyncSession):
        await db.execute(
            insert(AuditLog).values(
                user_id=user_id, action=action,
                resource_type=resource.split("/")[0] if "/" in resource else resource,
                resource_id=resource.split("/")[1] if "/" in resource else None,
                details=details, ip_address=ip,
            )
        )
```

### 3.8 可观测性层

**LangGraph 自带 tracing，不需要手写 Tracer。** 接 LangSmith 可视化。

```python
# src/coursepilot/observability/metrics.py
# 基于 LangGraph checkpoint + LangSmith 数据聚合

class AgentMetrics:
    """从 LangGraph checkpoint 和 LangSmith 读取指标"""
    
    async def session_summary(self, thread_id: str) -> dict:
        """从 checkpoint 读取单次会话指标"""
        state = await checkpointer.aget({"configurable": {"thread_id": thread_id}})
        return {
            "intent": state.get("intent"),
            "token_count": state.get("token_count", 0),
            "retry_count": state.get("retry_count", 0),
            "status": state.get("status"),
            "estimated_cost": self._calc_cost(state.get("token_count", 0)),
        }
    
    async def course_analytics(self, course_id: int) -> dict:
        """课程级别的 Agent 使用分析"""
        return {
            "total_sessions": ...,
            "active_students": ...,
            "avg_session_duration": ...,
            "most_used_intent": ...,
            "common_weak_kps": ...,    # 全班普遍薄弱点
            "total_token_cost": ...,
            "eval_pass_rate": ...,     # 全课程评估通过率
        }
```

**LangSmith 集成**（可选，可视化追踪）：

```python
# .env
LANGSMITH_API_KEY=xxx
LANGSMITH_PROJECT=coursepilot

# LangGraph 自动上报到 LangSmith
# 可在 UI 中查看：
# - 每次会话的完整执行图
# - 每个节点的输入/输出/耗时/Token
# - 条件边的路由决策
# - 重试次数和失败原因
```

---

## 四、完整数据流示例

### 场景：学生说"帮我复习二叉树"

```
1. POST /api/v1/agent/chat
   {message: "帮我复习二叉树", course_id: 3}
   → 鉴权：JWT → user_id=42, role=student
   → 权限检查：RBAC → "agent_chat" ✓

2. 创建 agent_sessions 记录
   → {user_id: 42, course_id: 3, langgraph_thread_id: "uuid-xxx", status: "running"}

3. LangGraph 图执行开始
   → graph_app.ainvoke({user_query, user_id, course_id}, config={thread_id})

4. build_context 节点
   → 从 user_profiles 读取学生画像
   → 从 knowledge_points 读取知识点树
   → 从 qa_records 读取最近 5 条对话
   → state.context = {mastery: {...}, kp_tree: {...}, ...}

5. classify 节点
   → 意图分类：intent = "review"
   → 条件边路由到 get_mastery

6. get_mastery 节点
   → 提取 kp_path = "数据结构/树/二叉树"
   → 查询 user_profiles → {rate: 0.6, level: "中等", weak_subtopics: ["AVL旋转", "哈夫曼树"]}

7. query_rag 节点
   → 调用现有 Retriever 检索 "AVL旋转" + "哈夫曼树"
   → 3 个知识单元，来自《数据结构》第4章
   → 调用现有 Generator 生成摘要

8. generate_quiz 节点
   → 基于 materials 生成 3 道练习题 + 答案key

9. evaluate_quiz 节点（Harness：独立验证）
   → Q1 验证：PASS
   → Q2 验证：FAIL（答案与课程定义不一致）
   → 条件边路由：retry_count=1 < 2 → 回到 generate_quiz

10. generate_quiz 节点（重试，带反馈）
    → 重新生成 Q2（带 evaluator 反馈"答案与课程定义不一致"）

11. evaluate_quiz 节点（第二次验证）
    → Q2 验证：PASS
    → 条件边路由：passed=True → create_plan

12. create_plan 节点
    → 生成 3 天复习计划
    → 存入 review_plans 表

13. finalize 节点
    → 更新 qa_records
    → 异步触发 profile_updater 更新 user_profiles
    → state.status = "completed"

14. LangGraph checkpoint 自动持久化每步状态

15. 返回给学生
    → 你的二叉树掌握度：60%（中等）
    → 薄弱点：AVL旋转、哈夫曼树
    → 复习材料：来自《数据结构》第4章（3个知识单元）
    → 练习题：3 道（等你作答）
    → 复习计划：Day1 AVL旋转 / Day2 哈夫曼树 / Day3 综合练习

16. 更新 agent_sessions
    → status: "completed", token_count: 4800, estimated_cost: ¥0.024
```

---

## 五、文件清单

| 文件路径 | 操作 | 职责 |
|---------|------|------|
| `AGENTS.md` | 新增 | 项目指令文件（Agent 宪法） |
| `src/coursepilot/agent/__init__.py` | 新增 | Agent 模块（初始化 graph_app） |
| `src/coursepilot/agent/graph.py` | 新增 | **LangGraph StateGraph 定义** |
| `src/coursepilot/agent/state.py` | 新增 | AgentState TypedDict 定义 |
| `src/coursepilot/agent/nodes.py` | 新增 | 图节点函数 |
| `src/coursepilot/agent/routing.py` | 新增 | 条件路由函数 |
| `src/coursepilot/agent/context.py` | 新增 | 会话上下文管理 |
| `src/coursepilot/agent/memory_manager.py` | 新增 | 记忆管理（短期+长期+知识） |
| `src/coursepilot/agent/profile_updater.py` | 新增 | 学生画像异步预计算 |
| `src/coursepilot/agent/skills/__init__.py` | 新增 | Skill 注册中心 |
| `src/coursepilot/agent/skills/query_rag.py` | 新增 | Skill：封装现有 RAG |
| `src/coursepilot/agent/skills/generate_quiz.py` | 新增 | Skill：练习生成 |
| `src/coursepilot/agent/skills/evaluate_quiz.py` | 新增 | Skill：独立验证答案key |
| `src/coursepilot/agent/skills/grade_answers.py` | 新增 | Skill：批改学生答案 |
| `src/coursepilot/agent/skills/diagnose.py` | 新增 | Skill：学情诊断 |
| `src/coursepilot/agent/skills/review_plan.py` | 新增 | Skill：复习规划 |
| `src/coursepilot/agent/skills/code_help.py` | 新增 | Skill：代码辅导（CS专用） |
| `src/coursepilot/agent/skills/get_mastery.py` | 新增 | Skill：查询掌握度 |
| `src/coursepilot/agent/skills/classify_intent.py` | 新增 | Skill：意图分类 |
| `src/coursepilot/agent/skills/update_qa_record.py` | 新增 | Skill：更新QA记录 |
| `src/coursepilot/tools/code_sandbox.py` | 新增 | Docker 代码沙箱 |
| `src/coursepilot/governance/rbac.py` | 新增 | RBAC 权限控制 |
| `src/coursepilot/governance/guardrails.py` | 新增 | 确定性护栏 |
| `src/coursepilot/governance/audit.py` | 新增 | 审计日志 |
| `src/coursepilot/observability/metrics.py` | 新增 | 指标与成本分析 |
| `src/coursepilot/mcp/server.py` | 新增 | MCP Server |
| `src/coursepilot/api/agent.py` | 新增 | Agent API 路由 |
| `src/coursepilot/db/models/agent_session.py` | 新增 | agent_sessions 模型 |
| `src/coursepilot/db/models/user_profile.py` | 新增 | user_profiles 模型 |
| `src/coursepilot/db/models/audit_log.py` | 新增 | audit_logs 模型 |
| `src/coursepilot/main.py` | 改动 | 注册 agent 路由 |
| `src/coursepilot/rag/*` | **不改** | Retriever/Generator/Reranker 零改动 |
| `src/coursepilot/ingestion/*` | **不改** | 解析器/管道 零改动 |
| `src/coursepilot/knowledge/*` | **不改** | 知识点树/KPSplitter 零改动 |

**总计**：新增 30 个文件，改动 1 个文件，**核心现有代码零改动**。

**新增依赖**：

```
langgraph >= 0.2
langgraph-checkpoint-postgres
langsmith (可选，可视化追踪)
```

---

## 六、数据库变更

新增 3 张表（agent_sessions / user_profiles / audit_logs）+ LangGraph checkpoint 表（PostgresSaver 自动创建），现有 11 张表不变。

---

## 七、实施路线图

### Phase 1：LangGraph 骨架（2 周）

| 周 | 任务 | 产出 |
|----|------|------|
| W1 | AGENTS.md + agent/ 目录 + AgentState 定义 + PostgresSaver 配置 | 骨架就绪 |
| W2 | build_context + classify + query_rag 节点 + question 路径打通 | 端到端问答可用 |

**里程碑**：学生通过 Agent 入口提问，LangGraph 图执行 build_context → classify → query_rag → finalize → 返回。

### Phase 2：教学核心能力（3 周）

| 周 | 任务 | 产出 |
|----|------|------|
| W3 | get_mastery + generate_quiz + evaluate_quiz 节点 + 条件边重试循环 | 练习生成+验证可用 |
| W4 | diagnose 节点 + user_profiles 表 + profile_updater | 学情诊断+长期记忆 |
| W5 | create_plan 节点 + review 完整路径 | 复习规划完整闭环 |

**里程碑**：学生说"帮我复习二叉树"，LangGraph 图执行完整 review 路径，含验证重试。

### Phase 3：企业级加固（3 周）

| 周 | 任务 | 产出 |
|----|------|------|
| W6 | code_help 节点 + Docker 沙箱 | 代码辅导可用 |
| W7 | RBAC + 护栏 + 审计日志 + interrupt 人类接管 | 安全治理就绪 |
| W8 | LangSmith 集成 + Metrics + MCP Server | 可观测+外部可调用 |

**里程碑**：企业级教学 Agent，多角色、可审计、可观测、可扩展。

---

## 八、风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| LangGraph 学习曲线 | 中 | 中 | Deep Agents 教程有完整示例；先跑通 question 路径再扩展 |
| evaluate_quiz 与 generate_quiz 同模型"护短" | 中 | 高 | evaluate_quiz 用不同 prompt 策略；未来可接入第二模型交叉验证 |
| DeepSeek 练习题质量不稳定 | 中 | 高 | evaluate_quiz 独立验证 + 最多 2 次重试 + RAGAS 评估 |
| LangGraph checkpoint 与 agent_sessions 数据不一致 | 低 | 低 | agent_sessions 只存业务元数据，checkpoint 存执行细节，各管各的 |
| user_profiles 预计算延迟导致画像过时 | 低 | 中 | 每次 Agent 会话结束后触发增量更新 + 每小时全量刷新 |
| 代码沙箱安全风险 | 低 | 高 | Docker 隔离 + 5s 超时 + 禁网 + 128MB + 只读 |
| Token 成本失控 | 低 | 中 | 单会话 50K 上限 + 单日 500K 上限 + 成本追踪告警 |
| 意图分类错误 | 中 | 中 | 通用回退到 question 路径 + 澄清追问 |

---

## 九、成功指标

| 指标 | 基线 | Phase 2 目标 | Phase 3 目标 |
|------|------|-------------|-------------|
| 问答响应时间 | ~3s | ≤5s（含意图分类） | ≤4s |
| 练习生成时间 | 不支持 | ≤10s（含验证） | ≤8s |
| evaluate_quiz 答案验证通过率 | 不支持 | ≥85% | ≥90% |
| 练习题答案正确率 | 不支持 | ≥85% | ≥90% |
| 学情诊断准确率 | 不支持 | ≥75% | ≥80% |
| 跨会话记忆命中率 | 0% | ≥50% | ≥70% |
| 学生满意度 | 未测量 | ≥3.8/5 | ≥4.2/5 |
| 单次会话 Token 成本 | 未追踪 | ≤¥0.05 | ≤¥0.03 |
| LangGraph 追踪覆盖率 | 0% | 100% | 100% |
| 断点恢复成功率 | 不支持 | ≥95% | ≥99% |

---

*文档版本：v4.0（LangGraph 有向状态图）*
*设计日期：2026年6月28日*
