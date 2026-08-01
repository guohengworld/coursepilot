# CoursePilot 上下文窗口与记忆层设计文档

> 文档范围：P0–P5 企业级上下文管理实现
> 核心目标：把"无界拼接对话"改造成"分层、有预算、可压缩、可观测"的上下文工程体系
> 适用模型：DeepSeek 系列（默认 64K 上下文），可通过 `llm_context_budget` 配置扩展到其它模型

---

## 一、改造前的核心问题

| 层级 | 原实现 | 问题 |
|---|---|---|
| 存储层 | `agent_sessions.conversation` JSONB 无限追加 | 无界增长、无压缩、无生命周期管理 |
| 加载层 | `agent.py` 把整段 conversation 塞进 `state["messages"]` | `messages` 只进不出，下游 generator 不消费 |
| 注入层 | `agent/context.py` 固定取最近 5 条 QA、答案硬截断 200 字符 | 只看 recency，不讲 relevance；200 字符会切断公式 |
| 生成层 | `rag/generator.py` 每次只发 `[system, user]` | 多轮对话实际是单轮，"上一道题"必翻车 |
| 预算层 | 全项目只有 `kp_max_tokens`（给切块用的） | 没有任何 prompt 级 token 预算、计数、超限降级策略 |
| 缓存层 | `SYSTEM_PROMPT` 把每轮都变的 RAG sources 拼进 system | 放在 messages 最前面，导致 DeepSeek prompt caching 每轮 miss |

---

## 二、总体架构：四层记忆 + 预算装配

我们参考 MemGPT、Claude memory、Trae 的上下文管理思路，把对话上下文抽象成四层：

```
L0 工作记忆    当前 turn 的 AgentState（LangGraph 状态机，已有）
L1 短期记忆    最近 N 轮原文（滑动窗口，逐字保留，保证连贯性）
L2 情景记忆    滚动摘要：窗口溢出的老轮次被增量压缩成 summary
L3 语义记忆    结构化事实：从 QA 中抽取"已掌握/薄弱/常见错误/学习风格"
              写入 user_profiles.memory_facts，带 provenance
L4 归档记忆    原始全量 QARecord 留库，支持语义检索召回（memory-as-RAG）
```

对应的代码映射：

| 层级 | 存储 | 代码入口 |
|---|---|---|
| L0 | `AgentState`（运行时 dict） | `agent/state.py` |
| L1 | `agent_sessions.conversation` JSONB | `agent/nodes.py` `_update_session_intent` |
| L2 | `agent_sessions.rolling_summary` TEXT | `agent/nodes.py` `_maybe_compact_session` |
| L3 | `user_profiles.memory_facts` JSONB | `agent/memory/extractor.py` |
| L4 | `qa_records` 表 + `embedding`/`importance` | `agent/memory/retriever.py` |

---

## 三、ContextManager：预算制装配器

### 3.1 为什么必须按预算装配

大模型上下文不是"能装多少装多少"。无界拼接会导致：
- 偶然性爆窗：某次 RAG 返回长片段 + 长对话 = 直接超限
- 成本失控：长 prompt 的输入 token 单价虽然低，但累计惊人
- 注意力稀释：末尾的当前 query 被淹没在冗余历史里
- cache miss：变化内容放在前面，把稳定前缀挤出缓存前缀

因此每次调 LLM 前，先按预算分配，再决定每一层放多少。

### 3.2 预算配置

配置集中在 `src/coursepilot/config.py`：

```python
llm_context_budget: dict = {
    "total_tokens": 64_000,          # 模型总窗口
    "reserved_output": 4_096,        # 给模型输出预留
    "safety_margin": 1_024,          # 估算误差安全余量
    "max_recent_turns": 6,           # L1 最多保留轮数（一对 QA 算一轮）
    "rolling_summary_max_tokens": 1_500,
    "user_profile_max_tokens": 400,
    "rag_default_max_tokens": 8_000,
}
```

可用预算计算公式：

```
available = total_tokens - reserved_output - safety_margin
          = 64_000 - 4_096 - 1_024 = 58_880 token
```

所有层的实际占用都必须从 `available` 里扣。

### 3.3 装配顺序：缓存友好

`ContextManager.build_view()` 按以下顺序产出 `ContextView`：

```
1. system_prefix   （system prompt + 课程上下文 + 学生画像，放最前）
2. rolling_summary （L2 滚动摘要）
3. recent_turns    （L1 最近轮次）
4. rag_context     （当前检索上下文）
5. current_query   （当前用户 query）
```

这样排序的原因：
- DeepSeek / OpenAI 的 prompt caching 按**消息前缀**命中
- 前 1–3 项在每轮之间高度稳定（同一课程、同一学生），cache hit 概率最大
- RAG sources 和 current_query 每轮都变，放在最后，不影响前面稳定前缀的缓存命中

### 3.4 节点差异化视图

不是所有节点都需要全部记忆。`ContextManager.NODE_CONFIGS` 为不同节点定义策略：

| 节点 | RAG | 滚动摘要 | 学生画像 | RAG 预算 |
|---|---|---|---|---|
| classify | 否 | 是 | 否 | 0 |
| query_rag | 是 | 是 | 是 | 8_000 |
| generate_quiz | 是 | 是 | 是 | 6_000 |
| evaluate_quiz | 是 | 是 | 否 | 4_000 |
| diagnose | 否 | 是 | 是 | 0 |
| review_plan | 否 | 是 | 是 | 0 |

例如 `classify` 只做意图分类，不需要 RAG 和学生画像，只保留 system + rolling_summary + 最近轮次，降低调用成本和延迟。

### 3.5 各层预算分配细节

#### 1) system_prefix（固定前缀）

- 由 `SYSTEM_PROMPT.format(course_context=...)` 得到
- 如果节点启用学生画像，会把 `_fmt_user_profile()` 输出的摘要拼接在 system 后面
- 这层**不压缩**，但学生画像会按 `user_profile_max_tokens=400` 截断
- token 占用直接计入已用预算

#### 2) rolling_summary（L2）

```python
remaining = budget.available - used
summary_max = min(self.rolling_summary_max, max(0, remaining // 4))
```

即：L2 最多占剩余预算的 25%，且不超过 `rolling_summary_max_tokens=1500`。

#### 3) recent_turns（L1）

```python
recent_turns = self._trim_turns_to_budget(
    conversation,
    max_turns=self.max_recent_turns,     # 6
    max_tokens=max(0, remaining // 3),   # 剩余预算的 1/3
)
```

两步裁剪：
1. 先按数量截断：只保留最近 6 轮（12 条消息）
2. 再按 token 截断：从最近往前遍历，累计超过 `remaining//3` 时停止

保留的是最近轮次，因为最近的对话对当前 query 最直接相关。

#### 4) rag_context（动态检索上下文）

```python
remaining = budget.available - used
rag_max = min(rag_max, max(0, remaining - estimate_tokens(current_query) - 200))
rag_context = self._compact_text(rag_context, rag_max)
```

RAG 预算 = 节点配置预算 与 "剩余预算扣掉 current_query 和 200 token 缓冲" 的较小值。
如果 RAG 内容超长，按 `_compact_text()` 截断。

#### 5) current_query

- 必须保留
- 但异常长输入会被截断到 2_000 token

### 3.6 截断策略：避免切断公式

`_compact_text()` 使用二分查找找到合适长度，截断后检查：
- 如果单 `$` 未闭合，补一个 `$`
- 如果 `$$` 未闭合，补一个 `$$`
- 末尾加 `...（已截断）`

这样能保证截断后的 Markdown/LaTeX 仍然可解析，不会留下半个公式。

---

## 四、Token 估算方法

我们没有引入 tiktoken，而是用一个轻量纯 Python 估算器：

```python
def estimate_tokens(text: str) -> int:
    ascii_words = len(re.findall(r"[a-zA-Z0-9_]+", text))
    ascii_len = sum(len(w) for w in re.findall(r"[a-zA-Z0-9_]+", text))
    cn_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    other_chars = len(text) - ascii_len - cn_chars
    return int(ascii_words * 1.3 + cn_chars * 0.6 + max(0, other_chars) * 0.5)
```

经验系数来源：
- 英文单词：GPT/DeepSeek tokenizer 平均 0.75 word/token，取 1.3 token/word 偏保守
- 中文字符：常见 tokenizer 约 0.5–0.7 token/字，取 0.6
- 其它字符（标点、空白、LaTeX 符号）：取 0.5

**注意**：这是估算，实际用量以 LLM 返回的 `usage` 为准。估算器的目的是做预算截断决策，不要求绝对精确。`safety_margin=1024` 就是用来吸收估算误差的。

---

## 五、短期记忆 L1：滑动窗口

### 5.1 数据格式

`agent_sessions.conversation` 存储：

```json
[
  {"role": "user", "content": "..., "intent": null},
  {"role": "assistant", "content": "...", "intent": "question", "sources": [...], "query": "..."},
  ...
]
```

每轮 finalize 节点会追加两条：
- user query
- assistant answer（带 intent、sources、query 回填）

### 5.2 为什么这样设计

- 保留原文，保证"上一轮那道题"的追问能直接命中
- 不保存完整 AgentState，只保存 conversation 角色消息，减小 JSONB 体积
- `sources` 和 `query` 字段作为锚点，压缩后仍可追溯

### 5.3 数量 vs Token 双截断

L1 不是简单"保留最近 5 条"。`ContextManager._trim_turns_to_budget()` 同时考虑：
- 轮数上限 `max_recent_turns=6`
- token 上限 `remaining // 3`

例如某轮用户贴了 3000 token 的长代码，即使只超一点点，也会只保留 1 轮，而不是机械保留 6 轮导致爆窗。

---

## 六、滚动摘要 L2：增量压缩

### 6.1 触发时机

`finalize_node` 调用 `_maybe_compact_session()`，判断条件：

```python
history_tokens = estimate_tokens(所有 conversation 文本) + estimate_tokens(rolling_summary)
return history_tokens > self.total * threshold_ratio   # threshold_ratio=0.75
```

当 L1+L2 超过总窗口的 75% 时触发压缩。

### 6.2 压缩策略

`compact_conversation()`：

1. 取 conversation 的前 50% 作为老轮次
2. 对每条 assistant 消息调用 `micro_compact_turn()`，提取：
   - query 摘要（≤200 字符）
   - answer 核心结论（≤200 token）
   - 关键公式片段
   - 涉及的知识点路径 kp_paths
3. 对 user 消息只保留前 120 字符
4. 拼接到 existing_summary 后面
5. 如果 summary 超过 `max_summary_tokens=1500`，从头部弹出行，保留最近事件

### 6.3 Micro-compact 的设计原则

保留"可回答后续追问"的最小信息：
- 结论性语句
- 关键公式/定理
- 引用来源 kp_path
- 去除客套话、问候语

例如长回答被压成：

```
[question] 怎么求二重积分 -> 先确定积分区域，再选择直角/极坐标；关键公式：$\iint_D f(x,y)d\sigma$ (kp: 1.2/3)
```

### 6.4 保留锚点

压缩后：
- `agent_session.rolling_summary` = 新摘要
- `agent_session.conversation = conversation[compacted_count:]`

被压缩的原始轮次从 L1 移除，但摘要里保留了 intent、query 摘要、kp_path。如果用户追问"刚才那道题"，LLM 仍能从 L2 摘要中推断上下文，必要时再去 L4 QARecord 打捞原文。

---

## 七、语义记忆 L3：结构化事实抽取

### 7.1 存储

新增字段 `user_profiles.memory_facts` JSONB：

```json
{
  "mastered_kps": [
    {"kp_path": "1.2/3", "confidence": 0.85, "evidence": "...", "provenance": {"qa_id": "...", "session_id": "..."}}
  ],
  "weak_kps": [...],
  "common_mistakes": [
    {"category": "概念混淆", "pattern": "...", "count": 2, "provenance": {...}}
  ],
  "learning_style": {"style": "visual", "evidence": "...", "provenance": {...}}
}
```

### 7.2 抽取流程

`finalize_node` 通过 `asyncio.create_task(extract_facts_for_session(...))` 异步触发：

1. 查询该 session 下所有 QARecord
2. 对每条 QARecord 调用 `extract_facts_from_qa()`
3. LLM 按 `_EXTRACTION_SYSTEM` 输出 JSON
4. `_merge_facts()` 合并到 `UserProfile.memory_facts`
5. 控制每项最多保留 20 条，避免无限增长

### 7.3 为什么用异步

- 抽取是"写后分析"，不影响当前回答延迟
- 失败不影响主流程
- 每条 QA 独立调用，便于后续做批处理优化

### 7.4 Provenance

每个事实都带 `provenance.qa_id` 和 `provenance.session_id`，用于：
- 可观测：知道这个判断来自哪次对话
- 溯源：学生质疑"为什么说我薄弱"时可以召回原始 QA
- 去重/更新：相同 kp_path 出现新证据时更新 confidence

---

## 八、归档记忆 L4：语义召回

### 8.1 存储

新增字段：
- `qa_records.embedding`：BGE-M3 dense 向量
- `qa_records.importance`：0.0–1.0 重要性评分

### 8.2 评分公式

```python
score = alpha * recency + beta * relevance + gamma * importance
```

默认权重：
- `alpha=0.25`
- `beta=0.55`
- `gamma=0.20`

recency：
```python
recency = exp(-delta_days / tau_days)   # tau_days=30
```

relevance：
```python
relevance = cosine(query_embedding, qa_record.embedding)
```

importance：
- 规则估算（轻量，不调用 LLM）
- 薄弱信号词 +0.3
- 答案含公式且较长 +0.2
- 问候类短句 -0.2

### 8.3 召回流程

`recall_memory_turns()`：
1. 编码当前 query
2. 拉取该用户/课程最近 200 条 QARecord
3. 对缺失 embedding 的记录即时编码（冷启动）
4. 逐条计算 score
5. 返回 top_k，带 score breakdown

### 8.4 当前未接入生成链路

L4 召回目前只通过 admin API `/admin/memory/recall` 暴露，供人工测试。后续 P+ 阶段可以把它作为 RAG 的额外候选源，或在 `query_rag` 中合并进 context。

---

## 九、Prompt 最终组装

`generator.py` 的 `_build_messages()` 把 `ContextView` 转成 LLM messages：

```python
messages = [
    {"role": "system", "content": system_prefix.replace("{sources}", rag_context)},
    {"role": "system", "content": "以下是对话历史摘要：\n{rolling_summary}"},  # 若存在
    # recent_turns...
    {"role": "user", "content": current_query},
]
```

注意：
- `SYSTEM_PROMPT` 中保留 `{sources}` 占位符，由 `_build_messages()` 在最后替换
- 这样 `ContextManager` 不需要关心 RAG 格式，只负责预算分配
- system 消息合并成一条，避免多 system 消息影响某些模型的缓存前缀匹配

---

## 十、可观测性 P5

### 10.1 运行时指标

每次 LLM 调用返回的 `token_info` 包含：

```json
{
  "prompt_tokens": 1234,
  "completion_tokens": 567,
  "total_tokens": 1801,
  "context_budget": {
    "total": 64000,
    "available": 58880,
    "used": 15200,
    "node": "query_rag",
    "recent_turns_count": 4,
    "rag_truncated": false,
    "cache_friendly": true
  },
  "layer_tokens": {
    "system_prefix": 4200,
    "rolling_summary": 800,
    "recent_turns": 3200,
    "rag_context": 6000,
    "current_query": 1000
  },
  "cache_hit_estimated": {
    "estimated_hit_rate": 0.72,
    "stable_tokens": 4200,
    "actual_prompt_tokens": 1234,
    "note": "基于稳定前缀占比启发式估算"
  }
}
```

### 10.2 Admin 控制台

`api/admin.py` 提供三个端点：

| 端点 | 功能 |
|---|---|
| `GET /admin/memory/dashboard?course_id=...` | 课程级记忆层仪表盘 |
| `GET /admin/memory/session/{session_id}` | 单次会话的记忆详情 |
| `GET /admin/memory/recall?user_id=...&course_id=...&query=...` | 测试 L4 召回 |

前端对应 `frontend/src/views/AdminMemoryConsole.vue`，仅限 superuser 访问。

### 10.3 压缩次数观测

`finalize_node` 触发压缩后，会在 `llm_calls` 中追加一条：

```json
{"node": "compaction", "compacted_turns": 3, "prompt_tokens": 0, ...}
```

---

## 十一、数据库迁移

迁移文件：`alembic/versions/94e81d0d679b_add_memory_facts_qa_embedding_.py`

新增字段：
- `qa_records.embedding` ARRAY(Float)
- `qa_records.importance` Float
- `user_profiles.memory_facts` JSONB

同时调整了一些字段 comment，并清理了旧的 `checkpoints*` 表（因为 LangGraph PostgresSaver 会自动建自己的 checkpoint 表）。

---

## 十二、当前不确定 / 需要测试的项

以下是我目前无法从静态代码中完全确认、需要实测验证的点：

### 12.1 Token 估算系数

- 当前 `estimate_tokens()` 对中文取 0.6、英文单词取 1.3
- **需测试**：用真实 DeepSeek usage 回归，看估算值/实际值的平均误差是否在 ±15% 以内
- 如果误差大，需要把系数调为 0.55/1.4 或引入 tiktoken

### 12.2 Prompt Caching 实际效果

- 当前缓存命中率是启发式估算，非 DeepSeek 官方返回
- **需测试**：连续两轮同一课程同一学生的调用，观察 prompt_tokens 是否显著下降
- 如果 DeepSeek 返回 `cached_tokens` 字段，应替换启发式逻辑

### 12.3 L2 压缩质量

- `compact_conversation()` 目前基于规则，未调用 LLM
- **需测试**：压缩后摘要是否保留足够信息支撑追问（如"刚才那道题第二步怎么算的"）
- 如果质量差，可启用 `compact_with_llm()` 占位函数

### 12.4 L3 抽取准确性

- `extract_facts_from_qa()` 调用 LLM 输出 JSON
- **需测试**：
  - JSON 格式是否稳定（已设 `response_format={"type": "json_object"}`）
  - confidence 分布是否合理
  - 是否会出现幻觉事实

### 12.5 L4 召回权重

- `alpha=0.25, beta=0.55, gamma=0.20` 是经验值
- **需测试**：调整权重看召回结果是否符合直觉
  - 例如"帮我复习上周的内容"应该让 recency 权重更高
  - "我之前问过一道相似的题"应该让 relevance 权重更高

### 12.6 节点配置是否合理

- `classify` 节点目前仍会用 rolling_summary
- **需测试**：意图分类准确率在有/无 rolling_summary 时的差异
- `diagnose` 节点禁用 RAG，但保留 user_profile，是否足够？

### 12.7 多轮对话真实效果

- **需测试**：连续 10 轮同一话题的对话，观察 L1 是否被正确截断、L2 是否正确累积
- 特别测试边界：第 7 轮触发压缩后，用户追问"你刚才说的那个定理"

---

## 十三、文件清单

| 文件 | 作用 |
|---|---|
| `src/coursepilot/agent/memory/context_manager.py` | ContextManager、ContextView、预算装配、token 估算 |
| `src/coursepilot/agent/memory/compactor.py` | L2 滚动摘要、micro-compact |
| `src/coursepilot/agent/memory/extractor.py` | L3 语义记忆抽取 |
| `src/coursepilot/agent/memory/retriever.py` | L4 归档记忆召回、importance 估算 |
| `src/coursepilot/agent/memory/__init__.py` | 统一导出 |
| `src/coursepilot/agent/nodes.py` | finalize 中触发 L1/L2 维护、L3/L4 异步任务 |
| `src/coursepilot/agent/state.py` | AgentState 定义上下文预算字段 |
| `src/coursepilot/agent/skills/query_rag.py` | 把 conversation/rolling_summary/user_profile 传入 Generator |
| `src/coursepilot/agent/skills/classify_intent.py` | 消费最近轮次做意图分类 |
| `src/coursepilot/rag/generator.py` | 用 ContextManager 装配 messages，返回 token_info |
| `src/coursepilot/api/admin.py` | Admin 记忆层 API |
| `src/coursepilot/observability/metrics.py` | 上下文预算、记忆层指标聚合 |
| `src/coursepilot/models/agent_session.py` | conversation、rolling_summary 字段 |
| `src/coursepilot/models/user_profile.py` | memory_facts 字段 |
| `src/coursepilot/models/qa_record.py` | embedding、importance 字段 |
| `src/coursepilot/config.py` | llm_context_budget 配置 |
| `frontend/src/views/AdminMemoryConsole.vue` | 前端 Admin 控制台 |
| `alembic/versions/94e81d0d679b_*.py` | 数据库迁移 |

---

## 十四、后续建议

1. **接入真实缓存标记**：DeepSeek 如果返回 `cached_tokens`，替换 `Generator._estimate_cache_hit()` 的启发式逻辑
2. **L4 并入 RAG**：把 `recall_memory_turns()` 的结果作为 RAG 候选源之一，实现 memory-augmented RAG
3. **LLM 级压缩**：当规则压缩质量不达标时，启用 `compact_with_llm()`
4. **更细粒度预算**：给不同模型（32K/128K/256K）分别配置 budget profile
5. **持久化 llm_calls**：目前 `llm_calls` 只在运行时 state 中存在，建议新增 `agent_session.llm_calls` JSONB 字段，便于事后分析
