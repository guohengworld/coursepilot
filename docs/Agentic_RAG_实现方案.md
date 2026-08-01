> 状态：方案已锁定，核心功能（问答、练习、诊断、复习）已实现。
> 相关代码：`src/coursepilot/agent/`、`src/coursepilot/rag/`。
> 最后更新：2026-08-01

# Agentic RAG 实现方案

> 基于《Agentic RAG 深度解析》的核心思想，结合 CoursePilot 现有架构的可行路线

---

## 一、现状 vs 目标

### 当前 RAG 流程（单次检索，无回退）

```
query → 改写 → 编码 → 检索 → 重排序 → KP扩展 → LLM生成 → 结束
```

**问题**：
- 检索结果好不好，没有验证环节
- 信息不够时，不会自动补搜
- 简单问题和复杂问题走同样的全量流程

### 目标：Agentic RAG 流程（带质量闭环）

```
query → 路由判断
  ├─ 简单问题 → 快速RAG → 结束
  └─ 复杂问题 → 规划/分解 → 并行检索 → 质检
       ├─ 通过 → 合成 → 结束
       └─ 不足 → 补搜(≤N轮) → 质检
```

---

## 二、重点实现（按优先级）

### [P0] 1. 智能路由 + 快速通道

**改动范围**：`agent/routing.py` + `rag/config.py`

**思路**：用规则+LLM 快速判断问题复杂度，决定走快慢通道。

**在 classify_intent 中增加复杂度判断**：

```python
# agent/skills/classify_intent.py 新增字段
{
  "intent": "question",
  "complexity": "simple",  # simple | complex
  "reasoning": "单知识点事实性问题",
  "requires_search": False,  # 是否需网络搜索补充
}
```

**判断依据**（规则+LLM混合）：
- **simple**: 单知识点、事实性、一句话可答（如"什么是极限"）
- **complex**: 多知识点比较、需要推理、信息可能分散（如"极限和连续有什么关系"）

**路由变化**（`agent/graph.py`）：

```
classify → [route_by_complexity]
  ├─ simple → query_rag → finalize
  └─ complex → agentic_rag  (带质检+多轮)
```

**新增 RAGConfig 参数**：

```python
# rag/config.py
enable_routing: bool = True           # 启用智能路由
simple_top_k: int = 3                 # 简单通道取 top-3 即可
complex_max_rounds: int = 3           # 复杂问题最多3轮补搜
context_sufficiency_threshold: float = 0.7  # 质检通过阈值
```

---

### [P1] 2. 上下文充足性质检（核心创新）

**改动范围**：新增 `agent/skills/check_sufficiency.py`

**思路**：在生成答案之前，先判断检索到的上下文是否足够回答问题。如果不足，明确指出缺什么，触发补搜。

```python
# agent/skills/check_sufficiency.py

from typing import Any
from coursepilot.rag.config import config

CHECK_PROMPT = """你是一个RAG质量检验员。你的任务不是回答问题，
而是判断给定的【教材内容】是否足以回答【用户问题】。

【教材内容】
{context}

【用户问题】
{query}

请输出JSON：
{{
  "sufficient": true/false,
  "confidence": 0.0-1.0,
  "missing_info": "如果不足，明确指出缺失的信息类型；如果充足则为空字符串",
  "missing_kp": "如果不足，推测缺少的知识点路径；如果充足则为空字符串",
  "covered_aspects": ["已覆盖的方面1", "已覆盖的方面2"],
  "uncovered_aspects": ["未覆盖的方面1"]
}}
"""

async def check_sufficiency(
    query: str,
    context: str,
    kp_paths: list[str],
) -> dict[str, Any]:
    """质检：判断检索到的上下文是否足够回答用户问题。
    
    Returns:
        {"sufficient": bool, "confidence": float, "missing_info": str, 
         "missing_kp": str, "covered_aspects": [...], "uncovered_aspects": [...]}
    """
    # 快速规则过滤
    if not context.strip():
        return {"sufficient": False, "confidence": 0.0, 
                "missing_info": "未检索到任何教材内容",
                "missing_kp": "", "covered_aspects": [], 
                "uncovered_aspects": ["需要检索相关教材内容"]}
    
    # 对比类问题：检查是否两个概念都有覆盖
    if _is_comparison_query(query):
        concepts = _extract_concepts(query)
        missing = [c for c in concepts if not _kp_contains(kp_paths, c)]
        if missing:
            return {"sufficient": False, "confidence": 0.3,
                    "missing_info": f"缺少概念「{'」「'.join(missing)}」的相关内容",
                    "missing_kp": missing[0], "covered_aspects": [],
                    "uncovered_aspects": [f"缺少{m}的内容" for m in missing]}
    
    # LLM 判断（仅当规则无法决定时）
    llm_check = await _llm_check(query, context)
    return llm_check
```

**调用位置**：拆 `query_rag_node` 为多个子节点

```python
# agent/nodes.py 新增节点

async def query_rag_node(state: dict) -> dict:
    """RAG 检索（不生成答案）"""
    # ... 现有检索逻辑，返回 context, metadata
    return {"rag_context": context, "rag_metadata": metadata}

async def check_sufficiency_node(state: dict) -> dict:
    """质检：判断检索结果是否足够"""
    result = await check_sufficiency(
        query=state["query"],
        context=state.get("rag_context", ""),
        kp_paths=state.get("rag_metadata", {}).get("source_kp_paths", []),
    )
    retry_count = state.get("retrieval_retry_count", 0)
    return {
        "sufficiency": result,
        "retrieval_retry_count": retry_count + (0 if result["sufficient"] else 1),
    }

async def synthesize_node(state: dict) -> dict:
    """合成最终答案（质检通过后）"""
    # ... 调用 Generator.generate()
    pass
```

**图结构变化**（`agent/graph.py`）：

```
START → build_context → classify → [route_by_complexity]
  ├─ simple → query_rag_once → finalize
  └─ complex → retrieve → check_sufficiency
       ├─ sufficient → synthesize → finalize
       └─ insufficient + retry<N → rewrite_query → retrieve (回退)

query_rag_once = retrieve + synthesize（无质检）
retrieve → check_sufficiency → synthesize（带质检链路）
```

---

### [P2] 3. 复杂查询分解（多跳推理）

**改动范围**：新增 `agent/skills/decompose_query.py`

**思路**：遇到"极限和连续有什么关系"这类多跳问题，LLM 拆成子问题分别检索，结果合并后再质检。

```python
# agent/skills/decompose_query.py

DECOMPOSE_PROMPT = """将复杂问题拆解为多个独立的子问题。
每个子问题应能通过单次检索找到答案。

【用户问题】
{query}

【课程知识点】
{knowledge_points}

输出JSON:
{{
  "sub_queries": [
    {{"id": 1, "query": "极限的定义", "target_concept": "极限", "reason": "需要先了解极限的基本定义"}},
    {{"id": 2, "query": "连续的定义", "target_concept": "连续", "reason": "需要先了解连续的基本定义"}},
    {{"id": 3, "query": "极限和连续的关系", "target_concept": "极限/连续", "reason": "对比分析两者的联系"}}
  ],
  "aggregation_type": "compare"  # compare | sequential | compose
}}
"""

async def decompose_query(query: str, kp_tree: dict) -> dict:
    """将复杂查询拆解为子问题列表"""
    # 规则优先：检测是否为对比类
    concepts = _extract_comparison_concepts(query)
    if concepts:
        return {
            "sub_queries": [
                {"id": 1, "query": concepts[0], "target_concept": concepts[0]},
                {"id": 2, "query": concepts[1], "target_concept": concepts[1]},
            ],
            "aggregation_type": "compare",
        }
    
    # LLM 分解
    return await _llm_decompose(query, kp_tree)
```

**并行检索**：利用 asyncio.gather 同时检索多个子问题

```python
async def parallel_retrieve(session, sub_queries: list, course_id: str) -> list:
    """并行检索多个子问题"""
    retriever = Retriever()
    tasks = [
        retriever.retrieve(session, sq["query"], course_id)
        for sq in sub_queries
    ]
    results = await asyncio.gather(*tasks)
    merged_context = "\n".join(ctx for ctx, _ in results)
    merged_metadata = _merge_metadata([m for _, m in results])
    return merged_context, merged_metadata
```

---

### [P3] 4. 搜索分发器（外部知识补充）

**改动范围**：新增 `agent/tools/web_search.py`

**思路**：当教材内容不足以回答时，通过 MCP 或直接 API 调用网络搜索补充信息。

```python
# agent/tools/web_search.py

class WebSearchTool:
    """网络搜索工具（通过 MCP 或 SerpAPI）"""
    
    async def search(self, query: str, max_results: int = 3) -> list[dict]:
        """搜索并返回结果片段"""
        if self._mcp_client:
            return await self._mcp_search(query, max_results)
        return await self._direct_api_search(query, max_results)
```

**集成到质检决策中**：

```
质检 → 信息不足，需要最新资料 → 触发网络搜索 → 结果加入上下文 → 再次质检
质检 → 信息不足，教材有但未命中 → 查询改写 → 重新检索教材 → 再次质检
```

---

## 三、已有资源复用

| 已有模块 | 复用到 Agentic RAG |
|----------|-------------------|
| `rag/retriever.py` 六阶段检索 | 直接用于多轮检索循环 |
| `rag/query_rewriter.py` 查询改写 | 补搜时自动改写查询 |
| `rag/bm25.py` 关键词检索 | 配合向量检索确保召回 |
| `rag/reranker.py` 重排序 | 每轮检索后精排 |
| `agent/skills/classify_intent.py` | 扩展为复杂度+意图双分类 |
| `agent/context.py` 上下文管理 | 多轮检索上下文的 Token 预算控制 |
| `governance/guardrails.py` 护栏 | 质检通过的答案再经过护栏验证 |
| `evaluation/rag_eval.py` RAGAS | 自动化评估 Agentic RAG vs 传统 RAG |

---

## 四、评估计划

1. **离线对比**：用现有 `eval/questions/` 数据集，对比传统 RAG vs Agentic RAG
2. **核心指标**：Answer Correctness（正确率）、Faithfulness（忠实度）、Context Sufficiency（一次通过率）
3. **成本指标**：每回答的 LLM 调用次数、Token 消耗量、总延迟

---

## 五、实施路线

| 阶段 | 内容 | 工作量 | 效果 |
|------|------|--------|------|
| **P0 智能路由** | classify 扩复杂度、graph 加条件边、简单快速通道 | 1-2天 | 简单问题响应快，降低成本 |
| **P1 质检+多轮** | check_sufficiency 节点、retrieve→synthesize 拆分、重试循环 | 3-4天 | 复杂问题准确率显著提升 |
| **P2 查询分解** | decompose 节点、并行检索、结果合并 | 2-3天 | 多跳/对比问题质量飞跃 |
| **P3 外部搜索** | WebSearchTool、MCP 扩展 | 2天 | 补全教材之外的知识空白 |
