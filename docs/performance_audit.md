# 性能耗时拆解审计报告

> 基线：`baseline_20260627_075005` | 11 题 | 总耗时 1128.95s（18 分 49 秒）| 平均每题 102.6s

---

## 每道题耗时分布

| # | 耗时 (s) | Recall | Faith | 检索 UUID 数 | 瓶颈判断 |
|---|----------|--------|-------|-------------|----------|
| Q1 | 104.3 | 1.0 | 1.0 | ~121 | Judge |
| Q2 | 98.7 | 1.0 | 1.0 | ~121 | Judge |
| Q3 | 54.4 | 1.0 | 0.833 | ~19 | 均衡 |
| Q4 | 95.3 | 0.0 | 0.273 | ~107 | Judge |
| Q5 | 86.6 | 1.0 | 0.800 | ~17 | Judge |
| Q6 | 121.3 | 1.0 | 0.852 | ~73 | Judge |
| Q7 | 92.0 | 1.0 | 0.308 | ~121 | Judge |
| Q8 | 110.4 | 1.0 | 0.154 | ~121 | Judge |
| Q9 | 62.8 | 1.0 | 0.765 | ~57 | 均衡 |
| Q10 | 115.5 | 1.0 | 0.706 | ~91 | Judge |
| Q11 | 176.7 | 1.0 | 0.0 | ~73 | **Judge（最严重）** |
| **总计** | **1128.95** | | | | |

---

## 耗时拆解模型

每道题耗时 = 检索 + 生成 + Judge × (1 + N_units + 2) + 题间休眠

```
检索阶段 (~5-10s)
├── Query Rewrite (DeepSeek API)       ~3-5s
├── Dense + Sparse Encoding (BGE-M3)   ~1-2s
├── Milvus Hybrid Search               ~0.5s
├── Rerank (bge-reranker-v2-m3)       ~0.5-2s
└── KP Expand (DB query)              ~0.1s

生成阶段 (~10-30s)
└── DeepSeek LLM Generate             ~10-30s

Judge 阶段 (~40-150s) ← 主导瓶颈
├── Context Precision: N_units × ~1s  (per-unit LLM judge)
├── Faithfulness: 1 × ~5-10s          (claim decomposition)
└── Answer Relevancy: 1 × ~2-3s       (0-1 scoring)

题间休眠: 1.0s × 10 = 10s
```

### 耗时占比估算

| 阶段 | 估算占比 | 说明 |
|------|----------|------|
| Judge (Context Precision) | **60-75%** | per-unit LLM judge，每题 15-121 次调用 |
| LLM Generate | 12-20% | 单次 DeepSeek 生成调用 |
| Retrieval | 5-10% | 改写+编码+检索+重排序+KP 扩展 |
| Faithfulness Judge | 3-5% | 单次 claim 分解 |
| 题间休眠 | ~1% | asyncio.sleep(1.0) × 10 |

---

## 关键发现

### 1. Context Precision per-unit judge 是绝对瓶颈

**一道题 121 个 unit → 121 次 LLM API 调用**，每次 ~1s → 仅此一项每题 40-120s。

Q11 极端情况：73 个 unit × ~1.5s（含重试）+ faithfulness + relevancy = **176.7s**。

当前设计：
```python
# 对每个检索到的 unit 逐一调 LLM judge
for content in units:          # 17-121 iterations
    answer = await self._llm_judge(prompt, max_tokens=5)  # ~1s each
    relevant_count += 1 if "yes" else 0
```

**改进方案**：改为 KP 级判定
- 将同一 KP 下的所有 unit 拼接为一段完整上下文（≤ 2000 字）
- 每个 KP 只调用 1 次 judge
- 典型场景：5 个 KP → 5 次调用，每题省 50-100s
- **预估总耗时：1128s → ~180s（减少 84%）**

### 2. 耗时与 unit 数量正相关（但非线性）

| 题目 | UUID 数 | 总耗时 | 每 UUID 均摊 |
|------|---------|--------|-------------|
| Q3 | ~19 | 54.4s | 2.86s |
| Q6 | ~73 | 121.3s | 1.66s |
| Q11 | ~73 | 176.7s | 2.42s |
| Q1 | ~121 | 104.3s | 0.86s |

低 unit 数时固定开销（检索+生成）占主导（Q3: 54.4s 仅处理 19 个 unit）。高 unit 数时 judge 开销线性增长，但 Q1 的 121 个 unit 反而比 Q11 的 73 个更快 → 说明 **LLM API 延迟方差很大**（冷启动、限流、网络波动）。

### 3. LLM Generate 延迟差异大

| 题目 | 生成耗时估算 | answer_chars |
|------|-------------|-------------|
| Q3 | ~7s | 短答案 |
| Q1 | ~14s | 中等 |
| Q10 | ~18s | 长答案（偏导数vs全微分） |

concept 类题（定义题）生成快，comparison 类题（对比题）生成慢，因为需要更长的对比性回答。

### 4. 题间休眠可削减

当前 `asyncio.sleep(1.0)` × 10 题 = 10s。改为 `asyncio.sleep(0.1)` 可省 9s，影响微小但零成本。

---

## 优化路线图

| 优先级 | 措施 | 预期节省 | 实现难度 |
|--------|------|----------|----------|
| **P0** | Context Precision 改为 KP 级判定 | 每题省 50-100s，总耗时 → ~3 分钟 | 低（修改 judge 循环） |
| **P1** | 批量 LLM judge 调用（asyncio.gather 并发） | 每题省 30-60s | 中（需注意 API 限流） |
| P2 | 减少题间休眠 1.0s → 0.1s | 省 9s | 极低 |
| P2 | Faithfulness 用更小模型（如 DeepSeek v4-lite） | 省 ~2s/题 | 低 |
| P3 | 缓存检索结果（同 KP 问题共享检索） | 省 ~5s/题 | 中 |

### P0 详细方案

```python
# 替代当前 _judge_context_precision 中的 per-unit 循环
async def _judge_context_precision_v2(
    self, question: str, top_kp_ids: list[str], session: AsyncSession
) -> float:
    """KP 级 Context Precision：每个 KP 一条完整上下文，一次 judge"""
    relevant_count = 0
    for kp_id in top_kp_ids[:5]:  # 只 judge top-5 KP
        units = await self._load_units_by_kp(session, kp_id)
        kp_context = "\n\n".join(u[:500] for u in units)[:2000]
        answer = await self._llm_judge(
            JUDGE_CONTEXT_PRECISION_PROMPT.format(
                question=question, context=kp_context
            ),
            max_tokens=5,
        )
        relevant_count += 1 if answer.strip().lower().startswith("yes") else 0
    return relevant_count / min(5, len(top_kp_ids))
```

---

## 硬件/环境说明

- **GPU**：NVIDIA (BGE-M3 编码 + bge-reranker-v2-m3)
- **LLM API**：DeepSeek v4-flash（远程 API，网络延迟 ~200ms + 推理时间）
- **向量库**：Milvus Lite（本地文件，`data/milvus/milvus.db`）
- **数据库**：PostgreSQL（本地/远程）
- **gRPC keepalive**：已修复（60s 间隔，解决 GOAWAY "too_many_pings"）
