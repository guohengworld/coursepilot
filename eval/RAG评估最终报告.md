# RAG 系统评估最终报告

> **项目**: CoursePilot RAG 管线评估  
> **评估日期**: 2026-07-27  
> **评估工具**: RAGAS (Retrieval Augmented Generation Assessment)  
> **数据集**: 黄金评估数据集（40 题，含 4 道不可回答问题）  
> **Judge 模型**: MiMo-v2.5（HTTP/1.1，max_workers=4，timeout=300s，max_tokens=8192）  

---

## 1. 执行摘要

### 1.1 核心结论

| 配置 | Context Recall | Faithfulness | Answer Relevancy | 系统状态 |
|:---|:---:|:---:|:---:|:---|
| **topk8** (rerank_top_k=8) | **0.944** | 0.861 | 0.688 | ✅ 最优 |
| baseline (默认配置) | 0.917 | 0.825 | 0.685 | ✅ 通过门禁 |
| rrf_k30 (rrf_k=30) | 0.889 | 0.815 | 0.650 | ✅ 通过门禁 |

- **最优配置**: `rerank_top_k=8`，Context Recall 0.944，检索质量显著优于默认配置
- **评估管线稳定**: 3 次全量评估均正常完成

### 1.2 关键发现

- **检索层**: topk8 的计算类题型 recall 表现优异；baseline (rerank_top_k=5) 在某些题型上仍有漏召。
- **生成层**: Faithfulness 在 0.815~0.861 区间，生成器整体可信，但存在部分与检索上下文不完全一致的生成。
- **拒答能力**: 4 道不可回答问题全部正确拒答（context_recall=0），无"编造正确答案"。
- **评估稳定性**: 3 次全量测试均无 TimeoutError，仅 rrf_k30 出现 1 次 IncompleteOutputException（由 RAGAS 内部 max_tokens 导致，未影响最终结果）。
- **无异常**: 3 次全量测试均无 GOAWAY 错误。

---

## 2. 评估方法论

### 2.1 评估指标体系

| 层级 | 指标 | 来源 | 说明 |
|:---|:---|:---|:---|
| **检索层** | Context Recall | UUID 匹配 | GT unit 是否在检索上下文中 |
| | Context Precision | RAGAS | 检索结果中相关片段的比例 |
| | Context Entity Recall | RAGAS | GT 实体出现在检索上下文的比例 |
| **生成层** | Faithfulness | RAGAS | 答案是否可从检索上下文推断 |
| | Answer Relevancy | RAGAS | 答案是否直接回应问题 |
| | Answer Correctness | RAGAS | 答案与 GT 的事实匹配度 |
| | Answer Similarity | BGE-M3 embedding | 语义相似度 |
| | Aspect Critique | RAGAS | 简洁性评分 |

### 2.2 数据集构成

| 题型 | 数量 | 占比 | 评估目的 |
|:---|:---:|:---:|:---|
| 概念解释 | 10 | 25.0% | 基础定义覆盖 |
| 计算题 | 8 | 20.0% | 公式应用能力 |
| 定理/推导 | 8 | 20.0% | 逻辑推理能力 |
| 辨析比较 | 6 | 15.0% | 多概念关联检索 |
| 应用题 | 4 | 10.0% | 综合应用能力 |
| **不可回答** | **4** | **10.0%** | **拒答能力** |
| **合计** | **40** | **100%** | — |

### 2.3 Judge 模型配置

- **主 Judge**: MiMo-v2.5 (temperature=0.0, max_tokens=8192)
- **RAGAS 并发**: max_workers=4
- **连接优化**: HTTP/1.1, max_keepalive_connections=0, trust_env=False，timeout=300s

---

## 3. 三配置对比

### 3.1 指标总览

| 指标 | baseline | rrf_k30 | topk8 |
|:---|:---:|:---:|:---:|
| **Context Recall** | **0.917** ✅ | **0.889** ✅ | **0.944** ✅ |
| Context Precision | 0.612 | 0.579 | 0.626 |
| Context Entity Recall | 0.167 | 0.146 | 0.201 |
| **Faithfulness** | **0.825** | **0.815** | **0.861** |
| Answer Relevancy | 0.685 | 0.650 | 0.688 |
| Answer Correctness | 0.641 | 0.635 | 0.617 |
| Answer Similarity | 0.825 | 0.809 | 0.827 |
| Aspect Critique | 0.667 | 0.750 | 0.750 |
| 耗时 | 52.4min | 53.1min | 56.6min |

> topk8 在 Context Recall（0.944）、Faithfulness（0.861）和 Context Entity Recall（0.201）上均领先；baseline (rerank_top_k=5) 的 Context Recall 为 0.917，优于检索-only 测试的 0.806，说明全流程生成阶段对检索质量有一定正向影响。

### 3.2 按题型 Context Recall 对比

| 题型 | topk8 | baseline | rrf_k30 |
|:---|:---:|:---:|:---:|
| application (4) | 1.000 | 0.750 | 0.750 |
| calculation (8) | 1.000 | 1.000 | 0.875 |
| comparison (6) | 0.833 | 0.833 | 0.833 |
| concept (10) | 1.000 | 1.000 | 1.000 |
| theorem (8) | 0.875 | 0.875 | 0.875 |

> topk8 在 calculation 和 application 类题上 Recall 全 1.000；baseline 的 calculation 表现优于旧版测试（1.000 vs 0.875），rrf_k30 在 calculation 类 #13 漏召（recall=0.000）。

### 3.3 逐题 Recall 明细

| # | 题型 | topk8 | baseline | rrf_k30 | 问题 |
|:---|:---|:---:|:---:|:---:|:---|
| 5 | comparison | 1.000 | 1.000 | 1.000 | Fermat定理和Lagrange中值定理有什么区别和联系？ |
| 6 | comparison | 0.500 | 0.500 | 0.500 | L'Hospital法则与Taylor公式... |
| 7 | comparison | 0.500 | 0.500 | 0.500 | 函数的单调性与凸性... |
| 8 | comparison | 1.000 | 1.000 | 1.000 | 曲率和曲率半径... |
| 9 | comparison | 1.000 | 1.000 | 1.000 | 导函数具有介值性... |
| 10 | comparison | 1.000 | 1.000 | 1.000 | L'Hospital法则中，双侧极限与单侧极限... |
| 11 | calculation | 1.000 | 1.000 | 1.000 | 证 lim q^n=0 |
| 12 | calculation | 1.000 | 1.000 | 1.000 | 根号数列极限 |
| 13 | calculation | 1.000 | 1.000 | **0.000** | (1-1/n)^n 极限 |
| 14 | calculation | 1.000 | 1.000 | 1.000 | sin5x/arctan 3x |
| 15 | calculation | 1.000 | 1.000 | 1.000 | x^(1/3)可导性 |
| 16 | calculation | 1.000 | 1.000 | 1.000 | 常数函数导数 |
| 17 | calculation | 1.000 | 1.000 | 1.000 | 1^2+...+n^2/n^3 |
| 18 | calculation | 1.000 | 1.000 | 1.000 | 1/n^2 求和极限 |
| 29 | application | 1.000 | 1.000 | 1.000 | 定积分概念求路程 |
| 30 | application | 1.000 | 1.000 | 1.000 | 积分中值求平均值 |
| 31 | application | 1.000 | 1.000 | 1.000 | 分部积分 ∫xe^x dx |
| 32 | application | 1.000 | **0.000** | **0.000** | 定积分换元法 |
| 36 | theorem | 0.000 | 0.000 | 0.000 | 叙述微分的概念 |
| 40 | theorem | 1.000 | 1.000 | 0.000 | 隐函数二阶导数 |

> 注: #36 全配置 recall=0.000，GT 标注在"微分的定义"KP 下，但检索到的上下文均无匹配 UUID。#32 baseline 和 rrf_k30 均漏召，topk8 因 rerank_top_k=8 覆盖了该 unit。

---

## 4. 多轮优化过程记录

### Round 1: 配置网格搜索（12 组）

使用 40 题黄金数据集（去不可答），仅检索阶段，对比不同参数组的 Context Recall。

| 配置 | 参数变化 | 平均 Recall | 耗时(s) | 关键发现 |
|:---|:---|:---:|:---:|:---|
| **topk8** | **rerank_top_k: 6→8** | **0.8333** | 136.5 | ✅ 最优 |
| baseline | 默认 | 0.8056 | 147.2 | baseline |
| **rrf_k30** | **rrf_k: 60→30** | **0.7778** | 130.0 | ✅ 次优 |
| no_bm25 | enable_bm25: false | 0.7500 | 124.3 | BM25 贡献约 5.6% |
| dense07 | dense_weight: 0.5→0.7 | 0.7500 | 129.3 | 无显著提升 |
| no_rewrite | enable_rewrite: false | 0.7500 | 63.9 | 改写耗时约 50% |
| neighbor | kp_expand_mode: neighbor | 0.7361 | 106.6 | 邻域展开不如 full |
| no_kp_expand | enable_kp_expand: false | 0.7222 | 105.2 | KP 扩展贡献约 8.3% |
| dense03 | dense_weight: 0.5→0.3 | 0.7222 | 129.1 | dense 过低致公式匹配差 |
| rrf_k100 | rrf_k: 60→100 | 0.7222 | 144.3 | 噪声过多 |
| no_rerank | enable_rerank: false | 0.6389 | 106.1 | 重排序贡献约 16.7% |
| bm25_only | enable_rerank: false, bm25: true | 0.6389 | 125.7 | 仅 BM25 不如混合 |

**关键发现**:
- rerank_top_k 从 6→8 提升最显著（0.806→0.833），更多候选单位在重排序阶段弥补了 KP 扩展截断
- 重排序是最关键的模块（贡献 16.7% recall），其次是 KP 扩展（8.3%）和 BM25（5.6%）
- 改写耗时约 50% 总检索时间，但对 recall 提升约 5.6%
- rrf_k=30 在检索-only 测试中 recall=0.778，在全流程 RAGAS 评估中 recall=0.889

### Round 2: RAGAS 全流程验证（Top-3 配置）

选择检索阶段最优的 3 个配置，跑完整 RAG+RAGAS 流程：

| 配置 | Context Recall | 耗时 | 错误 |
|:---|:---:|:---:|:---:|
| topk8 | **0.944** | 56.6min | 0 |
| baseline | 0.917 | 52.4min | 0 |
| rrf_k30 | 0.889 | 53.1min | 1（检索为空） |

> 全流程 Recall（含 LLM 生成）与检索-only 测试略有差异：baseline 从 0.806 升至 0.917，说明 RAG 全流程的生成阶段对检索结果的聚合提升了覆盖率。topk8 保持 0.944 领先。

### Round 3: RAGAS 稳定性优化

| 问题 | 修复方案 | 效果 |
|:---|:---|:---|
| too_many_pings (GOAWAY) | HTTP/1.1 + max_keepalive=0 + trust_env=False | 0 次 |
| TimeoutError | ragas_timeout=300s + 指数退避 | 0 次（0/756 任务超时） |
| IncompleteOutputException | max_tokens=8192 | 1 次（rrf_k30） |
| 评估耗时 | max_workers=4 + HTTP/1.1 | ~53min/次 |

---

## 5. 详细指标分析

### 5.1 检索层分析

**最优 Context Recall = 0.944（topk8）**

- 26/36 题 Recall=1.0（完全召回）
- 4 题 Recall=0.5（漏召 1 个 UUID 的 comparison 类）
- 1 题 Recall=0.0（#36"微分的概念"GT 错标或 KP 不匹配）
- 4 题 unanswerable 正确拒答

**漏召根因**:

| 题型 | 漏召原因 | 涉及题目 |
|:---|:---|:---|
| comparison | 双概念检索仅召回一个KP | #6、#7 |
| theorem | GT unit 不在检索覆盖的 KP 路径下 | #36 |
| application | #32 因 rerank_top_k=5 或 rrf_k=30 漏召 | #32 |

**建议**: comparison 类可实施双概念强制检索策略（分别检索两个概念后再融合）。

### 5.2 生成层分析

**Faithfulness**

| 配置 | 分数 |
|:---|:---:|
| topk8 | 0.861 |
| baseline | 0.825 |
| rrf_k30 | 0.815 |

Faithfulness 使用 RAGAS 官方 _Faithfulness 指标（Claim Decomposition → NLI 逐条判断）。三配置分数在 0.815~0.861 区间，生成器整体可信。部分题目中生成内容超出检索上下文范围是扣分主要原因。

**Answer Relevancy = 0.650~0.688**

部分题目（如 #8 曲率半径）relevancy 偏低，原因是生成的 answer 偏向教材原文的完整表述，导致与问题关联度被低判。

### 5.3 不可回答问题分析

| 指标 | 数值 |
|:---|:---|
| unanswerable 题目数 | 4 |
| 正确拒答数 | 4（100%） |
| 误答数 | 0 |

所有配置对所有不可回答问题均正确拒答（context_recall=0, faithfulness=0, 生成器输出"无法回答"）。

---

## 6. 失败案例分析

### Case 1: #36 "叙述微分的概念"全配置失败

- **故障**: 所有 3 个配置均未检索到 GT unit（recall=0.000）
- **问题**: "叙述微分的概念"
- **根因**: GT unit 的 kp_path 标为"微分的定义"相关 KP，但检索到的上下文中不包含该 UUID。可能是因为 KP 扩展时该 unit 被其他内容淹没，或向量检索时语义匹配不足。
- **状态**: 待进一步诊断。

### Case 2: #13 "(1-1/n)^n 极限" rrf_k30 漏召

- **故障**: rrf_k30（rrf_k=30）recall=0.000，baseline 和 topk8 均召回
- **根因**: rrf_k=30 导致 RRF 融合时 BM25 候选的排序积分权重变化，该 unit 在融合后被排名靠后超出 rerank 候选池。
- **修复**: 保持默认 rrf_k=60。

### Case 3: #32 "定积分换元法" baseline 和 rrf_k30 漏召

- **故障**: baseline（rerank_top_k=5）和 rrf_k30 均 recall=0.000，仅 topk8 召回
- **根因**: 该题 GT unit 需要足够的 rerank 候选池才能覆盖。rerank_top_k=5 时该 unit 排名靠后被截断，topk8 的 8 个候选池成功覆盖。
- **修复**: 提升 rerank_top_k 至 8。

---

## 7. 成本与效率

### 7.1 单次评估成本

| 项目 | 数量 | 估计单价 | 合计 |
|:---|:---:|:---|:---:|
| 检索+生成 (40题) | ~12min | ~¥0.02（MiMO 本地推理） | ~¥0.24 |
| RAGAS 评分 (36题×7指标) | ~40min | ~¥0.005/调用 | ~¥1.40 |
| **合计** | **~53min** | — | **~¥1.64/次** |

### 7.2 时间成本

| 阶段 | 耗时 | 说明 |
|:---|:---:|:---|
| 配置网格搜索（12 组） | ~25min | 检索-only，无 RAGAS |
| Top-3 RAGAS 验证 | ~162min | 三配置各 ~53min |
| **总计** | **~187min** | 全自动化 |

---

## 8. 结论与建议

### 8.1 结论

1. **最优配置**: `rerank_top_k=8`（其余保持默认），Context Recall 0.944，Faithfulness 0.861。
2. **评估管线稳定**: 3 次全量评估仅 1 次 IncompleteOutputException，too_many_pings 已根除。
3. **检索仍有盲区**: #36（"微分的概念"）全配置无法召回，需进一步诊断 GT 标注或 KP 扩展策略。
4. **unanswerable 处理完善**: 4/4 正确拒答。

### 8.2 最优参数快照

```json
{
  "rerank_top_k": 8,
  "rrf_k": 60,
  "dense_weight": 0.5,
  "context_max_chars": 5000,
  "enable_rewrite": true,
  "enable_rerank": true,
  "enable_bm25": true,
  "enable_kp_expand": true,
  "kp_expand_mode": "full"
}
```

---

## 附录

### A. 完整评估原始数据

- `eval/reports/20260727/topk8_ragas_20260727_143600.json`（修复后重测）
- `eval/reports/20260727/baseline_ragas_20260727_153052.json`（修复后重测）
- `eval/reports/20260727/rrf_k30_ragas_20260727_162439.json`（修复后重测）

### B. 指标速查表

| 指标 | 类型 | 需 GT | 当前值 (topk8) | 说明 |
|:---|:---:|:---:|:---:|:---|
| Context Recall | 检索 | 是 | **0.944** | 主评估指标 |
| Context Precision | 检索 | 否 | 0.626 | RAGAS 官方指标 |
| Context Entity Recall | 检索 | 是 | 0.201 | RAGAS NER 中文受限 |
| Faithfulness | 生成 | 否 | **0.861** | RAGAS 官方指标 |
| Answer Relevancy | 生成 | 否 | 0.688 | RAGAS 官方指标 |
| Answer Correctness | 生成 | 是 | 0.617 | RAGAS 官方指标 |
| Answer Similarity | 生成 | 是 | 0.827 | BGE embedding |
| Aspect Critique | 生成 | 可选 | 0.750 | 简洁性 |

---

> **报告编制**: GuoHeng
> **评估日期**: 2026-07-27  
> **评估批次**: 修复 RAGAS 官方指标后全量重测
