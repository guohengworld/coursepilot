# RAG 评估体系构建指南（2026.7 版）

## 1. 黄金评估数据集构建

### 路线：LLM 生成 + 人工校验（唯一推荐）

**适用场景**：所有规模。2026 年 LLM 生成质量已足够高，人工校验成本远低于纯手动编写。

**具体做法：**

1. **文档预处理**：将教材按 KP/Unit 切片，保留 `uuid` + `content`（完整文本）+ `page_ref` + `kp_path`。
2. **LLM 生成问题**：使用 DeepSeek/GPT-4o 读取每个 Unit 内容，按 5 种问题类型各生成 2-3 题。
3. **人工校验**：
   - 删除模糊/歧义问题
   - 修正答案中的事实错误
   - 确认 `ground_truth_contexts` 精确指向相关 Unit 的 UUID
   - 标注"不可回答"问题（测试拒答能力）

**规模建议**：

- 单课程：30-50 题（覆盖全部 KP，每 KP 至少 2 题）
- 必须包含 10% "不可回答"问题（答案不在知识库中）

| 类型 | 占比 | 示例 |
| :--- | :--- | :--- |
| 概念解释 | 25% | "函数极限的定义" |
| 计算题 | 20% | "解下列绝对值不等式 |x+1| < |2x - 3|" |
| 定理/推导 | 20% | "叙述定积分的分部积分公式" |
| 辨析比较 | 15% | "无穷小和无穷大有什么区别" |
| 应用题 | 10% | "用微分方程求连续复利计算问题" |
| 不可回答 | 10% | "请解释量子纠缠在微积分中的应用"（知识库无此内容） |

---

## 2. 评估维度与指标体系

### 2.1 完整指标矩阵（8 项）

RAGAS 2026 提供 **8 大核心指标**，分为检索层与生成层，按是否需要参考答案分类：

#### 检索层（Retriever Quality）

| 指标 | 含义 | 需要 Ground Truth | 评估方式 |
| :--- | :--- | :--- | :--- |
| **Context Precision** | 召回的 chunk 中相关 chunk 的比例，且相关 chunk 是否排在前面 | 否（LLM 判断）/ 是（更准） | LLM-as-judge 或字符串匹配 |
| **Context Recall** | 参考答案中的信息被检索上下文覆盖的比例 | **是** | 客观指标，字符串/UUID 匹配 |
| **Context Entity Recall** | 参考答案中的实体出现在检索上下文中的比例 | **是** | 实体抽取 + 匹配 |

#### 生成层（Generation Quality）

| 指标 | 含义 | 需要 Ground Truth | 评估方式 |
| :--- | :--- | :--- | :--- |
| **Faithfulness** | 答案中的每个 claim 是否都能从检索上下文中推断（防幻觉） | **否** | 两阶段流水线：Claim 分解 → NLI 推理 |
| **Answer Relevancy** | 答案是否直接回应问题，有无离题或冗长 | **否** | LLM 反推问题 + 相似度计算 |
| **Answer Correctness** | 答案与标准答案在事实与语义上的综合匹配度 | **是** | F1 事实匹配 + 语义相似度加权 |
| **Answer Similarity** | 答案与标准答案的语义相似度 | **是** | Embedding 余弦相似度 |
| **Aspect Critique** | 对答案在特定维度（有害性、偏见、简洁性等）的自定义评判 | 可选 | 自定义 prompt + LLM 打分 |

### 2.2 两阶段评估流水线（RAGAS 核心机制）

Faithfulness 不依赖简单 prompt 打分，而是结构化流水线：

```
生成答案
    ↓
[阶段 1] Claim Decomposition（声明分解）
    → 将答案拆分为原子化、自包含的独立声明
    → 例："The plan costs $40/month and includes 10 seats"
       分解为："costs $40/month" + "includes 10 seats"
    ↓
[阶段 2] Natural Language Inference（自然语言推理）
    → 对每个原子声明，判断检索上下文是：
       • entail（蕴含）✓
       • contradict（矛盾）✗
       • neutral（中性）?
    ↓
Faithfulness = 被蕴含声明数 / 总声明数
```

**优势**：能捕捉"四个正确句子中夹带一个伪造数字"的部分幻觉，避免整体打分掩盖问题。

### 2.3 指标选择决策树

```
是否有标注的黄金 chunk？
    ├─ 有 → 加入 Context Precision, Context Recall, Context Entity Recall
    └─ 无 → 使用 LLM-judge 版 Context Precision / Context Relevance

知识库是宽域还是窄域？
    ├─ 宽域（多主题）→ 优先保证 Context Recall（召回不足是主要风险）
    └─ 窄域（单领域）→ 优先保证 Context Precision（噪声是主要风险）

错误答案的代价？
    ├─ 高 stakes（法律/医疗/合规）→ 追踪全部 8 项指标
    │   → Faithfulness + Groundedness 作为发布门禁
    │   → 运行时加护栏（拒绝低置信度回答）
    └─ 低 stakes（FAQ/内部搜索）→ Faithfulness + Answer Relevance 即可
        → 跳过重指标以保证速度
```

---

## 3. 标注数据格式

```json
{
  "question": "二重积分的几何意义是什么",
  "answer": "二重积分 ∫∫_D f(x,y) dσ 的几何意义是...",
  "ground_truth": "二重积分 ∫∫_D f(x,y) dσ 表示以 D 为底、z=f(x,y) 为顶的曲顶柱体体积...",
  "ground_truth_contexts": ["uuid-aaa", "uuid-bbb"],
  "course_id": "40f17aac-bfc5-4737-9dba-638a3e46fb63",
  "question_type": "concept",
  "kp_path": "微积分/.../9.1 重积分的概念和性质",
  "unanswerable": false
}
```

> `ground_truth_contexts` 是评估核心——定义"标准答案应该引用哪些 unit"。该字段质量直接决定 Context Recall 可信度。

---

## 4. RAGAS 集成方式

### 4.1 基础评估

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness, ResponseRelevancy,
    LLMContextPrecisionWithReference, LLMContextRecall,
    ContextEntityRecall, AnswerCorrectness,
    AnswerSimilarity, AspectCritique
)
from ragas.llms import LangchainLLMWrapper

metrics = [
    Faithfulness(llm=evaluator_llm),
    ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    LLMContextPrecisionWithReference(llm=evaluator_llm),
    LLMContextRecall(llm=evaluator_llm),
    ContextEntityRecall(),
    AnswerCorrectness(llm=evaluator_llm),
    AnswerSimilarity(embeddings=evaluator_embeddings),
    AspectCritique(name="conciseness", definition="答案是否简洁，无冗余信息"),
]

ds = Dataset.from_list(results)
scores = evaluate(ds, metrics=metrics, llm=LangchainLLMWrapper(evaluator_llm))
```

### 4.2 Judge 模型选择

| 场景 | 模型 |
| :--- | :--- |
| 默认 | DeepSeek-v4-pro |
| 备选 | MiMo-v2.5-pro |

### 4.3 与 DeepEval / LangSmith 的组合方案

RAGAS 是**指标库**，不解决测试框架和可观测性。2026 年最佳实践是组合使用：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   RAGAS     │    │  DeepEval   │    │  LangSmith   │
│  (指标计算)  │ →  │ (测试框架)  │ →  │ (生产监控)  │
│  8大核心指标 │    │ Pytest断言  │    │ 在线追踪    │
│  检索/生成   │    │ CI门禁      │    │ 实时告警    │
└─────────────┘    └─────────────┘    └─────────────┘
```

- **RAGAS**：提供 RAG 专项指标（Faithfulness, Context Recall 等）
- **DeepEval**：提供 `assert_test()` 断言、`@pytest.mark.parametrize` 参数化测试、Red-teaming
- **Langfuse / LangSmith**：生产链路追踪、在线评估、延迟/成本监控

---

## 5. 参数配置说明与网格搜索

### 5.1 完整参数清单

RAG 管线所有可配置参数在 `src/coursepilot/rag/config.py` 的 `RAGConfig` 中集中管理，共 **18 项**，按功能分为 5 类：

#### 5.1.1 功能开关（Function Switches）

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `enable_rewrite` | `True` | 查询改写开关。关闭后直接用原始 query 检索，不经过 LLM 改写 |
| `enable_sparse` | `True` | 稀疏检索开关。关闭后只用 dense 向量检索（Milvus ANNS） |
| `enable_bm25` | `True` | BM25 关键词检索开关。关闭后只走 Milvus 混合检索 |
| `enable_rerank` | `True` | 重排序开关。关闭后 RRF 融合结果直接按 score 取 top_k，跳过 cross-encoder |
| `enable_kp_expand` | `True` | KP 文档金字塔扩展开关。关闭后只使用检索直接命中的 unit 文本 |
| `kp_expand_mode` | `"full"` | KP 扩展模式：`"full"` 拉取同 KP 下全部 unit；`"neighbor"` 只取命中 unit 前后各 N 个相邻 unit |
| `kp_neighbor_window` | `2` | `neighbor` 模式下前后各取 N 个相邻 unit（仅当 `kp_expand_mode="neighbor"` 时生效） |

#### 5.1.2 阈值类（Thresholds）

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `reranker_min_score` | `0.3` | 重排序分数低于此值的 source 直接丢弃，过滤低质量候选 |
| `context_max_chars` | `5000` | 送入 LLM 的上下文软上限（字符数），超过后截断 |

#### 5.1.3 检索参数（Retrieval Params）

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `dense_top_k` | `20` | Dense 向量检索返回的候选条数 |
| `sparse_top_k` | `20` | Sparse 向量检索返回的候选条数（仅 `enable_sparse=True` 时生效） |
| `rrf_k` | `60` | RRF 融合参数 `k`，控制稀疏/稠密检索结果的排序平衡 |
| `dense_weight` | `0.5` | Dense 检索在 RRF 融合中的权重（0~1），sparse 权重自动为 `1 - dense_weight` |
| `rerank_top_k` | `5` | 重排序后最终送入生成器的 chunk 数量 |

#### 5.1.4 BM25 参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `bm25_top_k` | `20` | BM25 关键词检索返回的候选条数（仅 `enable_bm25=True` 时生效） |
| `bm25_cache_ttl` | `600` | BM25 索引缓存有效期（秒） |

#### 5.1.5 编码参数（Encoding Params）

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `batch_size` | `32` | BGE-M3 编码器的 batch size，影响编码速度和显存占用 |
| `dim` | `1024` | BGE-M3 dense 向量维度，需与 Milvus collection schema 一致 |

> **更新参数后**：直接修改 `config.py` 中的默认值即可全局生效。网格搜索时通过 `config_overrides` 临时覆盖，不会改动原始值。

### 5.2 网格搜索空间

网格搜索聚焦 **4 个对 RAG 效果影响最大的参数**，其余参数保持默认值：

| 参数 | 默认值 | 搜索范围 | 影响 |
| :--- | :--- | :--- | :--- |
| `rrf_k` | 60 | [20, 40, 60, 100] | 融合稀疏与稠密检索的权重平衡 |
| `rerank_top_k` | 5 | [3, 5, 8, 10] | 重排序后送入生成器的 chunk 数 |
| `context_max_chars` | 5000 | [4000, 6000, 8000, 10000] | 生成器上下文窗口利用率 |
| `dense_weight` | 0.5 | [0.3, 0.5, 0.7] | 稠密检索在 RRF 中的权重 |

> 如需搜索其他参数（如 `reranker_min_score`、`bm25_top_k` 等），可通过 `--params` 自定义网格：  
> `PYTHONPATH=src python -m eval.eval_ragas grid --params '{"reranker_min_score":[0.2,0.3,0.4]}'`

### 5.3 务实搜索策略（避免 1024 组合爆炸）

```
Round 1: rrf_k × dense_weight（4×3=12 组合）
    → 固定 rerank_top_k=5, context_max_chars=6000
    → 目标：找到最优检索融合策略

Round 2: rerank_top_k × context_max_chars（4×4=16 组合）
    → 固定 Round 1 最优 rrf_k + dense_weight
    → 目标：找到最优生成输入规模

总计：12 + 16 = 28 次评估
```

### 5.4 成本估算

| 项目 | 单价 | 数量 | 总价 |
| :--- | :--- | :--- | :--- |
| 单次评估（30 题，GPT-4o judge） | ~¥1.5 | 31 次 | ~¥46.5 |
| 数据集生成（LLM + 人工校验） | ~¥20/题 | 30 题 | ~¥600 |
| **总计** | | | **~¥650** |

> 什么玩意儿，这么贵？
> 开发环境使用本地模型可将评估成本降至 ~¥0，但 CI 门禁仍用 DeepSeek-v4-pro。

---

## 6. CI/CD 质量门禁

### 6.1 阈值设定

```python
THRESHOLDS = {
    "faithfulness": 0.85,        # 幻觉容忍度
    "answer_relevancy": 0.80,    # 跑题容忍度
    "context_recall": 0.85,      # 检索召回底线（根因指标）
    "context_precision": 0.75,   # 检索精度底线
    "answer_correctness": 0.80,  # 事实正确性底线
}
```

### 6.2 门禁脚本

```python
import sys
from ragas import evaluate

def quality_gate(dataset, metrics, thresholds):
    result = evaluate(dataset, metrics=metrics)
    means = result.to_pandas().mean(numeric_only=True)

    failures = []
    for metric, threshold in thresholds.items():
        actual = means.get(metric, 0)
        if actual < threshold:
            failures.append(f"{metric}: {actual:.3f} < {threshold}")

    if failures:
        print("❌ RAG quality gate FAILED:")
        for f in failures:
            print(f"   • {f}")
        sys.exit(1)
    else:
        print("✅ RAG quality gate PASSED")
        sys.exit(0)
```

### 6.3 CI 实践

- **PR 门禁**：使用代表性子集（10 题，快速反馈，~30 秒）
- **Nightly 完整评估**：全量 30-50 题
- **阈值策略**：设定略低于当前基线（如基线 0.88，阈值 0.85），避免正常波动阻塞发布
- **方差控制**：每次评估运行 2-3 次取平均，降低 LLM judge 随机性

---

## 7. 文件结构设计

```text
src/coursepilot/evaluation/
├── __init__.py
├── rag_eval.py              # RAGAS 评估核心类
├── dataset_generator.py     # LLM 生成 + 人工校验流水线
├── quality_gate.py          # CI/CD 门禁脚本
└── metrics_config.py        # 指标与阈值配置

scripts/
├── eval_ragas.py            # CLI: 单次评估 / 网格搜索 / 报告
├── generate_eval_dataset.py # CLI: 生成标注数据集
└── quality_gate.py          # CLI: CI 门禁

tests/fixtures/
├── eval_questions.json      # 30-50 条标注数据
└── eval_thresholds.yaml     # 门禁阈值配置
```

### 核心接口

```python
# evaluation/rag_eval.py
class RAGEvaluator:
    def __init__(
        self,
        llm: LangchainLLMWrapper,
        embeddings: Embeddings,
        metrics: list | None = None,
    ): ...

    async def evaluate_single(
        self, question: str, course_id: str
    ) -> dict: ...

    async def run_eval(
        self, eval_data: list[dict], session: AsyncSession
    ) -> dict: ...

    def grid_search(
        self, eval_data: list[dict], param_grid: dict
    ) -> pd.DataFrame: ...

# evaluation/dataset_generator.py
class EvalDatasetGenerator:
    def __init__(self, llm: BaseLLM): ...

    async def generate_from_unit(
        self, unit: dict, num_questions: int = 3
    ) -> list[dict]: ...

    async def generate_batch(
        self, units: list[dict], questions_per_unit: int = 3
    ) -> list[dict]: ...
```

### CLI 命令

```bash
# 生成标注数据集
PYTHONPATH=src python -m scripts.generate_eval_dataset     --course-id 40f17aac-bfc5-4737-9dba-638a3e46fb63     --output tests/fixtures/eval_questions.json

# 单次 baseline 评估
PYTHONPATH=src python -m scripts.eval_ragas --baseline

# 网格搜索
PYTHONPATH=src python -m scripts.eval_ragas --grid-search

# 指定参数
PYTHONPATH=src python -m scripts.eval_ragas     --rrf-k 40 --rerank-top-k 8 --dense-weight 0.7

# CI 门禁
PYTHONPATH=src python -m scripts.quality_gate
```

---

## 8. 执行清单

| 步骤 | 任务 | 预计时间 | 交付物 |
| :--- | :--- | :--- | :--- |
| 1 | 安装依赖：`pip install ragas datasets deepeval` | 5 min | 环境就绪 |
| 2 | 运行数据集生成脚本，人工校验 30 题 | 2-3 h | `eval_questions.json` |
| 3 | 实现 `rag_eval.py` + `dataset_generator.py` | 3-4 h | 核心模块 |
| 4 | 跑 Baseline，记录 8 项指标初始值 | 10 min | 基线报告 |
| 5 | 执行 3 轮网格搜索（31 次评估） | 1-2 h | 最优参数组合 |
| 6 | 设定门禁阈值，配置 CI 流水线 | 1 h | `.github/workflows/rag-quality.yml` |
| 7 | 接入 Langfuse 生产监控（可选） | 2 h | 在线追踪面板 |

---

## 附录：指标速查表

| 指标 | 类型 | 需要 GT | 生产监控 | CI 门禁 | 根因定位 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Context Precision | 检索 | 否/是 | ✓ | ✓ | 检索噪声 |
| Context Recall | 检索 | **是** | ✗ | **✓** | **检索漏召** |
| Context Entity Recall | 检索 | **是** | ✗ | ✓ | 实体覆盖 |
| Faithfulness | 生成 | **否** | **✓** | **✓** | **幻觉** |
| Answer Relevancy | 生成 | **否** | **✓** | ✓ | 跑题 |
| Answer Correctness | 生成 | **是** | ✗ | **✓** | 事实错误 |
| Answer Similarity | 生成 | **是** | ✗ | ✗ | 语义偏差 |
| Aspect Critique | 生成 | 可选 | ✓ | ✓ | 自定义维度 |
