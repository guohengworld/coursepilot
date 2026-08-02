> 状态：路线图已部分实施，混合检索（dense + sparse + RRF）已上线。
> 后续优化项详见 `eval/RAG评估最终报告.md`。
> 最后更新：2026-08-01

# CoursePilot RAG 含金量提升方案

> 文档生成时间：2026-07-14
> 基于代码版本：六阶段检索（改写 → BGE-M3 编码 → Milvus 混合检索 → bge-reranker 重排 → KP 文档金字塔扩展）+ DeepSeek 生成

---

## 1. 当前架构总览

CoursePilot 的 RAG 已经具备一个**相对完整、可扩展的检索管线**：

```text
用户 query
  → 阶段0：DeepSeek 查询改写
  → 阶段1：BGE-M3（dense + sparse）编码
  → 阶段2：Milvus 混合检索 + RRF 融合
  → 阶段3：bge-reranker-v2-m3 精排
  → 阶段4：KP 文档金字塔扩展（同 KP 拉全量 unit → dense 粗排 → reranker 精排）
  → XML 格式上下文 + 课程大纲 → DeepSeek 生成
```

配套能力：摘要桥（SummaryBridge）、引用验证（citation.py）、RAGAS 风格评估（rag_eval.py）、学生画像/最近 QA 上下文注入。

---

## 2. 含金量诊断：强点 vs 弱点

### 2.1 已做得不错的地方

| 模块 | 亮点 |
|------|------|
| **数据切分** | 数学块感知切分（`$$...$$` / `$...$` 保护），段落优先，避免公式被切断。 |
| **索引** | BGE-M3 同时产出 dense + learned sparse，Milvus 做混合检索，已覆盖语义 + 词汇两条路。 |
| **重排** | 使用 cross-encoder 精排，并加入层级惩罚（越深路径轻微扣分）。 |
| **KP 扩展** | 不满足于 top-5 单 unit，而是拉取同 KP 全部 unit 做二次精排，上下文更完整。 |
| **生成 prompt** | 已强制要求 `<ref id="N" />` 引用、LaTeX 公式、边界声明、启发式教学风格。 |
| **评估** | 有 Context Recall / Precision / Faithfulness / Answer Relevancy 四大指标 + LLM-as-Judge。 |

### 2.2 目前明显可提升的短板

| 短板 | 影响 | 证据（代码） |
|------|------|-------------|
| **检索没有真正利用学生画像** | 用户画像（薄弱 KP、掌握度）只在 `build_context` 中加载，未进入 Retriever 的打分或过滤。 | `agent/context.py` 查出 profile 后未传给 `retriever.retrieve`；`retriever.py` 的排序逻辑无 personalization。 |
| **Milvus 混合检索缺少“真” BM25** | `enable_sparse` 依赖 BGE-M3 的 learned sparse，对精确术语/公式名召回不如传统 BM25 稳定。 | `vector_store.py` 仅 dense + sparse vector 两路。 |
| **KP 扩展会引入噪声** | 阶段4把同 KP 下所有 unit 都拉进来，再粗排/精排；如果 KP 本身很大，容易把不相关段落塞进上下文。 | `_kp_expand` 按 KP 拉全量，再截断到 `context_max_chars`。 |
| **生成后没有引用修正** | `citation.py` 只能验证引用 id 是否合法，无法把 LLM 引错的内容纠正或移除。 | `validate_citations` 只返回 `ok/hallucinated`。 |
| **公式/图表/例题未结构化** | 解析出的图片（parsed/*.jpg）和表格目前没有被向量化或索引；例题没有单独标记。 | 图片、表格未进入 `knowledge_units` 或 Milvus schema。 |
| **多轮对话引用消解弱** | 当前只把最近 5 条 QA 拼进 prompt，没有显式做“共指消解”或“子问题继承”。 | `agent/context.py` 简单截断答案到 200 字。 |
| **评估数据没有回流到模型/参数优化** | `rag_eval.py` 输出报告，但未形成自动 bad-case 收集、参数网格搜索后写入最佳配置。 | 网格搜索靠手动 `config_overrides`。 |
| **工程上全链路 CPU 推理** | BGE-M3 + bge-reranker 都是 CPU，延迟会成为高并发瓶颈。 | `encoder.py` / `reranker.py` 都 `device="cpu"`。 |

---

## 3. 含金量提升路线图（按优先级）

### 第一阶段：立竿见影（2-3 周）

| 优先级 | 进化点 | 预期收益 | 关键改动 |
|--------|--------|----------|----------|
| P0 | **PG 全文检索补充 Milvus** | 提升精确术语、公式名、定理名的召回率 | 给 `knowledge_units` 加 `tsvector` 或 `pg_trgm` 索引，检索时三路融合（dense + sparse + BM25） |
| P0 | **检索利用用户画像 boost** | 对学生薄弱 KP 的 unit 加权，提升个性化 | `Retriever.retrieve` 接收 `user_profile`，在 reranker 打分后加 personalized boost |
| P0 | **引用修正后处理** | 显著降低幻觉引用、提升答案可信度 | 生成后解析 `<ref>`，把对应 source 片段再与答案核对，低相关引用移除或重新标号 |
| P1 | **KP 扩展内加入“上下文窗口”** | 减少同 KP 噪声，保留命题/定理的完整逻辑链 | 扩展时只取命中 unit 前后相邻 unit，而非全 KP 拉取 |
| P1 | **检索结果置信度门控** | 避免在检索不相关时还硬生成 | 若 top rerank score < 阈值，直接返回“教材未涉及” |

### 第二阶段：结构性提升（1-2 个月）

| 优先级 | 进化点 | 预期收益 | 关键改动 |
|--------|--------|----------|----------|
| P1 | **例题/定理/公式 结构化元数据** | 学生问“求极限例题”时能精准命中例题，而非概念段 | 在 `knowledge_units` 加 `unit_type`（theorem/example/formula/text）字段，并在 chunking/摘要时识别 |
| P1 | **多查询扩展 + HyDE** | 应对复杂/多知识点问题，提升 recall | 用 LLM 把 query 拆成 2-3 个子问题，各自检索后合并；或对 query 生成假设答案再编码检索 |
| P1 | **查询意图路由** | 区分“概念解释 / 计算题 / 例题 / 证明”，走不同检索策略 | 在意图分类后增加 `question_subtype`，Retriever 按子类型调整 top_k / 是否扩展 / 是否启用 CoT |
| P2 | **reranker 领域微调** | 数学教材的语义匹配更准确 | 用课程 QA 数据构造（query, positive, negative）三元组，微调 bge-reranker 或换 gte-reranker |
| P2 | **层级父子索引** | 父 KP 包含子 KP 内容时，能自动提升父级召回 | 在 Milvus 中增加 `ancestor_kp_paths` 多值字段，检索时做 OR 过滤 |

### 第三阶段：系统级护城河（3-6 个月）

| 优先级 | 进化点 | 预期收益 | 关键改动 |
|--------|--------|----------|----------|
| P2 | **图/表/公式 多模态索引** | 回答能引用教材原图、原表 | 把 MinerU 提取的图片、表格 caption 向量化，建立 image-table-text 联合索引 |
| P2 | **检索-生成联合优化（子问题递归）** | 复杂证明题可拆多步，每步检索 + 生成 | 在 LangGraph 中增加 `decompose` 节点，对复杂题拆分子问题串行/并行检索 |
| P3 | **在线评估 + 反馈闭环** | 持续自动优化 RAG 参数 | 收集用户点赞/点踩、计算在线指标，定期触发 `rag_eval.py` 网格搜索并自动应用最佳配置 |
| P3 | **GPU/量化推理部署** | 降低延迟、提升吞吐 | BGE-M3 转 ONNX/TensorRT，reranker 量化；或接入 vllm/text-embedding-inference 服务 |

---

## 4. 具体模块进化点详解

### 4.1 数据层：让 chunk 更“有结构”

当前 `parser_utils._split_text_v2` 按段落/数学块切分，很合理，但**缺少对教材内容类型的识别**。建议：

- **增加 `unit_type` 字段**：在解析时通过规则识别 `theorem` / `example` / `formula` / `definition` / `text`。
  - 规则示例：包含“例 \d+”“例如”“例题” → example；包含“定理”“引理”“推论” → theorem；包含“定义” → definition；纯 `$$` 块 → formula。
- **公式元数据**：把公式里的 LaTeX 单独提取，作为 `formula_latex` 字段，同时保留 `summary_bridge` 自然语言描述。
- **图片/表格**：MinerU 已经提取到 `parsed/*.jpg` 和 md 表格，应该把它们 caption 化后作为独立 unit 入向量库。
- **chunk overlap 策略**：当前 `chunk_overlap=50` 是全局配置，建议按内容类型动态调整：定理/公式 0 overlap，例题保持完整，叙述文本可 overlap。

### 4.2 检索层：从“六阶段”到“自适应多路”

建议把 `Retriever.retrieve` 升级为**策略化检索**：

```python
class RetrievalStrategy:
    intent: str          # question / practice / diagnose / review
    subtype: str         # concept / calculation / example / proof
    use_hyde: bool
    multi_query_count: int
    expand_mode: str     # "none" | "sibling" | "full_kp"
    boost_weak_kps: bool
```

关键改动：

1. **PG 全文检索补充**：
   ```sql
   ALTER TABLE knowledge_units ADD COLUMN search_vec tsvector
     GENERATED ALWAYS AS (to_tsvector('chinese', coalesce(content,'') || ' ' || coalesce(summary,''))) STORED;
   CREATE INDEX idx_ku_search ON knowledge_units USING GIN (search_vec);
   ```
   检索时先 pg 全文粗排 top-50，再与 Milvus 结果做 RRF 三 fusion。

2. **多查询扩展**：
   - 对复杂问题，用 LLM 生成 2-3 个等价/子问题，分别检索，合并后 rerank。
   - 对计算题，提取“已知条件”和“求解目标”分别生成子查询。

3. **HyDE（假设文档嵌入）**：
   - 用 LLM 为 query 生成一段“理想教材答案”，对这段答案做 embedding，再用它检索真实 unit。
   - 适合概念性问题，能提升 recall 5-15%。

4. **个性化 boost**：
   ```python
   # 在 reranker 打分后，对用户薄弱 KP 轻微加分，已掌握 KP 轻微减分
   for c in candidates:
       if c["kp_path"] in weak_kps:
           c["rerank_score"] += 0.05
   ```

5. **KP 扩展改 neighbor-window**：
   - 不再全 KP 拉取，而是对每个命中 unit，按 `seq_order` 取前后各 2 个相邻 unit，组成“逻辑段落”再精排。
   - 对例题类型，优先取完整例题 block（可能跨多个 unit）。

### 4.3 生成层：从“能回答”到“可验证”

当前 `Generator` 已经做得不错，但可以再加三层保险：

1. **引用修正后处理（Citations Post-Processor）**：
   - 生成答案后，提取所有 `<ref id="N" />`。
   - 对每个引用，把对应 source 片段与答案中的陈述做 mini-reranker / LLM 一致性判断。
   - 如果引用与答案无关，则移除该引用；如果答案缺少引用，则补充最近相关 source 的引用。

2. **答案忠实度自检（Self-Check）**：
   - 在生成 prompt 里增加 instruction：“回答完成后，用一行 [CHECK: 每个论断都能在 <sources> 中找到依据]”。
   - 或独立调用一个轻量 LLM 做 `faithfulness` 二分类，低于阈值则重新生成。

3. **拒绝策略**：
   - 当 top rerank score < 0.3 或上下文与 query 关键词 Jaccard < 0.1 时，直接返回“教材中未找到相关内容，请换个问法”。
   - 避免把无关上下文塞给 LLM 导致幻觉。

4. **多轮消解**：
   - 把 `recent_qa` 中的实体/知识点提取出来，生成“共指消解后的完整 query”再进入检索。
   - 例如学生问“它怎么求？”→ 消解为“泰勒展开的求法”。

### 4.4 评估与运营层：从“离线报告”到“在线闭环”

1. **在线指标看板**：
   - 每次 Agent 调用记录：recall@K、precision@K、rerank score 分布、latency、p99。
   - 用 `observability/metrics.py` 扩展 Prometheus/OpenTelemetry 格式。

2. **Bad case 回流**：
   - 用户点“不满意”时，记录 query、answer、context、引用 id，定期人工/自动标注 ground truth。
   - 形成 `eval/bad_cases.jsonl`，作为 reranker 微调数据。

3. **自动网格搜索**：
   - 每周用 `rag_eval.py` 对黄金数据集做参数扫描（`rrf_k`、`dense_top_k`、`rerank_top_k`、`context_max_chars`、`level_penalty`）。
   - 取 Context Recall ≥ 0.85 且 Faithfulness 最高的配置，自动写回 `rag/config.py` 或配置表。

4. **A/B 实验**：
   - 对查询改写、KP 扩展、HyDE 等开关，按用户分流做 A/B，比较 answer relevancy 和满意度。

---

## 5. 推荐下一步：先做这 3 件事

如果你只能做三件事，建议按这个顺序：

1. **PG 全文检索补充 Milvus 混合检索**（P0，1 周）
   - 改动最小，收益最稳：对“精确术语/公式名”召回提升明显。
2. **生成后引用修正**（P0，1-2 周）
   - 直接提升答案可信度，是学生最感知得到的“含金量”。
3. **Retriever 接入 user_profile 做个性化 boost**（P0，1 周）
   - 让学生感觉“它知道我哪里不会”，这是教育场景的核心差异点。

这三件事完成后，再考虑例题结构化、HyDE、reranker 微调等中长期进化点。

---

## 6. 附：文件改动清单

| 目标能力 | 主要改动文件 |
|----------|--------------|
| PG 全文检索 | `src/coursepilot/rag/vector_store.py` 或新增 `src/coursepilot/rag/fulltext_store.py`；Alembic 迁移给 `knowledge_units` 加 `tsvector` |
| 个性化 boost | `src/coursepilot/rag/retriever.py`（接收 `user_profile`）、`src/coursepilot/agent/skills/query_rag.py`（传 profile） |
| 引用修正 | 新增 `src/coursepilot/rag/citation_post_processor.py`；`src/coursepilot/rag/generator.py` 调用后处理 |
| 内容类型识别 | `src/coursepilot/ingestion/parser_utils.py` 或新增 `src/coursepilot/ingestion/unit_classifier.py`；更新 `models/knowledge_unit.py` |
| 多查询/HyDE | `src/coursepilot/rag/query_rewriter.py` 升级；`src/coursepilot/rag/retriever.py` 支持多查询合并 |
| 在线指标/bad case | `src/coursepilot/observability/metrics.py`；`src/coursepilot/api/agent.py` 增加反馈入口 |
