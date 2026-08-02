> 状态：设计已锁定，核心检索管线已实现。
> 代码对应：`src/coursepilot/rag/`（encoder、retriever、reranker、generator、vector_store）。
> 最后更新：2026-08-01

# CoursePilot RAG 引擎设计 v1.0

> 最终方案 — 达成后不计划大改。
> 讨论日期：2026-06-19 ~ 2026-06-20
> 实施状态：核心检索管线已实现，详见上方状态栏。

---

## 目录

1. [总体架构](#1-总体架构)
2. [数据层：文档解析与分块](#2-数据层文档解析与分块)
3. [嵌入层：BGE-M3 统一编码](#3-嵌入层bge-m3-统一编码)
4. [向量存储层：Milvus Lite](#4-向量存储层milvus-lite)
5. [检索层：六阶段管线](#5-检索层六阶段管线)
6. [生成层：Prompt 工程与引用锚定](#6-生成层prompt-工程与引用锚定)
7. [评估层：RAGAS](#7-评估层ragas)
8. [运维层：日志、监控、降级](#8-运维层日志监控降级)
9. [文件结构总览](#9-文件结构总览)
10. [实施序列](#10-实施序列)

---

## 1. 总体架构

### 1.1 设计原则

- **无条件混合检索**：不分简单/复杂查询，始终 dense+sparse 双路并行，由 RRF+reranker 自然决定每条查询的主导信号
- **文档金字塔**：细粒度 unit（~400 tokens）做向量检索，命中后按 `kp_id` 扩展到该知识点全部 unit 送入 LLM
- **CPU 优先**：所有本地模型（bge-m3、bge-reranker-v2-m3）运行在 CPU 上，LLM 调用 DeepSeek API
- **一体式嵌入**：BGE-M3 一次 forward pass 同时输出 dense 和 learned sparse 向量，不单独维护 BM25

### 1.2 组件选型

| 层 | 组件 | 选型 | 理由 |
|----|------|------|------|
| 嵌入 | bge-m3 | 本地 CPU 推理 | 1024 维，中英文 + 数学术语覆盖好，同时输出 dense+sparse |
| 向量库 | Milvus Lite | 嵌入部署 | 与 pymilvus 统一 API，支持 hybrid_search + 内置 RRF |
| 重排序 | bge-reranker-v2-m3 | 本地 CPU 推理 | cross-encoder 精度高，20 条 pair 约 1~2s |
| LLM | DeepSeek | API | 便宜、中文推理强、OpenAI 兼容 |
| 查询改写 | DeepSeek | API（同 LLM） | temperature=0，成本 < 0.001 元/次 |
| 评估 | RAGAS | 离线脚本 | 4 指标覆盖检索+生成质量，社区标准 |

### 1.3 数据流

#### 查询时数据流（RAG 检索）

```
学生口语化查询
    │
    ▼
┌──────────────────────────────────────────────────┐
│ 阶段0: 查询改写 (DeepSeek, ~0.5s)                  │
│   → 补充学科术语、消解指代、标准化表述              │
├──────────────────────────────────────────────────┤
│ 阶段1: BGE-M3 统一编码 (CPU, ~0.1s)               │
│   → dense_vec (1024维, L2归一化)                   │
│   → sparse_vec (learned lexical weights)           │
├──────────────────────────────────────────────────┤
│ 阶段2: Milvus 混合检索 + 内置 RRF (~0.05s)         │
│   dense ANN (IP, nprobe=16) + sparse (IP)          │
│   filter: course_id == "xxx"                       │
│   RRFRerank(k=60) → top-20 候选                    │
├──────────────────────────────────────────────────┤
│ 阶段3: bge-reranker-v2-m3 重排序 (~1.5s)          │
│   cross-encoder 对 20 条逐对打分                    │
│   + 层级惩罚：更深的 kp_path 轻微扣分               │
│   → top-5 unit                                     │
├──────────────────────────────────────────────────┤
│ 阶段4: KP 文档金字塔扩展 (~0.02s)                  │
│   top-5 → 按 kp_id 分组 → 从 PG 拉取同 KP 全部 unit │
│   → 按 seq_order 排序 → 去重 → 拼接                │
│   context 软上限: 8000 字符                         │
├──────────────────────────────────────────────────┤
│ 阶段5: LLM 生成 (DeepSeek, ~3s)                    │
│   System prompt + sources + 问题 → 带引用的回答     │
│   引用格式: <ref id="N" />                          │
│   公式: LaTeX $...$ / $$...$$                      │
└──────────────────────────────────────────────────┘

总延迟预估: ~5-6 秒
单次成本: < 0.01 元人民币
```

#### 导入时数据流（Ingestion Pipeline）【已实施】

```
PDF/DOCX/MD 上传
    │
    ▼
┌──────────────────────────────────────────────────┐
│ B0: 自动构建知识点树（新增）                       │
│   从 content_list 提取标题（text_level ≤ 4）       │
│   构建 kp_path 层级结构 → 合并到已有 KP 树         │
│   幂等：已存在的 kp_path 跳过                      │
├──────────────────────────────────────────────────┤
│ B1: 文件解析                                      │
│   MinerU (PDF) / python-docx / 自定义 Markdown     │
│   → content_list [{type, text, text_level, page}]  │
├──────────────────────────────────────────────────┤
│ B2: 文本切分（阶段 A 改造）                        │
│   _filter_garbage → _split_by_headings             │
│   → _split_text_v2（数学块感知 + 段落边界优先）     │
├──────────────────────────────────────────────────┤
│ B3: KP 分配                                       │
│   KPSplitter.assign() → 每 unit 匹配到知识点       │
├──────────────────────────────────────────────────┤
│ B4: SummaryBridge 摘要生成                         │
│   DeepSeek 为每 unit 生成 ≤80 字中文摘要           │
├──────────────────────────────────────────────────┤
│ B5: BGE-M3 编码 + Milvus 入库                     │
│   encode() → dense+sparse → vector_store.insert()  │
├──────────────────────────────────────────────────┤
│ B6: PostgreSQL 入库                               │
│   KnowledgeUnit 批量 INSERT → Document.status=ready │
└──────────────────────────────────────────────────┘
```

### 1.4 暂不实施（已存档，后续迭代）

- 多轮对话历史管理
- 分类型 Prompt 策略（概念/计算/证明/比较/例题）
- 回答质量保障（引用覆盖率检查 / 检索相关性阈值 / 空回答防护）
- 图形/图片处理
- 线上监控/闭环优化

---

## 2. 数据层：文档解析与分块

### 2.1 当前问题（已通过数据库查询确认）

文件: `src/coursepilot/ingestion/parser_utils.py`

| 问题 | 根因 | 影响 |
|------|------|------|
| 76% unit 被截断在 768 字符 | `_split_text()` 中 `chunk_size = max_tokens * 1.5`，对 512 token 目标来说过小 | 知识碎片化，失去上下文完整性 |
| meta_data 只存 `{"text_level": N}` | `_split_by_headings` 未追踪当前标题文本 | 无法从 meta_data 知道 unit 属于哪个章节 |
| 垃圾入库 | CIP 数据、封面、目录被当作正文解析 | 检索可能召回无效内容 |
| LaTeX 公式可能被切断 | `_split_text()` 按字符数暴力切分 | 一个证明跨 3 个 unit，公式语义丢失 |

### 2.2 修正方案：文档金字塔分块

#### 两级粒度

| 层级 | 粒度 | 用途 | 存储 |
|------|------|------|------|
| Layer 1 (粗) | KP 级完整上下文，~3000-5000 字符 | LLM 最终消费 | PostgreSQL `knowledge_units`（按 kp_id + seq_order 聚合） |
| Layer 2 (细) | ~400 token / ~800 中文字符 | 向量检索匹配 | Milvus + PostgreSQL `knowledge_units` |

#### 分块策略

```
PDF/DOCX/MD 解析
    │
    ▼ content_list [{type, text, text_level, page_idx}, ...]
    │
    ▼ _split_by_headings()   ← 按标题边界切（text_level ≤ 4 触发新 block）
    │                           修复：追踪当前标题文本，写入 meta_data
    │
    ▼ 垃圾过滤               ← 新增：黑名单规则（CIP、封面、目录页、空白页）
    │
    ▼ _split_text_v2()       ← 重写：数学块感知 + 段落边界优先
    │   ├── 检测 $$...$$ / $...$，标记为原子不可分割块
    │   ├── 优先在段落边界（双换行）切分
    │   ├── 次优在句边界（。！？\n）切分
    │   └── 目标: ~800 字符/unit，硬下限 400，硬上限 1200
    │
    ▼ Summary Bridge         ← 新增：每条 unit 调用 DeepSeek 生成中文摘要
    │   index_text = f"{unit.summary}\n{unit.content[:200]}"
    │
    ▼ KnowledgeUnit INSERT
```

#### 垃圾过滤规则

```python
GARBAGE_PATTERNS = [
    r"^[A-Z]{3,}\s.* CIP",           # CIP 数据
    r"^图书在版编目",                   # 中国 CIP
    r"^内容简介|^本书.*编写",           # 出版信息
    r"^封面|^扉页|^版权",               # 页面类型
    r"^\s*$",                          # 空行（已在现有逻辑中跳过）
]
GARBAGE_PAGE_RANGE = (0, 2)            # 前 3 页（封面+目录）整体跳过
```

#### _split_by_headings 修复

```python
# 修复前
current_meta = {"text_level": 99}

# 修复后
current_heading = "未知章节"  # 追踪当前所在标题
# 遇到标题行时更新 current_heading = item["text"]
# block 的 meta_data 写入 {"text_level": N, "heading": current_heading}
```

### 2.3 Summary Bridge

**为什么需要**：LaTeX 公式（如 `$\int_a^b f(x)dx$`）和自然语言在不同嵌入空间，纯内容检索对公式不敏感。LLM 生成的自然语言摘要将公式"翻译"为可检索的描述性文本。

**index_text 组成**：
```
{summary}           ← LLM 生成的中文摘要，描述本 unit 讲了什么
                     ← 如: "介绍牛顿-莱布尼茨公式的推导过程，说明定积分与原函数的对应关系"
{content[:200]}     ← 正文前 200 字符作为精确匹配锚点
```

**摘要生成 prompt**：
```
用一句话（不超过80字）概括以下教材片段的核心内容。
如果是公式/定理，说明名称和作用；如果是例题，说明题型和用到的定理。

教材内容：
{unit.content[:500]}

摘要：
```

**成本**：55 条 unit × ~0.001 元 ≈ 0.06 元，可忽略。

### 2.4 管道集成

`pipeline.py` 的 `run_ingestion()` 完整流程（已实施）：

```
B0: _ensure_kp_tree()        ← 新增：从标题自动构建/合并知识点树
B1: 文件解析                  ← 已有（pdf_parser / docx_parser / markdown_parser）
B2: extract_knowledge_units() ← 阶段A改造：垃圾过滤 + heading追踪 + 数学块感知
B3: KPSplitter.assign()      ← 已有
B4: SummaryBridge.run()      ← 新增：为每条 unit 生成摘要
B5: _encode_units()          ← 新增：BGE-M3 编码 → Milvus insert
B6: KnowledgeUnit INSERT     ← PG 入库
```

**关键设计决策**：B0 步骤使得上传 PDF 时不再需要手动预建知识点树。`_ensure_kp_tree()` 按 `kp_path` 去重，支持多卷教材逐本上传自动合并。

---

## 3. 嵌入层：BGE-M3 统一编码

### 3.1 模型信息

| 属性 | 值 |
|------|-----|
| 模型 | BAAI/bge-m3 |
| 路径 | `settings.embedding_model_path` |
| 加载位置 | CPU（首次调用时惰性加载） |
| 内存占用 | ~2 GB |
| 稠密维度 | 1024 |
| 稀疏维度 | 动态（learned token weights） |
| 归一化 | L2 normalize（配合 Milvus IP 度量） |
| batch_size | 32 |

### 3.2 改造：同时输出 dense + sparse

现有 `encoder.py` 只返回 dense vector。改造后增加 sparse 输出：

```python
class Encoder:
    """BGE-M3 编码器，一次 forward 出 dense + sparse"""

    def __init__(self) -> None:
        global _encoder_instance
        if _encoder_instance is None:
            _encoder_instance = _load_model()
        self._model = _encoder_instance

    def encode(self, texts: list[str]) -> list[dict]:
        """将文本列表转为向量列表

        返回: [{"dense": list[float], "sparse": dict[int, float]}, ...]
        """
        if not texts:
            return []

        output = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )

        return [
            {
                "dense": output["dense_vecs"][i].tolist(),
                "sparse": self._format_sparse(output["lexical_weights"][i]),
            }
            for i in range(len(texts))
        ]

    def encode_queries(self, queries: list[str]) -> list[dict]:
        """查询编码（与文档用同一方法，语义上一致）"""
        return self.encode(queries)

    @staticmethod
    def _format_sparse(weights: dict) -> dict:
        """BGE-M3 返回 token_id → weight，需要转为 Milvus sparse 格式"""
        # Milvus sparse vector: {token_id: weight} 或使用 scipy.sparse
        return {int(k): float(v) for k, v in weights.items()}

    @property
    def dim(self) -> int:
        return 1024
```

### 3.3 dense 与 sparse 的互补关系

| | Dense (1024维) | Sparse (learned lexical) |
|---|---|---|
| 擅长 | 语义相似、同义改写、跨语言 | 关键词精确匹配、术语召回 |
| 数学场景 | 召回"导数定义"相关变体表述 | 精确匹配"拉格朗日中值定理" |
| 短板 | 罕见术语可能被平滑掉 | 同义词、改写句式无法匹配 |

BGE-M3 的 learned sparse 比 BM25 的统计权重更聪明——模型自己学到了哪些 token 对区分语义重要，而不是依赖 IDF 统计。

---

## 4. 向量存储层：Milvus Lite

### 4.1 Collection Schema

```python
COLLECTION_NAME = "knowledge_units"

schema = {
    "fields": [
        {"name": "id",          "dtype": "INT64", "is_primary": True, "auto_id": True},
        {"name": "uuid",        "dtype": "VARCHAR", "max_length": 36},
        {"name": "dense_vec",   "dtype": "FLOAT_VECTOR", "dim": 1024},
        {"name": "sparse_vec",  "dtype": "SPARSE_FLOAT_VECTOR"},
        {"name": "kp_id",       "dtype": "VARCHAR", "max_length": 36},
        {"name": "course_id",   "dtype": "VARCHAR", "max_length": 36},
        {"name": "kp_path",     "dtype": "VARCHAR", "max_length": 512},
        {"name": "content",     "dtype": "VARCHAR", "max_length": 8192},  # index_text
    ]
}
```

### 4.2 索引配置

| 向量字段 | 索引类型 | 度量 | 参数 |
|----------|---------|------|------|
| `dense_vec` | IVF_FLAT | IP (Inner Product) | nlist=128 |
| `sparse_vec` | SPARSE_INVERTED_INDEX | IP | drop_ratio_build=0.2 |

### 4.3 操作接口

```python
class VectorStore:
    """Milvus Lite 向量存储，封装 CRUD + 混合检索"""

    def __init__(self, db_path: str = "./data/milvus.db"):
        self.client = MilvusClient(db_path)

    def create_collection(self) -> None:
        """创建 collection + 双索引，幂等"""

    def insert(self, units: list[dict]) -> list[str]:
        """批量插入向量

        units: [{"uuid":..., "dense_vec":..., "sparse_vec":...,
                  "kp_id":..., "course_id":..., "kp_path":..., "content":...}, ...]
        返回: insert_ids
        """

    def hybrid_search(
        self,
        dense_vec: list[float],
        sparse_vec: dict[int, float],
        course_id: str,
        top_k: int = 20,
    ) -> list[dict]:
        """混合检索 + 内置 RRF

        返回: [{"uuid":..., "kp_id":..., "kp_path":..., "content":..., "score":...}, ...]
        """

    def delete_by_uuids(self, uuids: list[str]) -> None:
        """按 uuid 批量删除"""

    def delete_by_course(self, course_id: str) -> None:
        """删除某课程全部向量"""

    def count(self) -> int:
        """collection 中的向量总数"""

    def drop_collection(self) -> None:
        """删除 collection（用于全量重建）"""
```

### 4.4 hybrid_search 实现细节

```python
def hybrid_search(self, dense_vec, sparse_vec, course_id, top_k=20):
    results = self.client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[
            AnnSearchRequest(
                data=[dense_vec],
                anns_field="dense_vec",
                limit=top_k,
                params={"metric_type": "IP", "params": {"nprobe": 16}},
            ),
            AnnSearchRequest(
                data=[sparse_vec],
                anns_field="sparse_vec",
                limit=top_k,
                params={"metric_type": "IP"},
            ),
        ],
        rerank=RRFRerank(k=60),
        filter=f'course_id == "{course_id}"',
        output_fields=["uuid", "kp_id", "kp_path", "content"],
        limit=top_k,
    )
    return results[0]  # 单 query → 取第一个结果列表
```

**注意**：Milvus Lite 需要确认是否完整支持 `hybrid_search` + `RRFRerank`。如果不支持，退回到两次独立 `search()` 调用 + 手写 RRF 融合。

---

## 5. 检索层：六阶段管线

### 5.0 查询改写（Query Rewriting）

**目的**：将学生口语化查询转为适合检索的标准化表述。

```python
REWRITE_PROMPT = """你是课程助教。将学生的口语化问题改写为适合检索的表述。

规则：
- 补充学科关键术语，但不要编造问题中没提到的内容
- 简单明确的问题保持原样，不过度扩展
- 消解指代词（"它"→具体概念名）
- 只输出改写后的问题，不加任何解释

学生问题：{query}
改写后："""

class QueryRewriter:
    def __init__(self, client: openai.AsyncOpenAI):
        self.client = client

    async def rewrite(self, query: str) -> str:
        response = await self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
            temperature=0,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
```

改写示例：

| 原始 | 改写后 |
|------|--------|
| "泰勒展开咋用来求极限" | "使用泰勒展开式求函数极限的原理、步骤与例题" |
| "这个公式咋推出来的" | "该公式的推导过程" |
| "它有什么几何意义" | "该概念的几何意义" |
| "什么时候用洛必达" | "洛必达法则的适用条件和使用场景" |

### 5.1 阶段1: BGE-M3 编码

```python
vecs = encoder.encode_queries([rewritten_query])[0]
dense_vec = vecs["dense"]     # list[float] × 1024
sparse_vec = vecs["sparse"]   # dict[int, float]
```

### 5.2 阶段2: Milvus 混合检索

阶段 2 和阶段 3（RRF）在 Milvus 内一次完成。

```python
candidates = vector_store.hybrid_search(
    dense_vec=dense_vec,
    sparse_vec=sparse_vec,
    course_id=course_id,
    top_k=20,  # RRF 后返回 top-20
)
# candidates: [{uuid, kp_id, kp_path, content, score(rrf)}, ...] × 20
```

### 5.3 阶段3: 重排序（Reranker）

```python
class Reranker:
    def __init__(self):
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(
            settings.reranker_model_path,  # bge-reranker-v2-m3
            use_fp16=True,
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """对候选列表逐对打分，返回 top-k"""
        pairs = [[query, c["content"]] for c in candidates]
        scores = self.model.compute_score(pairs, normalize=True)

        # 层级惩罚：更深的 kp_path 给轻微加分
        for i, c in enumerate(candidates):
            depth = c["kp_path"].count("/") + 1
            scores[i] += min((depth - 1) * 0.02, 0.1)

        # 合并得分并排序
        for i, c in enumerate(candidates):
            c["rerank_score"] = scores[i]

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]
```

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入数量 | 20 | RRF 后候选取前 20 |
| 输出数量 | 5 | 送入 KP 扩展 |
| 层级惩罚 | +0.02/级，上限 +0.1 | 只在分数接近时打破平局 |
| 推理耗时 | ~1.5s | CPU 上 20 对 cross-encoder |

### 5.4 阶段4: KP 文档金字塔扩展

```python
async def kp_expand(
    session: AsyncSession,
    top_units: list[dict],
    max_chars: int = 8000,
) -> str:
    """拉取 top-5 unit 所在 KP 的全部 unit，组装为 LLM 上下文"""
    kp_ids = list({u["kp_id"] for u in top_units})

    # 从 PG 拉取这些 KP 下的所有 unit
    stmt = (
        select(KnowledgeUnit)
        .where(KnowledgeUnit.kp_id.in_(kp_ids))
        .order_by(KnowledgeUnit.kp_id, KnowledgeUnit.seq_order)
    )
    result = await session.execute(stmt)
    all_units = result.scalars().all()

    # 按 kp_id 分组，reranker 得分最高的 KP 排前面
    kp_order = {kp_id: i for i, kp_id in enumerate(kp_ids)}
    grouped = {}
    for u in all_units:
        grouped.setdefault(u.kp_id, []).append(u)

    # 组装 context，超上限截断
    parts = []
    total_chars = 0
    ref_id = 0
    for kp_id in kp_ids:
        units = grouped.get(kp_id, [])
        if not units:
            continue
        kp_path = units[0].kp_path
        parts.append(f'## {kp_path}\n')
        for u in units:
            ref_id += 1
            if total_chars > max_chars:
                break
            source_block = (
                f'<source id="{ref_id}" path="{u.kp_path}" '
                f'pages="{u.page_ref}" book="{u.document.name}">\n'
                f'{u.summary or ""}\n{u.content}\n'
                f'</source>\n'
            )
            parts.append(source_block)
            total_chars += len(source_block)

    return "\n".join(parts)
```

### 5.5 检索编排

```python
class Retriever:
    """六阶段检索编排器"""

    def __init__(
        self,
        encoder: Encoder,
        vector_store: VectorStore,
        reranker: Reranker,
        rewriter: QueryRewriter,
    ):
        self.encoder = encoder
        self.vector_store = vector_store
        self.reranker = reranker
        self.rewriter = rewriter

    async def retrieve(
        self,
        session: AsyncSession,
        query: str,
        course_id: str,
        enable_rewrite: bool = True,
    ) -> tuple[str, list[dict]]:
        """执行完整检索管线

        返回: (formatted_context, retrieval_metadata)
        """
        # 阶段0: 查询改写
        rewritten = await self.rewriter.rewrite(query) if enable_rewrite else query

        # 阶段1: 编码
        vecs = self.encoder.encode_queries([rewritten])[0]

        # 阶段2: 混合检索 + RRF
        candidates = self.vector_store.hybrid_search(
            vecs["dense"], vecs["sparse"], course_id, top_k=20
        )

        # 阶段3: 重排序
        top_units = self.reranker.rerank(rewritten, candidates, top_k=5)

        # 阶段4: KP 扩展
        context = await kp_expand(session, top_units)

        metadata = {
            "query_raw": query,
            "query_rewritten": rewritten,
            "top_rerank_scores": [u["rerank_score"] for u in top_units],
            "source_kp_paths": [u["kp_path"] for u in top_units],
            "candidate_count": len(candidates),
        }
        return context, metadata
```

---

## 6. 生成层：Prompt 工程与引用锚定

### 6.1 上下文格式化

检索结果组装为结构化 XML 格式送入 LLM：

```xml
<source id="1" path="微积分/定积分/牛顿-莱布尼茨公式" pages="p45-47" book="同济高等数学 第八版 上册">
设函数 f(x) 在区间 [a,b] 上连续，F(x) 是 f(x) 的一个原函数，则
∫_a^b f(x)dx = F(b) - F(a)
这就是牛顿-莱布尼茨公式，它将定积分与原函数联系起来。
</source>

<source id="2" path="微积分/定积分/定积分的线性性质" pages="p48-49" book="同济高等数学 第八版 上册">
定积分具有线性性质：
∫_a^b [αf(x) + βg(x)]dx = α∫_a^b f(x)dx + β∫_a^b g(x)dx
</source>
```

**为什么用 XML 标签**：DeepSeek 对 `<source id="N">...</source>` 格式的遵循度比纯数字标记好，训练语料中有大量结构化标签。

### 6.2 System Prompt

```python
SYSTEM_PROMPT = """你是 CoursePilot 课程助教，为大学生解答数学问题。

## 回答规则

1. **基于教材**：回答必须依据下面 <sources> 中提供的教材内容
2. **引用格式**：涉及教材内容时使用 <ref id="N" /> 引用，N 为 source id
3. **公式正确**：所有数学公式使用 LaTeX 语法，行内 $...$，独立行 $$...$$
4. **区分边界**：教材中有的内容正常回答；超出教材范围的，明确说"教材未涉及此内容"并提供已知的相关知识点
5. **启发思考**：先给出关键思路，再展示详细步骤，鼓励学生自己先尝试
6. **概念串联**：主动关联相关知识点，帮助学生建立知识网络
7. **语言风格**：简洁清晰，不啰嗦

## 当前课程

{course_context}

## 参考教材内容

{sources}
"""
```

### 6.3 course_context 构建

```python
async def build_course_context(session, course_id: str) -> dict:
    """查询课程的 KP 树，构建层级大纲"""
    course = await session.get(Course, course_id)
    kps = await session.execute(
        select(KnowledgePoint)
        .where(KnowledgePoint.course_id == course_id)
        .order_by(KnowledgePoint.kp_path)
    )
    kp_list = kps.scalars().all()

    # 提取顶层 KP 作为"已学章节"
    chapters = [kp.name for kp in kp_list if kp.parent_id is None]

    return {
        "name": course.name,
        "textbook": course.textbook or "未知教材",
        "chapters": chapters,
    }
```

### 6.4 LLM 生成器

```python
class Generator:
    """DeepSeek LLM 调用封装"""

    def __init__(self, client: openai.AsyncOpenAI):
        self.client = client

    async def generate(
        self,
        query: str,
        context: str,
        course_context: dict,
        stream: bool = False,
    ) -> str:
        system = SYSTEM_PROMPT.format(
            course_context=self._format_course(course_context),
            sources=context,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]

        if stream:
            return self._stream_response(messages)
        else:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
            )
            return response.choices[0].message.content

    async def _stream_response(self, messages) -> AsyncGenerator:
        """SSE 流式生成，供 FastAPI StreamingResponse 消费"""
        stream = await self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @staticmethod
    def _format_course(ctx: dict) -> str:
        chapters = "、".join(ctx["chapters"])
        return f"当前课程：{ctx['name']}\n教材：{ctx['textbook']}\n已学章节：{chapters}"
```

### 6.5 引用验证

```python
import re

def validate_citations(answer: str, valid_source_ids: set[int]) -> tuple[bool, set[int]]:
    """验证回答中的引用是否都来自合法的 source id"""
    cited = set()
    for m in re.finditer(r'<ref id="(\d+)" />', answer):
        cited.add(int(m.group(1)))
    hallucinated = cited - valid_source_ids
    return len(hallucinated) == 0, hallucinated
```

### 6.6 前端交互

| 元素 | 实现 |
|------|------|
| 引用渲染 | 前端解析 `<ref id="N" />`，hover 显示 tooltip（kp_path + page_ref + 原文摘要），点击跳转教材阅读器 |
| 公式渲染 | KaTeX 或 MathJax 渲染 `$...$` 和 `$$...$$` |
| Streaming | FastAPI `StreamingResponse` + SSE |

---

## 7. 评估层：RAGAS

### 7.1 评估框架

使用 [RAGAS](https://docs.ragas.io/) 框架，4 个核心指标：

| 指标 | 评估对象 | 含义 | 及格线 | 目标 |
|------|---------|------|--------|------|
| Context Precision | 检索 | 检索到的内容有多少是相关的 | > 0.70 | > 0.80 |
| Context Recall | 检索 | 相关内容有多少被检索到了 | > 0.80 | > 0.85 |
| Faithfulness | 生成 | 回答是否完全基于检索内容（不编造） | > 0.85 | > 0.90 |
| Answer Relevancy | 生成 | 回答是否紧扣问题（不跑题） | > 0.75 | > 0.80 |

### 7.2 测试集构建

50 条标注问答，覆盖 5 门课程，每门 10 条。

| 问题类型 | 占比 | 示例 |
|----------|------|------|
| 概念解释 | 30% | "什么是定积分的几何意义" |
| 计算题 | 25% | "用牛顿-莱布尼茨公式求 ∫_0^1 x² dx" |
| 证明推导 | 20% | "证明拉格朗日中值定理" |
| 辨析比较 | 15% | "定积分和不定积分有什么区别" |
| 应用题 | 10% | "用定积分求旋转体体积" |

标注格式：

```json
{
    "question": "什么是定积分的几何意义",
    "answer": "定积分的几何意义是曲边梯形的面积...",
    "ground_truth_contexts": ["uuid-001", "uuid-003"],
    "course_id": "1257e794-...",
    "question_type": "concept"
}
```

### 7.3 评估脚本

```python
# scripts/eval_ragas.py
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

async def run_evaluation(eval_data_path: str = "tests/fixtures/eval_questions.json"):
    eval_data = load_json(eval_data_path)

    results = []
    for item in eval_data:
        context, metadata = await retriever.retrieve(
            session, item["question"], item["course_id"]
        )
        answer = await generator.generate(
            item["question"], context, metadata["course_context"]
        )
        results.append({
            "question": item["question"],
            "answer": answer,
            "contexts": [context],
            "ground_truth": item["answer"],
        })

    ds = Dataset.from_list(results)
    scores = evaluate(ds, metrics=[
        faithfulness, answer_relevancy, context_precision, context_recall
    ])
    return scores
```

### 7.4 执行频率

| 时机 | 说明 |
|------|------|
| 开发阶段 | 每次改动检索/生成代码后手动跑 |
| PR 阶段 | CI 自动跑，任一指标降 >5% 阻断合并 |
| 定期 | 每周跑一次，跟踪指标趋势 |

成本：50 条 × RAGAS 4 指标 ≈ 0.5-1 元/次。

---

## 8. 运维层：日志、监控、降级

### 8.1 结构化查询日志

每条查询打一条 JSON 日志：

```json
{
    "trace_id": "abc123",
    "timestamp": "2026-06-20T10:30:00",
    "user_id": "uuid",
    "course_id": "uuid",
    "query_raw": "泰勒展开咋用来求极限",
    "query_rewritten": "泰勒展开式求函数极限的原理与步骤",
    "stages": {
        "rewrite_ms": 480,
        "encode_ms": 120,
        "hybrid_search_ms": 35,
        "rerank_ms": 1800,
        "kp_expand_ms": 15,
        "generate_ms": 3200
    },
    "top_rerank_scores": [0.85, 0.72, 0.68, 0.55, 0.42],
    "source_kp_paths": [
        "微积分/导数的应用/泰勒公式",
        "微积分/极限与连续/极限的计算方法"
    ],
    "citation_count": 3,
    "answer_length": 450
}
```

存储方式：存入 PostgreSQL 的 `query_logs` 表（或扩展现有 `QARecord`），同时输出到日志文件用于实时排查。

### 8.2 关键监控查询

```sql
-- 每日查询量和延迟趋势
SELECT DATE(timestamp), COUNT(*),
       AVG((stages->>'total_ms')::int) AS avg_total_ms
FROM query_logs
GROUP BY 1 ORDER BY 1;

-- 检索质量预警（最高 rerank 分 < 0.5 的比例）
SELECT DATE(timestamp),
       COUNT(*) FILTER (WHERE (top_rerank_scores->>0)::float < 0.5)::float / COUNT(*) AS low_quality_rate
FROM query_logs
GROUP BY 1;

-- 知识点盲区（哪些 KP 从未被命中）
SELECT kp.kp_path
FROM knowledge_points kp
LEFT JOIN query_logs ql ON ql.source_kp_paths::jsonb @> to_jsonb(kp.kp_path)
WHERE ql.id IS NULL;

-- 索引一致性检查
SELECT
    (SELECT COUNT(*) FROM knowledge_units) AS pg_count,
    (SELECT COUNT(*) FROM milvus_collection_stats('knowledge_units')) AS milvus_count;
```

### 8.3 降级开关

每个环节可独立 bypass：

```python
@dataclass
class RAGConfig:
    # 功能开关
    enable_rewrite: bool = True       # 关闭 → 直接用原始 query
    enable_sparse: bool = True        # 关闭 → 只用 dense
    enable_rerank: bool = True        # 关闭 → RRF 后直接取 top-5
    enable_kp_expand: bool = True     # 关闭 → 只用检索到的 unit 本身

    # 阈值
    reranker_min_score: float = 0.3   # 低于此分的 source 丢弃
    context_max_chars: int = 8000     # 送入 LLM 的上下文软上限

    # 检索参数
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 5
```

### 8.4 索引维护

| 操作 | 触发条件 | 做法 |
|------|---------|------|
| 增量索引 | 新文档上传 | `pipeline.py` 在 `run_ingestion()` 最后自动调用 `vector_store.insert()` |
| 增量删除 | 文档被删除 | `vector_store.delete_by_uuids()` 清理对应 unit 的向量 |
| 全量重建 | KP 树变更 / 模型更换 / 索引参数调整 | `drop_collection()` → 遍历全部 unit 重新 `encode()` + `insert()` |
| 一致性检查 | 定时（每周）或手动 `scripts/check_index_health.py` | PG `COUNT` vs Milvus `num_entities` |

### 8.5 用户反馈闭环

```
回答末尾两个按钮:
  [有帮助 👍]  [不准确 👎]

前端埋点:
  - 悬停引用标签次数 → 对哪段原文最感兴趣
  - 复制公式次数 → 哪个回答最有用
```

反馈存入 `QARecord` 表，定期抽样差评 case 做根因分析（rerank 分低？context 缺关键信息？LLM 幻觉？）。

### 8.6 成本估算

| 调用 | Token (in/out) | 单次成本 |
|------|---------------|---------|
| 查询改写 | ~50 / ~30 | < 0.001 元 |
| 摘要生成（导入时） | ~500 / ~80 | < 0.003 元/条 |
| 生成回答 | ~3000 / ~500 | < 0.005 元 |
| RAGAS 评估（单条） | ~2000 / ~200 | < 0.003 元 |

单次查询总成本约 0.005-0.01 元。月活 1000 次约 5-10 元。55 条 unit 首次摘要生成约 0.17 元。

---

## 9. 文件结构总览

```
src/coursepilot/rag/
├── __init__.py           # 空
├── config.py             # RAGConfig (降级开关、阈值)
├── encoder.py            # BGE-M3 dense + sparse 统一编码（改造现有）
├── vector_store.py       # Milvus Lite CRUD + hybrid_search
├── query_rewriter.py     # DeepSeek 查询改写（阶段0）
├── reranker.py           # bge-reranker-v2-m3 重排序（阶段3）
├── retriever.py          # 六阶段检索编排 + KP 扩展（阶段1-4）
├── generator.py          # DeepSeek LLM 调用 + prompt 组装（阶段5）
├── citation.py           # <ref> 标签解析与验证
├── logger.py             # 结构化查询日志
└── pipeline.py           # 导入时索引构建（注入 ingestion pipeline）

src/coursepilot/ingestion/
├── parser_utils.py       # 改造：文档金字塔分块
└── pipeline.py           # 改造：B6 SummaryBridge + B7 encode→Milvus

scripts/
├── eval_ragas.py         # RAGAS 离线评估
└── check_index_health.py # 索引健康检查

database/
└── (新建 query_logs 表 或在 QARecord 扩展字段)
```

---

## 10. 实施序列

### 阶段 A：数据层修复（parser_utils.py + Summary Bridge）✅ 已完成

| 步骤 | 内容 | 文件 | 状态 |
|------|------|------|------|
| A0 | 新增 `_ensure_kp_tree()`：从标题自动构建/合并知识点树 | `ingestion/pipeline.py` | ✅ 已完成 |
| A1 | 重写 `_split_by_headings`：追踪当前标题文本，写入 meta_data | `parser_utils.py` | ✅ 已完成 |
| A2 | 新增垃圾过滤函数 `_filter_garbage()` | `parser_utils.py` | ✅ 已完成 |
| A3 | 重写 `_split_text_v2`：数学块原子检测 + 段落边界优先 + 死循环 guard | `parser_utils.py` | ✅ 已完成 |
| A4 | 新增 `SummaryBridge` 类：调用 DeepSeek 生成摘要 | `rag/summary_bridge.py` | ✅ 已完成 |
| A5 | 更新 `run_ingestion()`：B0 自动 KP + B4 SummaryBridge + B5 编码入库 | `ingestion/pipeline.py` | ✅ 已完成 |
| A6 | 重新运行 ingestion，验证数据质量 | 数据库查询 | ✅ 已完成 |

### 阶段 B：RAG 引擎核心 ✅ 已完成

| 步骤 | 内容 | 文件 | 状态 |
|------|------|------|------|
| B1 | 改造 `encoder.py`：`encode()` 同时返回 dense + sparse | `rag/encoder.py` | ✅ 已完成 |
| B2 | 实现 `vector_store.py`：CRUD + hybrid_search | `rag/vector_store.py` | ✅ 已完成 |
| B3 | 实现 `query_rewriter.py` | `rag/query_rewriter.py` | ✅ 已完成 |
| B4 | 实现 `reranker.py` | `rag/reranker.py` | ✅ 已完成 |
| B5 | 实现 `retriever.py`：六阶段编排 + KP 扩展 | `rag/retriever.py` | ✅ 已完成 |
| B6 | 实现 `generator.py` + `citation.py` | `rag/generator.py`, `rag/citation.py` | ✅ 已完成 |
| B7 | 实现 `rag/config.py` 降级开关 | `rag/config.py` | ✅ 已完成 |
| B8 | 实现 `rag/logger.py` 结构化日志 | `rag/logger.py` | ✅ 已完成 |

### 阶段 C：评估与运维 ← 当前

| 步骤 | 内容 | 文件 | 状态 |
|------|------|------|------|
| C1 | 编写 50 条标注问答 | `tests/fixtures/eval_questions.json` | 🔲 待实施 |
| C2 | 实现 `eval_ragas.py` | `scripts/eval_ragas.py` | 🔲 待实施 |
| C3 | 实现 `check_index_health.py` | `scripts/check_index_health.py` | 🔲 待实施 |
| C4 | 建 `query_logs` 表 | Alembic migration | 🔲 待实施 |

### 阶段 D：集成与调优 ✅ 已完成

| 步骤 | 内容 | 状态 |
|------|------|------|
| D1 | 将 RAG 管线接入 `/api/v1/courses/{id}/ask` 端点 | ✅ 已完成 |
| D2 | 端到端手动验证（10+ 条典型查询） | ✅ 已完成 |
| D3 | 跑 RAGAS 评估，确认 4 指标达标 | 🔲 待阶段 C |
| D4 | 根据评估结果微调参数（nprobe, rrf_k, rerank_top_k） | 🔲 待阶段 C |

---

## 附录：关键决策记录

| 决策 | 选项 A | 选项 B | 最终 | 理由 |
|------|--------|--------|------|------|
| 条件式 vs 无条件检索 | 分类后选策略 | 始终双路并行 | **B** | 半术语半模糊查询无法可靠分类 |
| 分块策略 | 递归分块 | 文档金字塔 | **B** | KP 树是最佳语义结构，代替通用递归 |
| 嵌入输出 | 仅 dense | dense + sparse 一体 | **B** | BGE-M3 原生支持，一次 forward 出两种 |
| 稀疏检索 | BM25 + jieba | BGE-M3 learned sparse | **B** | 少维护一套分词+词典，语义对齐更好 |
| RRF 实现 | 手写 | Milvus 内置 | **B** | hybrid_search 原生支持 RRF |
| 查询改写 | 无 | DeepSeek 前置改写 | **B** | 对学生口语化查询有必要 |
| unit 粒度 | 512 token (768 字符) | ~400 token (800 字符) | **B** | 当前 76% 被截断，扩大后保证段落完整 |
| 公式检索 | 纯 content 嵌入 | Summary Bridge | **B** | LaTeX 与自然语言不在同一嵌入空间 |
| KP 扩展范围 | N±1 兄弟 unit | 同 KP 全部 unit | **B** | KP 内连续讲解，截断丢失上下文 |
| LLM | 本地模型 | DeepSeek API | **B** | 便宜、中文强、无需 GPU 运维 |
