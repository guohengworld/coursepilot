# 评估逻辑演进

## 初始方案

### Context Precision（per-unit judge）
```
对每个检索到的 unit（17-121 个）：
  → 取前 500 字
  → 调 LLM judge 问"这段内容是否有助于回答该问题"（yes/no）
CP = 返回 yes 的 unit 数 / 总 unit 数
```
- 每题 50-200 次 LLM 调用，耗时 50-120s
- 单个 unit 是碎片化段落，脱离上下文 LLM 难以判断 → 大量 false negative → CP ≈ 0.02

### KP 扩展（全量拉取）
```
top-5 KP → 拉取全部 unit → 按 seq_order 拼接到 8000 字符上限
```
- 全量覆盖无遗漏，Recall=0.909
- 噪声大（整章 unit 全塞进去），Faithfulness 受影响

---

## 当前方案

### Context Precision（KP 级 judge）
```
对 top-5 KP，每个 KP：
  → 拉取全部 unit
  → 关键词 Jaccard 粗排取 top-8 最相关 unit
  → 拼接截断 3000 字
  → 调 LLM judge 一次
CP = 返回 yes 的 KP 数 / 5
```
- 每题 5 次 LLM 调用，耗时 ~3s
- CP=0.400，远优于 per-unit 方案

### KP 扩展（语义粗排 + 精排）
```
top-5 KP → 拉取全部 unit
  → BGE-M3 dense embedding 余弦相似度粗排 top-30
  → bge-reranker cross-encoder 精排
  → 按得分拼接到 8000 字符上限
```
- Recall=1.0（全量覆盖无遗漏）
- Faith=0.788（精排过滤噪声）
- 粗排 1-2s + 精排 ~9s = ~11s

---

## 关键差异

| | 初始 | 当前 |
|------|------|------|
| CP 判定粒度 | per-unit（碎片） | KP 级（聚合） |
| CP LLM 调用/题 | 50-200 次 | 5 次 |
| CP 得分 | 0.021 | 0.400 |
| KP 扩展粗排 | 无 | BGE-M3 dense cosine |
| KP 扩展精排 | 无 | bge-reranker cross-encoder |
| 总耗时 | 1129s | 385s |
| Recall | 0.909 | 1.000 |
