# RAG 评估对比报告（最新）

> 评估时间：2026-07-15  
> 数据集：`eval/questions/eval_questions.json`（11 题）  
> 对比策略：baseline / kp_full / kp_neighbor  
> 原始报告：`eval/reports/kp_expand_20260715_111535.json`

---

## 总览

| 指标 | Baseline | KP-Full | KP-Neighbor | 最优 |
|------|----------|---------|-------------|------|
| Context Recall | 0.591 | **0.818** | 0.591 | KP-Full |
| Context Precision | 0.345 | **0.377** | 0.309 | KP-Full |
| Faithfulness | 0.661 | **0.754** | 0.585 | KP-Full |
| Answer Relevancy | 1.000 | 1.000 | 1.000 | 持平 |
| Avg Context Len | 7158 | 8471 | 8662 | — |
| 总耗时 | 252s | 318s | 280s | Baseline |

---

## 关键结论

### 1. KP-Full 是当前最优策略

- **Recall 提升 22.7%**（0.591 → 0.818），说明 KP 扩展有效补全了同知识点下的相关 unit。
- **Precision 未下降**（0.345 → 0.377），反而略有提升，说明扩展引入的噪声在可控范围。
- **Faithfulness 提升 9.3%**（0.661 → 0.754），上下文更完整后，LLM 生成的回答更忠实于教材。

### 2. KP-Neighbor 假设不成立

- Recall 与 Baseline 持平（0.591），没有发挥召回补偿作用。
- Precision 和 Faithfulness 均低于 Baseline，说明窗口截断反而丢失了有用信息。

### 3. 距离门禁还差 3.2%

当前 KP-Full Recall = 0.818，距离设定的 0.85 门禁还有约 0.032 差距。

---

## 与历史版本对比

| 指标 | 历史最佳（docs/eval_comparison.md） | 当前 KP-Full | 说明 |
|------|-------------------------------------|--------------|------|
| Context Recall | 1.000 | 0.818 | 数据集不同，当前 11 题更难 |
| Context Precision | 0.400 | 0.377 | 基本持平 |
| Faithfulness | 0.788 | 0.754 | 接近 |

> 注：历史报告使用另一批 eval_questions，Recall 达到 1.0；当前 11 题召回更难，但 Precision/Faithfulness 接近历史最佳水平。

---

## Bad Case 方向（需逐题明细进一步定位）

由于本次 `kp_expand_20260715_111535.json` 仅保存了汇总指标，缺少逐题 UUID 命中详情，下一步建议：

1. 重新运行评估并保存逐题结果：
   ```bash
   PYTHONPATH=src .venv/Scripts/python -m eval.eval_ragas baseline
   ```
2. 运行诊断脚本查看最终上下文命中：
   ```bash
   PYTHONPATH=src .venv/Scripts/python -m scripts.diagnose_rag
   ```
3. 重点观察 Recall < 1.0 的题目，确认是阶段3（reranker top-k）截断还是阶段4（KP 扩展）丢失。

---

## 下一步调参建议

| 优先级 | 动作 | 预期收益 |
|--------|------|----------|
| P0 | 提升 `rerank_top_k` 到 10~12 | 直接提升阶段3 GT 命中率 |
| P0 | 跑 RRF 权重网格搜索（`grid --stage 4`） | 让 BM25 把术语/公式顶上来 |
| P1 | 检查 `reranker_min_score=0.3` | 若边缘正确项被过滤，降到 0.15 或取消 |
| P1 | 优化查询改写对数学问题的稳定性 | 避免“叉积”“曲线积分”等关键词被改写弱化 |

---

## 配置推荐

基于本次实验，建议默认配置调整为：

```python
# src/coursepilot/rag/config.py
kp_expand_mode: str = "full"
rrf_weights: tuple[float, ...] = (1.0, 1.0)  # 待 grid stage 4 调优
rerank_top_k: int = 8  # 可继续上调到 10
```
