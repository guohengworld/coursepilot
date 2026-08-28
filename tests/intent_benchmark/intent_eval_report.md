# 意图识别评测报告（MIMO）

- 生成时间：2026-08-28 20:28:39
- 模型：`mimo-v2.5`  @  `https://api.xiaomimimo.com/v1`
- 评测范围：仅意图识别（question/practice/diagnose/review/none），不评测复杂度
- 数据集：58 条（normal/boundary/oos 三类）
- 运行次数：1（投票取多数；temperature=0.3）
- 提示词：复用 `classify_intent.CLASSIFY_SYSTEM`（与生产一致）

## 一、总体指标

- **整体准确率 Accuracy**：96.6%
- **Macro-F1**：97.1%（平等看待每个意图，重点盯低频/拒识类）
- **Weighted-F1**：96.5%（按流量加权）

> 结论判读：Macro-F1 与 Weighted-F1 差距越大，说明少数类（diagnose/review/none）越被多数类（question）掩盖。

- **4 类学习意图（排除 oos/none）准确率**：44/46 = 95.7%（误判见第六节；none 拒识见第四节）

## 二、每类 Precision / Recall / F1

| 意图 | Precision | Recall | F1 | 样本数(support) |
|---|---|---|---|---|
| question | 100.0% | 87.5% | 93.3% | 16 |
| practice | 92.3% | 100.0% | 96.0% | 12 |
| diagnose | 100.0% | 100.0% | 100.0% | 9 |
| review | 100.0% | 100.0% | 100.0% | 9 |
| none | 92.3% | 100.0% | 96.0% | 12 |

## 三、混淆矩阵

| 真实 \ 预测 | question | practice | diagnose | review | none |
|---|---|---|---|---|---|
| question | 14 | 1 | 0 | 0 | 1 |
| practice | 0 | 12 | 0 | 0 | 0 |
| diagnose | 0 | 0 | 9 | 0 | 0 |
| review | 0 | 0 | 0 | 9 | 0 |
| none | 0 | 0 | 0 | 0 | 12 |

> 行=真实意图，列=模型预测；对角线为判对，非对角线为误判方向。

## 四、拒识专项（none 意图）

- none 类：Precision=92.3%，Recall=100.0%，F1=96.0%，样本数=12
  - Precision<100%：存在**误拒**（真实学习问题被判成 none），见第六节；常因 none 定义过宽或课程上下文过窄。
- MIMO 实际返回 `none` 的条数：13
- 其中预期为 none（即被正确拒识）的条数：12

## 五、降级陷阱专项

- API 调用异常（`<error>`）条数：0（生产中 `classify_intent` 对 API 异常未捕获，会向上抛出，需补 try/except 与降级标记）
- LLM 返回但 JSON 解析失败条数：0（生产中会静默降级为 `question`）
- 本批 MIMO 原始意图不在 5 类中的条数：0（若出现 practice_generation / chit-chat / 证明类 等自定义标签，生产 `valid_intents` 仅认 5 类会强制降级为 `question`）

### 语义降级（缺失 none 导致闲聊进入 RAG）
- 预期为 none 的 12 条中：0 条仍被误判为 `question`、0 条被误判为 `practice`，未被拒识 0 条。
- 语义降级已消除：所有越界用例均被正确判为 none，不再进入 RAG 链路。
- 本次 MIMO 实际返回 `none` 的条数：13；生产 `valid_intents` 现已含 `none`，模型输出的 none 不会被强制降级。
- 剩余降级风险：(a) API 异常未捕获需补 try/except + 降级标记；(b) 自定义标签仍会被强转 question。

## 六、错误用例清单（预测 ≠ 期望）

| ID | 类别 | 期望 | 预测 | 用户问题 |
|---|---|---|---|---|
| Q12 | normal | question | practice | 用拉格朗日中值定理证明这个不等式。 |
| Q13 | boundary | question | none | 这道题应该怎么解？ |

## 七、结论与建议

1. 整体 Macro-F1=97.1%，Weighted-F1=96.5%；4 类学习意图（排除 oos）准确率 44/46=95.7%，none/拒识 Recall=12/12=100.0%。
2. 拒识：Recall=100.0%（越界全拒），但 Precision=92.3% 存在误拒（真实问题被判 none），需放宽 none 定义或补全课程上下文。
3. 降级陷阱：none 已接入 `valid_intents`，语义降级（闲聊进 RAG）已消除；但 (a) `classify_intent` 对 API 异常未捕获会向上抛出（本批 0 例 `<error>`），需补 try/except + 降级标记；(b) 自定义标签仍会被强转 question；(c) 生产 max_tokens 已提至 800，避免推理型模型思考被截断。
4. 多意图（如 review+practice）当前只能取单标签，属已知设计缺口，不在本次拒识/降级范围。
5. 复现与稳健性：本评测 temperature=0.3 单次运行，边界与误拒用例建议 `--runs 3` 取多数投票验证稳定性；生产 `classify_intent` max_tokens 已提至 800，API 异常捕获与降级标记仍待补。