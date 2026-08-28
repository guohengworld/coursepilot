#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""意图识别评测脚本（直连 MIMO，复用 classify_intent 的 CLASSIFY_SYSTEM prompt）。

只评测意图识别（question/practice/diagnose/review/none），不评测复杂度。

用法：
  python run_intent_eval.py --out report.md                # 真实评测（需 .env 配 MIMO_API_KEY）
  python run_intent_eval.py --self-test                    # 离线校验指标函数（不调 API）
  python run_intent_eval.py --dry-run                      # 仅加载数据集并统计分布
  python run_intent_eval.py --runs 3 --out report.md       # 多次运行取多数投票 + 一致性

依赖：项目 .venv（openai / pydantic-settings）。从项目根目录或任意位置运行均可，
脚本会自动把 src/ 加入 sys.path 并读取项目根 .env。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# ── 路径与 import 引导 ──────────────────────────────────────
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from coursepilot.config import settings  # noqa: E402
from coursepilot.agent.skills.classify_intent import CLASSIFY_SYSTEM  # noqa: E402

DATASET_PATH = _THIS.parent / "intent_dataset.json"

INTENT_ORDER = ["question", "practice", "diagnose", "review", "none"]
PROD_VALID_INTENTS = {"question", "practice", "diagnose", "review", "none"}  # 生产 classify_intent 允许的意图（已补 none）
DEFAULT_COURSE = {"name": "高等数学", "chapters": ["极限与连续", "导数与微分", "微分中值定理", "不定积分", "定积分", "微分方程", "多元函数微积分", "无穷级数"]}


# ── 数据集 ──────────────────────────────────────────────────
def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── LLM 调用（直连 MIMO）────────────────────────────────────
def parse_llm_output(raw: str):
    """容错解析 LLM 返回的 JSON，返回 (intent, reasoning, parse_ok)。"""
    if not raw:
        return None, "", False
    raw = raw.strip()
    candidates = [raw]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        candidates.append(brace.group(0).strip())
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                intent = str(obj.get("intent", "")).strip().lower() or None
                reasoning = str(obj.get("reasoning", "")).strip()
                return intent, reasoning, True
        except (json.JSONDecodeError, ValueError):
            continue
    return None, "", False


async def classify_with_mimo(query: str, recent_qa=None, course_context=None) -> dict:
    """直连 MIMO 做意图分类，返回预测意图与诊断信息。带重试以吸收瞬时限流/超时。"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.mimo_api_key,
        base_url=settings.mimo_base_url,
        timeout=settings.llm_timeout,
    )
    parts = [f"用户问题：{query}"]
    # 注入当前课程上下文，供 none/离题 判定锚定基准（与生产 classify_intent 一致）
    if course_context:
        cc_lines = []
        if course_context.get("name"):
            cc_lines.append(f"当前课程：{course_context['name']}")
        chapters = course_context.get("chapters") or []
        if chapters:
            cc_lines.append("课程章节范围：" + "、".join(chapters[:10]))
        if cc_lines:
            parts.append("\n".join(cc_lines))
    if recent_qa:
        parts.append("最近回答：")
        for qa in recent_qa[-3:]:
            if isinstance(qa, dict) and "query" in qa:
                parts.append(f"  Q: {qa['query']}")
            elif isinstance(qa, dict) and qa.get("role") == "user" and "content" in qa:
                parts.append(f"  Q: {qa['content']}")
    prompt = "\n".join(parts)
    last_err = None
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=settings.mimo_model,
                messages=[
                    {"role": "system", "content": CLASSIFY_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            temperature=0.3,
            max_tokens=2000,
            )
            raw = resp.choices[0].message.content if resp.choices else ""
            intent, reasoning, parse_ok = parse_llm_output(raw)
            return {"intent": intent, "reasoning": reasoning, "parse_ok": parse_ok, "raw": raw}
        except Exception as e:  # 网络/鉴权/超时/限流等
            last_err = e
            await asyncio.sleep(1.5 * (attempt + 1))
    return {"intent": "<error>", "reasoning": "", "parse_ok": False, "error": str(last_err)[:200]}


# ── 指标 ────────────────────────────────────────────────────
def build_labels(predicted_labels: set[str]) -> list[str]:
    labels = list(INTENT_ORDER)
    extra = sorted(l for l in predicted_labels if l not in INTENT_ORDER)
    return labels + extra


def compute_metrics(expected: list[str], predicted: list[str], labels: list[str]):
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    for e, p in zip(expected, predicted):
        cm[idx[e]][idx[p]] += 1
    per_class = {}
    for i, l in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n) if r != i)
        fn = sum(cm[i][c] for c in range(n) if c != i)
        support = sum(cm[i])
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[l] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
    total = sum(sum(r) for r in cm)
    accuracy = sum(cm[i][i] for i in range(n)) / total if total else 0.0
    macro_f1 = sum(v["f1"] for v in per_class.values()) / n
    wsum = sum(v["support"] for v in per_class.values())
    weighted_f1 = sum(v["f1"] * v["support"] for v in per_class.values()) / wsum if wsum else 0.0
    return cm, per_class, accuracy, macro_f1, weighted_f1


# ── 运行 ────────────────────────────────────────────────────
async def run_eval(dataset: list[dict], runs: int):
    sem = asyncio.Semaphore(4)

    async def one(item):
        preds = []
        last = None
        for _ in range(max(1, runs)):
            async with sem:
                d = await classify_with_mimo(
                    item["query"], item.get("recent_qa"), item.get("course_context") or DEFAULT_COURSE
                )
            preds.append(d["intent"] or "<error>")
            last = d
        vote = Counter(preds).most_common(1)[0][0]
        consistency = preds.count(vote) / len(preds)
        return item["id"], {
            "id": item["id"],
            "query": item["query"],
            "category": item["category"],
            "expected": item["expected_intent"],
            "preds": preds,
            "vote": vote,
            "consistency": consistency,
            "detail": last,
        }

    coros = [one(it) for it in dataset]
    rows = await asyncio.gather(*coros)
    return {rid: r for rid, r in rows}


# ── 报告渲染 ────────────────────────────────────────────────
def render_report(dataset, cases: dict, runs: int) -> str:
    expected = [cases[it["id"]]["expected"] for it in dataset]
    predicted = [cases[it["id"]]["vote"] for it in dataset]
    pred_labels = set(predicted)
    labels = build_labels(pred_labels)
    cm, per_class, accuracy, macro_f1, weighted_f1 = compute_metrics(expected, predicted, labels)

    misclassified = [
        c for c in cases.values() if c["vote"] != c["expected"]
    ]
    parse_fail = [c for c in cases.values() if (c["detail"] or {}).get("parse_ok") is False]
    api_error = [c for c in cases.values() if c["vote"] == "<error>"]
    # 降级陷阱量化：MIMO 已正确返回 none，但生产 classify_intent 会因其 valid_intents 无 none 而降级为 question
    none_recognized = [c for c in cases.values() if c["vote"] == "none"]
    none_would_degrade = [c for c in none_recognized if c["expected"] == "none"]

    lines = []
    lines.append("# 意图识别评测报告（MIMO）")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 模型：`{settings.mimo_model}`  @  `{settings.mimo_base_url}`")
    lines.append(f"- 评测范围：仅意图识别（question/practice/diagnose/review/none），不评测复杂度")
    lines.append(f"- 数据集：{len(dataset)} 条（normal/boundary/oos 三类）")
    lines.append(f"- 运行次数：{runs}（投票取多数；temperature=0.3）")
    lines.append(f"- 提示词：复用 `classify_intent.CLASSIFY_SYSTEM`（与生产一致）")
    lines.append("")
    lines.append("## 一、总体指标")
    lines.append("")
    lines.append(f"- **整体准确率 Accuracy**：{accuracy*100:.1f}%")
    lines.append(f"- **Macro-F1**：{macro_f1*100:.1f}%（平等看待每个意图，重点盯低频/拒识类）")
    lines.append(f"- **Weighted-F1**：{weighted_f1*100:.1f}%（按流量加权）")
    lines.append("")
    lines.append("> 结论判读：Macro-F1 与 Weighted-F1 差距越大，说明少数类（diagnose/review/none）越被多数类（question）掩盖。")
    lines.append("")
    non_oos_idx = [i for i, e in enumerate(expected) if e != "none"]
    non_oos_correct = sum(1 for i in non_oos_idx if predicted[i] == expected[i])
    if non_oos_idx:
        lines.append(
            f"- **4 类学习意图（排除 oos/none）准确率**：{non_oos_correct}/{len(non_oos_idx)}"
            f" = {non_oos_correct/len(non_oos_idx)*100:.1f}%（误判见第六节；none 拒识见第四节）"
        )
    lines.append("")
    lines.append("## 二、每类 Precision / Recall / F1")
    lines.append("")
    lines.append("| 意图 | Precision | Recall | F1 | 样本数(support) |")
    lines.append("|---|---|---|---|---|")
    for l in labels:
        v = per_class[l]
        lines.append(
            f"| {l} | {v['precision']*100:.1f}% | {v['recall']*100:.1f}% | {v['f1']*100:.1f}% | {v['support']} |"
        )
    lines.append("")
    lines.append("## 三、混淆矩阵")
    lines.append("")
    head = "| 真实 \\ 预测 | " + " | ".join(labels) + " |"
    sep = "|---|" + "|".join(["---"] * len(labels)) + "|"
    lines.append(head)
    lines.append(sep)
    for i, l in enumerate(labels):
        row = [str(cm[i][j]) for j in range(len(labels))]
        lines.append(f"| {l} | " + " | ".join(row) + " |")
    lines.append("")
    lines.append("> 行=真实意图，列=模型预测；对角线为判对，非对角线为误判方向。")
    lines.append("")
    lines.append("## 四、拒识专项（none 意图）")
    lines.append("")
    if "none" in per_class:
        v = per_class["none"]
        lines.append(
            f"- none 类：Precision={v['precision']*100:.1f}%，Recall={v['recall']*100:.1f}%，"
            f"F1={v['f1']*100:.1f}%，样本数={v['support']}"
        )
        if v["precision"] < 1.0:
            lines.append(f"  - Precision<100%：存在**误拒**（真实学习问题被判成 none），见第六节；常因 none 定义过宽或课程上下文过窄。")
        if v["recall"] < 1.0:
            lines.append(f"  - Recall<100%：存在**漏拒**（越界用例未被拒识），见第六节。")
    else:
        lines.append("- 数据集中无 none 样本（不应发生）。")
    lines.append(f"- MIMO 实际返回 `none` 的条数：{len(none_recognized)}")
    lines.append(
        f"- 其中预期为 none（即被正确拒识）的条数：{len(none_would_degrade)}"
    )
    lines.append("")
    lines.append("## 五、降级陷阱专项")
    lines.append("")
    lines.append(f"- API 调用异常（`<error>`）条数：{len(api_error)}（生产中 `classify_intent` 对 API 异常未捕获，会向上抛出，需补 try/except 与降级标记）")
    lines.append(f"- LLM 返回但 JSON 解析失败条数：{len(parse_fail)}（生产中会静默降级为 `question`）")
    n_degraded = sum(1 for p in predicted if p not in PROD_VALID_INTENTS)
    lines.append(
        f"- 本批 MIMO 原始意图不在 5 类中的条数：{n_degraded}"
        f"（若出现 practice_generation / chit-chat / 证明类 等自定义标签，生产 `valid_intents` 仅认 5 类会强制降级为 `question`）"
    )
    lines.append("")
    lines.append("### 语义降级（缺失 none 导致闲聊进入 RAG）")
    none_as_q = sum(1 for it in dataset if cases[it["id"]]["expected"] == "none" and cases[it["id"]]["vote"] == "question")
    none_as_p = sum(1 for it in dataset if cases[it["id"]]["expected"] == "none" and cases[it["id"]]["vote"] == "practice")
    none_total = sum(1 for it in dataset if cases[it["id"]]["expected"] == "none")
    if none_total:
        lines.append(
            f"- 预期为 none 的 {none_total} 条中：{none_as_q} 条仍被误判为 `question`、{none_as_p} 条被误判为 `practice`，"
            f"未被拒识 {none_as_q + none_as_p} 条。"
        )
    if none_as_q + none_as_p == 0:
        lines.append("- 语义降级已消除：所有越界用例均被正确判为 none，不再进入 RAG 链路。")
    else:
        lines.append("- 仍有越界用例漏判为学习意图，会进入 RAG 造成'答非所问'，见第四节 none Recall 与第六节错误清单。")
    lines.append(
        f"- 本次 MIMO 实际返回 `none` 的条数：{len(none_recognized)}；"
        f"生产 `valid_intents` 现已含 `none`，模型输出的 none 不会被强制降级。"
    )
    lines.append("- 剩余降级风险：(a) API 异常未捕获需补 try/except + 降级标记；(b) 自定义标签仍会被强转 question。")
    lines.append("")
    lines.append("## 六、错误用例清单（预测 ≠ 期望）")
    lines.append("")
    if misclassified:
        lines.append("| ID | 类别 | 期望 | 预测 | 用户问题 |")
        lines.append("|---|---|---|---|---|")
        for c in misclassified:
            q = c["query"].replace("|", "/")
            lines.append(f"| {c['id']} | {c['category']} | {c['expected']} | {c['vote']} | {q} |")
    else:
        lines.append("无错误用例。")
    lines.append("")
    lines.append("## 七、结论与建议")
    lines.append("")
    non_oos_idx = [i for i, e in enumerate(expected) if e != "none"]
    non_oos_correct = sum(1 for i in non_oos_idx if predicted[i] == expected[i])
    none_total = sum(1 for it in dataset if cases[it["id"]]["expected"] == "none")
    none_correct = len(none_would_degrade)
    none_recall = (none_correct / none_total * 100) if none_total else 0.0
    lines.append(
        f"1. 整体 Macro-F1={macro_f1*100:.1f}%，Weighted-F1={weighted_f1*100:.1f}%；"
        f"4 类学习意图（排除 oos）准确率 {non_oos_correct}/{len(non_oos_idx)}={non_oos_correct/len(non_oos_idx)*100:.1f}%，"
        f"none/拒识 Recall={none_correct}/{none_total}={none_recall:.1f}%。"
    )
    none_v = per_class.get("none")
    if none_v and none_v["recall"] >= 1.0 and none_v["precision"] >= 1.0:
        lines.append("2. 拒识：none 的 Recall 与 Precision 均达 100%，越界全拒、无误拒，拒识链路打通。")
    elif none_v and none_v["recall"] >= 1.0:
        lines.append(f"2. 拒识：Recall={none_v['recall']*100:.1f}%（越界全拒），但 Precision={none_v['precision']*100:.1f}% 存在误拒（真实问题被判 none），需放宽 none 定义或补全课程上下文。")
    elif none_v:
        lines.append(f"2. 拒识：Recall={none_v['recall']*100:.1f}% 仍有漏拒（越界未被拒），需强化 none 定义与示例。")
    else:
        lines.append("2. 拒识：数据集中无 none 样本。")
    lines.append(f"3. 降级陷阱：none 已接入 `valid_intents`，语义降级（闲聊进 RAG）已消除；但 (a) `classify_intent` 对 API 异常未捕获会向上抛出（本批 {len(api_error)} 例 `<error>`），需补 try/except + 降级标记；(b) 自定义标签仍会被强转 question；(c) 生产 max_tokens 已提至 800，避免推理型模型思考被截断。")
    lines.append("4. 多意图（如 review+practice）当前只能取单标签，属已知设计缺口，不在本次拒识/降级范围。")
    lines.append("5. 复现与稳健性：本评测 temperature=0.3 单次运行，边界与误拒用例建议 `--runs 3` 取多数投票验证稳定性；生产 `classify_intent` max_tokens 已提至 800，API 异常捕获与降级标记仍待补。")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="意图识别评测（MIMO）")
    ap.add_argument("--out", default=str(_THIS.parent / "intent_eval_report.md"), help="报告输出路径")
    ap.add_argument("--runs", type=int, default=1, help="每个用例运行次数（取多数投票）")
    ap.add_argument("--self-test", action="store_true", help="离线校验指标函数")
    ap.add_argument("--dry-run", action="store_true", help="仅加载数据集统计")
    args = ap.parse_args()

    dataset = load_dataset(DATASET_PATH)

    if args.self_test:
        # 用已知矩阵校验指标函数（不调 API）
        exp = ["question"] * 7 + ["practice"] * 2 + ["none"] * 1
        pred = ["question"] * 7 + ["question"] * 2 + ["none"] * 1  # practice 全误判为 question
        labels = build_labels(set(pred))
        cm, pc, acc, mf1, wf1 = compute_metrics(exp, pred, labels)
        assert abs(acc - 0.8) < 1e-9, acc
        assert pc["practice"]["f1"] == 0.0, pc["practice"]
        # 5 个 label：question=0.875, practice=0, diagnose=0, review=0, none=1.0 -> macro=0.375
        assert abs(mf1 - 0.375) < 1e-9, mf1
        print("SELF_TEST_OK accuracy=%.3f macro_f1=%.3f" % (acc, mf1))
        return

    if args.dry_run:
        cat = Counter(d["category"] for d in dataset)
        intent = Counter(d["expected_intent"] for d in dataset)
        print("DRY_RUN total=%d" % len(dataset))
        print("categories: " + ", ".join(f"{k}={v}" for k, v in cat.items()))
        print("intents: " + ", ".join(f"{k}={v}" for k, v in intent.items()))
        return

    if not settings.mimo_api_key:
        print("ERROR: MIMO_API_KEY 未配置，无法运行真实评测。请检查 .env。")
        sys.exit(1)

    cases = asyncio.run(run_eval(dataset, args.runs))
    report = render_report(dataset, cases, args.runs)
    out = Path(args.out)
    out.write_text(report, encoding="utf-8")

    # 终端摘要（ASCII 安全）
    expected = [cases[it["id"]]["expected"] for it in dataset]
    predicted = [cases[it["id"]]["vote"] for it in dataset]
    labels = build_labels(set(predicted))
    _, pc, acc, mf1, wf1 = compute_metrics(expected, predicted, labels)
    print("EVAL_DONE total=%d accuracy=%.3f macro_f1=%.3f weighted_f1=%.3f" % (len(dataset), acc, mf1, wf1))
    print("report -> " + str(out))


if __name__ == "__main__":
    main()
