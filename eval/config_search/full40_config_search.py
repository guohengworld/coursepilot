"""40题黄金数据集 — 多组检索配置对比测试（检索只读模式，不调用LLM）。

输出:
  temp_results/full40_config_results.json   — 逐配置逐题完整结果
  temp_results/full40_config_report.txt     — 可读汇总报告
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coursepilot.config import settings
from coursepilot.db import get_session_etx
from coursepilot.evaluation.rag_eval import RAGEvaluator, EvalReport
from coursepilot.rag.config import config as rag_config

# ── 40题黄金数据集 ─────────────────────────────────────────
DATASET = Path(__file__).parent.parent / "eval" / "questions" / "20260726" / "eval_questions.json"

# ── 待测试的检索配置 ───────────────────────────────────────
CONFIGS: list[dict] = [
    # (名称, overrides)
    # 1) 基线：当前默认参数
    {"name": "baseline", "overrides": {}},
    # 2) 禁用 BM25（纯 Milvus 向量检索）
    {"name": "no_bm25", "overrides": {"enable_bm25": False}},
    # 3) 禁用重排序
    {"name": "no_rerank", "overrides": {"enable_rerank": False}},
    # 4) 禁用 KP 扩展
    {"name": "no_kp_expand", "overrides": {"enable_kp_expand": False}},
    # 5) dense 权重 0.7（更依赖向量）
    {"name": "dense07", "overrides": {"dense_weight": 0.7}},
    # 6) dense 权重 0.3（更依赖 BM25）
    {"name": "dense03", "overrides": {"dense_weight": 0.3}},
    # 7) neighbor 模式（滑动窗口）
    {"name": "neighbor", "overrides": {"kp_expand_mode": "neighbor", "kp_neighbor_window": 3}},
    # 8) 更大 top-k：rerank_top_k=8
    {"name": "topk8", "overrides": {"rerank_top_k": 8}},
    # 9) RRF k=30（融合更激进）
    {"name": "rrf_k30", "overrides": {"rrf_k": 30}},
    # 10) RRF k=100（融合更平滑）
    {"name": "rrf_k100", "overrides": {"rrf_k": 100}},
    # 11) 仅 BM25，不启用向量重排序（验证纯关键词检索效果）
    {"name": "bm25_only", "overrides": {"enable_rerank": False, "enable_bm25": True}},
    # 12) 禁用查询改写
    {"name": "no_rewrite", "overrides": {"enable_rewrite": False}},
]


async def run_single_config(
    session,
    name: str,
    overrides: dict,
    questions: list[dict],
    course_id: str,
) -> dict:
    """运行一个配置并返回统计结果。"""
    print(f"\n{'#'*60}")
    print(f"  配置: {name}")
    print(f"  overrides: {overrides}")
    print(f"{'#'*60}")

    t0 = time.monotonic()

    evaluator = RAGEvaluator(
        config_overrides=overrides,
        use_mimo=True,
    )

    # 只做检索，不生成、不 RAGAS
    report = await evaluator.evaluate_dataset(
        session, DATASET,
        skip_generation=True,  # 检索+UUID recall，不调用 LLM
    )

    elapsed = time.monotonic() - t0

    # 按题型分组统计
    type_stats: dict[str, dict] = {}
    for r in report.results:
        if r.question_type == "unanswerable":
            continue
        if r.question_type not in type_stats:
            type_stats[r.question_type] = {
                "count": 0, "recall_sum": 0.0, "recall_1_count": 0, "recall_0_count": 0,
            }
        ts = type_stats[r.question_type]
        ts["count"] += 1
        ts["recall_sum"] += r.context_recall
        if r.context_recall >= 1.0:
            ts["recall_1_count"] += 1
        if r.context_recall == 0.0:
            ts["recall_0_count"] += 1

    for qt in type_stats:
        ts = type_stats[qt]
        ts["recall_avg"] = round(ts["recall_sum"] / ts["count"], 4) if ts["count"] else 0.0
        ts["recall_full_rate"] = round(ts["recall_1_count"] / ts["count"], 4) if ts["count"] else 0.0
        ts["recall_zero_rate"] = round(ts["recall_0_count"] / ts["count"], 4) if ts["count"] else 0.0

    total_answerable = sum(ts["count"] for ts in type_stats.values())
    total_recall_sum = sum(ts["recall_sum"] for ts in type_stats.values())

    result = {
        "config_name": name,
        "overrides": overrides,
        "elapsed_seconds": round(elapsed, 1),
        "total_answerable": total_answerable,
        "avg_context_recall": round(total_recall_sum / total_answerable, 4) if total_answerable else 0.0,
        "per_type": type_stats,
        "per_question": [
            {
                "idx": i + 1,
                "type": r.question_type,
                "recall": r.context_recall,
                "question": r.question[:50],
                "kp_path": r.kp_path,
                "retrieved_uuids": r.retrieved_uuids,
                "ground_truth_uuids": r.ground_truth_uuids,
            }
            for i, r in enumerate(report.results)
            if r.question_type != "unanswerable"
        ],
    }

    # 打印本配置摘要
    print(f"\n  结果 [{name}]:")
    print(f"    总耗时: {elapsed:.1f}s")
    print(f"    总 recall: {result['avg_context_recall']:.4f}")
    for qt, ts in sorted(type_stats.items()):
        print(f"    {qt}: recall={ts['recall_avg']:.4f}  full_rate={ts['recall_full_rate']:.2f}  zero_rate={ts['recall_zero_rate']:.2f}  count={ts['count']}")

    return result


async def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("[full40] 强制 CPU 模式")

    if not DATASET.exists():
        print(f"[ERROR] 数据集不存在: {DATASET}")
        return

    questions = json.loads(DATASET.read_text(encoding="utf-8"))
    course_id = questions[0].get("course_id", "")
    if not course_id:
        raise ValueError("数据集中缺少 course_id")

    total_answerable = sum(1 for q in questions if q.get("question_type") != "unanswerable")
    print(f"[full40] 加载 {len(questions)} 题 (含 {len(questions) - total_answerable} 题不可回答, 有效 {total_answerable} 题)")

    all_results = []

    async with get_session_etx() as session:
        for cfg in CONFIGS:
            result = await run_single_config(session, cfg["name"], cfg["overrides"], questions, course_id)
            all_results.append(result)

    # ── 汇总排序 ────────────────────────────────────────────
    sorted_by_recall = sorted(all_results, key=lambda x: x["avg_context_recall"], reverse=True)

    output = {
        "dataset": str(DATASET),
        "total_questions": len(questions),
        "total_answerable": total_answerable,
        "config_count": len(CONFIGS),
        "configs": sorted_by_recall,
        "ranked": [c["config_name"] for c in sorted_by_recall],
    }

    # 保存 JSON
    out_json = Path(__file__).parent / "full40_config_results.json"
    out_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 生成可读报告
    report_lines = generate_report(sorted_by_recall, total_answerable)
    report_text = "\n".join(report_lines)

    out_txt = Path(__file__).parent / "full40_config_report.txt"
    out_txt.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\n结果已保存:\n  {out_json}\n  {out_txt}")


def generate_report(sorted_results: list[dict], total_answerable: int) -> list[str]:
    lines = [
        "=" * 80,
        "  40题黄金数据集 — 检索配置对比报告",
        "=" * 80,
        "",
        f"总题数: {total_answerable} (排除不可回答)",
        f"测试配置数: {len(sorted_results)}",
        "",
        "─" * 80,
        "  配置排名 (按平均 Context Recall 降序)",
        "─" * 80,
        "",
    ]

    # 表头
    header = (
        f"  {'排名':<4} {'配置名称':<14} {'平均Recall':<10} "
        f"{'耗时(s)':<8} "
    )
    # 类型列
    type_names = sorted({qt for r in sorted_results for qt in r.get("per_type", {})})
    for tn in type_names:
        header += f"{tn[:6]:<11} "
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for rank, r in enumerate(sorted_results, 1):
        row = (
            f"  {rank:<4} {r['config_name']:<14} {r['avg_context_recall']:<10.4f} "
            f"{r['elapsed_seconds']:<8.1f} "
        )
        for tn in type_names:
            ts = r.get("per_type", {}).get(tn, {})
            avg = ts.get("recall_avg", 0)
            row += f"{avg:<11.4f} "
        lines.append(row)

    lines += [
        "",
        "─" * 80,
        "  各配置详细数据",
        "─" * 80,
        "",
    ]

    for r in sorted_results:
        lines.append(f"  ▶ {r['config_name']}")
        lines.append(f"    参数: {json.dumps(r['overrides'], ensure_ascii=False)}")
        lines.append(f"    平均 Recall: {r['avg_context_recall']:.4f}")
        lines.append(f"    耗时: {r['elapsed_seconds']:.1f}s")
        for qt, ts in sorted(r["per_type"].items()):
            lines.append(
                f"    [{qt}] count={ts['count']}  recall={ts['recall_avg']:.4f}  "
                f"full={ts['recall_full_rate']:.2f}  zero={ts['recall_zero_rate']:.2f}"
            )

        # 逐题 recall 明细
        lines.append("    逐题 Recall:")
        for pq in r["per_question"]:
            gt = pq.get("ground_truth_uuids", [])
            ret = pq.get("retrieved_uuids", [])
            hit = len(set(gt) & set(ret))
            lines.append(
                f"      #{pq['idx']:<2} [{pq['type']:<10}] recall={pq['recall']:<.4f}  "
                f"hit={hit}/{len(gt)}  {pq['question']}"
            )
        lines.append("")

    lines.append("=" * 80)
    lines.append("  报告结束")
    lines.append("=" * 80)
    return lines


if __name__ == "__main__":
    asyncio.run(main())
