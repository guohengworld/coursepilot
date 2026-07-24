"""
KP 扩展策略对比实验：baseline / kp_full / kp_neighbor

用法：
    PYTHONPATH=src .venv/Scripts/python scripts/eval_kp_expand.py

输出各策略的 Context Recall / Precision / Faithfulness / Relevancy / Context Length，
判断 KP 扩展是否引入噪声以及邻居模式是否更优。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from coursepilot.db import get_session_etx
from coursepilot.evaluation.rag_eval import RAGEvaluator
from coursepilot.rag.config import config

DATASET = Path("eval/questions/eval_questions.json")

# 三种策略的配置覆盖
STRATEGIES = {
    "baseline": {"enable_kp_expand": False},
    "kp_full": {"enable_kp_expand": True, "kp_expand_mode": "full"},
    "kp_neighbor": {"enable_kp_expand": True, "kp_expand_mode": "neighbor", "kp_neighbor_window": 2},
}


def _apply_overrides(overrides: dict) -> dict:
    """应用策略配置覆盖，返回原始值用于恢复。"""
    saved = {}
    for k, v in overrides.items():
        saved[k] = getattr(config, k, None)
        setattr(config, k, v)
    return saved


def _restore(saved: dict) -> None:
    for k, v in saved.items():
        if v is not None:
            setattr(config, k, v)


async def run_strategy(name: str, overrides: dict) -> dict:
    """运行一种策略并返回聚合指标。"""
    saved = _apply_overrides(overrides)
    try:
        evaluator = RAGEvaluator()
        async with get_session_etx() as session:
            report = await evaluator.evaluate_dataset(
                session, DATASET, skip_generation=False,
            )
        result = {
            "context_recall": report.avg_context_recall,
            "context_precision": report.avg_context_precision,
            "faithfulness": report.avg_faithfulness,
            "answer_relevancy": report.avg_answer_relevancy,
            "avg_context_length": report.avg_context_length,
            "elapsed_seconds": report.elapsed_seconds,
            "error_count": report.error_count,
        }
        print(f"\n>>> {name}: recall={result['context_recall']:.3f}, "
              f"precision={result['context_precision']:.3f}, "
              f"faithfulness={result['faithfulness']:.3f}, "
              f"len={result['avg_context_length']:.0f}")
        return result
    finally:
        _restore(saved)


async def main():
    print("=" * 70)
    print("KP 扩展策略对比实验")
    print(f"数据集: {DATASET}")
    print("策略: baseline (无扩展), kp_full (全量), kp_neighbor (±2 邻居)")
    print("=" * 70)

    all_results: dict[str, dict] = {}
    for name, overrides in STRATEGIES.items():
        t0 = time.monotonic()
        print(f"\n{'─'*70}\n策略 [{name}] 开始...\n{'─'*70}")
        all_results[name] = await run_strategy(name, overrides)
        elapsed = time.monotonic() - t0
        print(f"策略 [{name}] 完成, 耗时={elapsed:.0f}s")

    # 汇总表格
    print("\n\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    header = f"{'策略':<15} {'Recall':<9} {'Precision':<11} {'Faithfulness':<14} {'Relevancy':<11} {'AvgLen':<8} {'耗时':<8}"
    sep = "─" * len(header)
    print(header)
    print(sep)
    for name, r in all_results.items():
        print(
            f"{name:<15} "
            f"{r['context_recall']:<9.3f} "
            f"{r['context_precision']:<11.3f} "
            f"{r['faithfulness']:<14.3f} "
            f"{r['answer_relevancy']:<11.3f} "
            f"{r['avg_context_length']:<8.0f} "
            f"{r['elapsed_seconds']:<8.0f}"
        )

    # 判断结论
    r_base = all_results["baseline"]
    r_full = all_results["kp_full"]
    r_neighbor = all_results.get("kp_neighbor")

    print("\n" + "=" * 70)
    print("结论分析")
    print("=" * 70)
    if r_full["context_recall"] > r_base["context_recall"] + 0.05:
        recall_msg = f"KP-Full Recall 比 Baseline 高 {r_full['context_recall'] - r_base['context_recall']:.1%}，扩展有助于找到更多相关 unit"
    else:
        recall_msg = f"KP-Full Recall 与 Baseline 接近 ({r_full['context_recall']:.1%} vs {r_base['context_recall']:.1%})，扩展收益有限"
    print(f"  Recall:   {recall_msg}")

    if r_full["context_precision"] < r_base["context_precision"] - 0.05:
        print(f"  Precision: KP-Full ({r_full['context_precision']:.1%}) < Baseline ({r_base['context_precision']:.1%}) ⚠ 扩展引入了噪声")
    else:
        print(f"  Precision: KP-Full ({r_full['context_precision']:.1%}) ≈ Baseline ({r_base['context_precision']:.1%})，噪声不明显")

    if r_neighbor:
        print(f"\n  KP-Neighbor 与 KP-Full 对比:")
        recall_delta = r_neighbor["context_recall"] - r_full["context_recall"]
        prec_delta = r_neighbor["context_precision"] - r_full["context_precision"]
        print(f"    Recall delta:     {recall_delta:+.1%}")
        print(f"    Precision delta:  {prec_delta:+.1%}")
        print(f"    AvgLen delta:     {r_neighbor['avg_context_length'] - r_full['avg_context_length']:+.0f} chars")
        if recall_delta >= -0.03 and prec_delta > 0:
            print(f"    ✅ 邻居模式在保持 Recall 的同时提升了 Precision")
        elif r_neighbor["avg_context_length"] < r_full["avg_context_length"] * 0.7:
            print(f"    📦 邻居模式上下文更紧凑，质量待进一步验证")

    # 保存 JSON 报告到 eval/reports/
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"eval/reports/kp_expand_{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n完整结果已保存: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
