"""
RAGAS 评估 CLI —— 基线评估 + 参数网格搜索 + 报告输出

用法：
    # 基线评估（默认参数）
    PYTHONPATH=src .venv/Scripts/python -m scripts.eval_ragas baseline

    # 仅评估检索质量（跳过 LLM 生成，更快）
    PYTHONPATH=src .venv/Scripts/python -m scripts.eval_ragas baseline --skip-generation

    # 三轮网格搜索（门禁调优）
    PYTHONPATH=src .venv/Scripts/python -m scripts.eval_ragas grid --stage 1   # rrf_k
    PYTHONPATH=src .venv/Scripts/python -m scripts.eval_ragas grid --stage 2   # rerank_top_k
    PYTHONPATH=src .venv/Scripts/python -m scripts.eval_ragas grid --stage 3   # context_max_chars

    # 自定义网格搜索
    PYTHONPATH=src .venv/Scripts/python -m scripts.eval_ragas grid --params '{"rrf_k":[30,60,90]}'

    # 从 JSON 报告对比两次评估
    PYTHONPATH=src .venv/Scripts/python -m scripts.eval_ragas compare report1.json report2.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from coursepilot.db import get_session_etx
from coursepilot.evaluation.rag_eval import RAGEvaluator, EvalReport


DATASET_PATH = Path("eval/questions/eval_questions.json")
OUTPUT_DIR = Path("eval/reports")


# ══════════════════════════════════════════════════════════════
# 三轮网格搜索的预设参数
# ══════════════════════════════════════════════════════════════

GRID_STAGES = {
    1: {"rrf_k": [30, 60, 90, 120]},
    2: {"rerank_top_k": [3, 5, 8, 10]},
    3: {"context_max_chars": [4000, 6000, 8000, 10000, 12000]},
    4: {"rrf_weights": [[1.0, 1.0], [1.5, 1.0], [1.0, 1.5], [2.0, 1.0], [1.0, 2.0]]},
}

STAGE_NAMES = {
    1: "rrf_k",
    2: "rerank_top_k",
    3: "context_max_chars",
    4: "rrf_weights",
}


# ══════════════════════════════════════════════════════════════
# 命令实现
# ══════════════════════════════════════════════════════════════

async def cmd_baseline(args):
    """基线评估 —— 使用当前 RAG 配置评估全部题目"""
    if not DATASET_PATH.exists():
        print(f"[ERROR] 数据集不存在: {DATASET_PATH}")
        sys.exit(1)

    print(f"加载数据集: {DATASET_PATH}")
    print(f"跳过生成: {args.skip_generation}")

    evaluator = RAGEvaluator()
    async with get_session_etx() as session:
        report = await evaluator.evaluate_dataset(
            session, DATASET_PATH, skip_generation=args.skip_generation
        )

    print()
    print(report.summary())

    # 保存报告
    output_path = _save_report(report, "baseline")
    print(f"\n报告已保存: {output_path}")

    # 门禁检查
    if report.avg_context_recall < 0.85:
        print("\n[GATE FAIL] Context Recall 未达标 (< 0.85)，建议执行网格搜索调优")
        sys.exit(1)
    else:
        print("\n[GATE PASS] Context Recall 达标 (>= 0.85)")


async def cmd_grid(args):
    """网格搜索 —— 在参数空间内搜索最佳配置"""
    if args.params:
        param_grid = json.loads(args.params)
        stage_name = "custom"
    elif args.stage in GRID_STAGES:
        param_grid = GRID_STAGES[args.stage]
        stage_name = STAGE_NAMES[args.stage]
    else:
        print(f"[ERROR] 无效的 stage: {args.stage}，可选: {list(GRID_STAGES.keys())}")
        sys.exit(1)

    if not DATASET_PATH.exists():
        print(f"[ERROR] 数据集不存在: {DATASET_PATH}")
        sys.exit(1)

    # 展开参数网格
    combinations = _expand_grid(param_grid)
    print(f"网格搜索: {stage_name}")
    print(f"参数空间: {param_grid}")
    print(f"组合数:   {len(combinations)}")
    print(f"跳过生成: {args.skip_generation}")
    print()

    best_report: EvalReport | None = None
    best_score = -1.0
    all_summaries: list[str] = []

    for i, overrides in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] {overrides}")

        evaluator = RAGEvaluator(config_overrides=overrides)
        async with get_session_etx() as session:
            report = await evaluator.evaluate_dataset(
                session, DATASET_PATH, skip_generation=args.skip_generation
            )

        score = report.avg_context_recall  # 主要优化目标
        status = "PASS" if score >= 0.85 else "FAIL"
        print(f"  Recall={score:.3f} Prec={report.avg_context_precision:.3f} "
              f"Faith={report.avg_faithfulness:.3f} Relev={report.avg_answer_relevancy:.3f} [{status}]")

        all_summaries.append(
            f"  {overrides}: Recall={score:.3f} Prec={report.avg_context_precision:.3f} "
            f"Faith={report.avg_faithfulness:.3f} Relev={report.avg_answer_relevancy:.3f}"
        )

        if score > best_score:
            best_score = score
            best_report = report

    # 输出最终结果
    print()
    print("=" * 60)
    print("网格搜索完成")
    print("=" * 60)
    print(f"参数空间: {stage_name}")
    for s in all_summaries:
        print(s)
    print()

    if best_report:
        print("最佳配置:")
        print(f"  参数: {best_report.config}")
        print(f"  Context Recall:     {best_report.avg_context_recall:.3f}")
        print(f"  Context Precision:  {best_report.avg_context_precision:.3f}")
        print(f"  Faithfulness:       {best_report.avg_faithfulness:.3f}")
        print(f"  Answer Relevancy:   {best_report.avg_answer_relevancy:.3f}")

        output_path = _save_report(best_report, f"grid-{stage_name}")
        print(f"\n报告已保存: {output_path}")

        if best_score >= 0.85:
            print("\n[GATE PASS] 最佳 Context Recall 达标 (>= 0.85)")
        else:
            print("\n[GATE FAIL] 所有配置的 Context Recall 均未达标 (< 0.85)")


async def cmd_compare(args):
    """对比两次评估报告"""
    report_paths = [Path(p) for p in args.reports]
    for p in report_paths:
        if not p.exists():
            print(f"[ERROR] 报告不存在: {p}")
            sys.exit(1)

    reports_data = []
    for p in report_paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        reports_data.append((p.name, data))

    print("=" * 60)
    print("评估报告对比")
    print("=" * 60)
    header = f"  {'指标':<25}"
    for name, _ in reports_data:
        header += f" {name:<20}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    metrics = [
        "context_recall",
        "context_precision",
        "faithfulness",
        "answer_relevancy",
    ]
    labels = {
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
    }

    for metric in metrics:
        line = f"  {labels[metric]:<25}"
        for _, data in reports_data:
            avg = data.get("averages", {}).get(metric, 0)
            line += f" {avg:<20.3f}"
        print(line)

    print()
    # 逐题差异
    if len(reports_data) == 2:
        print("逐题 Context Recall 差异:")
        results_a = reports_data[0][1]["results"]
        results_b = reports_data[1][1]["results"]
        for i, (ra, rb) in enumerate(zip(results_a, results_b)):
            diff = ra["context_recall"] - rb["context_recall"]
            marker = " >>>" if abs(diff) > 0.1 else ""
            print(f"  Q{i+1}: {ra['context_recall']:.3f} vs {rb['context_recall']:.3f} "
                  f"(diff={diff:+.3f}){marker}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def _expand_grid(param_grid: dict) -> list[dict]:
    """展开参数网格为组合列表"""
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations


def _save_report(report: EvalReport, prefix: str) -> Path:
    """保存评估报告为 JSON 文件"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    path = OUTPUT_DIR / filename
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


# ══════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RAGAS 评估 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s baseline                    基线评估
  %(prog)s baseline --skip-generation  仅评估检索（跳过生成）
  %(prog)s grid --stage 1              第一轮网格搜索（rrf_k）
  %(prog)s grid --stage 2              第二轮网格搜索（rerank_top_k）
  %(prog)s grid --stage 3              第三轮网格搜索（context_max_chars）
  %(prog)s grid --params '{"rrf_k":[30,60]}'  自定义网格
  %(prog)s compare r1.json r2.json     对比两次报告
        """,
    )
    sub = parser.add_subparsers(dest="command")

    # baseline
    p_base = sub.add_parser("baseline", help="基线评估")
    p_base.add_argument(
        "--skip-generation", action="store_true",
        help="跳过 LLM 生成（仅评估检索指标）"
    )

    # grid
    p_grid = sub.add_parser("grid", help="参数网格搜索")
    p_grid.add_argument(
        "--stage", type=int, choices=[1, 2, 3],
        help="预设搜索阶段: 1=rrf_k, 2=rerank_top_k, 3=context_max_chars"
    )
    p_grid.add_argument(
        "--params", type=str,
        help='JSON 格式的自定义参数网格，如 \'{"rrf_k":[30,60,90]}\''
    )
    p_grid.add_argument(
        "--skip-generation", action="store_true",
        help="跳过 LLM 生成（更快，仅评估检索质量）"
    )

    # compare
    p_cmp = sub.add_parser("compare", help="对比评估报告")
    p_cmp.add_argument("reports", nargs="+", help="报告 JSON 文件路径")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "baseline":
        asyncio.run(cmd_baseline(args))
    elif args.command == "grid":
        asyncio.run(cmd_grid(args))
    elif args.command == "compare":
        asyncio.run(cmd_compare(args))


if __name__ == "__main__":
    main()
