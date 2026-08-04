"""
RAGAS 评估 CLI —— 基线评估 + 参数网格搜索 + 报告输出

用法：
    # 基线评估（默认参数）
    PYTHONPATH=src .venv/Scripts/python -m eval.eval_ragas baseline

    # 仅评估检索质量（跳过 LLM 生成，更快）
    PYTHONPATH=src .venv/Scripts/python -m eval.eval_ragas baseline --skip-generation

    # 两轮网格搜索（参考 docs/rag/RAG评估体系构建.md 5.2）
    PYTHONPATH=src .venv/Scripts/python -m eval.eval_ragas grid --stage 1   # rrf_k × dense_weight
    PYTHONPATH=src .venv/Scripts/python -m eval.eval_ragas grid --stage 2   # rerank_top_k × context_max_chars

    # 自定义网格搜索
    PYTHONPATH=src .venv/Scripts/python -m eval.eval_ragas grid --params '{"rrf_k":[30,60,90]}'

    # 从 JSON 报告对比两次评估
    PYTHONPATH=src .venv/Scripts/python -m eval.eval_ragas compare report1.json report2.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from coursepilot.db import get_session_etx
from coursepilot.evaluation.metrics_config import (
    GRID_FIXED_DEFAULTS,
    GRID_SEARCH_PLAN,
    METRIC_NAMES,
    THRESHOLDS,
)
from coursepilot.evaluation.rag_eval import EvalReport, RAGEvaluator


DATASET_PATH = Path("eval/questions/20260726/eval_questions.json")
OUTPUT_DIR = Path("eval/reports")


# ══════════════════════════════════════════════════════════════
# 命令实现
# ══════════════════════════════════════════════════════════════

async def cmd_baseline(args):
    """基线评估 —— 使用当前 RAG 配置评估全部题目。"""
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
    failures = []
    for metric, threshold in THRESHOLDS.items():
        actual = getattr(report, f"avg_{metric}", 0.0)
        if actual < threshold:
            failures.append(f"{METRIC_NAMES.get(metric, metric)}: {actual:.3f} < {threshold:.3f}")

    if failures:
        print("\n[GATE FAIL] 以下指标未达标:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\n[GATE PASS] 所有门禁指标达标")


async def cmd_grid(args):
    """网格搜索 —— 在参数空间内搜索最佳配置。"""
    if args.params:
        try:
            param_grid = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(f"[ERROR] --params JSON 解析失败: {e}")
            sys.exit(1)
        stage_name = "custom"
        fixed_params = {}
    elif args.stage in GRID_SEARCH_PLAN:
        param_grid = GRID_SEARCH_PLAN[args.stage]
        stage_name = ", ".join(param_grid.keys())
        # 固定本轮不搜索的参数为默认值
        fixed_params = {
            k: v
            for k, v in GRID_FIXED_DEFAULTS.items()
            if k not in param_grid
        }
    else:
        print(f"[ERROR] 无效的 stage: {args.stage}，可选: {list(GRID_SEARCH_PLAN.keys())}")
        sys.exit(1)

    if not DATASET_PATH.exists():
        print(f"[ERROR] 数据集不存在: {DATASET_PATH}")
        sys.exit(1)

    # 展开参数网格
    combinations = _expand_grid(param_grid)
    print(f"网格搜索: stage {args.stage} ({stage_name})")
    print(f"参数空间: {param_grid}")
    print(f"固定参数: {fixed_params}")
    print(f"组合数:   {len(combinations)}")
    print(f"跳过生成: {args.skip_generation}")
    print()

    best_report: EvalReport | None = None
    best_score = -1.0
    best_overrides: dict = {}
    all_summaries: list[tuple[dict, EvalReport]] = []

    for i, search_overrides in enumerate(combinations):
        overrides = {**fixed_params, **search_overrides}
        print(f"[{i+1}/{len(combinations)}] {overrides}")

        evaluator = RAGEvaluator(config_overrides=overrides)
        async with get_session_etx() as session:
            report = await evaluator.evaluate_dataset(
                session, DATASET_PATH, skip_generation=args.skip_generation
            )

        score = report.avg_context_recall  # 主要优化目标
        status = "PASS" if score >= THRESHOLDS.get("context_recall", 0.85) else "FAIL"
        print(
            f"  Recall={score:.3f} Prec={report.avg_context_precision:.3f} "
            f"Ent={report.avg_context_entity_recall:.3f} "
            f"Faith={report.avg_faithfulness:.3f} Relev={report.avg_answer_relevancy:.3f} "
            f"Corr={report.avg_answer_correctness:.3f} Sim={report.avg_answer_similarity:.3f} "
            f"Crit={report.avg_aspect_critique:.3f} [{status}]"
        )

        all_summaries.append((overrides, report))

        if score > best_score:
            best_score = score
            best_report = report
            best_overrides = overrides

    # 输出最终结果
    print()
    print("=" * 70)
    print("网格搜索完成")
    print("=" * 70)
    print(f"参数空间: stage {args.stage} ({stage_name})")
    for overrides, report in all_summaries:
        print(
            f"  {overrides}: Recall={report.avg_context_recall:.3f} "
            f"Prec={report.avg_context_precision:.3f} "
            f"Faith={report.avg_faithfulness:.3f} Relev={report.avg_answer_relevancy:.3f} "
            f"Corr={report.avg_answer_correctness:.3f}"
        )
    print()

    if best_report:
        print("最佳配置:")
        print(f"  参数: {best_overrides}")
        print(f"  Context Recall:        {best_report.avg_context_recall:.3f}")
        print(f"  Context Precision:     {best_report.avg_context_precision:.3f}")
        print(f"  Context Entity Recall: {best_report.avg_context_entity_recall:.3f}")
        print(f"  Faithfulness:          {best_report.avg_faithfulness:.3f}")
        print(f"  Answer Relevancy:      {best_report.avg_answer_relevancy:.3f}")
        print(f"  Answer Correctness:    {best_report.avg_answer_correctness:.3f}")
        print(f"  Answer Similarity:     {best_report.avg_answer_similarity:.3f}")
        print(f"  Aspect Critique:       {best_report.avg_aspect_critique:.3f}")

        output_path = _save_report(best_report, f"grid-stage{args.stage}")
        print(f"\n报告已保存: {output_path}")

        if best_score >= THRESHOLDS.get("context_recall", 0.85):
            print("\n[GATE PASS] 最佳 Context Recall 达标")
        else:
            print("\n[GATE FAIL] 所有配置的 Context Recall 均未达标")


async def cmd_compare(args):
    """对比两次评估报告。"""
    report_paths = [Path(p) for p in args.reports]
    for p in report_paths:
        if not p.exists():
            print(f"[ERROR] 报告不存在: {p}")
            sys.exit(1)

    reports_data = []
    for p in report_paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        reports_data.append((p.name, data))

    print("=" * 70)
    print("评估报告对比")
    print("=" * 70)
    header = f"  {'指标':<25}"
    for name, _ in reports_data:
        header += f" {name:<20}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    metrics = list(METRIC_NAMES.keys())
    for metric in metrics:
        line = f"  {METRIC_NAMES[metric]:<25}"
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
            print(
                f"  Q{i+1}: {ra['context_recall']:.3f} vs {rb['context_recall']:.3f} "
                f"(diff={diff:+.3f}){marker}"
            )
    print("=" * 70)


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def _expand_grid(param_grid: dict) -> list[dict]:
    """展开参数网格为组合列表。"""
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations


def _save_report(report: EvalReport, prefix: str) -> Path:
    """保存评估报告为 JSON 文件，按日期分目录。"""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    dated_dir = OUTPUT_DIR / date_str
    dated_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    path = dated_dir / filename
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
  %(prog)s grid --stage 1              第一轮网格搜索（rrf_k × dense_weight）
  %(prog)s grid --stage 2              第二轮网格搜索（rerank_top_k × context_max_chars）
  %(prog)s grid --params '{"rrf_k":[30,60]}'  自定义网格
  %(prog)s compare r1.json r2.json     对比两次报告
        """,
    )
    sub = parser.add_subparsers(dest="command")

    # baseline
    p_base = sub.add_parser("baseline", help="基线评估")
    p_base.add_argument(
        "--skip-generation",
        action="store_true",
        help="跳过 LLM 生成（仅评估检索指标）",
    )

    # grid
    p_grid = sub.add_parser("grid", help="参数网格搜索")
    p_grid.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        help="预设搜索阶段: 1=rrf_k×dense_weight, 2=rerank_top_k×context_max_chars",
    )
    p_grid.add_argument(
        "--params",
        type=str,
        help='JSON 格式的自定义参数网格，如 \'{"rrf_k":[30,60,90]}\'',
    )
    p_grid.add_argument(
        "--skip-generation",
        action="store_true",
        help="跳过 LLM 生成（更快，仅评估检索质量）",
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
