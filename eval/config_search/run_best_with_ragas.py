"""用最优配置（topk8）完整测试 40 题 RAG + RAGAS。

目标：
  1. 检查 RAGAS 流程是否存在超时、限流、降级等问题
  2. 逐题记录错误和异常
  3. 对比检索-only vs 全流程的结果差异

用法：
  cd f:\all-projs\coursepilot
  $env:PYTHONPATH="src"; .venv\Scripts\python eval\config_search\run_best_with_ragas.py

报告输出：
  eval/reports/20260727/best_topk8_ragas_YYYYMMDD_HHMMSS.json
  eval/reports/20260727/best_topk8_ragas_YYYYMMDD_HHMMSS.txt
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from coursepilot.config import settings
from coursepilot.db import get_session_etx
from coursepilot.evaluation.rag_eval import RAGEvaluator, EvalReport

DATASET = Path("eval/questions/20260726/eval_questions.json")
OUTPUT_DIR = Path("eval/reports") / "20260727"

# 最优配置
BEST_CONFIG = {
    "rerank_top_k": 8,
    "rrf_k": 60,
    "dense_weight": 0.5,
    "enable_bm25": True,
    "enable_rerank": True,
    "enable_kp_expand": True,
    "kp_expand_mode": "full",
    "context_max_chars": 5000,
}

# 稳定性参数
RAGAS_MAX_TOKENS = 4096    # 剩余 RAGAS 指标不需要太长
RAGAS_TIMEOUT = 120.0      # 单次请求超时（无 keepalive，更快断开）
RAGAS_MAX_WORKERS = 4      # 并发 4（HTTP/1.1 + 无 keepalive）


async def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if not DATASET.exists():
        print(f"[ERROR] 数据集不存在: {DATASET}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    questions = json.loads(DATASET.read_text(encoding="utf-8"))
    total = len(questions)
    answerable = sum(1 for q in questions if q.get("question_type") != "unanswerable")
    print(f"[run_best] 数据集: {DATASET}")
    print(f"[run_best] 总题数: {total} (有效 {answerable})")
    print(f"[run_best] 最佳配置: {BEST_CONFIG}")
    print(f"[run_best] RAGAS 参数: max_tokens={RAGAS_MAX_TOKENS}, timeout={RAGAS_TIMEOUT}s, max_workers={RAGAS_MAX_WORKERS}")
    print()

    evaluator = RAGEvaluator(
        config_overrides=BEST_CONFIG,
        use_mimo=True,
        ragas_max_tokens=RAGAS_MAX_TOKENS,
        ragas_timeout=RAGAS_TIMEOUT,
        ragas_max_workers=RAGAS_MAX_WORKERS,
    )

    t0 = time.monotonic()

    async with get_session_etx() as session:
        print("=" * 70)
        print("阶段1: 检索 + 生成（无 RAGAS）")
        print("=" * 70)
        report = await evaluator.evaluate_dataset(
            session, DATASET,
            skip_generation=False,  # 生成
            skip_ragas=False,       # 跑 RAGAS
        )

    elapsed = time.monotonic() - t0

    # ═══ 输出结果 ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  结果汇总")
    print("=" * 70)
    print(report.summary())
    print(f"\n总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ═══ 错误分析 ═══════════════════════════════════════════
    error_results = [r for r in report.results if r.error]
    if error_results:
        print("\n" + "=" * 70)
        print(f"  [WARN] 共 {len(error_results)} 道题有错误")
        print("=" * 70)
        for r in error_results:
            print(f"  Q: {r.question[:60]}")
            print(f"  类型: {r.question_type}")
            print(f"  错误: {r.error}")
            print()
    else:
        print("\n[OK] 无错误")

    # ═══ RAGAS 异常检测 ═══════════════════════════════════════
    # 检查 RAGAS 评分为 0.0 或异常值的题目（可能被降级）
    low_quality = []
    for r in report.results:
        if r.question_type == "unanswerable":
            continue
        issues = []
        if r.faithfulness == 0.0 and not r.error:
            issues.append("faithfulness=0.0")
        if r.answer_relevancy == 0.0 and not r.error:
            issues.append("answer_relevancy=0.0")
        if r.answer_correctness == 0.0 and not r.error:
            issues.append("answer_correctness=0.0")
        if r.context_precision == 0.0 and r.context_recall > 0 and not r.error:
            issues.append("context_precision=0.0 (recall>0)")
        if r.context_entity_recall == 0.0 and r.context_recall > 0 and not r.error:
            issues.append("entity_recall=0.0 (recall>0)")
        if issues:
            low_quality.append((r, issues))

    if low_quality:
        print("\n" + "=" * 70)
        print(f"  [ANOMALY] {len(low_quality)} 道题可能被降级/误判")
        print("=" * 70)
        for r, issues in low_quality:
            print(f"  Q: {r.question[:60]}")
            print(f"  类型: {r.question_type}  异常: {', '.join(issues)}")
            print(f"    recall={r.context_recall:.3f} faith={r.faithfulness:.3f} "
                  f"relev={r.answer_relevancy:.3f} corr={r.answer_correctness:.3f} "
                  f"prec={r.context_precision:.3f} ent={r.context_entity_recall:.3f}")
            print()
    else:
        print("\n[OK] 无异常 RAGAS 评分")

    # ═══ 按题型统计 ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  按题型统计")
    print("=" * 70)
    type_stats: dict[str, dict] = {}
    for r in report.results:
        if r.question_type == "unanswerable":
            continue
        if r.question_type not in type_stats:
            type_stats[r.question_type] = {
                "count": 0, "recall": 0.0, "faith": 0.0, "correct": 0.0,
                "prec": 0.0, "ent": 0.0, "relev": 0.0, "sim": 0.0, "crit": 0.0,
                "latency_ms": 0.0,
            }
        ts = type_stats[r.question_type]
        ts["count"] += 1
        ts["recall"] += r.context_recall
        ts["faith"] += r.faithfulness
        ts["correct"] += r.answer_correctness
        ts["prec"] += r.context_precision
        ts["ent"] += r.context_entity_recall
        ts["relev"] += r.answer_relevancy
        ts["sim"] += r.answer_similarity
        ts["crit"] += r.aspect_critique
        ts["latency_ms"] += r.latency_ms

    for qt, ts in sorted(type_stats.items()):
        n = ts["count"]
        print(f"  [{qt:<12}] count={n:2d}  recall={ts['recall']/n:.4f}  "
              f"faith={ts['faith']/n:.4f}  correct={ts['correct']/n:.4f}  "
              f"prec={ts['prec']/n:.4f}  ent={ts['ent']/n:.4f}  "
              f"relev={ts['relev']/n:.4f}  sim={ts['sim']/n:.4f}  "
              f"crit={ts['crit']/n:.4f}  latency={ts['latency_ms']/n:.0f}ms")

    # ═══ 保存 ═══════════════════════════════════════════════════
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"best_topk8_ragas_{timestamp}.json"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    txt_path = OUTPUT_DIR / f"best_topk8_ragas_{timestamp}.txt"
    txt_path.write_text(report.summary(), encoding="utf-8")

    print(f"\n报告已保存:")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")


if __name__ == "__main__":
    asyncio.run(main())
