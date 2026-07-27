"""运行指定配置的 RAG + RAGAS 评估。

用法：
  cd f:\all-projs\coursepilot
  $env:PYTHONPATH="src"; .venv\Scripts\python eval\config_search\run_config.py baseline
  $env:PYTHONPATH="src"; .venv\Scripts\python eval\config_search\run_config.py rrf_k30
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
from coursepilot.evaluation.rag_eval import RAGEvaluator

DATASET = Path("eval/questions/20260726/eval_questions.json")
OUTPUT_DIR = Path("eval/reports") / "20260727"

# ── 各配置参数 ──────────────────────────────────────────────
CONFIGS = {
    "baseline": {},   # 全默认
    "topk8": {"rerank_top_k": 8},
    "rrf_k30": {"rrf_k": 30},
}

# 稳定性参数
RAGAS_MAX_WORKERS = 4
RAGAS_TIMEOUT = 300.0
RAGAS_MAX_TOKENS = 8192


async def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else ""
    if config_name not in CONFIGS:
        print(f"用法: python {sys.argv[0]} [{'|'.join(CONFIGS.keys())}]")
        print(f"可用配置: {list(CONFIGS.keys())}")
        return 1

    config = CONFIGS[config_name]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if not DATASET.exists():
        print(f"[ERROR] 数据集不存在: {DATASET}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    questions = json.loads(DATASET.read_text(encoding="utf-8"))
    total = len(questions)
    answerable = sum(1 for q in questions if q.get("question_type") != "unanswerable")
    print(f"[run_config] 配置: {config_name} = {config}")
    print(f"[run_config] 数据集: {DATASET} ({total} 题, 有效 {answerable})")
    print()

    evaluator = RAGEvaluator(
        config_overrides=config,
        use_mimo=True,
        ragas_max_tokens=RAGAS_MAX_TOKENS,
        ragas_timeout=RAGAS_TIMEOUT,
        ragas_max_workers=RAGAS_MAX_WORKERS,
    )

    t0 = time.monotonic()
    async with get_session_etx() as session:
        print("=" * 70)
        print("阶段1: 检索 + 生成")
        print("=" * 70)
        report = await evaluator.evaluate_dataset(
            session, DATASET,
            skip_generation=False,
            skip_ragas=False,
        )
    elapsed = time.monotonic() - t0

    # ═══ 输出结果 ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"  结果汇总 - {config_name}")
    print("=" * 70)
    print(report.summary())
    print(f"\n总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ═══ 错误分析 ═══════════════════════════════════════════
    error_results = [r for r in report.results if r.error]
    if error_results:
        print(f"\n[WARN] 共 {len(error_results)} 道题有错误")
        for r in error_results:
            print(f"  Q: {r.question[:60]}  err: {r.error}")
    else:
        print("\n[OK] 无错误")

    # ═══ 保存 ═══════════════════════════════════════════════════
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"{config_name}_ragas_{timestamp}.json"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    txt_path = OUTPUT_DIR / f"{config_name}_ragas_{timestamp}.txt"
    txt_path.write_text(report.summary(), encoding="utf-8")
    print(f"\n报告已保存:")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
