"""Agentic RAG vs 传统 RAG 的 RAGAS 对比评估 CLI（P3.2/P3.3）。

P3.2：同一黄金数据集，分别用
  - 传统管线（Retriever + Generator，即 CRAG 的检索生成主链路）
  - Agentic RAG（agentic_rag_node ReAct 循环）
跑出回答，再用同一套 RAGAS 指标评分，逐项对比。

P3.3：输出 Agentic RAG 的成本/延迟画像：平均步数、每步 token、
总 token、成本估算、工具调用分布（方案 §10 验证点 #5 的数据收集）。

用法：
    PYTHONPATH=src .venv/Scripts/python -m eval.eval_agentic_rag
    PYTHONPATH=src .venv/Scripts/python -m eval.eval_agentic_rag --dataset eval/questions/20260726/eval_questions.json --limit 3
    PYTHONPATH=src .venv/Scripts/python -m eval.eval_agentic_rag --only crag    # 只跑传统管线
    PYTHONPATH=src .venv/Scripts/python -m eval.eval_agentic_rag --only agent   # 只跑 Agentic RAG
    PYTHONPATH=src .venv/Scripts/python -m eval.eval_agentic_rag --skip-ragas   # 不跑 RAGAS 评分（只出问答+统计）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from coursepilot.agent.rag_agent import agentic_rag_node
from coursepilot.db import get_session_etx
from coursepilot.evaluation.rag_eval import EvalReport, EvalResult, RAGEvaluator
from coursepilot.rag.generator import build_course_context

DEFAULT_DATASET = Path("eval/questions/20260726/eval_questions.json")
OUTPUT_DIR = Path("eval/reports")

# 假身份（评估不落库，仅满足 agent 循环对 state 的字段要求）
_FAKE_USER_ID = "00000000-0000-0000-0000-000000000000"


# ══════════════════════════════════════════════════════════════
# 两种评估模式的单题执行
# ══════════════════════════════════════════════════════════════

def _unanswerable_result(question: dict) -> EvalResult:
    """不可回答问题：不检索不生成，直接返回标准答案（与 RAGEvaluator 一致）。"""
    return EvalResult(
        question=question["question"],
        question_type=question.get("question_type", ""),
        kp_path=question.get("kp_path", ""),
        ground_truth_uuids=question.get("ground_truth_contexts", []),
        ground_truth=question.get("ground_truth", question.get("answer", "")),
        answer=question.get("ground_truth", question.get("answer", "教材未涉及此内容，无法回答")),
    )


async def _run_crag(
    evaluator: RAGEvaluator,
    session,
    question: dict,
    course_id: str,
    course_context: dict,
) -> EvalResult:
    """传统管线：Retriever + Generator（复用 RAGEvaluator 的准备逻辑）。"""
    if question.get("question_type") == "unanswerable" or question.get("unanswerable"):
        return _unanswerable_result(question)
    return await evaluator._prepare_result(
        session, question, course_id, _cached_course_context=course_context,
    )


async def _run_agent(
    evaluator: RAGEvaluator,
    session,
    question: dict,
    course_id: str,
    course_context: dict,
) -> tuple[EvalResult, dict]:
    """Agentic RAG：agentic_rag_node ReAct 循环，闭环契约转成 EvalResult。

    返回 (EvalResult, trace)，trace 供 P3.3 成本/延迟统计。
    """
    if question.get("question_type") == "unanswerable" or question.get("unanswerable"):
        return _unanswerable_result(question), {}

    t0 = time.monotonic()
    state = {
        "query": question["question"],
        "course_id": course_id,
        "user_id": _FAKE_USER_ID,
        "session_id": str(uuid4()),
        "course_context": course_context,
        "user_profile": None,
        "conversation": [],
        "rolling_summary": "",
        "llm_calls": [],
        "agent_steps": [],
        "tool_history": [],
    }
    try:
        result = await agentic_rag_node(state)
    except Exception as e:
        result = {"error": str(e)}
    latency_ms = (time.monotonic() - t0) * 1000

    result_ctx = result.get("context", "")
    metadata = result.get("retrieved_metadata", {})
    citation_map = metadata.get("citation_map", {})
    retrieved_uuids = [
        v["uuid"] for v in citation_map.values() if v.get("uuid")
    ]
    gt_uuids = question.get("ground_truth_contexts", [])

    r = EvalResult(
        question=question["question"],
        question_type=question.get("question_type", ""),
        kp_path=question.get("kp_path", ""),
        ground_truth_uuids=gt_uuids,
        ground_truth=question.get("ground_truth", question.get("answer", "")),
        retrieved_uuids=retrieved_uuids,
        answer=result.get("answer", ""),
        context_length=len(result_ctx),
        latency_ms=latency_ms,
    )
    r.context_recall = RAGEvaluator._compute_context_recall(gt_uuids, retrieved_uuids)
    if result.get("error"):
        r.error = str(result["error"])
    if r.retrieved_uuids:
        r.retrieved_contexts = await evaluator._load_units(session, r.retrieved_uuids)

    llm_calls = result.get("llm_calls") or []
    trace = {
        "step_count": len(result.get("agent_steps") or []),
        "tools_used": [s.get("tool") for s in (result.get("agent_steps") or [])],
        "llm_calls": llm_calls,
        "total_tokens": sum(c.get("total_tokens", 0) for c in llm_calls),
        "degraded": bool(result.get("degraded_mode")),
    }
    return r, trace


# ══════════════════════════════════════════════════════════════
# P3.3 Agent 成本/延迟画像
# ══════════════════════════════════════════════════════════════

def _agent_stats(results: list[EvalResult], all_agent_data: list[dict]) -> str:
    """从 agent 模式结果统计：平均步数、每步 token、总 token、成本估算、工具分布。"""
    if not all_agent_data:
        return "（无 Agentic RAG 数据）"

    steps_per_q = [d.get("step_count", 0) for d in all_agent_data]
    tokens_per_q = [d.get("total_tokens", 0) for d in all_agent_data]
    degraded = sum(1 for d in all_agent_data if d.get("degraded"))

    # 每步 token（只统计 agent_step 节点，排除 finalize 生成）
    step_tokens = [
        c.get("total_tokens", 0)
        for d in all_agent_data
        for c in d.get("llm_calls", [])
        if c.get("node") == "agent_step"
    ]
    step_count_total = max(len(step_tokens), 1)

    # 工具调用分布
    tool_dist: dict[str, int] = {}
    for d in all_agent_data:
        for t in d.get("tools_used", []):
            tool_dist[t] = tool_dist.get(t, 0) + 1

    # 成本估算：按总 token 粗略估算（以 deepseek 类约 ¥1/百万 token 为参考）
    total_tokens = sum(tokens_per_q)
    est_cost_yuan = total_tokens / 1_000_000 * 1.0

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    lines = [
        "Agentic RAG 成本/延迟画像（P3.3）:",
        f"  评估题目数:            {len(all_agent_data)}",
        f"  平均步数:              {_mean(steps_per_q):.2f} 步/题（最大 {max(steps_per_q)}）",
        f"  平均每步 token:        {_mean(step_tokens):.0f}",
        f"  平均总 token:          {_mean(tokens_per_q):.0f}/题",
        f"  总 token:              {total_tokens}",
        f"  降级次数(guardrail):   {degraded}",
        f"  估算成本:              ¥{est_cost_yuan:.4f}（按 ¥1/百万 token 粗略估算）",
        f"  工具调用分布:          {json.dumps(tool_dist, ensure_ascii=False)}",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

async def run(dataset_path: Path, *, only: str, skip_ragas: bool, limit: int) -> None:
    questions = json.loads(dataset_path.read_text(encoding="utf-8"))
    if limit and limit > 0:
        questions = questions[:limit]
    if not questions:
        print("[ERROR] 数据集为空")
        sys.exit(1)
    course_id = questions[0].get("course_id", "")
    if not course_id:
        print("[ERROR] 数据集中缺少 course_id")
        sys.exit(1)

    print(f"数据集: {dataset_path}（{len(questions)} 题）")
    print(f"模式: {'/'.join(x for x in ('crag', 'agent') if only in ('', x))}")
    print(f"跳过 RAGAS: {skip_ragas}")

    evaluator = RAGEvaluator()
    async with get_session_etx() as session:
        course_context = await build_course_context(session, course_id)

        reports: dict[str, EvalReport] = {}
        all_agent_data: list[dict] = []

        if only in ("", "crag"):
            print("\n" + "=" * 40 + "\n[crag] 传统管线评估中...\n" + "=" * 40)
            crag_results = [
                await _run_crag(evaluator, session, q, course_id, course_context)
                for q in questions
            ]
            reports["crag"] = EvalReport(
                results=crag_results,
                config={"mode": "crag"},
            )

        if only in ("", "agent"):
            print("\n" + "=" * 40 + "\n[agent] Agentic RAG 评估中...\n" + "=" * 40)
            agent_results: list[EvalResult] = []
            for i, q in enumerate(questions):
                print(f"[agent] [{i+1}/{len(questions)}] {q['question'][:50]}")
                r, trace = await _run_agent(evaluator, session, q, course_id, course_context)
                agent_results.append(r)
                all_agent_data.append(trace)
                print(f"  -> answer_chars={len(r.answer)}, error={r.error!r}")
            reports["agent"] = EvalReport(
                results=agent_results,
                config={"mode": "agent"},
            )

        # RAGAS 评分（可跳过）
        if not skip_ragas:
            print("\n[ragas] 开始 RAGAS 评分...")
            for mode, report in reports.items():
                print(f"[ragas] 评分模式: {mode}")
                scores = await evaluator._run_ragas(report.results)
                for r, s in zip(report.results, scores):
                    evaluator._merge_ragas_scores(r, s)

        # 汇总输出
        for mode, report in reports.items():
            print()
            print(report.summary())
            _save_report(report, mode)

    # 对比表 + P3.3（在 session 之外输出）
    _print_comparison(reports)
    print()
    print(_agent_stats(list(reports.get("agent", EvalReport(results=[], config={})).results), all_agent_data))


def _print_comparison(reports: dict[str, EvalReport]) -> None:
    if "crag" not in reports or "agent" not in reports:
        return
    metrics = [
        ("context_recall", "Context Recall"),
        ("context_precision", "Context Precision"),
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevancy"),
        ("answer_correctness", "Answer Correctness"),
        ("answer_similarity", "Answer Similarity"),
        ("aspect_critique", "Aspect Critique"),
        ("context_length", "Context Length"),
        ("latency_ms", "延迟(ms)"),
    ]
    crag, agent = reports["crag"], reports["agent"]
    print("\n" + "=" * 70)
    print("对比：传统管线 vs Agentic RAG（P3.2）")
    print("=" * 70)
    header = f"  {'指标':<18}{'crag':<12}{'agent':<12}{'差值':<10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key, label in metrics:
        va = _report_avg(crag, key)
        vb = _report_avg(agent, key)
        diff = vb - va
        marker = "  <<< Agent 更优" if diff > 0.05 else ("  >>> CRAG 更优" if diff < -0.05 else "")
        print(f"  {label:<18}{va:<12.3f}{vb:<12.3f}{diff:<+10.3f}{marker}")


def _report_avg(report: EvalReport, key: str) -> float:
    return getattr(report, f"avg_{key}", 0.0)


def _save_report(report: EvalReport, mode: str) -> None:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    dated_dir = OUTPUT_DIR / date_str
    dated_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = dated_dir / f"agentic_vs_crag_{mode}_{timestamp}.json"
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n报告已保存: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG vs 传统 RAG 的 RAGAS 对比评估（P3.2/P3.3）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 题（调试用）")
    parser.add_argument("--only", choices=["crag", "agent"], default="",
                        help="只跑某一种模式（默认两种都跑）")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="跳过 RAGAS 评分（只出问答与统计）")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"[ERROR] 数据集不存在: {args.dataset}")
        sys.exit(1)

    asyncio.run(run(args.dataset, only=args.only, skip_ragas=args.skip_ragas, limit=args.limit))


if __name__ == "__main__":
    main()
