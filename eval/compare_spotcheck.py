"""
聚焦验证：传统 RAG vs Agentic RAG 在对比类问题上的表现。

测试仅 2 个问题，看传统 RAG 是否"只检索到一边"而 Agentic RAG 能否同时覆盖两边。
"""
from __future__ import annotations

import asyncio
import sys

# Windows 修正：psycopg3 不支持 ProactorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import json
import time
from pathlib import Path

from coursepilot.db import get_session_etx
from coursepilot.evaluation.rag_eval import RAGEvaluator, EvalResult
from coursepilot.models.knowledge_unit import KnowledgeUnit
from coursepilot.models.knowledge_point import KnowledgePoint
from sqlalchemy import select

OUTPUT_PATH = Path("eval/reports/20260728/comparison_spotcheck.json")


def load_spotcheck_questions() -> list[dict]:
    """从现有 topk8 报告中提取两个对比问题"""
    report_path = Path("eval/reports/20260727/topk8_ragas_20260727_132921.json")
    report = json.loads(report_path.read_text("utf-8"))

    questions = []
    for entry in report["results"]:
        if entry["question_type"] != "comparison":
            continue
        if "L'Hospital" in entry["question"] or "单调性" in entry["question"]:
            questions.append({
                "question": entry["question"],
                "answer": entry.get("answer", ""),
                "question_type": "comparison",
                "kp_path": entry.get("kp_path", ""),
                "course_id": "e7a20f2f-c98e-4ff3-9938-04351616e66d",
                "ground_truth_contexts": entry["ground_truth_uuids"],
            })
    return questions


async def resolve_uuids(uuids: list[str]) -> dict[str, str]:
    """查 UUID 对应的 KP 路径"""
    async with get_session_etx() as s:
        resolved = {}
        for uid in uuids:
            try:
                from uuid import UUID
                uuid_val = UUID(uid)
            except ValueError:
                resolved[uid] = uid
                continue
            r = await s.execute(
                select(KnowledgePoint.kp_path)
                .join(KnowledgeUnit, KnowledgeUnit.kp_id == KnowledgePoint.id)
                .where(KnowledgeUnit.id == uuid_val)
            )
            kp_path = r.scalar_one_or_none()
            resolved[uid] = kp_path if kp_path else "(未在数据库中)"
        return resolved


async def run_traditional_rag(question: dict) -> dict:
    """运行传统 RAG 单题（直接复用 query_rag skill）"""
    from coursepilot.agent.skills.query_rag import query_rag
    from coursepilot.rag.generator import build_course_context

    query = question["question"]
    course_id = question.get("course_id", "")

    t0 = time.monotonic()

    async with get_session_etx() as session:
        course_context = await build_course_context(session, course_id)
        answer, context, metadata, sources, token_info = await query_rag(
            session=session,
            query=query,
            course_id=course_id,
            course_context=course_context,
        )

        # 从 metadata 中提取 KP 分布
        top_kp_paths = metadata.get("source_kp_paths", [])
        latency = (time.monotonic() - t0) * 1000

        return {
            "context_len": len(context),
            "top_kp_paths": top_kp_paths,
            "answer": answer,
            "latency_ms": round(latency),
            "token_info": token_info,
        }


async def run_agentic_rag(question: dict) -> dict:
    """运行 Agentic RAG 单题"""
    from coursepilot.agent.graph import build_agent_graph

    query = question["question"]
    course_id = question.get("course_id", "")

    t0 = time.monotonic()
    try:
        graph = await build_agent_graph()
        state = await graph.ainvoke({
            "query": query,
            "course_id": course_id,
            "user_id": "eval_spotcheck",
        }, config={"configurable": {"thread_id": "eval_spotcheck", "checkpoint_ns": "", "checkpoint_id": ""}})

        answer = state.get("answer", "")
        context = state.get("context", "")
        agent_steps = state.get("agent_steps") or []
        tool_history = state.get("tool_history") or []
        degraded_mode = state.get("degraded_mode", False)

        latency = (time.monotonic() - t0) * 1000

        # 从决策轨迹统计检索行为（P1 后 CRAG 字段已删除，改用 agent_steps）
        has_decompose = any(s.get("tool") == "plan" for s in agent_steps)
        has_web_search = any(s.get("tool") == "web_search" for s in agent_steps)

        return {
            "context_len": len(context),
            "answer": answer,
            "decomposed": has_decompose,
            "agent_step_count": len(agent_steps),
            "web_search_used": has_web_search,
            "degraded_mode": degraded_mode,
            "agent_steps": agent_steps,
            "tool_history": tool_history,
            "latency_ms": round(latency),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "latency_ms": round((time.monotonic() - t0) * 1000),
        }


async def main():
    questions = load_spotcheck_questions()
    print(f"加载 {len(questions)} 个对比问题\n")

    for q in questions:
        print("=" * 70)
        print(f"问题: {q['question']}")
        print(f"类型: {q['question_type']}")
        print(f"参考 UUIDs: {q['ground_truth_contexts']}")
        print()

        # 解析 UUID → KP 路径
        kp_map = await resolve_uuids(q["ground_truth_contexts"])
        print(f"参考 KP 分布:")
        for uid, kp in kp_map.items():
            print(f"  {uid[:8]}... → {kp}")

        print()

        # === 传统 RAG ===
        print("▶ 传统 RAG:")
        trad = await run_traditional_rag(q)
        if "answer" in trad:
            print(f"  检索候选: {len(trad.get('top_kp_paths', []))} 条")
            retrieved_kps = set(k for k in trad.get("top_kp_paths", []) if k)
            print(f"  覆盖 KP: {len(retrieved_kps)} 个")
            for kp in sorted(retrieved_kps):
                print(f"    · {kp}")
            print(f"  回答预览: {trad['answer'][:200]}...")
            print(f"  延迟: {trad['latency_ms']}ms")

        print()

        # === Agentic RAG ===
        print("▶ Agentic RAG:")
        agent = await run_agentic_rag(q)
        if "answer" in agent:
            print(f"  查询分解(plan 工具): {agent['decomposed']}")
            print(f"  决策步数: {agent['agent_step_count']}")
            for s in agent["agent_steps"]:
                print(f"    · {s.get('tool')}: {s.get('args')}")
            print(f"  网络搜索: {agent['web_search_used']}")
            print(f"  降级模式: {agent['degraded_mode']}")
            print(f"  回答预览: {agent['answer'][:200]}...")
            print(f"  延迟: {agent['latency_ms']}ms")
        else:
            print(f"  ❌ 失败: {agent.get('error', '未知')}")

        print()

    # 保存
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({
            "config": {"agentic_rag_version": "P1（主图切换）"},
            "questions": questions,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
