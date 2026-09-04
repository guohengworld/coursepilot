"""LangGraph 节点函数

每个节点接收 AgentState，返回状态更新字典（只写自己负责的字段）
"""
import asyncio
import logging
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.agent.context import build_context as build_context_logic
from coursepilot.agent.profile_updater import update_profile
from coursepilot.agent.skills.classify_intent import classify_intent
from coursepilot.agent.skills.diagnose import diagnose, generate_llm_analysis
from coursepilot.agent.skills.evaluate_quiz import evaluate_quiz
from coursepilot.agent.skills.generate_quiz import generate_quiz
from coursepilot.agent.skills.get_mastery import get_mastery
from coursepilot.agent.skills.query_rag import query_rag
from coursepilot.agent.skills.review_plan import review_plan
from coursepilot.agent.skills.update_qa_record import update_qa_record
from coursepilot.agent.state_models import QUIZ_FALLBACK
from coursepilot.db import async_session_factory
from coursepilot.models import AgentSession
from coursepilot.rag.config import config as rag_config

logger = logging.getLogger(__name__)

# 后台任务强引用登记：asyncio 只持有弱引用，裸 create_task 可能在事件循环
# 某次迭代后被 GC（尤其任务尚未 await 完成时），导致副作用静默丢失。
_BACKGROUND_TASKS: set[asyncio.Task] = set()


def _spawn_background(coro: Any) -> asyncio.Task:
    """登记强引用启动后台任务；完成后经 done_callback 自动移出集合。"""
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task

async def build_context_node(state: dict) -> dict:
    """构建上下文：课程信息（KP数+教材名） + 学生画像 + 最近问答"""
    try:
        async with async_session_factory() as session:
            course_ctx, profile, recent_qa = await build_context_logic(
                session,
                user_id=state["user_id"],
                course_id=state["course_id"]
            )
        return {
            "course_context": course_ctx,
            "user_profile": profile,
            "recent_qa": recent_qa,
            "error": None,
        }
    except Exception as e:
        logger.exception("build_context 节点异常")
        return {"course_context": {}, "user_profile": None, "recent_qa": [], "error": str(e)}

async def classify_node(state: dict) -> dict:
    """意图分类 + 复杂度判断（P0: Agentic RAG 智能路由）"""
    try:
        # 优先使用 conversation 最近轮次；未接入时回退 recent_qa
        conversation = state.get("conversation") or state.get("recent_qa", [])
        cls_result = await classify_intent(
            query=state["query"],
            course_context=state.get("course_context", {}),
            recent_qa=conversation,
        )
        # 兼容 2-元组（旧）和 3-元组（P0 新增 complexity）
        if len(cls_result) == 3:
            intent, complexity, token_info = cls_result
        else:
            intent, token_info = cls_result
            complexity = "simple"
        llm_calls = list(state.get("llm_calls", []))
        llm_calls.append({"node": "classify", **token_info})
        logger.info("classify 结果: intent=%s complexity=%s", intent, complexity)
        return {
            "intent": intent,
            "complexity": complexity,
            "llm_calls": llm_calls,
            "error": None,
        }
    except Exception as e:
        logger.exception("classify 节点异常")
        return {
            "intent": "question",
            "complexity": "simple",
            "classify_degraded": True,
            "error": str(e),
        }


# 路由兜底固定引导文案：覆盖寒暄 / 离题 / 欠指定 / 分类异常等无学习意图输入。
# 提供"重新描述"自救入口，避免对误判为 none 的知识提问直接拒答。
FALLBACK_ANSWER = (
    "抱歉，我还没理解你想让我做什么。请重新描述一下你的问题："
    "可以直接输入想了解的知识点（如「什么是进程调度」），"
    "或试试「帮我出几道练习题」「分析一下我的学习情况」「帮我制定复习计划」这类指令。"
)


async def fallback_node(state: dict) -> dict:
    """路由兜底：intent 无法进入任何业务分支时统一收口

    覆盖三类输入（flag orch_route_fallback 开启时由 route_by_intent 送入）：
      - classify 节点异常降级（classify_degraded=True）
      - classify 判 none（寒暄 / 离题 / 欠指定）
      - intent 缺失 / "" / ∉ VALID_INTENTS（防御性收口）

    产出固定引导文案（不调 LLM）；intent 收敛为 "none" 供会话状态记录；
    fallback_reason 区分收口来源（受控枚举），routing_notes 记录原始 intent
    或异常信息（观测用）。QA Record 不落库，见 finalize_node Step C。
    """
    intent = state.get("intent", "")
    degraded = state.get("classify_degraded", False)

    if degraded:
        reason = "classify_degraded"
    elif intent == "none":
        reason = "none"
    else:
        reason = "unclassified"

    notes = str(state.get("error") or "") if degraded else f"intent={intent!r}"

    return {
        "answer": FALLBACK_ANSWER,
        "intent": "none",
        "fallback_reason": reason,
        "routing_notes": notes or None,
        "error": None,
    }


async def query_rag_node(state: dict) -> dict:
    """RAG 检索 + LLM 生成（携带分层记忆）。

    根据 complexity 切换快慢通道：
    - simple: 轻量检索（rerank_top_k=simple_top_k），保留 KP 扩展
    - complex: 全量检索（rerank_top_k=8，保留现有完整管道）
    """
    complexity = state.get("complexity", "simple")

    # ── 简单通道：临时切换 rerank_top_k 为轻量参数 ──
    if complexity == "simple" and rag_config.enable_routing:
        saved_rerank_top_k = rag_config.rerank_top_k
        rag_config.rerank_top_k = rag_config.simple_top_k
        logger.info("简单通道: rerank_top_k=%d (保留 KP 扩展)", rag_config.rerank_top_k)
        _need_restore = True
    else:
        _need_restore = False

    try:
        async with async_session_factory() as session:
            answer, context, metadata, sources, token_info = await query_rag(
                session=session,
                query=state["query"],
                course_id=state["course_id"],
                course_context=state.get("course_context", {}),
                conversation=state.get("conversation"),
                rolling_summary=state.get("rolling_summary", ""),
                user_profile=state.get("user_profile"),
            )
        # 在 token_info 中标记路由信息
        token_info["routing_complexity"] = complexity
        llm_calls = list(state.get("llm_calls", []))
        llm_calls.append({"node": "query_rag", **token_info})
        return {
            "answer": answer,
            "context": context,
            "retrieved_metadata": metadata,
            "sources": sources,
            "llm_calls": llm_calls,
            "error": None,
        }
    except Exception as e:
        logger.exception("query_rag 节点异常")
        return {
            "answer": f"抱歉，检索知识库时出错了：{e}",
            "error": str(e),
        }
    finally:
        # 恢复原始配置
        if _need_restore:
            rag_config.rerank_top_k = saved_rerank_top_k


async def finalize_node(state: dict) -> dict:
    """持久化 + 会话更新 + 滚动摘要 + 异步副作用（④ 拆分后仅做职责编排）

    Phase 3 增强：
      - 汇总 llm_calls 写入真实 token 计数和成本估算
      - 维护 conversation（L1）与 rolling_summary（L2）
      - 末尾异步触发 profile_updater.update_profile()
    """
    try:
        # 汇总所有 LLM 调用的 token 用量（供 QA 记录与 P5 快照）
        llm_calls = state.get("llm_calls", [])
        total_tokens = sum(c.get("total_tokens", 0) for c in llm_calls)
        total_prompt = sum(c.get("prompt_tokens", 0) for c in llm_calls)
        total_completion = sum(c.get("completion_tokens", 0) for c in llm_calls)

        # Step A: guardrails（纯计算，事务前；异常则整体失败，与旧序一致）
        guard_issues = _run_guardrails(state)

        # Step C/D/E: 唯一主事务（QA 记录 / 会话状态 / 诊断报告）
        await _persist_core(state, total_tokens, total_prompt, total_completion)

        # 滚动压缩（C4b：移出主事务，独立事务 + best-effort）——失败只丢压缩，
        # 不影响已提交的核心持久化；下一轮 needs_compaction 仍为真自动重试。
        compaction_count = state.get("compaction_count", 0)
        compaction_count += await _compact_best_effort(state)

        # 记录压缩次数到 llm_calls 便于可观测（P5）
        if compaction_count > 0:
            llm_calls.append({
                "node": "compaction",
                "compacted_turns": compaction_count,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            })

        # Step F: 异步 audit 日志（不阻塞，不强等）
        _spawn_audit_tasks(state, guard_issues)

        # Step G/H/I: 异步副作用（profile 更新 / L3 抽取 / QA embedding 补全）
        _spawn_side_effects(state)

        # P5: 可观测快照写回 state，供 admin 控制台消费
        return _observability_snapshot(llm_calls, total_tokens, compaction_count)
    except Exception as e:
        logger.exception("finalize 节点异常")
        return {"error": str(e)}


def _run_guardrails(state: dict) -> list[str]:
    """Step A: guardrails 检查（纯计算）。

    输入 answer/context/sources，产出违规清单（仅告警日志，不阻断）。
    兜底轮次跳过：固定引导文案非 LLM 生成、无检索上下文，
    检查只会产出假警报（如「较长回答缺少引用标记」）。
    """
    if state.get("fallback_reason"):
        return []

    from coursepilot.governance.guardrails import guard_answer

    guard_issues = guard_answer(
        answer=state.get("answer", ""),
        context=state.get("context", ""),
        sources=state.get("sources", []),
    )
    if guard_issues:
        logger.warning("Guardrail 警告: %s", guard_issues)
    return guard_issues


async def _persist_core(
    state: dict,
    total_tokens: int,
    total_prompt: int,
    total_completion: int,
) -> None:
    """Step C/D/E：核心持久化（唯一主事务）。

    - Step C: QA Record（仅有效会话；路由兜底跳过 —— ① 遗留逻辑落点）
    - Step D: 会话状态更新（L1 conversation / status / quiz_data）
    - Step E: 学情诊断报告

    滚动压缩已移出（见 _compact_best_effort），本函数不包含任何 LLM 调用，
    事务短、失败域小。任何异常向外抛出 → finalize 返回 error。
    """
    # fallback_reason 非空 = 本轮是兜底引导（none/未知/降级收口），
    # update_qa_record 会触发 embedding + importance 计算，对引导文案无意义；
    # Step D 的会话状态更新仍须保留。
    is_fallback = bool(state.get("fallback_reason"))
    session_id = state.get("session_id", "")

    async with async_session_factory() as session:
        # ── Step C: 写入 QA Record ──
        if session_id and not is_fallback:
            await update_qa_record(
                session=session,
                user_id=state["user_id"],
                course_id=state["course_id"],
                query=state["query"],
                answer=state.get("answer", ""),
                kp_path=_first_kp_path(state.get("retrieved_metadata", {})),
                retrieved_units=state.get("retrieved_metadata", {}).get("top_uuids", []),
                citations=state.get("sources", []),
                session_id=session_id,
                token_count=total_tokens,
                prompt_tokens=total_prompt,
                completion_tokens=total_completion,
            )

        # ── Step D: 更新会话状态（含 L1/L2 记忆，仅在 session_id 有效时） ──
        if session_id:
            await _update_session_intent(
                session, session_id,
                state.get("intent", "question"),
                quiz_data=state.get("quiz_data"),
                answer=state.get("answer", ""),
                sources=state.get("sources", []),
                citation_map=state.get("retrieved_metadata", {}).get("citation_map", {})
                or {},
                query=state.get("query", ""),
            )

        # ── Step E: 保存学情诊断报告 ──
        diagnosis = state.get("diagnosis")
        if diagnosis and diagnosis.get("total_practiced", 0) > 0:
            from coursepilot.models import DiagnosisReport
            report = DiagnosisReport(
                user_id=UUID(state["user_id"]),
                course_id=UUID(state["course_id"]),
                session_id=UUID(state["session_id"]),
                overall_rate=diagnosis["overall_rate"],
                total_practiced=diagnosis["total_practiced"],
                kp_stats=diagnosis.get("kp_stats"),
                weak_kps=diagnosis.get("weak_kps", []),
                llm_analysis=diagnosis.get("llm_analysis"),
                recommendations=diagnosis.get("recommendations"),
            )
            session.add(report)

        # ── 提交事务 ──
        await session.commit()


async def _compact_best_effort(state: dict) -> int:
    """滚动压缩（C4b：主事务外独立事务，best-effort）。

    背景：压缩含一次 LLM 调用。若仍置于 commit 前的主事务内，LLM 异常/超时
    会把整轮核心持久化（QA 记录 / 会话状态 / 诊断报告）一起回滚 —— 用户已看到
    答案而会话永不 completed。移出后主事务先行 commit（核心数据必不丢），
    压缩独立提交，失败仅记日志返回 0；下一轮 needs_compaction 仍为真自动重试，自愈。

    返回本轮实际压缩轮数（无会话 / 无需压缩 / 失败均返回 0）。
    """
    session_id = state.get("session_id", "")
    if not session_id:
        return 0
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(AgentSession).where(AgentSession.id == UUID(session_id))
            )
            agent_session = result.scalar_one_or_none()
            if not agent_session:
                return 0
            count = await _maybe_compact_session(state, agent_session)
            if count > 0:
                await session.commit()
            return count
    except Exception:
        logger.exception(
            "滚动压缩失败（best-effort，下轮自动重试）session=%s", session_id
        )
        return 0


def _spawn_audit_tasks(state: dict, guard_issues: list[str]) -> None:
    """Step F: 异步 audit 日志（独立 session，不阻塞）。"""
    from coursepilot.governance.audit import (
        log_action,
        log_agent_chat,
        log_guardrail_violation,
    )

    _spawn_background(log_agent_chat(
        user_id=state["user_id"],
        session_id=state["session_id"],
        intent=state.get("intent", ""),
        query=state.get("query", ""),
    ))
    # P1: Agentic RAG 决策轨迹写入审计日志（agent_steps 非空时）
    agent_steps = state.get("agent_steps") or []
    if agent_steps:
        _spawn_background(log_action(
            user_id=state["user_id"],
            action="agent.rag_steps",
            resource_type="agent_session",
            resource_id=state["session_id"],
            details={
                "step_count": len(agent_steps),
                "tools_used": [s.get("tool") for s in agent_steps],
                "tool_history": state.get("tool_history") or [],
            },
        ))
    if guard_issues:
        _spawn_background(log_guardrail_violation(
            user_id=state["user_id"],
            session_id=state["session_id"],
            issues=guard_issues,
        ))


def _spawn_side_effects(state: dict) -> None:
    """Step G/H/I: profile 更新、L3 语义记忆抽取、QA embedding 补全。

    兜底轮次（none/未知/降级收口）整体跳过：元对话对画像与 L3 长期记忆
    是噪音（会污染后续个性化与记忆召回）；QA Record 未落库
    （见 _persist_core），embedding 补全也无事可做。
    """
    if state.get("fallback_reason"):
        return

    # ── Step G: Profile 更新 ──
    _spawn_background(update_profile(
        user_id=state["user_id"],
        course_id=state["course_id"],
    ))

    # ── Step H: L3 语义记忆抽取（P3） ──
    try:
        from coursepilot.agent.memory import extract_facts_for_session
        _spawn_background(extract_facts_for_session(
            user_id=state["user_id"],
            course_id=state["course_id"],
            session_id=state["session_id"],
        ))
    except Exception:
        logger.exception("触发 L3 抽取任务失败")

    # ── Step I: 同步触发一次 QA embedding 补全（P4） ──
    try:
        _spawn_background(ensure_qa_embeddings_for_user_course(
            user_id=state["user_id"],
            course_id=state["course_id"],
        ))
    except Exception:
        logger.exception("触发 QA embedding 补全失败")


def _observability_snapshot(
    llm_calls: list[dict],
    total_tokens: int,
    compaction_count: int,
) -> dict:
    """P5: 把可观测快照写回 state，供 admin 控制台消费。"""
    # 这里取最近一次 query_rag 的 budget 信息
    last_budget = None
    last_layer_tokens = None
    last_cache_hit = None
    for call in reversed(llm_calls):
        if call.get("node") == "query_rag":
            last_budget = call.get("context_budget")
            last_layer_tokens = call.get("layer_tokens")
            last_cache_hit = call.get("cache_hit_estimated")
            break

    return {
        "token_count": total_tokens,
        "context_budget": last_budget,
        "layer_tokens": last_layer_tokens,
        "cache_hit_estimated": last_cache_hit,
        "compaction_count": compaction_count,
        "error": None,
    }

async def _update_session_intent(
    session: AsyncSession, session_id: str, intent: str,
    quiz_data: dict | None = None,
    answer: str | None = None,
    sources: list[dict] | None = None,
    citation_map: dict | None = None,
    query: str | None = None,
) -> AgentSession | None:
    """更新 agent_session 的 intent、answer、sources、quiz_data、conversation 等字段。

    citation_map 为 {ref_id: {kp_path, page_ref, uuid}}（回答正文 <ref id="N" />
    的引用来源），随该轮 assistant 一起写入 conversation JSONB，供前端渲染
    可点击引用上标与来源面板。旧会话无此键时前端自然降级。

    返回更新后的 AgentSession 实例，供调用方继续修改（如滚动摘要）。
    """
    result = await session.execute(
        select(AgentSession).where(AgentSession.id == UUID(session_id))
    )
    agent_session = result.scalar_one_or_none()
    if not agent_session:
        return None

    agent_session.intent = intent
    if answer is not None:
        agent_session.answer = answer
    if sources is not None:
        agent_session.sources = sources
    if quiz_data and intent in ("practice", "review"):
        agent_session.quiz_data = quiz_data

    # 追加到多轮对话 L1
    conv = list(agent_session.conversation or [])
    if query:
        conv.append({"role": "user", "content": query, "intent": None})
    conv.append({
        "role": "assistant",
        "content": answer or "",
        "intent": intent,
        "sources": sources or [],
        "citations": citation_map or {},
        "query": query or "",
    })
    agent_session.conversation = conv
    agent_session.status = "completed"

    return agent_session


async def ensure_qa_embeddings_for_user_course(user_id: str, course_id: str) -> int:
    """后台任务：为指定用户/课程补全 QARecord embedding（P4）。"""
    from coursepilot.agent.memory import ensure_qa_embeddings
    from coursepilot.db import async_session_factory
    try:
        async with async_session_factory() as session:
            return await ensure_qa_embeddings(session)
    except Exception:
        logger.exception("QA embedding 补全失败 user=%s course=%s", user_id, course_id)
        return 0


async def _maybe_compact_session(state: dict, agent_session: AgentSession) -> int:
    """当 L1 记忆超过阈值时，把老轮次压缩进 L2 rolling_summary。

    返回实际压缩的轮数，供调用方记录可观测指标。
    """
    from coursepilot.agent.memory import ContextManager, compact_conversation

    cm = ContextManager()
    conversation = agent_session.conversation or []
    rolling_summary = str(agent_session.rolling_summary or "")
    if not cm.needs_compaction(conversation, rolling_summary):
        return 0

    new_summary, compacted_count = await compact_conversation(
        conversation,
        existing_summary=rolling_summary,
        max_summary_tokens=cm.rolling_summary_max,
    )
    if compacted_count == 0:
        return 0

    # 保留最近轮次作为 L1，老轮次转为 L2
    agent_session.rolling_summary = new_summary
    agent_session.conversation = conversation[compacted_count:]
    logger.info(
        "会话 %s 滚动压缩：%d 轮 -> summary，剩余 %d 轮在 L1",
        agent_session.id, compacted_count, len(agent_session.conversation)
    )
    return compacted_count

def _first_kp_path(metadata: dict) -> str | None:
    paths = metadata.get("source_kp_paths", [])
    return paths[0] if paths else None

async def get_mastery_node(state: dict) -> dict:
    """查询知识点掌握度 → state["mastery]"""
    try:
        async with async_session_factory() as session:
            mastery = await get_mastery(
                session=session,
                user_id=state["user_id"],
                course_id=state["course_id"],
            )
        return {"mastery": mastery, "error": None}
    except Exception as e:
        logger.exception("get_mastery_node 异常")
        return {"mastery": {}, "error": str(e)}

async def generate_quiz_node(state: dict) -> dict:
    """生成练习题 → state["quiz_data"]"""
    try:
        quiz_data, token_info = await generate_quiz(
            context=state.get("context", ""),
            course_context=state.get("course_context", {}),
            mastery=state.get("mastery", {}),
        )
        llm_calls = list(state.get("llm_calls", []))
        llm_calls.append({"node": "generate_quiz", **token_info})
        return {"quiz_data": quiz_data, "llm_calls": llm_calls, "error": None}
    except Exception as e:
        logger.exception("generate_quiz_node 异常")
        return {"quiz_data": dict(QUIZ_FALLBACK), "error": str(e)}

async def evaluate_quiz_node(state: dict) -> dict:
    """验证练习题质量 → state["eval_result"]，同时递增 retry_count"""
    try:
        result, token_info = await evaluate_quiz(
            quiz_data=state.get("quiz_data", {}),
            context=state.get("context", ""),
            course_context=state.get("course_context", {}),
        )
        retry_count = state.get("retry_count", 0)
        if result.get("status") == "FAIL":
            retry_count += 1
        llm_calls = list(state.get("llm_calls", []))
        llm_calls.append({"node": "evaluate_quiz", **token_info})
        return {
            "eval_result": result,
            "retry_count": retry_count,
            "llm_calls": llm_calls,
            "error": None,
        }
    except Exception as e:
        logger.exception("evaluate_quiz_node 异常")
        return {
            "eval_result": {"status": "FAIL", "score": 0.0},
            "retry_count": state.get("retry_count", 0) + 1,
            "error": str(e),
        }

async def create_plan_node(state: dict) -> dict:
    """practice 路径终点：将生成的 quiz 写入 answer，准备返回给用户"""
    quiz_data = state.get("quiz_data", {})
    # 默认值为 []：questions 缺失时与 {} 都是空，但类型应与实际结构一致
    questions = quiz_data.get("questions", [])
    answer_parts = [f"为你生成了 {len(questions)} 道练习题：\n"]
    for i, q in enumerate(questions, 1):
        opts = "\n".join(f"  {k}. {v}" for k, v in q.get("options", {}).items())
        answer_parts.append(f"{i}. {q['question_text']}\n{opts}\n")
    return {
        "answer": "\n".join(answer_parts),
        "sources": [{"kp_path": q.get("kp_path", "")} for q in questions if q.get("kp_path")],
        "error": None,
    }

async def diagnose_node(state: dict) -> dict:
    """学情诊断 → state["diagnosis"] + state["answer"]

    调用聚合统计 + LLM 分析，组合为完整诊断报告。
    如果用户查询了特定知识点（如"二重积分"），自动过滤到该子树。
    持久化由 finalize_node 统一处理。
    """
    try:
        async with async_session_factory() as session:
            diagnosis = await diagnose(
                session=session,
                user_id=state["user_id"],
                course_id=state["course_id"],
            )

            # 从用户查询中提取特定知识点
            query = state.get("query", "")
            topic_kp_path = await _find_topic_kp(
                session, state["course_id"], query,
            ) if query else ""

        # 找到了特定知识点，过滤统计到该子树
        if topic_kp_path and diagnosis["total_practiced"] > 0:
            filtered = {
                kp: stat for kp, stat in diagnosis["kp_stats"].items()
                if kp == topic_kp_path or kp.startswith(topic_kp_path + "/")
            }
            if filtered:
                diagnosis["kp_stats"] = filtered
                diagnosis["weak_kps"] = [
                    kp for kp in diagnosis["weak_kps"] if kp in filtered
                ]
                total_correct = sum(s["correct"] for s in filtered.values())
                total_count = sum(s["total"] for s in filtered.values())
                diagnosis["overall_rate"] = total_correct / total_count if total_count else 0.0
                diagnosis["total_practiced"] = total_count
            else:
                topic_name = topic_kp_path.rsplit("/", 1)[-1]
                answer = (
                    f"你还没有练习过「{topic_name}」相关的题目"
                    f"（知识点路径：{topic_kp_path}），"
                    f"暂时无法针对这个知识点进行学情诊断。"
                    f"先做一些相关练习题吧！"
                )
                diagnosis["llm_analysis"] = ""
                diagnosis["recommendations"] = ""
                return {"diagnosis": diagnosis, "answer": answer, "error": None}

        # 无练习记录时跳过 LLM
        if diagnosis["total_practiced"] == 0:
            answer = "暂无练习记录，请先做一些练习题再来进行学情诊断。"
            diagnosis["llm_analysis"] = ""
            diagnosis["recommendations"] = ""
            return {"diagnosis": diagnosis, "answer": answer, "error": None}

        # LLM 深度分析（传用户查询，让分析更有针对性）
        analysis, recommendations, token_info = await generate_llm_analysis(
            diagnosis, user_query=query, topic_kp_path=topic_kp_path,
        )
        llm_calls = list(state.get("llm_calls", []))
        llm_calls.append({"node": "diagnose", **token_info})

        # 组合回答文本
        answer_parts = [diagnosis.get("summary", "")]
        if topic_kp_path:
            answer_parts.insert(0, f"针对知识点「{topic_kp_path}」的分析：")
        if analysis:
            answer_parts.append(f"\n📊 深度分析\n{analysis}")
        if recommendations:
            answer_parts.append(f"\n📌 学习建议\n{recommendations}")
        answer = "\n\n".join(answer_parts)

        diagnosis["llm_analysis"] = analysis
        diagnosis["recommendations"] = recommendations

        return {
            "diagnosis": diagnosis,
            "answer": answer,
            "llm_calls": llm_calls,
            "error": None,
        }
    except Exception as e:
        logger.exception("diagnose_node 异常")
        return {"diagnosis": {}, "answer": "诊断失败，请稍后再试", "error": str(e)}


async def _find_topic_kp(session: AsyncSession, course_id: str, query: str) -> str:
    """从用户查询中提取匹配的知识点路径

    先用完整查询搜索 kp_path/title，未命中则去掉疑问词再试。
    返回最长匹配的 kp_path，未找到返回空字符串。
    """
    from sqlalchemy import or_

    from coursepilot.models import KnowledgePoint

    candidates = [
        query.strip(),
    ]
    # 加上去掉常见疑问词的版本
    simplified = query.replace("怎么样", "").replace("如何", "").replace(
        "什么", ""
    ).replace("怎么", "").replace("吗", "").strip()
    if simplified and simplified != query:
        candidates.append(simplified)

    for text in candidates:
        if not text:
            continue
        result = await session.execute(
            select(KnowledgePoint).where(
                KnowledgePoint.course_id == UUID(course_id),
                or_(
                    KnowledgePoint.title.ilike(f"%{text}%"),
                    KnowledgePoint.kp_path.ilike(f"%{text}%"),
                ),
            ).limit(5)
        )
        kps = result.scalars().all()
        if kps:
            kps.sort(key=lambda kp: len(kp.kp_path), reverse=True)
            return kps[0].kp_path

    return ""

async def review_plan_node(state: dict) -> dict:
    """生成复习计划 → state["review_plan"] + state["answer"]

    注意：review 路径不经 diagnose_node，所以 state["diagnosis"] 可能为空。
    此处自动调用 diagnose() 确保有数据。
    """
    try:
        async with async_session_factory() as session:
            diagnosis = state.get("diagnosis", {})
            if not diagnosis.get("kp_stats"):
                # review 路径没经过 diagnose，实时查
                from coursepilot.agent.skills.diagnose import diagnose as _diagnose
                diagnosis = await _diagnose(
                    session=session,
                    user_id=state["user_id"],
                    course_id=state["course_id"],
                )

            plan_data, token_info = await review_plan(
                session=session,
                user_id=state["user_id"],
                course_id=state["course_id"],
                diagnosis=diagnosis,
            )
        answer = plan_data.get("plan_summary", "")
        llm_calls = list(state.get("llm_calls", []))
        llm_calls.append({"node": "review_plan", **token_info})
        return {"review_plan": plan_data, "answer": answer, "llm_calls": llm_calls, "error": None}
    except Exception as e:
        logger.exception("review_plan_node 异常")
        return {"review_plan": {}, "answer": "生成复习计划失败", "error": str(e)}
