"""LangGraph 节点函数

每个节点接收 AgentState，返回状态更新字典（只写自己负责的字段）
"""
import logging
from uuid import UUID

from langgraph.types import interrupt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.agent.context import build_context as build_context_logic
from coursepilot.agent.profile_updater import update_profile
from coursepilot.agent.skills.check_sufficiency import check_sufficiency
from coursepilot.agent.skills.classify_intent import classify_intent
from coursepilot.agent.skills.decompose_query import decompose_query
from coursepilot.agent.skills.diagnose import diagnose, generate_llm_analysis
from coursepilot.agent.skills.evaluate_quiz import evaluate_quiz
from coursepilot.agent.skills.generate_quiz import generate_quiz
from coursepilot.agent.skills.get_mastery import get_mastery
from coursepilot.agent.skills.query_rag import query_rag
from coursepilot.agent.skills.review_plan import review_plan
from coursepilot.agent.skills.update_qa_record import update_qa_record
from coursepilot.agent.skills.web_search import format_web_context, web_search
from coursepilot.db import async_session_factory
from coursepilot.models import AgentSession, User
from coursepilot.rag.config import config as rag_config

logger = logging.getLogger(__name__)

HUMAN_REVIEW_INTENTS = {"practice", "review"}

async def build_context_node(state: dict) -> dict:
    """构建上下文：课程信息 + 学生画像 + 最近问答"""
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
        return {"intent": "question", "complexity": "simple", "error": str(e)}

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


async def decompose_query_node(state: dict) -> dict:
    """复杂查询分解：将多跳问题拆为子问题，供并行检索。"""
    try:
        result = await decompose_query(
            query=state["query"],
            course_context=state.get("course_context", {}),
        )
        sub_queries = result.get("sub_queries", [])
        dtype = result.get("decomposition_type", "single")

        if sub_queries:
            logger.info("查询分解 (%s): %d 个子问题", dtype, len(sub_queries))
            for sq in sub_queries:
                tc = sq.get("target_concept", "")
                logger.info("  子问题[%d]: %s (→ %s)", sq["id"], sq["query"], tc)

        return {
            "sub_queries": sub_queries,
            "error": None,
        }
    except Exception as e:
        logger.exception("decompose_query_node 异常")
        return {"sub_queries": [], "error": str(e)}


async def web_search_node(state: dict) -> dict:
    """网络搜索节点：教材检索不足时，最后一次尝试改用网络搜索（P3）。

    1. 从 sufficiency 质检结果中取 missing_info + 用户原问题 拼搜索词
    2. 搜索 DuckDuckGo，格式化结果追加到现有 context 前
    3. 结果继续走 check_sufficiency 质检
    """
    try:
        query = state["query"]
        sufficiency = state.get("sufficiency", {})
        missing_info = sufficiency.get("missing_info", "")

        # 拼搜索词
        search_query = query
        if missing_info:
            search_query = f"{missing_info} {query}"

        results = await web_search(search_query, top_k=5)
        if not results:
            logger.info("web_search 无结果，跳过")
            return {"error": None}  # 无搜索结果，但不上报错误

        web_context = format_web_context(results, query)

        # 追加到现有 context 之前
        existing_context = state.get("context", "")
        if existing_context:
            new_context = web_context + "\n\n" + existing_context
        else:
            new_context = web_context

        logger.info("web_search: 结果追加到 context (+%d chars)", len(web_context))

        return {
            "context": new_context,
            "error": None,
        }
    except Exception as e:
        logger.exception("web_search_node 异常")
        return {"error": str(e)}


async def retrieve_node(state: dict) -> dict:
    """RAG 检索（支持并行子问题检索 P2）。

    如果 state 中存在 sub_queries，并行检索每个子问题后合并结果；
    否则退化为单查询检索。
    """
    try:
        from coursepilot.rag.retriever import Retriever

        sub_queries = state.get("sub_queries", [])
        retriever = Retriever()
        async with async_session_factory() as session:
            if sub_queries:
                # P2: 并行检索多个子问题
                import asyncio

                async def _retrieve_one(sq: dict) -> tuple[str, dict]:
                    ctx, meta = await retriever.retrieve(
                        session, sq["query"], state["course_id"],
                    )
                    return ctx, meta

                tasks = [_retrieve_one(sq) for sq in sub_queries]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                contexts = []
                merged_metadata = {}
                all_kp_paths = []
                all_top_uuids = []
                for i, r in enumerate(results):
                    if isinstance(r, Exception):
                        logger.warning("子问题[%d] 检索失败: %s", i + 1, r)
                        continue
                    ctx, meta = r
                    contexts.append(ctx)
                    all_kp_paths.extend(meta.get("source_kp_paths", []))
                    all_top_uuids.extend(meta.get("top_uuids", []))

                context = "\n---\n".join(contexts) if contexts else ""
                merged_metadata = {
                    "source_kp_paths": all_kp_paths,
                    "top_uuids": all_top_uuids,
                    "parallel_count": len(sub_queries),
                }
            else:
                # 单查询检索（含 P1 补搜重试）
                context, merged_metadata = await retriever.retrieve(
                    session, state["query"], state["course_id"],
                )

        source_kp_paths = merged_metadata.get("source_kp_paths", [])
        sources = [{"kp_path": p} for p in source_kp_paths]

        retry_count = state.get("retrieval_retry_count", 0)
        logger.info("检索完成 (第%d轮): context=%d chars, sources=%d",
                    retry_count + 1, len(context), len(sources))

        return {
            "context": context,
            "retrieved_metadata": merged_metadata,
            "sources": sources,
            "error": None,
        }
    except Exception as e:
        logger.exception("retrieve_node 异常")
        return {"context": "", "retrieved_metadata": {}, "sources": [], "error": str(e)}


async def check_sufficiency_node(state: dict) -> dict:
    """质检：判断检索到的教材内容是否足够回答用户问题。

    不足时递增 retrieval_retry_count 触发补搜循环。
    """
    try:
        query = state["query"]
        context = state.get("context", "")
        metadata = state.get("retrieved_metadata", {})
        kp_paths = metadata.get("source_kp_paths", [])
        retry_count = state.get("retrieval_retry_count", 0)
        max_rounds = rag_config.complex_max_rounds

        result = await check_sufficiency(
            query=query, context=context, kp_paths=kp_paths,
        )

        sufficient = result.get("sufficient", True)
        confidence = result.get("confidence", 0.0)

        if not sufficient and retry_count < max_rounds:
            # 需要补搜
            new_retry = retry_count + 1
            degraded = False
            logger.info("质检不足 (第%d轮/%d轮): confidence=%.2f, missing=%s",
                        new_retry, max_rounds, confidence, result.get("missing_info", ""))
        else:
            new_retry = retry_count  # 充足或已达上限，不再递增
            degraded = not sufficient  # 达上限仍不足 → 降级
            if degraded:
                logger.warning("质检不足但已达最大轮数 %d，开启降级生成", max_rounds)

        return {
            "sufficiency": result,
            "retrieval_retry_count": new_retry,
            "degraded_mode": degraded,
            "error": None,
        }
    except Exception as e:
        logger.exception("check_sufficiency_node 异常")
        return {
            "sufficiency": {"sufficient": True, "confidence": 0.5, "missing_info": ""},
            "retrieval_retry_count": state.get("retrieval_retry_count", 0),
            "degraded_mode": False,
            "error": str(e),
        }


async def synthesize_node(state: dict) -> dict:
    """根据已有检索结果生成最终答案。

    如果 degraded_mode=True，在生成时追加免责声明
    （教材内容不足以完整回答，答案仅供参考）。
    """
    try:
        from coursepilot.rag.generator import Generator

        generator = Generator()
        degraded = state.get("degraded_mode", False)

        # ── 降级模式：在 context 前面加一段说明 ──
        context = state.get("context", "")
        if degraded and context:
            disclaimer = (
                "[注意] 以下教材内容可能不足以完整回答该问题。"
                "请结合教材原文和课堂笔记使用，以下回答仅供参考。\n"
            )
            context = disclaimer + context

        answer, token_info = await generator.generate(
            query=state["query"],
            context=context,
            course_context=state.get("course_context", {}),
            conversation=state.get("conversation"),
            rolling_summary=state.get("rolling_summary", ""),
            user_profile=state.get("user_profile"),
        )

        token_info["routing_complexity"] = "complex"
        token_info["degraded_mode"] = degraded
        llm_calls = list(state.get("llm_calls", []))
        llm_calls.append({"node": "synthesize", **token_info})

        return {
            "answer": answer,
            "llm_calls": llm_calls,
            "error": None,
        }
    except Exception as e:
        logger.exception("synthesize_node 异常")
        return {
            "answer": f"抱歉，生成答案时出错了：{e}",
            "error": str(e),
        }

async def finalize_node(state: dict) -> dict:
    """持久化 + 会话更新 + 滚动摘要 + 异步触发 profile_updater

    Phase 3 增强：
      - 汇总 llm_calls 写入真实 token 计数和成本估算
      - 维护 conversation（L1）与 rolling_summary（L2）
      - 末尾异步触发 profile_updater.update_profile()
    """
    try:
        # 汇总所有 LLM 调用的 token 用量
        llm_calls = state.get("llm_calls", [])
        total_tokens = sum(c.get("total_tokens", 0) for c in llm_calls)
        total_prompt = sum(c.get("prompt_tokens", 0) for c in llm_calls)
        total_completion = sum(c.get("completion_tokens", 0) for c in llm_calls)

        answer = state.get("answer", "")

        # Step A: Guardrails 检查
        from coursepilot.governance.guardrails import guard_answer
        guard_issues = guard_answer(
            answer=answer,
            context=state.get("context", ""),
            sources=state.get("sources", []),
        )
        if guard_issues:
            logger.warning(f"Guardrail 警告: {guard_issues}")

        async with async_session_factory() as session:
            # ── Step B: Audit 日志（独立 session 但复用同一个也可） ──
            from coursepilot.governance.audit import log_agent_chat, log_guardrail_violation

            # ── Step C: 写入 QA Record（仅在 session_id 有效时） ──
            session_id = state.get("session_id", "")
            if session_id:
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
            agent_session = None
            if session_id:
                agent_session = await _update_session_intent(
                    session, session_id,
                    state.get("intent", "question"),
                    human_review_result=state.get("human_review_result"),
                    quiz_data=state.get("quiz_data"),
                    answer=state.get("answer", ""),
                    sources=state.get("sources", []),
                    query=state.get("query", ""),
                )

            # 滚动压缩：当 L1 过长时，把老轮次压缩进 rolling_summary
        compaction_count = state.get("compaction_count", 0)
        if agent_session:
            compaction_count += await _maybe_compact_session(state, agent_session)

        # 记录压缩次数到 llm_calls 便于可观测（P5）
        if compaction_count > 0:
            llm_calls.append({
                "node": "compaction",
                "compacted_turns": compaction_count,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            })

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

        # ── Step F: 异步 audit 日志（独立 session，不阻塞） ──
        import asyncio
        asyncio.create_task(log_agent_chat(
            user_id=state["user_id"],
            session_id=state["session_id"],
            intent=state.get("intent", ""),
            query=state.get("query", ""),
        ))
        if guard_issues:
            asyncio.create_task(log_guardrail_violation(
                user_id=state["user_id"],
                session_id=state["session_id"],
                issues=guard_issues,
            ))

        # ── Step G: Profile 更新（已有） ──
        asyncio.create_task(update_profile(
            user_id=state["user_id"],
            course_id=state["course_id"],
        ))

        # ── Step H: L3 语义记忆抽取（P3） ──
        try:
            from coursepilot.agent.memory import extract_facts_for_session
            asyncio.create_task(extract_facts_for_session(
                user_id=state["user_id"],
                course_id=state["course_id"],
                session_id=state["session_id"],
            ))
        except Exception:
            logger.exception("触发 L3 抽取任务失败")

        # ── Step I: 同步触发一次 QA embedding 补全（P4） ──
        try:
            asyncio.create_task(ensure_qa_embeddings_for_user_course(
                user_id=state["user_id"],
                course_id=state["course_id"],
            ))
        except Exception:
            logger.exception("触发 QA embedding 补全失败")

        # P5: 把可观测快照写回 state，供 admin 控制台消费
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
    except Exception as e:
        logger.exception("finalize 节点异常")
        return {"error": str(e)}

async def _update_session_intent(
    session: AsyncSession, session_id: str, intent: str,
    human_review_result: str | None = None,
    quiz_data: dict | None = None,
    answer: str | None = None,
    sources: list[dict] | None = None,
    query: str | None = None,
) -> AgentSession | None:
    """更新 agent_session 的 intent、answer、sources、quiz_data、conversation 等字段。

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
        "query": query or "",
    })
    agent_session.conversation = conv
    if human_review_result == "rejected":
        agent_session.status = "rejected"
    else:
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
        return {"quiz_data": {"question": [], "error": str(e)}}

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
            "retry_count": state.get("retry_state", 0) + 1,
            "error": str(e),
        }

async def create_plan_node(state: dict) -> dict:
    """practice 路径终点：将生成的 quiz 写入 answer，准备返回给用户"""
    quiz_data = state.get("quiz_data", {})
    questions = quiz_data.get("questions", {})
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

async def human_review_node(state: dict) -> dict:
    """人类审批节点：在高风险操作前暂停等待确认

    教师和超级管理员自动跳过审批（auto-approve），
    学生则需要等待教师人工确认。

    使用 interrupt() 暂停图执行 → 保存 checkpoint
    恢复时传入 resume={"approved": True/False, "feedback": "..."}

    interrupt() 的参数是发送给调用者的消息（展示给前端）
    """
    intent = state.get("intent", "")
    user_id = state.get("user_id", "")

    # 查询用户角色，教师和超级管理员自动通过审批
    auto_approve = False
    if user_id:
        try:
            async with async_session_factory() as session:
                result = await session.execute(
                    select(User).where(User.id == UUID(user_id))
                )
                user = result.scalar_one_or_none()
                if user and user.role in ("teacher", "super"):
                    auto_approve = True
        except Exception:
            logger.warning("查询用户角色失败 user_id=%s", user_id)

    if intent in HUMAN_REVIEW_INTENTS and not auto_approve:
        # interrupt() 暂停执行，等待 resume 值
        # 返回值是调用 Command(resume=...) 时传来的数据
        approval = interrupt({
            "type": "human_review",
            "intent": intent,
            "query": state.get("query", "")[:200],
            "message": f"需要确认是否执行 {intent} 操作"
        })
    else:
        approval = {"approved": True}

    if not isinstance(approval, dict) or not approval.get("approved", False):
        # 人类拒绝 → 跳过后续节点，直接到 finalize
        return {
            "answer": f"{intent} 操作已被管理员暂停，请联系教师确认",
            "human_review_result": "rejected",
            "error": None,
        }

    return {"human_review_result": "approved", "error": None}
