"""LangGraph 节点函数

每个节点接收 AgentState，返回状态更新字典（只写自己负责的字段）
"""
import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.agent.context import build_context as build_context_logic
from coursepilot.agent.profile_updater import update_profile
from coursepilot.agent.skills.classify_intent import classify_intent
from coursepilot.agent.skills.diagnose import diagnose
from coursepilot.agent.skills.evaluate_quiz import evaluate_quiz
from coursepilot.agent.skills.generate_quiz import generate_quiz
from coursepilot.agent.skills.get_mastery import get_mastery
from coursepilot.agent.skills.query_rag import query_rag
from coursepilot.agent.skills.review_plan import review_plan
from coursepilot.agent.skills.update_qa_record import update_qa_record
from coursepilot.db import async_session_factory
from coursepilot.models import AgentSession

logger = logging.getLogger(__name__)

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
    """意图分类"""
    try:
        intent = await classify_intent(
            query=state["query"],
            course_context=state.get("course_context", {}),
            recent_qa=state.get("recent_qa", []),
        )
        return {"intent": intent, "error": None}
    except Exception as e:
        logger.exception("classify 节点异常")
        return {"intent": "question", "error": str(e)}

async def query_rag_node(state: dict) -> dict:
    """RAG 检索 + LLM 生成"""
    try:
        async with async_session_factory() as session:
            answer, context, metadata, sources = await query_rag(
                session=session,
                query=state["query"],
                course_id=state["course_id"],
                course_context=state.get("course_context", {}),
            )
        return {
            "answer": answer,
            "context": context,
            "retrieved_metadata": metadata,
            "sources": sources,
            "error": None,
        }
    except Exception as e:
        logger.exception("query_rag 节点异常")
        return {
            "answer": f"抱歉，检索知识库时出错了：{e}",
            "error": str(e),
        }

async def finalize_node(state: dict) -> dict:
    """持久化 + 会话更新 + 异步触发 profile_updater

    Phase 2 增强：
      - 支持各种 intent 类型的 answer 持久化
      - 末尾异步触发 profile_updater.update_profile()
    """
    try:
        async with async_session_factory() as session:
            token_count = await update_qa_record(
                session=session,
                user_id=state["user_id"],
                course_id=state["course_id"],
                query=state["query"],
                answer=state.get("answer", ""),
                kp_path=_first_kp_path(state.get("retrieved_metadata", {})),
                retrieved_units=state.get("retrieved_metadata", {}).get("top_uuids", []),
                citations=state.get("sources", []),
                session_id=state["session_id"],
            )
            # 更新 agent_session 的 intent 为实际值
            await _update_session_intent(
                session, state["session_id"], state.get("intent", "question")
            )

        # 异步触发 profile 更新（不阻塞回答）
        import asyncio
        asyncio.create_task(update_profile(
            user_id=state["user_id"],
            course_id=state["course_id"],
        ))

        return {"token_count": token_count, "error": None}
    except Exception as e:
        logger.exception("finalize 节点异常")
        return {"error": str(e)}

async def _update_session_intent(
    session: AsyncSession, session_id: str, intent: str
) -> None:
    """更新 agent_session 的 intent 字段"""
    result = await session.execute(
        select(AgentSession).where(AgentSession.id == UUID(session_id))
    )
    agent_session = result.scalar_one_or_none()
    if agent_session:
        agent_session.intent = intent
        agent_session.status = "completed"

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
        quiz_data = await generate_quiz(
            context=state.get("context", ""),
            course_context=state.get("course_context", {}),
            mastery=state.get("mastery", {}),
        )
        return {"quiz_data": quiz_data, "error": None}
    except Exception as e:
        logger.exception("generate_quiz_node 异常")
        return {"quiz_data": {"question": [], "error": str(e)}}

async def evaluate_quiz_node(state: dict) -> dict:
    """验证练习题质量 → state["eval_result"]，同时递增 retry_count"""
    try:
        result = await evaluate_quiz(
            quiz_data=state.get("quiz_data", {}),
            context=state.get("context", ""),
            course_context=state.get("course_context", {}),
        )
        retry_count = state.get("retry_count", 0)
        if result.get("status") == "FAIL":
            retry_count += 1
        return {"eval_result": result, "retry_count": retry_count, "error": None}
    except Exception as e:
        logger.exception("evaluate_quiz_node 异常")
        return {"eval_result": {"status": "FAIL", "score": 0.0},
                "retry_count": state.get("retry_state", 0) + 1,
                "error": str(e)}

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

async def diagnose_node(state:dict) -> dict:
    """学情诊断 → state["diagnosis"] + state["answer"]"""
    try:
        async with async_session_factory() as session:
            diagnosis = await diagnose(
                session=session,
                user_id=state["user_id"],
                course_id=state["course_id"]
            )
        answer = diagnosis.get("summary", "")
        return {"diagnosis": diagnosis, "answer": answer, "error": None}
    except Exception as e:
        logger.exception("diagnose_node 异常")
        return {"diagnosis": {}, "answer": "诊断失败，请稍后再试", "error": str(e)}

async def review_plan_node(state: dict) -> dict:
    """生成复习计划 → state["review_plan"] + state["answer"]"""
    try:
        async with async_session_factory() as session:
            plan = await review_plan(
                session=session,
                user_id=state["user_id"],
                course_id=state["course_id"],
                diagnosis=state.get("diagnosis", {})
            )
        answer = plan.get("plan_summary", "")
        return {"review_plan": plan, "answer": answer, "error": None}
    except Exception as e:
        logger.exception("review_plan_node 异常")
        return {"review_plan": {}, "answer": "生成复习计划失败", "error": str(e)}
