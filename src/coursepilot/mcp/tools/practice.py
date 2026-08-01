"""MCP Practice 工具实现。

包含：generate_practice, grade_answers（均为 P1 核心）
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import UUID

from mcp.types import CallToolResult, TextContent
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.agent.skills.generate_quiz import generate_quiz
from coursepilot.agent.skills.grade_answers import grade_answers as grade_answers_skill
from coursepilot.db import async_session_factory
from coursepilot.models import KnowledgePoint, PracticeRecord, Question
from coursepilot.rag.generator import build_course_context
from coursepilot.rag.retriever import Retriever

from coursepilot.mcp.shared.schemas import GeneratePracticeParams, GradeAnswersParams

_LOGGER = logging.getLogger(__name__)


async def _get_session() -> AsyncSession:
    """获取一个异步 DB 会话。"""
    return async_session_factory()


async def _find_kp_by_path(
    session: AsyncSession,
    course_id: str,
    kp_path: str,
) -> KnowledgePoint | None:
    """根据 kp_path 查找知识点，找不到则返回课程下任意一个 KP。"""
    if kp_path:
        result = await session.execute(
            select(KnowledgePoint)
            .where(
                KnowledgePoint.course_id == course_id,
                KnowledgePoint.kp_path == kp_path,
            )
            .limit(1)
        )
        kp = result.scalar_one_or_none()
        if kp:
            return kp

    fallback = await session.execute(
        select(KnowledgePoint)
        .where(KnowledgePoint.course_id == course_id)
        .limit(1)
    )
    return fallback.scalar_one_or_none()


async def generate_practice(params: GeneratePracticeParams) -> CallToolResult:
    """基于知识点生成练习题。

    [1-用途] 根据课程内容和知识点生成选择题，供学生练习。
    [2-限制] 一次最多生成 3 道题（受底层 skill 限制）；course_id 必须存在。
    [3-成本] 中，需要调用一次 LLM。
    [4-副作用] 会写入 Question 表。
    [5-输入格式] course_id: 课程 UUID；kp_path: 目标知识点路径；count: 题目数量（最大 10，当前底层固定 3）。
    [6-输出格式] 返回题目列表（不含答案），每题包含 question_id、题干、选项、类型。
    """
    session = await _get_session()
    try:
        course_id = str(params.course_id)
        course_context = await build_course_context(session, course_id)
        if not course_context:
            return _make_error(f"课程 {course_id} 不存在")

        kp = await _find_kp_by_path(session, course_id, params.kp_path)
        if not kp:
            return _make_error("课程下没有知识点，无法生成练习")

        query = params.kp_path or course_context.get("name", "全部内容")
        retriever = Retriever()
        context, _ = await retriever.retrieve(session, query, course_id)

        quiz_data, token_info = await generate_quiz(context, course_context, {})
        questions_data = quiz_data.get("questions", [])
        if not questions_data:
            return _make_error("题目生成失败，LLM 返回空结果")

        created_questions = []
        for q_data in questions_data:
            question = Question(
                kp_id=kp.id,
                question_text=q_data.get("question_text", ""),
                question_type=q_data.get("question_type", "choice_4"),
                options=q_data.get("options", {}),
                correct_answer=q_data.get("correct_answer", ""),
                explanation=q_data.get("explanation", ""),
                difficulty=params.difficulty,
                source="mcp",
                verified=True,
            )
            session.add(question)
            await session.flush()

            created_questions.append({
                "question_id": str(question.id),
                "question_text": question.question_text,
                "question_type": question.question_type,
                "options": question.options,
            })

        await session.commit()

        result = {
            "questions": created_questions,
            "count": len(created_questions),
            "tokens": token_info,
        }
        return _make_success(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        _LOGGER.exception("generate_practice failed: %s", e)
        await session.rollback()
        return _make_error(f"出题失败: {e}")
    finally:
        await session.close()


async def grade_answers(params: GradeAnswersParams) -> CallToolResult:
    """提交作答并批改。

    [1-用途] 根据 question_id 对应的题目，判断学生答案是否正确。
    [2-限制] 必须传入有效 question_id 和单字符答案（如 A/B/C/D）。
    [3-成本] 低，纯逻辑比对。
    [4-副作用] 会写入 PracticeRecord 表。
    [5-输入格式] question_id: 题目 UUID；answer: 学生答案。
    [6-输出格式] 返回批改结果、正确答案、解析和涉及知识点。
    """
    session = await _get_session()
    try:
        question_id = str(params.question_id)
        result = await session.execute(
            select(Question).where(Question.id == question_id)
        )
        question = result.scalar_one_or_none()
        if not question:
            return _make_error(f"题目 {question_id} 不存在")

        correct = params.answer == question.correct_answer
        record = PracticeRecord(
            user_id=params.user_id,
            question_id=question.id,
            user_answer=params.answer,
            correct_flag=correct,
            answered_at=datetime.now(timezone.utc),
        )
        session.add(record)
        await session.commit()

        result_data = {
            "question_id": question_id,
            "correct": correct,
            "student_answer": params.answer,
            "correct_answer": question.correct_answer,
            "explanation": question.explanation,
            "kp_path": "",
        }
        # 尝试获取 kp_path
        try:
            kp_result = await session.execute(
                select(KnowledgePoint.kp_path).where(KnowledgePoint.id == question.kp_id)
            )
            kp_path = kp_result.scalar_one_or_none()
            result_data["kp_path"] = kp_path or ""
        except Exception:
            pass

        return _make_success(json.dumps(result_data, ensure_ascii=False))
    except Exception as e:
        _LOGGER.exception("grade_answers failed: %s", e)
        await session.rollback()
        return _make_error(f"批改失败: {e}")
    finally:
        await session.close()


def _make_success(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=False)


def _make_error(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)
