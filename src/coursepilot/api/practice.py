"""Practice API：练习提交、批改、记录

依赖 AgentSession.quiz_data（由 finalize_node 保存），
在 practice/review 流程完成后，学生可通过此 API 提交答案。
"""
import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.api.deps import get_current_user
from coursepilot.db import get_session
from coursepilot.models import AgentSession, KnowledgePoint, PracticeRecord, Question, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/practice", tags=["practice"])


# ── Request / Response Models ──────────────────────────────────

class QuizQuestion(BaseModel):
    index: int
    question_text: str
    options: dict[str, str]
    kp_path: str

class QuizResponse(BaseModel):
    session_id: str
    questions: list[QuizQuestion]

class SubmitRequest(BaseModel):
    answers: dict[str, str] = Field(..., description='答案映射，如 {"0": "A", "1": "C"}')

class QuestionResult(BaseModel):
    index: int
    question_text: str
    correct: bool
    student_answer: str
    correct_answer: str
    explanation: str
    kp_path: str

class SubmitResponse(BaseModel):
    session_id: str
    total: int
    correct: int
    score: float
    results: list[QuestionResult]


# ── Helper ────────────────────────────────────────────────────

def _strip_quiz_answers(quiz_data: dict) -> list[QuizQuestion]:
    """从 quiz_data 中提取题目列表，移除正确答案（不暴露给学生）"""
    questions = []
    for i, q in enumerate(quiz_data.get("questions", [])):
        questions.append(QuizQuestion(
            index=i,
            question_text=q.get("question_text", ""),
            options=q.get("options", {}),
            kp_path=q.get("kp_path", ""),
        ))
    return questions


# ── Endpoints ─────────────────────────────────────────────────

@router.get("/{session_id}/quiz")
async def get_quiz(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db_session: AsyncSession = Depends(get_session),
) -> QuizResponse:
    """获取练习题（不含答案），供前端渲染选择题"""
    result = await db_session.execute(
        select(AgentSession).where(AgentSession.id == UUID(session_id))
    )
    agent_session = result.scalar_one_or_none()
    if not agent_session:
        raise HTTPException(status_code=404, detail="会话不存在")
    if agent_session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权访问此会话")

    quiz_data = agent_session.quiz_data
    if not quiz_data or not quiz_data.get("questions"):
        raise HTTPException(status_code=404, detail="该会话没有练习题数据")

    return QuizResponse(
        session_id=session_id,
        questions=_strip_quiz_answers(quiz_data),
    )


@router.post("/{session_id}/submit")
async def submit_answers(
    session_id: str,
    request: SubmitRequest,
    current_user: User = Depends(get_current_user),
    db_session: AsyncSession = Depends(get_session),
) -> SubmitResponse:
    """提交练习答案：批改 → 写 Question + PracticeRecord → 触发画像更新"""
    # 1. 加载会话
    result = await db_session.execute(
        select(AgentSession).where(AgentSession.id == UUID(session_id))
    )
    agent_session = result.scalar_one_or_none()
    if not agent_session:
        raise HTTPException(status_code=404, detail="会话不存在")
    if agent_session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权操作此会话")

    quiz_data = agent_session.quiz_data
    if not quiz_data or not quiz_data.get("questions"):
        raise HTTPException(status_code=400, detail="该会话没有练习题数据")

    questions_data = quiz_data["questions"]
    course_id = agent_session.course_id

    # 2. 批改
    from coursepilot.agent.skills.grade_answers import grade_answers
    grade_result = await grade_answers(quiz_data, request.answers)

    # 3. 持久化 Question + PracticeRecord
    for i, q_data in enumerate(questions_data):
        # 查找知识点
        kp_path = q_data.get("kp_path", "")
        kp = None
        if kp_path:
            kp_result = await db_session.execute(
                select(KnowledgePoint)
                .where(
                    KnowledgePoint.kp_path == kp_path,
                    KnowledgePoint.course_id == course_id,
                )
                .limit(1)
            )
            kp = kp_result.scalar_one_or_none()

        # 无匹配 KP 时回退到课程下任意一个 KP
        if not kp:
            fallback = await db_session.execute(
                select(KnowledgePoint)
                .where(KnowledgePoint.course_id == course_id)
                .limit(1)
            )
            kp = fallback.scalar_one_or_none()
            if not kp:
                raise HTTPException(status_code=500, detail="课程下没有知识点，无法创建练习记录")
            if kp_path:
                logger.warning("KP path=%s 未找到，回退到 kp=%s", kp_path, kp.id)

        # 创建 Question
        question = Question(
            kp_id=kp.id,
            question_text=q_data.get("question_text", ""),
            question_type=q_data.get("question_type", "choice_4"),
            options=q_data.get("options", {}),
            correct_answer=q_data.get("correct_answer", ""),
            explanation=q_data.get("explanation", ""),
            source="agent",
            verified=True,
        )
        db_session.add(question)
        await db_session.flush()

        # 获取批改结果
        r = grade_result.get("results", [])
        item = r[i] if i < len(r) else {"correct": False, "student_answer": request.answers.get(str(i), "")}

        # 创建 PracticeRecord
        record = PracticeRecord(
            user_id=current_user.id,
            question_id=question.id,
            user_answer=item.get("student_answer", request.answers.get(str(i), "")),
            correct_flag=item.get("correct", False),
            answered_at=datetime.now(timezone.utc),
        )
        db_session.add(record)

    await db_session.flush()

    # 4. 异步触发画像更新
    from coursepilot.agent.profile_updater import update_profile
    asyncio.create_task(update_profile(
        user_id=str(current_user.id),
        course_id=str(course_id),
    ))

    # 5. 组装返回
    results = []
    for item in grade_result.get("results", []):
        idx = item.get("index", 0)
        qd = questions_data[idx] if idx < len(questions_data) else {}
        results.append(QuestionResult(
            index=idx,
            question_text=item.get("question", "")[:80],
            correct=item.get("correct", False),
            student_answer=item.get("student_answer", ""),
            correct_answer=item.get("correct_answer", qd.get("correct_answer", "")),
            explanation=qd.get("explanation", ""),
            kp_path=item.get("kp_path", qd.get("kp_path", "")),
        ))

    return SubmitResponse(
        session_id=session_id,
        total=grade_result.get("total", 0),
        correct=grade_result.get("correct", 0),
        score=grade_result.get("score", 0.0),
        results=results,
    )
