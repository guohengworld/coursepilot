"""Practice API 单元测试"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coursepilot.api.practice import QuizQuestion, _strip_quiz_answers

SESSION_UUID = "00000000-0000-0000-0000-000000000001"


class TestStripQuizAnswers:
    """_strip_quiz_answers 从 quiz_data 提取题目，不暴露答案"""

    def test_strip_removes_correct_answer(self):
        quiz_data = {
            "questions": [
                {
                    "question_text": "1+1=?",
                    "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                    "correct_answer": "B",
                    "explanation": "很明显",
                    "kp_path": "math/basic",
                }
            ]
        }
        result = _strip_quiz_answers(quiz_data)
        assert len(result) == 1
        assert isinstance(result[0], QuizQuestion)
        assert result[0].question_text == "1+1=?"
        assert result[0].options == {"A": "1", "B": "2", "C": "3", "D": "4"}
        assert result[0].kp_path == "math/basic"
        assert not hasattr(result[0], "correct_answer")
        assert not hasattr(result[0], "explanation")

    def test_empty_questions(self):
        assert _strip_quiz_answers({}) == []
        assert _strip_quiz_answers({"questions": []}) == []

    def test_missing_fields_use_defaults(self):
        quiz_data = {"questions": [{}]}
        result = _strip_quiz_answers(quiz_data)
        assert result[0].question_text == ""
        assert result[0].options == {}
        assert result[0].kp_path == ""


def _make_db_session(agent_session_mock=None):
    """Helper: 创建一个 db_session mock 使 await execute() → scalar_one_or_none() → mock 正常工作"""
    session = AsyncMock()
    # session.execute 返回的 result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = agent_session_mock
    session.execute = AsyncMock(return_value=mock_result)
    return session


class TestGetQuiz:
    """GET /api/v1/practice/{session_id}/quiz"""

    @pytest.mark.asyncio
    async def test_returns_quiz_questions(self):
        from coursepilot.api.practice import get_quiz

        agent_session = MagicMock(
            user_id="user-uuid",
            quiz_data={
                "questions": [
                    {
                        "question_text": "1+1=?",
                        "options": {"A": "1", "B": "2"},
                        "kp_path": "math",
                    }
                ]
            },
        )
        session = _make_db_session(agent_session)
        user = MagicMock(id="user-uuid")

        result = await get_quiz(SESSION_UUID, user, session)
        assert len(result.questions) == 1
        assert result.questions[0].question_text == "1+1=?"

    @pytest.mark.asyncio
    async def test_raises_404_if_no_session(self):
        from fastapi import HTTPException
        from coursepilot.api.practice import get_quiz

        session = _make_db_session(None)
        user = MagicMock(id="user-uuid")

        with pytest.raises(HTTPException) as exc:
            await get_quiz(SESSION_UUID, user, session)
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_raises_403_if_wrong_user(self):
        from fastapi import HTTPException
        from coursepilot.api.practice import get_quiz

        agent_session = MagicMock(
            user_id="other-user",
            quiz_data={"questions": [{"question_text": "test"}]},
        )
        session = _make_db_session(agent_session)
        user = MagicMock(id="my-user")

        with pytest.raises(HTTPException) as exc:
            await get_quiz(SESSION_UUID, user, session)
        assert exc.value.status_code == 403


class TestSubmitAnswers:
    """POST /api/v1/practice/{session_id}/submit"""

    @pytest.mark.asyncio
    async def test_no_session_returns_404(self):
        from fastapi import HTTPException
        from coursepilot.api.practice import submit_answers

        session = _make_db_session(None)
        user = MagicMock(id="user-uuid")

        with pytest.raises(HTTPException) as exc:
            await submit_answers(SESSION_UUID, MagicMock(answers={}), user, session)
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_wrong_user_returns_403(self):
        from fastapi import HTTPException
        from coursepilot.api.practice import submit_answers

        agent_session = MagicMock(
            user_id="other-user",
            quiz_data={"questions": [{"question_text": "test"}]},
        )
        session = _make_db_session(agent_session)
        user = MagicMock(id="my-user")

        with pytest.raises(HTTPException) as exc:
            await submit_answers(SESSION_UUID, MagicMock(answers={}), user, session)
        assert exc.value.status_code == 403
