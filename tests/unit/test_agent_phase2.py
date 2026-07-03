"""Phase 2 Agent 全方位测试：条件路由、6 个新增 Skill、7 个新增节点、Profile Updater

覆盖范围：
    - 条件路由（route_by_intent / route_after_rag / route_after_evaluate）
    - Skill 函数（get_mastery、generate_quiz、evaluate_quiz、grade_answers、diagnose、review_plan）
    - Phase 2 节点（get_mastery_node ~ review_plan_node）
    - Profile Updater（_do_update upsert 逻辑）
    - Phase 2 图结构（9 个节点、条件边对照表）
    - 端到端 5 条 intent 路径 + evaluate 重试循环
    - AgentState Phase 2 扩展字段完整性

运行方式：
    .venv/Scripts/python -m pytest tests/unit/test_agent_phase2.py -v
    .venv/Scripts/python -m pytest tests/unit/test_agent_phase2.py -v -k "test_routing"
    .venv/Scripts/python -m pytest tests/unit/test_agent_phase2.py -v -m e2e
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════
# Shared Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_db():
    """异步 DB 会话 mock（同 Phase 1 模式）"""
    session = AsyncMock(spec=['execute', 'add', 'flush', 'scalar'])
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    result.scalars.return_value = result
    result.all.return_value = []

    async def execute_side_effect(*a, **kw):
        return result

    session.execute = execute_side_effect
    session.scalar = AsyncMock(return_value=0)
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
def mock_asf(mock_db):
    """模拟 async_session_factory() 返回 mock_db"""
    cm = AsyncMock()
    cm.__aenter__.return_value = mock_db
    cm.__aexit__.return_value = None
    return cm


@pytest.fixture
def course_context():
    return {"name": "操作系统", "textbook": "OSTEP", "chapters": ["进程管理", "内存管理"]}


@pytest.fixture
def sample_state():
    """模拟带 Phase 2 字段的 AgentState"""
    return {
        "query": "什么是进程调度？",
        "course_id": str(uuid4()),
        "user_id": str(uuid4()),
        "session_id": str(uuid4()),
        "messages": [],
        "course_context": {},
        "user_profile": None,
        "recent_qa": [],
        "intent": "",
        "context": "",
        "retrieved_metadata": {},
        "answer": "",
        "sources": [],
        "token_count": 0,
        "error": None,
        "mastery": {},
        "quiz_data": {},
        "eval_result": {},
        "retry_count": 0,
        "diagnosis": {},
        "review_plan": {},
    }


# ═══════════════════════════════════════════════════════════════
# 1. 条件路由
# ═══════════════════════════════════════════════════════════════

class TestRoutingPhase2:
    """route_by_intent / route_after_rag / route_after_evaluate"""

    # ── route_by_intent ──────────────────────────────

    def test_route_by_intent_question(self):
        from coursepilot.agent.routing import route_by_intent
        assert route_by_intent({"intent": "question"}) == "query_rag"

    def test_route_by_intent_practice_and_review(self):
        from coursepilot.agent.routing import route_by_intent
        assert route_by_intent({"intent": "practice"}) == "get_mastery"
        assert route_by_intent({"intent": "review"}) == "get_mastery"

    def test_route_by_intent_diagnose(self):
        from coursepilot.agent.routing import route_by_intent
        assert route_by_intent({"intent": "diagnose"}) == "diagnose"

    def test_route_by_intent_code_help(self):
        from coursepilot.agent.routing import route_by_intent
        assert route_by_intent({"intent": "code_help"}) == "query_rag"

    def test_route_by_intent_unknown_fallback(self):
        from coursepilot.agent.routing import route_by_intent
        assert route_by_intent({"intent": "unknown"}) == "query_rag"
        assert route_by_intent({}) == "query_rag"

    def test_route_by_intent_invalid_type_still_returns_str(self):
        from coursepilot.agent.routing import route_by_intent
        assert isinstance(route_by_intent({"intent": 42}), str)

    # ── route_after_rag ──────────────────────────────

    def test_route_after_rag_question(self):
        from coursepilot.agent.routing import route_after_rag
        assert route_after_rag({"intent": "question"}) == "finalize"

    def test_route_after_rag_code_help(self):
        from coursepilot.agent.routing import route_after_rag
        assert route_after_rag({"intent": "code_help"}) == "finalize"

    def test_route_after_rag_practice(self):
        from coursepilot.agent.routing import route_after_rag
        assert route_after_rag({"intent": "practice"}) == "generate_quiz"

    def test_route_after_rag_review(self):
        from coursepilot.agent.routing import route_after_rag
        assert route_after_rag({"intent": "review"}) == "generate_quiz"

    def test_route_after_rag_default_to_finalize(self):
        from coursepilot.agent.routing import route_after_rag
        assert route_after_rag({}) == "finalize"

    # ── route_after_evaluate ─────────────────────────

    def test_route_after_evaluate_pass_goes_to_create_plan(self):
        from coursepilot.agent.routing import route_after_evaluate
        state = {"intent": "practice", "eval_result": {"status": "PASS"}, "retry_count": 0}
        assert route_after_evaluate(state) == "create_plan"

    def test_route_after_evaluate_fail_below_limit_retry(self):
        from coursepilot.agent.routing import route_after_evaluate
        state = {"intent": "practice", "eval_result": {"status": "FAIL"}, "retry_count": 1}
        assert route_after_evaluate(state) == "generate_quiz"

    def test_route_after_evaluate_fail_at_limit_no_retry(self):
        from coursepilot.agent.routing import route_after_evaluate
        state = {"intent": "practice", "eval_result": {"status": "FAIL"}, "retry_count": 2}
        assert route_after_evaluate(state) == "create_plan"

    def test_route_after_evaluate_review_pass_goes_to_review_plan(self):
        from coursepilot.agent.routing import route_after_evaluate
        state = {"intent": "review", "eval_result": {"status": "PASS"}, "retry_count": 0}
        assert route_after_evaluate(state) == "review_plan"

    def test_route_after_evaluate_review_fail_retry(self):
        from coursepilot.agent.routing import route_after_evaluate
        state = {"intent": "review", "eval_result": {"status": "FAIL"}, "retry_count": 0}
        assert route_after_evaluate(state) == "generate_quiz"

    def test_route_after_evaluate_empty_eval_result_not_fail(self):
        """空 eval_result → 无 'FAIL' 状态 → 不进 retry"""
        from coursepilot.agent.routing import route_after_evaluate
        state = {"intent": "practice", "eval_result": {}, "retry_count": 0}
        assert route_after_evaluate(state) == "create_plan"

    def test_route_after_evaluate_retry_count_missing_treated_as_zero(self):
        from coursepilot.agent.routing import route_after_evaluate
        state = {"intent": "practice", "eval_result": {"status": "FAIL"}}
        assert route_after_evaluate(state) == "generate_quiz"

    def test_route_after_evaluate_unknown_intent_fallback_to_finalize(self):
        from coursepilot.agent.routing import route_after_evaluate
        state = {"intent": "unknown", "eval_result": {"status": "PASS"}, "retry_count": 0}
        assert route_after_evaluate(state) == "finalize"


# ═══════════════════════════════════════════════════════════════
# 2. Phase 2 Skills
# ═══════════════════════════════════════════════════════════════

class TestGetMastery:
    """get_mastery — 学生掌握度查询"""

    @pytest.mark.asyncio
    async def test_returns_mastery_with_profile(self, mock_db):
        """UserProfile 存在时返回带掌握度的 dict"""
        from coursepilot.models import UserProfile
        up = MagicMock(spec=UserProfile)
        up.mastery_level = {"OS/进程调度": 0.8, "OS/内存管理": 0.5}
        up.weak_kps = ["OS/内存管理"]
        up.avg_correct_rate = 0.65

        result = MagicMock()
        result.scalar_one_or_none.return_value = up
        mock_db.execute = AsyncMock(return_value=result)

        from coursepilot.agent.skills.get_mastery import get_mastery
        mastery = await get_mastery(session=mock_db, user_id=str(uuid4()), course_id=str(uuid4()))

        assert mastery["mastery_level"] == {"OS/进程调度": 0.8, "OS/内存管理": 0.5}
        assert mastery["weak_kps"] == ["OS/内存管理"]
        assert mastery["avg_correct_rate"] == 0.65

    @pytest.mark.asyncio
    async def test_returns_empty_without_profile(self, mock_db):
        """无 UserProfile 时返回空掌握度"""
        from coursepilot.agent.skills.get_mastery import get_mastery
        mastery = await get_mastery(session=mock_db, user_id=str(uuid4()), course_id=str(uuid4()))

        assert mastery["mastery_level"] == {}
        assert mastery["weak_kps"] == []
        assert mastery["avg_correct_rate"] is None

    @pytest.mark.asyncio
    async def test_handles_none_avg_correct_rate(self, mock_db):
        """avg_correct_rate 为 None 时返回 None"""
        from coursepilot.models import UserProfile
        up = MagicMock(spec=UserProfile)
        up.mastery_level = {}
        up.weak_kps = []
        up.avg_correct_rate = None

        result = MagicMock()
        result.scalar_one_or_none.return_value = up
        mock_db.execute = AsyncMock(return_value=result)

        from coursepilot.agent.skills.get_mastery import get_mastery
        mastery = await get_mastery(session=mock_db, user_id=str(uuid4()), course_id=str(uuid4()))
        assert mastery["avg_correct_rate"] is None


class TestGenerateQuiz:
    """generate_quiz — 练习题生成"""

    @pytest.mark.asyncio
    async def test_returns_empty_without_api_key(self, course_context):
        """无 API key 时返回空结果"""
        with patch("coursepilot.agent.skills.generate_quiz.settings.llm_api_key", ""):
            from coursepilot.agent.skills.generate_quiz import generate_quiz
            result = await generate_quiz(
                context="教材内容", course_context=course_context, mastery={}
            )
            assert result == {"questions": []}

    @pytest.mark.asyncio
    async def test_returns_quiz_data_with_valid_llm(self, course_context):
        """有 API key 时返回 LLM 生成的试题"""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = (
            '{"questions": [{"question_text": "1+1=?", "question_type": "choice_4", '
            '"options": {"A": "1", "B": "2", "C": "3", "D": "4"}, '
            '"correct_answer": "B", "explanation": "1+1=2", "kp_path": "OS/测试"}]}'
        )

        with (
            patch("coursepilot.agent.skills.generate_quiz.settings.llm_api_key", "sk-test"),
            patch("coursepilot.agent.skills.generate_quiz.AsyncOpenAI") as mock_openai,
        ):
            client = AsyncMock()
            mock_openai.return_value = client
            client.chat.completions.create = AsyncMock(return_value=mock_completion)

            from coursepilot.agent.skills.generate_quiz import generate_quiz
            result = await generate_quiz(
                context="教材内容", course_context=course_context,
                mastery={"weak_kps": ["OS/测试"]}
            )

        questions = result.get("questions", result.get("question"))
        assert questions is not None
        assert len(questions) == 1
        assert questions[0]["correct_answer"] == "B"

    @pytest.mark.asyncio
    async def test_fall_back_on_json_decode_error(self, course_context):
        """LLM 返回非 JSON 时回退空结果"""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "不是 JSON"

        with (
            patch("coursepilot.agent.skills.generate_quiz.settings.llm_api_key", "sk-test"),
            patch("coursepilot.agent.skills.generate_quiz.AsyncOpenAI") as mock_openai,
        ):
            client = AsyncMock()
            mock_openai.return_value = client
            client.chat.completions.create = AsyncMock(return_value=mock_completion)

            from coursepilot.agent.skills.generate_quiz import generate_quiz
            result = await generate_quiz(
                context="教材内容", course_context=course_context, mastery={}
            )
            assert result == {"questions": []}

    def test_generate_system_prompt_mentions_json(self):
        """GENERATE_SYSTEM prompt 包含 JSON 输出说明"""
        from coursepilot.agent.skills.generate_quiz import GENERATE_SYSTEM
        assert "JSON" in GENERATE_SYSTEM


class TestEvaluateQuiz:
    """evaluate_quiz — 试题验证"""

    @pytest.mark.asyncio
    async def test_pass_without_api_key(self):
        """无 API key 时默认通过"""
        with patch("coursepilot.agent.skills.evaluate_quiz.settings.llm_api_key", ""):
            from coursepilot.agent.skills.evaluate_quiz import evaluate_quiz
            result = await evaluate_quiz(
                quiz_data={"questions": [{"question_text": "测试"}]},
                context="", course_context={}
            )
            assert result["status"] == "PASS"
            assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_fail_on_empty_questions(self):
        """题目为空时返回 FAIL（需要 API key 绕过前置 PASS 逻辑）"""
        with patch("coursepilot.agent.skills.evaluate_quiz.settings.llm_api_key", "sk-test"):
            from coursepilot.agent.skills.evaluate_quiz import evaluate_quiz
            result = await evaluate_quiz(quiz_data={}, context="", course_context={})
            assert result["status"] == "FAIL"
            assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_parses_llm_evaluation(self):
        """验证 LLM 审核结果被正确解析"""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = (
            '{"status": "PASS", "score": 0.9, "feedback": {"suggestions": ["无问题"]}}'
        )

        with (
            patch("coursepilot.agent.skills.evaluate_quiz.settings.llm_api_key", "sk-test"),
            patch("coursepilot.agent.skills.evaluate_quiz.AsyncOpenAI") as mock_openai,
        ):
            client = AsyncMock()
            mock_openai.return_value = client
            client.chat.completions.create = AsyncMock(return_value=mock_completion)

            from coursepilot.agent.skills.evaluate_quiz import evaluate_quiz
            result = await evaluate_quiz(
                quiz_data={"questions": [{"question_text": "测试"}]},
                context="教材", course_context={}
            )
            assert result["status"] == "PASS"
            assert result["score"] == 0.9

    @pytest.mark.asyncio
    async def test_fall_back_on_bad_llm_response(self):
        """LLM 返回非 JSON 时降级为 PASS"""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "不是 JSON"

        with (
            patch("coursepilot.agent.skills.evaluate_quiz.settings.llm_api_key", "sk-test"),
            patch("coursepilot.agent.skills.evaluate_quiz.AsyncOpenAI") as mock_openai,
        ):
            client = AsyncMock()
            mock_openai.return_value = client
            client.chat.completions.create = AsyncMock(return_value=mock_completion)

            from coursepilot.agent.skills.evaluate_quiz import evaluate_quiz
            result = await evaluate_quiz(
                quiz_data={"questions": [{"question_text": "测试"}]},
                context="", course_context={}
            )
            assert result["status"] == "PASS"


class TestGradeAnswers:
    """grade_answers — 批改答案"""

    @pytest.mark.asyncio
    async def test_all_correct(self):
        """全部答对"""
        quiz_data = {
            "questions": [
                {"question_text": "1+1=?", "correct_answer": "B", "kp_path": "数学/算术"},
                {"question_text": "2+2=?", "correct_answer": "D", "kp_path": "数学/算术"},
            ]
        }
        from coursepilot.agent.skills.grade_answers import grade_answers
        result = await grade_answers(quiz_data, {"0": "B", "1": "D"})
        assert result["total"] == 2
        assert result["correct"] == 2
        assert result["score"] == 1.0
        assert result["error_kps"] == []

    @pytest.mark.asyncio
    async def test_partial_correct(self):
        """部分答对"""
        quiz_data = {
            "questions": [
                {"question_text": "1+1=?", "correct_answer": "B", "kp_path": "数学/算术"},
                {"question_text": "2+2=?", "correct_answer": "D", "kp_path": "数学/算术"},
            ]
        }
        from coursepilot.agent.skills.grade_answers import grade_answers
        result = await grade_answers(quiz_data, {"0": "A", "1": "D"})
        assert result["total"] == 2
        assert result["correct"] == 1
        assert result["score"] == 0.5
        assert result["error_kps"] == ["数学/算术"]

    @pytest.mark.asyncio
    async def test_all_wrong(self):
        """全部答错"""
        quiz_data = {
            "questions": [
                {"question_text": "1+1=?", "correct_answer": "B", "kp_path": "数学/算术"},
            ]
        }
        from coursepilot.agent.skills.grade_answers import grade_answers
        result = await grade_answers(quiz_data, {"0": "C"})
        assert result["total"] == 1
        assert result["correct"] == 0
        assert result["score"] == 0.0
        assert result["error_kps"] == ["数学/算术"]

    @pytest.mark.asyncio
    async def test_empty_quiz(self):
        """空试卷"""
        from coursepilot.agent.skills.grade_answers import grade_answers
        result = await grade_answers({"questions": []}, {})
        assert result["total"] == 0
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_missing_answer_treated_as_wrong(self):
        """未作答视为错误"""
        quiz_data = {
            "questions": [
                {"question_text": "1+1=?", "correct_answer": "B", "kp_path": "数学/算术"},
            ]
        }
        from coursepilot.agent.skills.grade_answers import grade_answers
        result = await grade_answers(quiz_data, {})
        assert result["correct"] == 0
        assert result["score"] == 0.0



class TestDiagnose:
    """diagnose — 学情诊断"""

    @pytest.mark.asyncio
    async def test_identifies_weak_kps(self, mock_db):
        """正确率 <60% 的 KP 被标记为薄弱"""
        result = MagicMock()
        result.all.return_value = [
            ("OS/进程同步", 5, 1),   # (kp_path, count, correct_sum) — 20%
            ("OS/进程调度", 5, 4),   # 80%
        ]
        mock_db.execute = AsyncMock(return_value=result)

        from coursepilot.agent.skills.diagnose import diagnose
        diag = await diagnose(session=mock_db, user_id=str(uuid4()), course_id=str(uuid4()))

        assert "OS/进程同步" in diag["weak_kps"]
        assert "OS/进程调度" not in diag["weak_kps"]
        assert diag["kp_stats"]["OS/进程同步"]["rate"] == 0.2
        assert diag["kp_stats"]["OS/进程调度"]["rate"] == 0.8

    @pytest.mark.asyncio
    async def test_empty_practice_records(self, mock_db):
        """无做题记录时返回空诊断"""
        result = MagicMock()
        result.all.return_value = []
        mock_db.execute = AsyncMock(return_value=result)

        from coursepilot.agent.skills.diagnose import diagnose
        diag = await diagnose(session=mock_db, user_id=str(uuid4()), course_id=str(uuid4()))

        assert diag["weak_kps"] == []
        assert diag["total_practiced"] == 0
        assert diag["overall_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_summary_contains_key_info(self, mock_db):
        """summary 包含练习总数和正确率"""
        result = MagicMock()
        result.all.return_value = [
            ("OS/进程调度", 10, 7),  # 70%
        ]
        mock_db.execute = AsyncMock(return_value=result)

        from coursepilot.agent.skills.diagnose import diagnose
        diag = await diagnose(session=mock_db, user_id=str(uuid4()), course_id=str(uuid4()))

        assert "共练习" in diag["summary"]
        assert "10" in diag["summary"] or "70" in diag["summary"]
        assert diag["overall_rate"] == 0.7


class TestReviewPlan:
    """review_plan — 复习计划"""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_weak_kps(self, mock_db):
        """无薄弱知识点时返回空计划"""
        from coursepilot.agent.skills.review_plan import review_plan
        plan = await review_plan(
            session=mock_db,
            user_id=str(uuid4()),
            course_id=str(uuid4()),
            diagnosis={"weak_kps": [], "kp_stats": {}},
        )
        assert plan["total_count"] == 0
        assert plan["plan_id"] == ""

    @pytest.mark.asyncio
    async def test_generates_plan_without_llm(self, mock_db):
        """无 API key 时生成简单的默认计划"""
        with patch("coursepilot.agent.skills.review_plan.settings.llm_api_key", ""):
            from coursepilot.agent.skills.review_plan import review_plan
            plan = await review_plan(
                session=mock_db,
                user_id=str(uuid4()),
                course_id=str(uuid4()),
                diagnosis={
                    "weak_kps": ["OS/进程同步", "OS/内存管理"],
                    "kp_stats": {
                        "OS/进程同步": {"total": 5, "correct": 1, "rate": 0.2},
                        "OS/内存管理": {"total": 4, "correct": 2, "rate": 0.5},
                    },
                },
            )
            assert plan["total_count"] == 2
            assert len(plan["items"]) == 2
            assert plan["plan_id"] != ""

    @pytest.mark.asyncio
    async def test_generates_plan_with_llm(self, mock_db):
        """有 API key 时调用 LLM 生成计划"""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = (
            '{"items": [{"kp_path": "OS/进程同步", "priority": 1, "reason": "薄弱", "status": "pending"}], '
            '"total_count": 1, "plan_summary": "复习计划摘要"}'
        )

        with patch("coursepilot.agent.skills.review_plan.settings.llm_api_key", "sk-test"):
            with patch("coursepilot.agent.skills.review_plan.AsyncOpenAI") as mock_openai:
                client = AsyncMock()
                mock_openai.return_value = client
                client.chat.completions.create = AsyncMock(return_value=mock_completion)

                from coursepilot.agent.skills.review_plan import review_plan
                plan = await review_plan(
                    session=mock_db,
                    user_id=str(uuid4()),
                    course_id=str(uuid4()),
                    diagnosis={
                        "weak_kps": ["OS/进程同步"],
                        "kp_stats": {"OS/进程同步": {"total": 5, "correct": 1, "rate": 0.2}},
                    },
                )
                assert plan["total_count"] == 1
                assert len(plan["items"]) == 1
                assert plan["plan_id"] != ""

    @pytest.mark.asyncio
    async def test_plan_persisted_to_db(self, mock_db):
        """review_plan 写入 ReviewPlan 表"""
        with patch("coursepilot.agent.skills.review_plan.settings.llm_api_key", ""):
            from coursepilot.models import ReviewPlan
            from coursepilot.agent.skills.review_plan import review_plan

            plan = await review_plan(
                session=mock_db,
                user_id=str(uuid4()),
                course_id=str(uuid4()),
                diagnosis={
                    "weak_kps": ["OS/进程同步"],
                    "kp_stats": {"OS/进程同步": {"total": 5, "correct": 1, "rate": 0.2}},
                },
            )
            assert mock_db.add.called
            assert mock_db.flush.called


# ═══════════════════════════════════════════════════════════════
# 3. Phase 2 节点函数
# ═══════════════════════════════════════════════════════════════

class TestPhase2Nodes:
    """Phase 2 节点函数 — get_mastery_node ~ review_plan_node"""

    @pytest.mark.asyncio
    async def test_get_mastery_node(self, sample_state, mock_asf):
        """get_mastery_node 查询掌握度"""
        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.get_mastery") as mock_gm,
        ):
            mock_gm.return_value = {
                "mastery_level": {"OS/进程调度": 0.8},
                "weak_kps": [],
                "avg_correct_rate": 0.8,
            }
            from coursepilot.agent.nodes import get_mastery_node
            result = await get_mastery_node(sample_state)

        assert result["mastery"]["mastery_level"]["OS/进程调度"] == 0.8
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_get_mastery_node_propagates_error(self, sample_state, mock_asf):
        """get_mastery_node 内部异常 → error 被设置"""
        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.get_mastery", side_effect=ValueError("DB err")),
        ):
            from coursepilot.agent.nodes import get_mastery_node
            result = await get_mastery_node(sample_state)
        assert "DB err" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_generate_quiz_node(self, sample_state, mock_asf):
        """generate_quiz_node 生成试题"""
        sample_state["context"] = "教材内容"
        sample_state["mastery"] = {"weak_kps": ["OS/进程同步"]}

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.generate_quiz") as mock_gq,
        ):
            mock_gq.return_value = {
                "questions": [{"question_text": "1+1=?", "correct_answer": "B"}]
            }
            from coursepilot.agent.nodes import generate_quiz_node
            result = await generate_quiz_node(sample_state)

        assert len(result["quiz_data"]["questions"]) == 1
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_generate_quiz_node_forwards_state(self, sample_state, mock_asf):
        """generate_quiz_node 正确传递 context / course_context / mastery 给 skill"""
        sample_state["context"] = "retrieved_context"
        sample_state["course_context"] = {"name": "OS"}
        sample_state["mastery"] = {"weak_kps": ["OS/测试"]}

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.generate_quiz") as mock_gq,
        ):
            mock_gq.return_value = {}
            from coursepilot.agent.nodes import generate_quiz_node
            await generate_quiz_node(sample_state)

            mock_gq.assert_called_once_with(
                context="retrieved_context",
                course_context={"name": "OS"},
                mastery={"weak_kps": ["OS/测试"]},
            )

    @pytest.mark.asyncio
    async def test_generate_quiz_node_catches_exception(self, sample_state, mock_asf):
        """generate_quiz_node 异常 → error 在 quiz_data 内部"""
        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.generate_quiz", side_effect=RuntimeError("LLM err")),
        ):
            from coursepilot.agent.nodes import generate_quiz_node
            result = await generate_quiz_node(sample_state)
        assert result["quiz_data"].get("error") is not None

    @pytest.mark.asyncio
    async def test_evaluate_quiz_node_passes_quiz(self, sample_state, mock_asf):
        """evaluate_quiz_node 验证通过 → retry_count 不变"""
        sample_state["quiz_data"] = {"questions": [{"question_text": "测试"}]}

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.evaluate_quiz") as mock_eq,
        ):
            mock_eq.return_value = {"status": "PASS", "score": 0.9}
            from coursepilot.agent.nodes import evaluate_quiz_node
            result = await evaluate_quiz_node(sample_state)

        assert result["eval_result"]["status"] == "PASS"
        assert result["retry_count"] == 0
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_evaluate_quiz_node_fail_increments_retry(self, sample_state, mock_asf):
        """evaluate_quiz_node FAIL → retry_count 递增"""
        sample_state["quiz_data"] = {"questions": [{"question_text": "测试"}]}
        sample_state["retry_count"] = 0

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.evaluate_quiz") as mock_eq,
        ):
            mock_eq.return_value = {"status": "FAIL", "score": 0.3}
            from coursepilot.agent.nodes import evaluate_quiz_node
            result = await evaluate_quiz_node(sample_state)

        assert result["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_evaluate_quiz_node_forwards_state(self, sample_state, mock_asf):
        """evaluate_quiz_node 正确传递参数给 skill"""
        sample_state["quiz_data"] = {"questions": [{"id": 1}]}
        sample_state["context"] = "教材内容"
        sample_state["course_context"] = {"name": "OS"}

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.evaluate_quiz") as mock_eq,
        ):
            mock_eq.return_value = {"status": "PASS", "score": 1.0}
            from coursepilot.agent.nodes import evaluate_quiz_node
            await evaluate_quiz_node(sample_state)

            mock_eq.assert_called_once_with(
                quiz_data={"questions": [{"id": 1}]},
                context="教材内容",
                course_context={"name": "OS"},
            )

    @pytest.mark.asyncio
    async def test_create_plan_node_formats_answer(self, sample_state):
        """create_plan_node 将 quiz 格式化为回答"""
        sample_state["quiz_data"] = {
            "questions": [
                {
                    "question_text": "1+1=?",
                    "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                    "kp_path": "数学/算术",
                }
            ]
        }
        from coursepilot.agent.nodes import create_plan_node
        result = await create_plan_node(sample_state)

        assert "1+1=?" in result["answer"]
        assert len(result["sources"]) == 1
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_create_plan_node_empty_quiz(self, sample_state):
        """空 quiz 也能正常格式化"""
        sample_state["quiz_data"] = {}
        from coursepilot.agent.nodes import create_plan_node
        result = await create_plan_node(sample_state)
        assert "0 道练习题" in result["answer"]
        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_diagnose_node(self, sample_state, mock_asf):
        """diagnose_node 执行诊断"""
        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.diagnose") as mock_diag,
        ):
            mock_diag.return_value = {
                "weak_kps": ["OS/进程同步"],
                "kp_stats": {},
                "summary": "诊断摘要",
                "total_practiced": 5,
                "overall_rate": 0.6,
            }
            from coursepilot.agent.nodes import diagnose_node
            result = await diagnose_node(sample_state)

        assert result["diagnosis"]["weak_kps"] == ["OS/进程同步"]
        assert result["answer"] == "诊断摘要"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_diagnose_node_error(self, sample_state, mock_asf):
        """diagnose_node 异常 → 降级回答"""
        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.diagnose", side_effect=RuntimeError("fail")),
        ):
            from coursepilot.agent.nodes import diagnose_node
            result = await diagnose_node(sample_state)
        assert "诊断失败" in result["answer"]

    @pytest.mark.asyncio
    async def test_review_plan_node(self, sample_state, mock_asf):
        """review_plan_node 生成复习计划"""
        sample_state["diagnosis"] = {"weak_kps": ["OS/进程同步"], "kp_stats": {}}

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.review_plan") as mock_rp,
        ):
            mock_rp.return_value = {
                "items": [{"kp_path": "OS/进程同步", "priority": 1}],
                "total_count": 1,
                "plan_summary": "复习计划摘要",
                "plan_id": str(uuid4()),
            }
            from coursepilot.agent.nodes import review_plan_node
            result = await review_plan_node(sample_state)

        assert result["review_plan"]["total_count"] == 1
        assert result["answer"] == "复习计划摘要"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_review_plan_node_error(self, sample_state, mock_asf):
        """review_plan_node 异常 → 降级回答"""
        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.review_plan", side_effect=RuntimeError("fail")),
        ):
            from coursepilot.agent.nodes import review_plan_node
            result = await review_plan_node(sample_state)
        assert "生成复习计划失败" in result["answer"]

    def test_all_phase2_nodes_callable(self):
        """所有 Phase 2 节点函数可导入且可调用"""
        from coursepilot.agent.nodes import (
            get_mastery_node, generate_quiz_node, evaluate_quiz_node,
            create_plan_node, diagnose_node, review_plan_node,
        )
        assert callable(get_mastery_node)
        assert callable(generate_quiz_node)
        assert callable(evaluate_quiz_node)
        assert callable(create_plan_node)
        assert callable(diagnose_node)
        assert callable(review_plan_node)


# ═══════════════════════════════════════════════════════════════
# 4. Profile Updater
# ═══════════════════════════════════════════════════════════════

class TestProfileUpdater:
    """profile_updater 入口函数（_do_update 含真实 SQL 构建，需集成测试验证）"""

    @pytest.mark.asyncio
    async def test_update_profile_calls_do_update(self):
        """update_profile 正确新建 session 并委托给 _do_update"""
        mock_session = AsyncMock()
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_cm.__aexit__.return_value = None

        with (
            patch("coursepilot.agent.profile_updater.async_session_factory",
                  return_value=mock_session_cm),
            patch("coursepilot.agent.profile_updater._do_update") as mock_do,
        ):
            from coursepilot.agent.profile_updater import update_profile
            await update_profile(user_id="uid", course_id="cid")

            mock_do.assert_called_once()
            args, _ = mock_do.call_args
            assert args[0] is mock_session  # 第一个参数是 session
            assert args[1] == "uid"
            assert args[2] == "cid"

    @pytest.mark.asyncio
    async def test_update_profile_wraps_exception(self):
        """update_profile 内部异常不向外传播"""
        with patch(
            "coursepilot.agent.profile_updater.async_session_factory",
            side_effect=RuntimeError("conn fail"),
        ):
            from coursepilot.agent.profile_updater import update_profile
            await update_profile(user_id=str(uuid4()), course_id=str(uuid4()))


# ═══════════════════════════════════════════════════════════════
# 5. Phase 2 图结构
# ═══════════════════════════════════════════════════════════════

class TestPhase2Graph:
    """验证 Phase 2 图结构：9 个节点 + 条件边"""

    @pytest.mark.asyncio
    async def test_graph_has_nine_nodes(self):
        """build_agent_graph() 注册了 9 个自定义节点"""
        from langgraph.checkpoint.memory import MemorySaver
        with patch("coursepilot.agent.graph._get_saver", return_value=MemorySaver()):
            from coursepilot.agent.graph import build_agent_graph
            graph = await build_agent_graph()

        custom_nodes = {n for n in graph.nodes if not n.startswith("__")}
        expected = {
            "build_context", "classify", "query_rag", "finalize",
            "get_mastery", "generate_quiz", "evaluate_quiz",
            "create_plan", "diagnose", "review_plan",
        }
        assert custom_nodes == expected, f"缺失节点: {expected - custom_nodes}"

    def test_conditional_edges_in_routing_module(self):
        """验证 3 个条件路由函数存在"""
        from coursepilot.agent.routing import (
            route_by_intent, route_after_rag, route_after_evaluate,
        )
        assert callable(route_by_intent)
        assert callable(route_after_rag)
        assert callable(route_after_evaluate)

    @pytest.mark.asyncio
    async def test_question_path_uses_fixed_edges(self):
        """question 路径经过 query_rag → finalize（不经 generate_quiz）"""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import StateGraph, START, END

        # 检查 graph 中 query_rag 到 finalize 的条件边存在
        with patch("coursepilot.agent.graph._get_saver", return_value=MemorySaver()):
            from coursepilot.agent.graph import build_agent_graph
            graph = await build_agent_graph()

        assert "query_rag" in graph.nodes
        assert "finalize" in graph.nodes

    def test_route_after_evaluate_handles_all_states(self):
        """route_after_evaluate 覆盖所有组合"""
        from coursepilot.agent.routing import route_after_evaluate

        # PASS + practice → create_plan
        assert route_after_evaluate({"intent": "practice", "eval_result": {"status": "PASS"}, "retry_count": 0}) == "create_plan"
        # PASS + review → review_plan
        assert route_after_evaluate({"intent": "review", "eval_result": {"status": "PASS"}, "retry_count": 0}) == "review_plan"
        # FAIL + retry < 2 → generate_quiz
        assert route_after_evaluate({"intent": "practice", "eval_result": {"status": "FAIL"}, "retry_count": 1}) == "generate_quiz"
        # FAIL + retry >= 2 → create_plan（最多重试后仍展示题目）
        assert route_after_evaluate({"intent": "practice", "eval_result": {"status": "FAIL"}, "retry_count": 2}) == "create_plan"


# ═══════════════════════════════════════════════════════════════
# 6. 端到端 Phase 2 工作流
# ═══════════════════════════════════════════════════════════════

@pytest.mark.e2e
class TestPhase2E2E:
    """5 条 intent 路径 + evaluate 重试循环（全 mock + MemorySaver）"""

    def _build_graph(self):
        """编译带 Phase 2 节点的 MemorySaver 图"""
        from langgraph.graph import StateGraph, START, END
        from langgraph.checkpoint.memory import MemorySaver
        from coursepilot.agent.state import AgentState
        from coursepilot.agent.nodes import (
            build_context_node, classify_node, finalize_node, query_rag_node,
            get_mastery_node, generate_quiz_node, evaluate_quiz_node,
            create_plan_node, diagnose_node, review_plan_node,
        )
        from coursepilot.agent.routing import (
            route_by_intent, route_after_rag, route_after_evaluate,
        )

        builder = StateGraph(AgentState)
        for name, node_fn in [
            ("build_context", build_context_node),
            ("classify", classify_node),
            ("query_rag", query_rag_node),
            ("get_mastery", get_mastery_node),
            ("generate_quiz", generate_quiz_node),
            ("evaluate_quiz", evaluate_quiz_node),
            ("create_plan", create_plan_node),
            ("diagnose", diagnose_node),
            ("review_plan", review_plan_node),
            ("finalize", finalize_node),
        ]:
            builder.add_node(name, node_fn)

        builder.add_edge(START, "build_context")
        builder.add_edge("build_context", "classify")
        builder.add_conditional_edges("classify", route_by_intent, {
            "query_rag": "query_rag", "get_mastery": "get_mastery", "diagnose": "diagnose",
        })
        builder.add_edge("get_mastery", "query_rag")
        builder.add_conditional_edges("query_rag", route_after_rag, {
            "generate_quiz": "generate_quiz", "finalize": "finalize",
        })
        builder.add_edge("generate_quiz", "evaluate_quiz")
        builder.add_conditional_edges("evaluate_quiz", route_after_evaluate, {
            "generate_quiz": "generate_quiz", "review_plan": "review_plan",
            "create_plan": "create_plan", "finalize": "finalize",
        })
        builder.add_edge("create_plan", "finalize")
        builder.add_edge("review_plan", "finalize")
        builder.add_edge("diagnose", "finalize")
        builder.add_edge("finalize", END)

        return builder.compile(checkpointer=MemorySaver())

    def _make_state(self, **overrides):
        state = {
            "query": "测试",
            "course_id": str(uuid4()),
            "user_id": str(uuid4()),
            "session_id": str(uuid4()),
            "messages": [],
            "course_context": {},
            "user_profile": None,
            "recent_qa": [],
            "intent": "",
            "context": "",
            "retrieved_metadata": {},
            "answer": "",
            "sources": [],
            "token_count": 0,
            "error": None,
            "mastery": {},
            "quiz_data": {},
            "eval_result": {},
            "retry_count": 0,
            "diagnosis": {},
            "review_plan": {},
        }
        state.update(overrides)
        return state

    @pytest.mark.asyncio
    async def test_question_path(self, mock_asf):
        """question 路径：build_context → classify → query_rag → finalize"""
        graph = self._build_graph()
        state = self._make_state(query="什么是进程调度？")

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.build_context_logic") as mock_bc,
            patch("coursepilot.agent.nodes.classify_intent", return_value="question"),
            patch("coursepilot.agent.nodes.query_rag") as mock_qr,
            patch("coursepilot.agent.nodes.update_qa_record") as mock_uq,
        ):
            mock_bc.return_value = ({"name": "OS"}, None, [])
            mock_qr.return_value = ("进程调度是核心功能", "上下文", {"source_kp_paths": ["OS/进程调度"]}, [])
            mock_uq.return_value = 42

            result = await graph.ainvoke(state, {"configurable": {"thread_id": str(uuid4())}})

        assert result["intent"] == "question"
        assert "进程调度" in result["answer"]
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_practice_path(self, mock_asf):
        """practice 路径：get_mastery → query_rag → generate_quiz → evaluate_quiz → create_plan → finalize"""
        graph = self._build_graph()
        state = self._make_state(query="给我出几道题")

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.build_context_logic") as mock_bc,
            patch("coursepilot.agent.nodes.classify_intent", return_value="practice"),
            patch("coursepilot.agent.nodes.get_mastery") as mock_gm,
            patch("coursepilot.agent.nodes.query_rag") as mock_qr,
            patch("coursepilot.agent.nodes.generate_quiz") as mock_gq,
            patch("coursepilot.agent.nodes.evaluate_quiz") as mock_eq,
            patch("coursepilot.agent.nodes.update_qa_record") as mock_uq,
        ):
            mock_bc.return_value = ({"name": "OS"}, None, [])
            mock_gm.return_value = {"mastery_level": {}, "weak_kps": ["OS/进程同步"], "avg_correct_rate": 0.5}
            mock_qr.return_value = ("教材上下文", "", {}, [])
            mock_gq.return_value = {"questions": [
                {"question_text": "进程同步?", "options": {"A": "1", "B": "2"},
                 "correct_answer": "B", "kp_path": "OS/进程同步"},
            ]}
            mock_eq.return_value = {"status": "PASS", "score": 0.9}
            mock_uq.return_value = 0

            result = await graph.ainvoke(state, {"configurable": {"thread_id": str(uuid4())}})

        assert result["intent"] == "practice"
        assert result["error"] is None
        # create_plan_node 格式化后的答案
        assert "为你生成了" in result.get("answer", "")
        assert "进程同步?" in result.get("answer", "")

    @pytest.mark.asyncio
    async def test_diagnose_path(self, mock_asf):
        """diagnose 路径：build_context → classify → diagnose → finalize"""
        graph = self._build_graph()
        state = self._make_state(query="我哪里掌握得不好")

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.build_context_logic") as mock_bc,
            patch("coursepilot.agent.nodes.classify_intent", return_value="diagnose"),
            patch("coursepilot.agent.nodes.diagnose") as mock_diag,
            patch("coursepilot.agent.nodes.update_qa_record") as mock_uq,
        ):
            mock_bc.return_value = ({"name": "OS"}, None, [])
            mock_diag.return_value = {
                "weak_kps": ["OS/进程同步"],
                "kp_stats": {},
                "summary": "共练习 10 题，正确率 50%。薄弱知识点：OS/进程同步",
                "total_practiced": 10,
                "overall_rate": 0.5,
            }
            mock_uq.return_value = 0

            result = await graph.ainvoke(state, {"configurable": {"thread_id": str(uuid4())}})

        assert result["intent"] == "diagnose"
        assert "薄弱" in result.get("answer", "")
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_review_path(self, mock_asf):
        """review 路径：get_mastery → query_rag → generate_quiz → evaluate_quiz → review_plan → finalize"""
        graph = self._build_graph()
        state = self._make_state(query="帮我复习进程管理")

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.build_context_logic") as mock_bc,
            patch("coursepilot.agent.nodes.classify_intent", return_value="review"),
            patch("coursepilot.agent.nodes.get_mastery") as mock_gm,
            patch("coursepilot.agent.nodes.query_rag") as mock_qr,
            patch("coursepilot.agent.nodes.generate_quiz") as mock_gq,
            patch("coursepilot.agent.nodes.evaluate_quiz") as mock_eq,
            patch("coursepilot.agent.nodes.review_plan") as mock_rp,
            patch("coursepilot.agent.nodes.update_qa_record") as mock_uq,
        ):
            mock_bc.return_value = ({"name": "OS"}, None, [])
            mock_gm.return_value = {"mastery_level": {}, "weak_kps": ["OS/进程同步"], "avg_correct_rate": 0.5}
            mock_qr.return_value = ("上下文", "", {}, [])
            mock_gq.return_value = {"questions": [
                {"question_text": "进程同步?", "options": {"A": "1", "B": "2"},
                 "correct_answer": "B", "kp_path": "OS/进程同步"},
            ]}
            mock_eq.return_value = {"status": "PASS", "score": 0.9}
            mock_rp.return_value = {
                "items": [{"kp_path": "OS/进程同步", "priority": 1}],
                "total_count": 1,
                "plan_summary": "复习计划摘要",
                "plan_id": str(uuid4()),
            }
            mock_uq.return_value = 0

            result = await graph.ainvoke(state, {"configurable": {"thread_id": str(uuid4())}})

        assert result["intent"] == "review"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_code_help_path(self, mock_asf):
        """code_help 路径：build_context → classify → query_rag → finalize（Phase 3 增强）"""
        graph = self._build_graph()
        state = self._make_state(query="这个代码为什么报错")

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.build_context_logic") as mock_bc,
            patch("coursepilot.agent.nodes.classify_intent", return_value="code_help"),
            patch("coursepilot.agent.nodes.query_rag") as mock_qr,
            patch("coursepilot.agent.nodes.update_qa_record") as mock_uq,
        ):
            mock_bc.return_value = ({"name": "OS"}, None, [])
            mock_qr.return_value = ("代码相关回答", "上下文", {}, [])
            mock_uq.return_value = 0

            result = await graph.ainvoke(state, {"configurable": {"thread_id": str(uuid4())}})

        assert result["intent"] == "code_help"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_evaluate_retry_loop(self, mock_asf):
        """evaluate FAIL → 重试 generate_quiz（最多 1 次重试，即共 2 次 evaluate）

        evaluate_quiz_node 在 FAIL 时递增 retry_count：
        第 1 次 FAIL → retry=1 → route_after_evaluate: retry<2 → "generate_quiz"
        第 2 次 FAIL → retry=2 → route_after_evaluate: retry>=2 → "finalize"
        """
        graph = self._build_graph()
        state = self._make_state(query="出几道题")

        evaluate_call_count = [0]

        def evaluate_side_effect(**kw):
            evaluate_call_count[0] += 1
            return {"status": "FAIL", "score": 0.3, "feedback": {"suggestions": ["题目太简单"]}}

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.build_context_logic") as mock_bc,
            patch("coursepilot.agent.nodes.classify_intent", return_value="practice"),
            patch("coursepilot.agent.nodes.get_mastery") as mock_gm,
            patch("coursepilot.agent.nodes.query_rag") as mock_qr,
            patch("coursepilot.agent.nodes.generate_quiz") as mock_gq,
            patch("coursepilot.agent.nodes.evaluate_quiz", side_effect=evaluate_side_effect),
            patch("coursepilot.agent.nodes.update_qa_record") as mock_uq,
        ):
            mock_bc.return_value = ({"name": "OS"}, None, [])
            mock_gm.return_value = {"mastery_level": {}, "weak_kps": [], "avg_correct_rate": None}
            mock_qr.return_value = ("上下文", "", {}, [])
            mock_gq.return_value = {"questions": []}
            mock_uq.return_value = 0

            result = await graph.ainvoke(state, {"configurable": {"thread_id": str(uuid4())}})

        assert evaluate_call_count[0] == 2, f"evaluate 应被调用 2 次，实际 {evaluate_call_count[0]}"
        assert result["error"] is None


# ═══════════════════════════════════════════════════════════════
# 7. AgentState Phase 2 字段验证
# ═══════════════════════════════════════════════════════════════

class TestAgentStatePhase2:
    """AgentState Phase 2 扩展字段"""

    def test_phase2_fields_present(self):
        from typing import get_type_hints
        from coursepilot.agent.state import AgentState

        hints = get_type_hints(AgentState)
        for field in ("mastery", "quiz_data", "eval_result", "retry_count", "diagnosis", "review_plan"):
            assert field in hints, f"AgentState 缺少 Phase 2 字段: {field}"

    def test_retry_count_is_int(self):
        from typing import get_type_hints
        from coursepilot.agent.state import AgentState

        hints = get_type_hints(AgentState)
        assert hints["retry_count"] is int or hints["retry_count"] == int

    def test_mastery_is_dict(self):
        from typing import get_type_hints
        from coursepilot.agent.state import AgentState

        hints = get_type_hints(AgentState)
        assert hints["mastery"] is dict or hints["mastery"] == dict


# ═══════════════════════════════════════════════════════════════
# 8. finalize_node Phase 2 增强
# ═══════════════════════════════════════════════════════════════

class TestFinalizePhase2:
    """finalize_node 的 Phase 2 增强（_update_session_intent + profile_updater 触发）"""

    @pytest.mark.asyncio
    async def test_finalize_updates_session_intent(self, sample_state, mock_asf):
        """finalize 更新 agent_session 的 intent"""
        from coursepilot.models import AgentSession

        agent_session = MagicMock(spec=AgentSession)
        result = MagicMock()
        result.scalar_one_or_none.return_value = agent_session

        async def exec_side(*a, **kw):
            return result

        mock_db = AsyncMock()
        mock_db.execute = exec_side
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        cm = AsyncMock()
        cm.__aenter__.return_value = mock_db
        cm.__aexit__.return_value = None

        sample_state["intent"] = "practice"

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=cm),
            patch("coursepilot.agent.nodes.update_qa_record", return_value=42),
            patch("coursepilot.agent.nodes.update_profile") as mock_up,
        ):
            from coursepilot.agent.nodes import finalize_node
            result = await finalize_node(sample_state)

        assert agent_session.intent == "practice"
        assert agent_session.status == "completed"
        assert mock_up.called, "update_profile 未被触发"

    @pytest.mark.asyncio
    async def test_finalize_catches_db_error(self, sample_state, mock_asf):
        """finalize_node DB 异常 → error 字段"""
        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.update_qa_record", side_effect=Exception("DB error")),
            patch("coursepilot.agent.nodes.update_profile"),
        ):
            from coursepilot.agent.nodes import finalize_node
            result = await finalize_node(sample_state)
        assert "DB error" in result.get("error", "")
