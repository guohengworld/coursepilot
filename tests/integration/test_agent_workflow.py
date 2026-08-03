"""Agent 工作流集成测试：用 MemorySaver + 真实 DB 验证完整路由

测试覆盖所有 5 条路径和重试逻辑。LLM skill 全部 mock。
"""
import uuid
from unittest.mock import patch, AsyncMock

import pytest
from langgraph.types import Command

from tests.integration.conftest import ZERO_TOKENS


# ── 每个 test 公用的 mock return values ──
MOCK_QA_ANSWER = (
    "进程调度是操作系统核心功能",
    "<context>CPU 调度</context>",
    {"source_kp_paths": ["OS/进程管理"], "top_uuids": [uuid.uuid4()]},
    [{"kp_path": "OS/进程管理"}],
    ZERO_TOKENS,
)
MOCK_GEN_QUIZ = (
    {"questions": [
        {"question_text": "什么是进程?", "options": {"A": "程序", "B": "进程", "C": "文件"}, "answer": "B", "kp_path": "OS/进程管理"},
    ]},
    ZERO_TOKENS,
)
MOCK_EVAL_PASS = ({"status": "PASS", "score": 0.85, "feedback": "合格"}, ZERO_TOKENS)
MOCK_EVAL_FAIL = ({"status": "FAIL", "score": 0.3, "feedback": "不合格"}, ZERO_TOKENS)
MOCK_DIAGNOSE = {"summary": "进程管理薄弱", "weak_kps": ["OS/进程管理"]}
MOCK_REVIEW_PLAN = ({"plan_summary": "复习计划: 进程管理", "items": []}, ZERO_TOKENS)


class TestQuestionWorkflow:
    """question 路径：build_context → classify → query_rag → finalize"""

    @pytest.mark.asyncio
    async def test_question_path(self, raw_session, std_data, real_asf):
        """完整 question workflow 返回 answer 且 DB 写入正常"""
        uid, cid = std_data["user_id"], std_data["course_id"]

        with (
            patch("coursepilot.agent.nodes.async_session_factory", real_asf),
            patch("coursepilot.agent.nodes.classify_intent",
                  return_value=("question", ZERO_TOKENS)) as mock_cls,
            patch("coursepilot.agent.nodes.query_rag",
                  return_value=MOCK_QA_ANSWER) as mock_qr,
            patch("coursepilot.agent.nodes.update_profile", new=AsyncMock()),
        ):
            from coursepilot.agent.graph import build_agent_graph
            from langgraph.checkpoint.memory import MemorySaver
            with patch("coursepilot.agent.graph._get_saver", return_value=MemorySaver()):
                graph = await build_agent_graph()

            state = {
                "query": "什么是进程调度？",
                "course_id": str(cid),
                "user_id": str(uid),
                "session_id": str(uuid.uuid4()),
                "messages": [], "course_context": {}, "user_profile": None,
                "recent_qa": [], "intent": "", "context": "",
                "retrieved_metadata": {}, "answer": "", "sources": [],
                "token_count": 0, "llm_calls": [], "error": None,
            }
            result = await graph.ainvoke(state, {"configurable": {"thread_id": str(uuid.uuid4())}})

        assert result["intent"] == "question"
        assert "进程调度" in result["answer"]
        assert len(result["sources"]) > 0
        assert isinstance(result["token_count"], int)
        mock_cls.assert_called_once()
        mock_qr.assert_called_once()

class TestDiagnoseWorkflow:
    """diagnose 路径：build_context → classify → diagnose → finalize"""

    @pytest.mark.asyncio
    async def test_diagnose_path(self, raw_session, std_data, real_asf):
        uid, cid = std_data["user_id"], std_data["course_id"]

        with (
            patch("coursepilot.agent.nodes.async_session_factory", real_asf),
            patch("coursepilot.agent.nodes.classify_intent",
                  return_value=("diagnose", ZERO_TOKENS)),
            patch("coursepilot.agent.nodes.update_profile", new=AsyncMock()),
            patch("coursepilot.agent.nodes.diagnose",
                  return_value=MOCK_DIAGNOSE) as mock_diag,
        ):
            from coursepilot.agent.graph import build_agent_graph
            from langgraph.checkpoint.memory import MemorySaver
            with patch("coursepilot.agent.graph._get_saver", return_value=MemorySaver()):
                graph = await build_agent_graph()

            state = {
                "query": "帮我诊断", "course_id": str(cid),
                "user_id": str(uid), "session_id": str(uuid.uuid4()),
                "messages": [], "course_context": {}, "user_profile": None,
                "recent_qa": [], "intent": "", "context": "",
                "retrieved_metadata": {}, "answer": "", "sources": [],
                "token_count": 0, "llm_calls": [], "error": None,
            }
            result = await graph.ainvoke(state, {"configurable": {"thread_id": str(uuid.uuid4())}})

        assert "进程管理薄弱" in result["answer"]
        mock_diag.assert_called_once()


class TestPracticeWorkflow:
    """practice 路径：含 human_review → get_mastery → query_rag → gen → eval → create_plan"""

    @pytest.mark.asyncio
    async def test_practice_path(self, raw_session, std_data, real_asf):
        """practice 路径通过 interrupt/resume 到达 create_plan"""
        uid, cid = std_data["user_id"], std_data["course_id"]

        with (
            patch("coursepilot.agent.nodes.async_session_factory", real_asf),
            patch("coursepilot.agent.nodes.classify_intent",
                  return_value=("practice", ZERO_TOKENS)),
            patch("coursepilot.agent.nodes.query_rag",
                  return_value=MOCK_QA_ANSWER),
            patch("coursepilot.agent.nodes.generate_quiz",
                  return_value=MOCK_GEN_QUIZ),
            patch("coursepilot.agent.nodes.evaluate_quiz",
                  return_value=MOCK_EVAL_PASS),
            patch("coursepilot.agent.nodes.update_profile", new=AsyncMock()),
        ):
            from coursepilot.agent.graph import build_agent_graph
            from langgraph.checkpoint.memory import MemorySaver
            with patch("coursepilot.agent.graph._get_saver", return_value=MemorySaver()):
                graph = await build_agent_graph()

            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            state = {
                "query": "帮我练习进程调度", "course_id": str(cid),
                "user_id": str(uid), "session_id": str(uuid.uuid4()),
                "messages": [], "course_context": {}, "user_profile": None,
                "recent_qa": [], "intent": "", "context": "",
                "retrieved_metadata": {}, "answer": "", "sources": [],
                "token_count": 0, "llm_calls": [], "error": None,
            }

            result = await graph.ainvoke(state, config)
            # result should contain the interrupt info
            # resume with approval
            result = await graph.ainvoke(
                Command(resume={"approved": True}),
                config,
            )

        assert result["intent"] == "practice"
        assert "练习题" in result.get("answer", "")
        assert len(result.get("sources", [])) > 0

    @pytest.mark.asyncio
    async def test_practice_rejected(self, raw_session, std_data, real_asf):
        """human_review 拒绝 → 直接到 finalize"""
        uid, cid = std_data["user_id"], std_data["course_id"]

        with (
            patch("coursepilot.agent.nodes.async_session_factory", real_asf),
            patch("coursepilot.agent.nodes.classify_intent",
                  return_value=("practice", ZERO_TOKENS)),
            patch("coursepilot.agent.nodes.update_profile", new=AsyncMock()),
        ):
            from coursepilot.agent.graph import build_agent_graph
            from langgraph.checkpoint.memory import MemorySaver
            with patch("coursepilot.agent.graph._get_saver", return_value=MemorySaver()):
                graph = await build_agent_graph()

            thread_id = str(uuid.uuid4())
            state = {
                "query": "帮我练习", "course_id": str(cid),
                "user_id": str(uid), "session_id": str(uuid.uuid4()),
                "messages": [], "course_context": {}, "user_profile": None,
                "recent_qa": [], "intent": "", "context": "",
                "retrieved_metadata": {}, "answer": "", "sources": [],
                "token_count": 0, "llm_calls": [], "error": None,
            }
            config = {"configurable": {"thread_id": thread_id}}

            result = await graph.ainvoke(state, config)
            result = await graph.ainvoke(
                Command(resume={"approved": False}),
                config,
            )

        assert "暂停" in result.get("answer", "")


class TestReviewWorkflow:
    """review 路径：含 human_review → ... → evaluate_quiz → review_plan"""

    @pytest.mark.asyncio
    async def test_review_path(self, raw_session, std_data, real_asf):
        uid, cid = std_data["user_id"], std_data["course_id"]

        with (
            patch("coursepilot.agent.nodes.async_session_factory", real_asf),
            patch("coursepilot.agent.nodes.classify_intent",
                  return_value=("review", ZERO_TOKENS)),
            patch("coursepilot.agent.nodes.query_rag",
                  return_value=MOCK_QA_ANSWER),
            patch("coursepilot.agent.nodes.generate_quiz",
                  return_value=MOCK_GEN_QUIZ),
            patch("coursepilot.agent.nodes.evaluate_quiz",
                  return_value=MOCK_EVAL_PASS),
            patch("coursepilot.agent.nodes.update_profile", new=AsyncMock()),
        ):
            from coursepilot.agent.graph import build_agent_graph
            from langgraph.checkpoint.memory import MemorySaver
            with patch("coursepilot.agent.graph._get_saver", return_value=MemorySaver()):
                graph = await build_agent_graph()

            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            state = {
                "query": "帮我复习进程管理", "course_id": str(cid),
                "user_id": str(uid), "session_id": str(uuid.uuid4()),
                "messages": [], "course_context": {}, "user_profile": None,
                "recent_qa": [], "intent": "", "context": "",
                "retrieved_metadata": {}, "answer": "", "sources": [],
                "token_count": 0, "llm_calls": [], "error": None,
            }

            result = await graph.ainvoke(state, config)
            result = await graph.ainvoke(
                Command(resume={"approved": True}),
                config,
            )

        assert result["intent"] == "review"
        # review path: evaluate_quiz passes → review_plan → finalize
        assert "llm_calls" in result
        assert len(result["llm_calls"]) > 0


class TestRetryLoop:
    """evaluate_quiz 重试循环"""

    @pytest.mark.asyncio
    async def test_retry_twice_then_pass(self, raw_session, std_data, real_asf):
        """前两次 FAIL，第三次 PASS → 进入 create_plan"""
        uid, cid = std_data["user_id"], std_data["course_id"]
        eval_results = [MOCK_EVAL_FAIL, MOCK_EVAL_FAIL, MOCK_EVAL_PASS]
        eval_counter = [0]

        async def eval_side_effect(**kwargs):
            idx = eval_counter[0]
            eval_counter[0] += 1
            return eval_results[idx]

        with (
            patch("coursepilot.agent.nodes.async_session_factory", real_asf),
            patch("coursepilot.agent.nodes.classify_intent",
                  return_value=("practice", ZERO_TOKENS)),
            patch("coursepilot.agent.nodes.query_rag",
                  return_value=MOCK_QA_ANSWER),
            patch("coursepilot.agent.nodes.generate_quiz",
                  return_value=MOCK_GEN_QUIZ),
            patch("coursepilot.agent.nodes.evaluate_quiz",
                  side_effect=eval_side_effect),
            patch("coursepilot.agent.nodes.update_profile", new=AsyncMock()),
        ):
            from coursepilot.agent.graph import build_agent_graph
            from langgraph.checkpoint.memory import MemorySaver
            with patch("coursepilot.agent.graph._get_saver", return_value=MemorySaver()):
                graph = await build_agent_graph()

            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            state = {
                "query": "练习", "course_id": str(cid),
                "user_id": str(uid), "session_id": str(uuid.uuid4()),
                "messages": [], "course_context": {}, "user_profile": None,
                "recent_qa": [], "intent": "", "context": "",
                "retrieved_metadata": {}, "answer": "", "sources": [],
                "token_count": 0, "llm_calls": [], "error": None,
            }

            result = await graph.ainvoke(state, config)
            result = await graph.ainvoke(
                Command(resume={"approved": True}),
                config,
            )

        assert result["intent"] == "practice"
        assert eval_counter[0] == 3  # 2 fail + 1 pass = 3 calls

    @pytest.mark.asyncio
    async def test_retry_exhausted_force_continue(self, raw_session, std_data, real_asf):
        """3 次 FAIL → 强制继续到 create_plan（retry_count >= 2 后强制通过）"""
        uid, cid = std_data["user_id"], std_data["course_id"]
        eval_counter = [0]

        async def always_fail(**kwargs):
            eval_counter[0] += 1
            return MOCK_EVAL_FAIL

        with (
            patch("coursepilot.agent.nodes.async_session_factory", real_asf),
            patch("coursepilot.agent.nodes.classify_intent",
                  return_value=("practice", ZERO_TOKENS)),
            patch("coursepilot.agent.nodes.query_rag",
                  return_value=MOCK_QA_ANSWER),
            patch("coursepilot.agent.nodes.generate_quiz",
                  return_value=MOCK_GEN_QUIZ),
            patch("coursepilot.agent.nodes.evaluate_quiz",
                  side_effect=always_fail),
            patch("coursepilot.agent.nodes.update_profile", new=AsyncMock()),
        ):
            from coursepilot.agent.graph import build_agent_graph
            from langgraph.checkpoint.memory import MemorySaver
            with patch("coursepilot.agent.graph._get_saver", return_value=MemorySaver()):
                graph = await build_agent_graph()

            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            state = {
                "query": "练习", "course_id": str(cid),
                "user_id": str(uid), "session_id": str(uuid.uuid4()),
                "messages": [], "course_context": {}, "user_profile": None,
                "recent_qa": [], "intent": "", "context": "",
                "retrieved_metadata": {}, "answer": "", "sources": [],
                "token_count": 0, "llm_calls": [], "error": None,
            }

            result = await graph.ainvoke(state, config)
            result = await graph.ainvoke(
                Command(resume={"approved": True}),
                config,
            )

        # 虽然始终 FAIL，但 retry_count >= 2 后强制继续到 create_plan
        assert result["intent"] == "practice"
        assert eval_counter[0] == 3  # gen(0) → eval(fail) → gen(1) → eval(fail) → gen(2) → eval(fail) → force


class TestHumanReview:
    """human-in-the-loop interrupt/resume"""

    @pytest.mark.asyncio
    async def test_interrupt_state_on_hold(self, raw_session, std_data, real_asf):
        """practice intent 触发 interrupt 后 state 保存 checkpoint"""
        uid, cid = std_data["user_id"], std_data["course_id"]

        with (
            patch("coursepilot.agent.nodes.async_session_factory", real_asf),
            patch("coursepilot.agent.nodes.classify_intent",
                  return_value=("practice", ZERO_TOKENS)),
            patch("coursepilot.agent.nodes.update_profile", new=AsyncMock()),
        ):
            from coursepilot.agent.graph import build_agent_graph
            from langgraph.checkpoint.memory import MemorySaver
            with patch("coursepilot.agent.graph._get_saver", return_value=MemorySaver()):
                graph = await build_agent_graph()

            thread_id = str(uuid.uuid4())
            state = {
                "query": "练习", "course_id": str(cid),
                "user_id": str(uid), "session_id": str(uuid.uuid4()),
                "messages": [], "course_context": {}, "user_profile": None,
                "recent_qa": [], "intent": "", "context": "",
                "retrieved_metadata": {}, "answer": "", "sources": [],
                "token_count": 0, "llm_calls": [], "error": None,
            }

            result = await graph.ainvoke(state, {"configurable": {"thread_id": thread_id}})
            # interrupt 后可以读取中间状态
            current = await graph.aget_state({"configurable": {"thread_id": thread_id}})
            assert current.values.get("intent") == "practice"
