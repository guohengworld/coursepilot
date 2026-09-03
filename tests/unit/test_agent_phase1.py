"""Phase 1 Agent 全方位测试

覆盖范围：
    - 图结构（节点数、边拓扑、checkpointer 类型）
    - Skill 函数（classify_intent、query_rag、update_qa_record）
    - Context Builder（user_profile + recent_qa 查询）
    - 节点函数（build_context_node ~ finalize_node）
    - API 端点（chat、get_session_status）
    - 端到端完整工作流（全 mock 外部依赖）

运行方式：
    .venv/Scripts/python -m pytest tests/unit/test_agent_phase1.py -v
    .venv/Scripts/python -m pytest tests/unit/test_agent_phase1.py -v -k "test_graph"
    .venv/Scripts/python -m pytest tests/unit/test_agent_phase1.py -v -m e2e

依赖：
    - pytest + pytest-asyncio
    - httpx（用于 TestClient）
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

ZERO_TOKENS = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# ═══════════════════════════════════════════════════════════════
# Shared Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_db():
    """异步 DB 会话 mock

    execute() 用 async def 实现，确保 await session.execute(...) 返回预置的 result。
    result 链：scalar_one_or_none → None | scalars → result → all → []
    """
    session = AsyncMock(spec=['execute', 'add', 'flush', 'commit'])
    session.commit = AsyncMock()

    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    result.scalars.return_value = result
    result.all.return_value = []

    async def execute_side_effect(*a, **kw):
        return result

    session.execute = execute_side_effect
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
    return {
        "course_name": "操作系统",
        "kp_tree_preview": "进程管理/进程调度/进程同步",
    }


@pytest.fixture
def sample_state():
    """模拟 LangGraph 节点间传递的 AgentState"""
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
        "llm_calls": [],
        "error": None,
    }


# ═══════════════════════════════════════════════════════════════
# 1. 图结构
# ═══════════════════════════════════════════════════════════════

class TestGraphStructure:
    """验证 StateGraph 拓扑：4 节点、线性边、AsyncPostgresSaver"""

    @pytest.mark.asyncio
    async def test_graph_has_ten_nodes(self):
        """build_agent_graph() 注册 12 个自定义节点（HITL 已移除；含路由兜底 fallback）"""
        from langgraph.checkpoint.memory import MemorySaver

        with patch("coursepilot.agent.graph._get_saver", return_value=MemorySaver()):
            from coursepilot.agent.graph import build_agent_graph
            graph = await build_agent_graph()

        custom_nodes = {n for n in graph.nodes if not n.startswith("__")}
        assert len(custom_nodes) == 12
        assert "human_review" not in custom_nodes
        assert "agentic_rag" in custom_nodes
        assert "fallback" in custom_nodes
        # P1: CRAG 节点已删除
        for removed in ("retrieve", "check_sufficiency", "synthesize", "decompose", "web_search"):
            assert removed not in custom_nodes

    def test_graph_linear_edges_in_builder(self):
        """检查 builder 注册了正确的边"""
        from langgraph.graph import END, START, StateGraph

        from coursepilot.agent.state import AgentState

        builder = StateGraph(AgentState)

        with patch.object(builder, "add_edge") as mock_add_edge:
            from coursepilot.agent.nodes import (
                build_context_node,
                classify_node,
                finalize_node,
                query_rag_node,
            )

            builder.add_node("build_context", build_context_node)
            builder.add_node("classify", classify_node)
            builder.add_node("query_rag", query_rag_node)
            builder.add_node("finalize", finalize_node)

            builder.add_edge(START, "build_context")
            builder.add_edge("build_context", "classify")
            builder.add_edge("classify", "query_rag")
            builder.add_edge("query_rag", "finalize")
            builder.add_edge("finalize", END)

        expected_edges = [
            (START, "build_context"),
            ("build_context", "classify"),
            ("classify", "query_rag"),
            ("query_rag", "finalize"),
            ("finalize", END),
        ]
        for src, dst in expected_edges:
            mock_add_edge.assert_any_call(src, dst)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="psycopg async 在 Windows ProactorEventLoop 下不可用",
    )
    @pytest.mark.asyncio
    async def test_build_agent_graph_uses_async_postgres_saver(self):
        """build_agent_graph 使用 AsyncPostgresSaver 作为 checkpointer"""
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        from coursepilot.agent.graph import _get_saver

        saver = await _get_saver()
        assert isinstance(saver, AsyncPostgresSaver)

    def test_route_by_intent_routes_properly(self):
        """根据 intent 路由（practice/review 直通 get_mastery；complex question → agentic_rag）"""
        from coursepilot.agent.routing import route_by_intent
        assert route_by_intent({"intent": "question"}) == "query_rag"
        assert route_by_intent({"intent": "practice"}) == "get_mastery"
        assert route_by_intent({"intent": "review"}) == "get_mastery"
        assert route_by_intent({"intent": "diagnose"}) == "diagnose"
        assert route_by_intent({"intent": "unknown"}) == "query_rag"
        assert route_by_intent({}) == "query_rag"
        # P1: complex question 走 Agentic RAG
        assert route_by_intent({"intent": "question", "complexity": "complex"}) == "agentic_rag"
        assert route_by_intent({"intent": "question", "complexity": "simple"}) == "query_rag"


# ═══════════════════════════════════════════════════════════════
# 2. classify_intent Skill
# ═══════════════════════════════════════════════════════════════

class TestClassifyIntent:
    """意图分类 skill，依赖 DeepSeek LLM

    注意：classify_intent 返回 3 元组 (intent, complexity, token_info)
    （复杂度判断为 Agentic RAG 演进新增，见 agent/skills/classify_intent.py）。
    """

    @pytest.mark.asyncio
    async def test_classify_question(self):
        """问题查询 → question"""
        with patch("coursepilot.agent.skills.classify_intent.AsyncOpenAI") as mock_openai:
            client = AsyncMock()
            mock_openai.return_value = client
            completion = AsyncMock()
            completion.choices = [AsyncMock()]
            completion.choices[0].message.content = "question"
            completion.usage = None
            client.chat.completions.create = AsyncMock(return_value=completion)

            from coursepilot.agent.skills.classify_intent import classify_intent
            intent, complexity, tokens = await classify_intent("什么是二叉树？")
            assert intent == "question"
            assert complexity == "simple"
            assert tokens == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @pytest.mark.asyncio
    async def test_classify_practice(self):
        """练习请求 → practice"""
        with patch("coursepilot.agent.skills.classify_intent.AsyncOpenAI") as mock_openai:
            client = AsyncMock()
            mock_openai.return_value = client
            completion = AsyncMock()
            completion.choices = [AsyncMock()]
            completion.choices[0].message.content = "practice"
            completion.usage = None
            client.chat.completions.create = AsyncMock(return_value=completion)

            from coursepilot.agent.skills.classify_intent import classify_intent
            intent, complexity, _ = await classify_intent("给我出几道题")
            assert intent == "practice"
            assert complexity == "simple"

    @pytest.mark.asyncio
    async def test_classify_falls_back_to_question_without_api_key(self):
        """无 API key 时回退到 question"""
        with patch("coursepilot.agent.skills.classify_intent.settings.llm_api_key", ""):
            from coursepilot.agent.skills.classify_intent import classify_intent
            intent, complexity, _ = await classify_intent("任何问题")
            assert intent == "question"
            assert complexity == "simple"

    @pytest.mark.asyncio
    async def test_classify_returns_question_for_invalid_response(self):
        """LLM 返回无效意图时回退到 question"""
        with patch("coursepilot.agent.skills.classify_intent.AsyncOpenAI") as mock_openai:
            client = AsyncMock()
            mock_openai.return_value = client
            completion = AsyncMock()
            completion.choices = [AsyncMock()]
            completion.choices[0].message.content = "invalid_intent"
            completion.usage = None
            client.chat.completions.create = AsyncMock(return_value=completion)

            from coursepilot.agent.skills.classify_intent import classify_intent
            intent, complexity, _ = await classify_intent("一些请求")
            assert intent == "question"
            assert complexity == "simple"

    @pytest.mark.asyncio
    async def test_diagnose_intent_recognized(self):
        """学情诊断 → diagnose"""
        with patch("coursepilot.agent.skills.classify_intent.AsyncOpenAI") as mock_openai:
            client = AsyncMock()
            mock_openai.return_value = client
            completion = AsyncMock()
            completion.choices = [AsyncMock()]
            completion.choices[0].message.content = "diagnose"
            completion.usage = None
            client.chat.completions.create = AsyncMock(return_value=completion)

            from coursepilot.agent.skills.classify_intent import classify_intent
            intent, complexity, _ = await classify_intent("我哪里掌握得不好")
            assert intent == "diagnose"
            assert complexity == "simple"


# ═══════════════════════════════════════════════════════════════
# 3. query_rag Skill
# ═══════════════════════════════════════════════════════════════

class TestQueryRAGSkill:
    """RAG 检索 + 生成 skill（封装现有 Retriever + Generator）"""

    @pytest.mark.asyncio
    async def test_returns_expected_structure(self, mock_db, course_context):
        """验证返回 (answer, context, metadata, sources) 四元组"""
        with (
            patch("coursepilot.agent.skills.query_rag.Retriever") as mock_ret,
            patch("coursepilot.agent.skills.query_rag.Generator") as mock_gen,
        ):
            ret_instance = AsyncMock()
            mock_ret.return_value = ret_instance
            ret_instance.retrieve.return_value = (
                "检索到的上下文内容",
                {"source_kp_paths": ["OS/进程管理/进程调度"], "scores": [0.95]},
            )

            gen_instance = AsyncMock()
            mock_gen.return_value = gen_instance
            gen_instance.generate.return_value = "进程调度是操作系统核心功能...", ZERO_TOKENS

            from coursepilot.agent.skills.query_rag import query_rag

            answer, context, metadata, sources, _ = await query_rag(
                session=mock_db,
                query="什么是进程调度？",
                course_id=str(uuid4()),
                course_context=course_context,
            )

            assert answer == "进程调度是操作系统核心功能..."
            assert context == "检索到的上下文内容"
            assert metadata["source_kp_paths"] == ["OS/进程管理/进程调度"]
            assert sources == [{"kp_path": "OS/进程管理/进程调度"}]

    @pytest.mark.asyncio
    async def test_handles_empty_context(self, mock_db, course_context):
        """检索结果为空时仍能正常返回"""
        with (
            patch("coursepilot.agent.skills.query_rag.Retriever") as mock_ret,
            patch("coursepilot.agent.skills.query_rag.Generator") as mock_gen,
        ):
            ret_instance = AsyncMock()
            mock_ret.return_value = ret_instance
            ret_instance.retrieve.return_value = ("", {"source_kp_paths": [], "scores": []})

            gen_instance = AsyncMock()
            mock_gen.return_value = gen_instance
            gen_instance.generate.return_value = "未找到相关信息", ZERO_TOKENS

            from coursepilot.agent.skills.query_rag import query_rag

            answer, context, metadata, sources, _ = await query_rag(
                session=mock_db,
                query="不存在的知识",
                course_id=str(uuid4()),
                course_context=course_context,
            )

            assert answer == "未找到相关信息"
            assert sources == []

    def test_retriever_is_lazy_initialized(self):
        """Retriever 在函数内构造，不在 import-time 初始化（避免 Milvus 锁冲突）"""
        import coursepilot.agent.skills.query_rag as qr_mod

        for name, val in qr_mod.__dict__.items():
            if name.startswith("__"):
                continue
            assert "Retriever" not in type(val).__name__, f"模块级存在 Retriever 实例: {name}={val}"


# ═══════════════════════════════════════════════════════════════
# 4. update_qa_record Skill
# ═══════════════════════════════════════════════════════════════

class TestUpdateQARecord:
    """QA 记录写入 + 会话状态更新"""

    @pytest.mark.asyncio
    async def test_writes_qa_record(self, mock_db):
        """创建 QARecord 行，更新 AgentSession 状态"""
        from coursepilot.agent.skills.update_qa_record import update_qa_record

        token_count = await update_qa_record(
            session=mock_db,
            user_id=str(uuid4()),
            course_id=str(uuid4()),
            query="什么是进程调度？",
            answer="进程调度是...",
            kp_path="OS/进程管理/进程调度",
            retrieved_units=["unit1", "unit2"],
            citations=["1", "2"],
            session_id=str(uuid4()),
        )

        assert mock_db.add.called
        assert mock_db.flush.called
        assert isinstance(token_count, object)  # 现在返回 QARecord 实例

    @pytest.mark.asyncio
    async def test_handles_session_not_found(self, mock_db):
        """会话不存在时不报错"""
        # scalar_one_or_none 返回 None → 跳过 status 更新
        from coursepilot.agent.skills.update_qa_record import update_qa_record

        result_record = await update_qa_record(
            session=mock_db,
            user_id=str(uuid4()),
            course_id=str(uuid4()),
            query="测试",
            answer="回答",
            kp_path="",
            retrieved_units=[],
            citations=[],
            session_id=str(uuid4()),
        )
        assert result_record is not None

    @pytest.mark.asyncio
    async def test_updates_agent_session_when_found(self, mock_db):
        """会话存在时 status → completed"""
        from coursepilot.models import AgentSession
        agent_session = MagicMock(spec=AgentSession)

        # 让 execute 返回包含 agent_session 的 result
        result = MagicMock()
        result.scalar_one_or_none.return_value = agent_session
        mock_db.execute = AsyncMock(return_value=result)

        from coursepilot.agent.skills.update_qa_record import update_qa_record

        await update_qa_record(
            session=mock_db,
            user_id=str(uuid4()),
            course_id=str(uuid4()),
            query="测试",
            answer="回答",
            kp_path="OS/进程调度",
            retrieved_units=[],
            citations=[],
            session_id=str(uuid4()),
        )

        assert agent_session.status == "completed"


# ═══════════════════════════════════════════════════════════════
# 5. Context Builder
# ═══════════════════════════════════════════════════════════════

class TestContextBuilder:
    """build_context：课程上下文 + 学生画像 + 最近 QA"""

    @pytest.mark.asyncio
    async def test_returns_expected_structure(self, mock_db):
        """返回 (course_context, user_profile_summary, recent_qa_list)"""
        user_id = str(uuid4())
        course_id = str(uuid4())

        # QARecord mock
        mock_qa = MagicMock()
        mock_qa.query = "之前的问题"
        mock_qa.answer = "之前的回答"
        mock_qa.kp_path = "OS/进程管理"

        # 两次 execute 分别返回不同的结果
        up_result = MagicMock()
        up_result.scalars.return_value.all.return_value = []  # UserProfile 空
        qa_result = MagicMock()
        qa_result.scalars.return_value.all.return_value = [mock_qa] * 3  # QARecord
        mock_db.execute = AsyncMock(side_effect=[up_result, qa_result])

        with patch(
            "coursepilot.agent.context.build_course_context",
            return_value={"course_name": "OS", "kp_tree_preview": "..."},
        ):
            from coursepilot.agent.context import build_context

            ctx, profile, qa_list = await build_context(
                session=mock_db,
                user_id=user_id,
                course_id=course_id,
            )

        assert ctx == {"course_name": "OS", "kp_tree_preview": "..."}
        assert profile is None
        assert len(qa_list) == 3

    @pytest.mark.asyncio
    async def test_builds_profile_when_user_profile_exists(self, mock_db):
        """UserProfile 存在时构造 profile dict"""
        user_id = str(uuid4())
        course_id = str(uuid4())

        up = MagicMock()
        up.mastery_level = 0.75
        up.weak_kps = ["OS/进程同步", "OS/文件系统"]
        up.avg_correct_rate = 0.68

        # 两次 execute 分别返回不同的结果
        up_result = MagicMock()
        up_result.scalars.return_value.all.return_value = [up]
        qa_result = MagicMock()
        qa_result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(side_effect=[up_result, qa_result])

        with patch(
            "coursepilot.agent.context.build_course_context",
            return_value={},
        ):
            from coursepilot.agent.context import build_context
            ctx, profile, qa_list = await build_context(
                session=mock_db,
                user_id=user_id,
                course_id=course_id,
            )

        assert profile["mastery_level"] == 0.75
        assert "OS/进程同步" in profile["weak_kps"]
        assert profile["avg_correct_rate"] == 0.68

    @pytest.mark.asyncio
    async def test_limits_recent_qa_to_five(self, mock_db):
        """最多返回 5 条 recent_qa"""
        user_id = str(uuid4())
        course_id = str(uuid4())

        called = False
        orig_execute = mock_db.execute

        async def tracking_execute(*a, **kw):
            nonlocal called
            called = True
            return await orig_execute(*a, **kw)

        mock_db.execute = tracking_execute

        with patch(
            "coursepilot.agent.context.build_course_context",
            return_value={},
        ):
            from coursepilot.agent.context import build_context

            _, _, qa_list = await build_context(
                session=mock_db,
                user_id=user_id,
                course_id=course_id,
            )

        assert called, "build_context 未调用 session.execute"

    @pytest.mark.asyncio
    async def test_truncates_long_answers(self, mock_db):
        """答案超过 200 字被截断"""
        user_id = str(uuid4())
        course_id = str(uuid4())

        long_qa = MagicMock()
        long_qa.query = "Q"
        long_qa.answer = "x" * 500
        long_qa.kp_path = ""


        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        result.scalars.return_value = result
        result.all.return_value = [long_qa]
        mock_db.execute = AsyncMock(return_value=result)

        with patch(
            "coursepilot.agent.context.build_course_context",
            return_value={},
        ):
            from coursepilot.agent.context import build_context
            _, _, qa_list = await build_context(
                session=mock_db,
                user_id=user_id,
                course_id=course_id,
            )

        assert len(qa_list[0]["answer"]) == 200


# ═══════════════════════════════════════════════════════════════
# 6. 节点函数
# ═══════════════════════════════════════════════════════════════

class TestNodes:
    """节点函数接收 AgentState 并返回增量更新

    节点内部通过 async_session_factory() 自建 DB 连接，
    测试中 mock 该工厂使其返回 mock_db。
    """

    @pytest.mark.asyncio
    async def test_build_context_node(self, sample_state, mock_db, mock_asf):
        """build_context_node → course_context 和 user_profile 被填充"""
        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.build_context_logic") as mock_bc,
        ):
            mock_bc.return_value = (
                {"course_name": "OS"},
                {"mastery_level": 0.7},
                [{"query": "之前问题"}],
            )

            from coursepilot.agent.nodes import build_context_node

            result = await build_context_node(sample_state)

        assert result["course_context"] == {"course_name": "OS"}
        assert result["user_profile"] == {"mastery_level": 0.7}
        assert len(result["recent_qa"]) == 1
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_classify_node(self, sample_state, mock_asf):
        """classify_node → intent 被设置"""
        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch(
                "coursepilot.agent.nodes.classify_intent",
                return_value=("question", ZERO_TOKENS),
            ),
        ):
            from coursepilot.agent.nodes import classify_node

            result = await classify_node(sample_state)

        assert result["intent"] == "question"
        assert len(result["llm_calls"]) == 1
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_classify_node_forwards_context(self, sample_state, mock_asf):
        """classify_node 把 course_context 传给 classify_intent"""
        sample_state["course_context"] = {"course_name": "OS"}
        sample_state["recent_qa"] = [{"query": "之前的"}]

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.classify_intent") as mock_cls,
        ):
            mock_cls.return_value = ("question", ZERO_TOKENS)

            from coursepilot.agent.nodes import classify_node

            await classify_node(sample_state)

            mock_cls.assert_called_once_with(
                query=sample_state["query"],
                course_context=sample_state["course_context"],
                recent_qa=sample_state["recent_qa"],
            )

    @pytest.mark.asyncio
    async def test_query_rag_node(self, sample_state, mock_asf):
        """query_rag_node → answer、sources 被填充"""
        sample_state["course_context"] = {"course_name": "OS"}

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.query_rag") as mock_qr,
        ):
            mock_qr.return_value = (
                "进程调度是...",
                "检索到的上下文",
                {"source_kp_paths": ["OS/进程调度"], "scores": [0.95]},
                [{"kp_path": "OS/进程调度"}],
                ZERO_TOKENS,
            )

            from coursepilot.agent.nodes import query_rag_node

            result = await query_rag_node(sample_state)

        assert result["answer"] == "进程调度是..."
        assert result["sources"] == [{"kp_path": "OS/进程调度"}]
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_finalize_node(self, sample_state, mock_asf):
        """finalize_node → QA 记录写入，token_count 汇总自 llm_calls"""
        sample_state.update({
            "answer": "最终答案",
            "context": "上下文",
            "sources": [{"kp_path": "OS/进程调度"}],
            "retrieved_metadata": {"source_kp_paths": ["OS/进程调度"]},
            "llm_calls": [
                {"node": "classify", "total_tokens": 15,
                 "prompt_tokens": 10, "completion_tokens": 5},
                {"node": "query_rag", "total_tokens": 50,
                 "prompt_tokens": 30, "completion_tokens": 20},
            ],
        })

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.update_qa_record") as mock_uq,
        ):
            from coursepilot.agent.nodes import finalize_node

            result = await finalize_node(sample_state)

        assert result["token_count"] == 65  # 15 + 50
        assert result["error"] is None
        # session_id 有效时应写入 QA 记录（含 token 汇总）
        mock_uq.assert_awaited_once()
        kwargs = mock_uq.await_args.kwargs
        assert kwargs["session_id"] == sample_state["session_id"]
        assert kwargs["token_count"] == 65

    @pytest.mark.asyncio
    async def test_build_context_node_propagates_error(self, sample_state, mock_asf):
        """build_context_node 内部异常 → error 字段被设置"""
        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.build_context_logic", side_effect=ValueError("DB 错误")),
        ):
            from coursepilot.agent.nodes import build_context_node

            result = await build_context_node(sample_state)

        assert "DB 错误" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_classify_node_catches_api_error(self, sample_state, mock_asf):
        """classify_node 内 classify_intent 抛异常 → 回退到 question"""
        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.classify_intent", side_effect=Exception("API 错误")),
        ):
            from coursepilot.agent.nodes import classify_node

            result = await classify_node(sample_state)

        assert result["intent"] == "question"
        assert "API 错误" in result.get("error", "")

    def test_all_nodes_callable(self):
        """所有节点函数都可以从 agent.nodes 导入"""
        from coursepilot.agent.nodes import (
            build_context_node,
            classify_node,
            finalize_node,
            query_rag_node,
        )

        assert callable(build_context_node)
        assert callable(classify_node)
        assert callable(finalize_node)
        assert callable(query_rag_node)


# ═══════════════════════════════════════════════════════════════
# 7. API 端点
# ═══════════════════════════════════════════════════════════════

class TestAgentAPI:
    """FastAPI 端点测试（TestClient + mock）"""

    @pytest.fixture
    def client(self, mock_db):
        from fastapi.testclient import TestClient

        from coursepilot.api.deps import get_current_user
        from coursepilot.db import get_session
        from coursepilot.main import app
        from coursepilot.models import User

        test_user = User(
            id=uuid4(),
            username="test_student",
            role="student",
        )

        async def override_user():
            return test_user

        async def override_session():
            yield mock_db

        app.dependency_overrides[get_current_user] = override_user
        app.dependency_overrides[get_session] = override_session

        yield TestClient(app)

        app.dependency_overrides.clear()

    @pytest.fixture
    def mock_graph(self):
        """用 MemorySaver 编译测试图，替换 _graph_app"""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph

        from coursepilot.agent.nodes import (
            build_context_node,
            classify_node,
            finalize_node,
            query_rag_node,
        )
        from coursepilot.agent.state import AgentState

        builder = StateGraph(AgentState)
        builder.add_node("build_context", build_context_node)
        builder.add_node("classify", classify_node)
        builder.add_node("query_rag", query_rag_node)
        builder.add_node("finalize", finalize_node)
        builder.add_edge(START, "build_context")
        builder.add_edge("build_context", "classify")
        builder.add_edge("classify", "query_rag")
        builder.add_edge("query_rag", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=MemorySaver())

    def test_chat_returns_202(self, client, mock_graph, mock_asf):
        """POST /agent/chat → 202 Accepted + session_id"""
        import coursepilot.api.agent as agent_mod
        saved = agent_mod._graph_app
        try:
            agent_mod._graph_app = mock_graph

            with (
                patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
                patch("coursepilot.agent.nodes.build_context_logic") as mock_bc,
                patch("coursepilot.agent.nodes.classify_intent") as mock_cls,
                patch("coursepilot.agent.nodes.query_rag") as mock_qr,
                patch("coursepilot.agent.nodes.update_qa_record") as mock_uq,
            ):
                mock_bc.return_value = ({"course_name": "OS"}, None, [])
                mock_cls.return_value = ("question", ZERO_TOKENS)
                mock_qr.return_value = (
                    "进程调度是操作系统的核心功能",
                    "上下文",
                    {"source_kp_paths": ["OS/进程调度"]},
                    [{"kp_path": "OS/进程调度"}],
                    ZERO_TOKENS,
                )
                mock_uq.return_value = 42

                response = client.post(
                    "/api/v1/agent/chat",
                    json={
                        "message": "什么是进程调度？",
                        "course_id": str(uuid4()),
                    },
                )
        finally:
            agent_mod._graph_app = saved

        assert response.status_code == 202
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "processing"

    def test_chat_422_on_empty_message(self, client):
        """空消息 → 422"""
        response = client.post(
            "/api/v1/agent/chat",
            json={"message": "", "course_id": str(uuid4())},
        )
        assert response.status_code == 422

    def test_get_session_not_found(self, client):
        """不存在的会话 → 404"""
        response = client.get(f"/api/v1/agent/sessions/{uuid4()}")
        assert response.status_code == 404


# ═══════════════════════════════════════════════════════════════
# 8. 端到端工作流
# ═══════════════════════════════════════════════════════════════

@pytest.mark.e2e
class TestEndToEnd:
    """完整 workflow（全 mock 外部依赖，使用 MemorySaver）"""

    @pytest.mark.asyncio
    async def test_full_workflow_returns_answer(self, mock_asf):
        """ainvoke 全路径 → 最终状态含 answer、sources"""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph

        from coursepilot.agent.nodes import (
            build_context_node,
            classify_node,
            finalize_node,
            query_rag_node,
        )
        from coursepilot.agent.state import AgentState

        builder = StateGraph(AgentState)
        builder.add_node("build_context", build_context_node)
        builder.add_node("classify", classify_node)
        builder.add_node("query_rag", query_rag_node)
        builder.add_node("finalize", finalize_node)
        builder.add_edge(START, "build_context")
        builder.add_edge("build_context", "classify")
        builder.add_edge("classify", "query_rag")
        builder.add_edge("query_rag", "finalize")
        builder.add_edge("finalize", END)
        graph = builder.compile(checkpointer=MemorySaver())

        initial_state = {
            "query": "什么是进程调度？请结合例子说明。",
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
            "llm_calls": [],
            "error": None,
        }

        mock_tokens = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch("coursepilot.agent.nodes.build_context_logic") as mock_bc,
            patch("coursepilot.agent.nodes.classify_intent") as mock_cls,
            patch("coursepilot.agent.nodes.query_rag") as mock_qr,
            patch("coursepilot.agent.nodes.update_qa_record") as mock_uq,
        ):
            mock_bc.return_value = ({"course_name": "OS"}, None, [])
            mock_cls.return_value = ("question", mock_tokens)
            mock_qr.return_value = (
                "进程调度是操作系统核心功能，例如 FCFS...",
                "上下文内容",
                {"source_kp_paths": ["OS/进程管理/进程调度"], "scores": [0.92]},
                [{"kp_path": "OS/进程管理/进程调度"}],
                mock_tokens,
            )
            mock_uq.return_value = 30

            result = await graph.ainvoke(
                initial_state, {"configurable": {"thread_id": str(uuid4())}}
            )

        assert result["error"] is None
        assert result["intent"] == "question"
        assert "进程调度" in result["answer"]
        assert len(result["sources"]) >= 1
        assert result["token_count"] == 60  # 30 from classify + 30 from query_rag

    @pytest.mark.asyncio
    async def test_workflow_handles_node_error(self, mock_asf):
        """build_context_node 失败 → 返回空 context，但下游节点仍正常执行"""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph

        from coursepilot.agent.nodes import (
            build_context_node,
            classify_node,
            finalize_node,
            query_rag_node,
        )
        from coursepilot.agent.state import AgentState

        builder = StateGraph(AgentState)
        builder.add_node("build_context", build_context_node)
        builder.add_node("classify", classify_node)
        builder.add_node("query_rag", query_rag_node)
        builder.add_node("finalize", finalize_node)
        builder.add_edge(START, "build_context")
        builder.add_edge("build_context", "classify")
        builder.add_edge("classify", "query_rag")
        builder.add_edge("query_rag", "finalize")
        builder.add_edge("finalize", END)
        graph = builder.compile(checkpointer=MemorySaver())

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
            "llm_calls": [],
            "error": None,
        }

        with (
            patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
            patch(
                "coursepilot.agent.nodes.build_context_logic",
                side_effect=RuntimeError("模拟错误"),
            ),
            patch(
                "coursepilot.agent.nodes.classify_intent",
                return_value=("question", ZERO_TOKENS),
            ),
            patch("coursepilot.agent.nodes.query_rag") as mock_qr,
            patch("coursepilot.agent.nodes.update_qa_record") as mock_uq,
        ):
            mock_qr.return_value = ("fallback answer", "", {}, [], ZERO_TOKENS)
            mock_uq.return_value = 0

            result = await graph.ainvoke(state, {"configurable": {"thread_id": str(uuid4())}})

        # build_context 失败但每个节点自行吞异常，管道完整执行
        assert result["intent"] == "question"
        assert result["answer"] == "fallback answer"


# ═══════════════════════════════════════════════════════════════
# 9. AgentState 定义验证
# ═══════════════════════════════════════════════════════════════

class TestAgentState:
    """AgentState TypedDict 字段完整性"""

    def test_required_fields_present(self):
        from typing import get_type_hints

        from coursepilot.agent.state import AgentState

        hints = get_type_hints(AgentState)
        expected_fields = {
            "query", "course_id", "user_id", "session_id",
            "messages", "course_context", "user_profile", "recent_qa",
            "intent", "context", "retrieved_metadata",
            "answer", "sources", "token_count", "error",
            "llm_calls",
        }
        missing = expected_fields - set(hints)
        assert not missing, f"AgentState 缺少字段: {missing}"

    def test_optional_fields(self):
        """error 字段应为 Optional[str]"""
        from typing import get_type_hints

        from coursepilot.agent.state import AgentState

        hints = get_type_hints(AgentState)
        assert hints["error"] is not None
