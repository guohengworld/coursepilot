"""编排层重构 · 行为快照测试（characterization tests）

目标：在重构前锁定"现有可观察行为"，重构期间每一步都必须保持这些测试全绿。
对应的重构方案步骤 0（锁行为）：docs/agent/编排层重构方案.md。

覆盖三个层次：
  1. 路由映射快照（纯函数）：route_by_intent / route_after_rag / route_after_evaluate /
     route_after_review 的输入 → 输出映射，锁定 intent×complexity 组合行为，
     显式锁定 none 意图"静默走 query_rag"的现状（未定义行为先锁住，重构后再决定改法）。
  2. 真实图拓扑快照：build_agent_graph() 编译后的自定义节点集合（13 个，含路由兜底
     fallback 节点——flag 关闭时无入边、不可达，但常驻注册）。
  3. E2E 行为快照：真实图 + MemorySaver + 全 mock 外部依赖，
     question / none / diagnose 三条完整路径的"节点执行序列 + 输出"，
     practice / review 锁到 human_review interrupt（审批前行为，含 resume 后完整路径）。

运行：
    .venv/Scripts/python -m pytest tests/unit/test_orchestration_behavior.py -v

注意：
    - 使用真实 build_agent_graph()（patch _get_saver → MemorySaver），
      与 tests/unit/test_agent_phase1.py 的自建简化图不同，锁的是生产图行为。
    - 不改动任何生产代码；测试全部 mock 外部依赖（DB / LLM / 后台任务）。
"""
from __future__ import annotations

import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

ZERO_TOKENS = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
MOCK_TOKENS = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


# ═══════════════════════════════════════════════════════════════
# Shared Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_db():
    """异步 DB 会话 mock（同 test_agent_phase1 模式）"""
    session = AsyncMock(spec=["execute", "add", "flush", "commit", "scalar"])
    session.commit = AsyncMock()
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
def base_state():
    """与 api/agent.py chat 新会话分支一致的初始 state"""
    return {
        "query": "什么是进程调度？",
        "course_id": str(uuid4()),
        "user_id": str(uuid4()),
        "session_id": str(uuid4()),
        "messages": [],
        "conversation": [],
        "rolling_summary": "",
        "course_context": {},
        "user_profile": None,
        "recent_qa": [],
        "intent": "",
        "complexity": "simple",
        "context": "",
        "retrieved_metadata": {},
        "answer": "",
        "sources": [],
        "token_count": 0,
        "llm_calls": [],
        "context_budget": None,
        "layer_tokens": None,
        "cache_hit_estimated": None,
        "compaction_count": 0,
        "error": None,
    }


@pytest.fixture
async def real_graph():
    """真实生产图：build_agent_graph()（async）+ MemorySaver"""
    from langgraph.checkpoint.memory import MemorySaver

    with patch("coursepilot.agent.graph._get_saver", return_value=MemorySaver()):
        from coursepilot.agent.graph import build_agent_graph

        return await build_agent_graph()


def _node_sequence(events: list[dict]) -> list[str]:
    """从 astream(stream_mode="updates") 的 chunk 中提取节点执行序列。

    过滤 __interrupt__ 伪节点（langgraph 内部机制，非业务节点），
    只保留真实业务节点的执行顺序。
    """
    return [
        name for chunk in events
        for name in chunk if not name.startswith("__")
    ]


@contextmanager
def _mock_external_deps(mock_asf, extra_patches=(), **overrides):
    """统一 patch 外部依赖：DB 会话、LLM、后台副作用任务。

    与 test_agent_phase1 的 E2E 模式一致，额外 patch 掉 finalize_node 触发的
    后台任务（audit / profile / L3 抽取 / QA embedding），保证测试快而稳。
    overrides 可替换 classify_intent / query_rag 等默认 mock；
    extra_patches 追加额外的 patch（如 diagnose / get_mastery）。
    """
    default_classify = AsyncMock(return_value=("question", "simple", MOCK_TOKENS))
    default_query_rag = AsyncMock(return_value=(
        "进程调度是操作系统的核心功能",
        "检索到的上下文",
        {"source_kp_paths": ["OS/进程管理/进程调度"], "scores": [0.95]},
        [{"kp_path": "OS/进程管理/进程调度"}],
        MOCK_TOKENS,
    ))

    patches = [
        patch("coursepilot.agent.nodes.async_session_factory", return_value=mock_asf),
        patch("coursepilot.agent.nodes.build_context_logic", AsyncMock(return_value=(
            {"name": "操作系统", "chapters": []}, None, [],
        ))),
        patch("coursepilot.agent.nodes.classify_intent",
              overrides.get("classify_intent", default_classify)),
        patch("coursepilot.agent.nodes.query_rag",
              overrides.get("query_rag", default_query_rag)),
        patch("coursepilot.agent.nodes.update_qa_record", AsyncMock(return_value=None)),
        # finalize 后台任务（audit / profile / L3 / embedding）全部隔离
        patch("coursepilot.governance.audit.async_session_factory", return_value=mock_asf),
        patch("coursepilot.agent.profile_updater.async_session_factory", return_value=mock_asf),
        patch("coursepilot.agent.memory.extract_facts_for_session", AsyncMock()),
        patch("coursepilot.agent.nodes.ensure_qa_embeddings_for_user_course", AsyncMock()),
        # 路由依赖的 RAG 开关：显式控制，避免受环境配置影响
        patch("coursepilot.rag.config.config.enable_routing", True),
    ]
    patches.extend(extra_patches)

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


# ═══════════════════════════════════════════════════════════════
# 1. 路由映射快照（纯函数）
# ═══════════════════════════════════════════════════════════════

class TestRoutingSnapshot:
    """route_by_intent 等 4 个路由函数的输入 → 输出快照"""

    def test_route_by_intent_snapshot(self):
        """route_by_intent 全组合映射（含 none 意图现状）"""
        from coursepilot.agent.routing import route_by_intent

        # question + simple → 快速通道
        assert route_by_intent({"intent": "question", "complexity": "simple"}) == "query_rag"
        # question + complex + 路由开关 → Agentic RAG
        with patch("coursepilot.rag.config.config.enable_routing", True):
            assert route_by_intent({"intent": "question", "complexity": "complex"}) == "agentic_rag"
        # question + complex + 开关关闭 → 回退快速通道
        with patch("coursepilot.rag.config.config.enable_routing", False):
            assert route_by_intent({"intent": "question", "complexity": "complex"}) == "query_rag"
        # practice / review → 生成前人工审批（锁盲审位置）
        assert route_by_intent({"intent": "practice"}) == "human_review"
        assert route_by_intent({"intent": "review"}) == "human_review"
        # diagnose → 直接诊断（只读，无需审批）
        assert route_by_intent({"intent": "diagnose"}) == "diagnose"
        # none 意图：当前静默走 query_rag（锁现状，重构方案阶段 4 再决定显式兜底）
        assert route_by_intent({"intent": "none"}) == "query_rag"
        # 未知 / 缺失 intent → 默认 query_rag
        assert route_by_intent({"intent": "unknown"}) == "query_rag"
        assert route_by_intent({}) == "query_rag"

    def test_route_by_intent_fallback_flag_on(self):
        """flag orch_route_fallback=True：none / "" / 未知值 / classify 降级 统一收口 fallback"""
        from coursepilot.agent.routing import route_by_intent
        from coursepilot.config import settings

        with patch.object(settings, "orch_route_fallback", True):
            # none → fallback（显式分支，不依赖 VALID_INTENTS 是否含 none）
            assert route_by_intent({"intent": "none"}) == "fallback"
            # ""（UNCLASSIFIED）/ 缺失 / 未知值（∉ VALID_INTENTS）→ fallback
            assert route_by_intent({"intent": ""}) == "fallback"
            assert route_by_intent({}) == "fallback"
            assert route_by_intent({"intent": "unknown"}) == "fallback"
            # classify 异常降级 → fallback（先于 intent 判断：降级时 intent 被写成 question）
            assert route_by_intent({
                "intent": "question", "classify_degraded": True,
            }) == "fallback"
            # 正常意图不受 flag 影响
            assert route_by_intent({"intent": "question", "complexity": "simple"}) == "query_rag"
            assert route_by_intent({"intent": "practice"}) == "human_review"
            assert route_by_intent({"intent": "diagnose"}) == "diagnose"

    def test_route_by_intent_fallback_off_classify_degraded_ignored(self):
        """flag 关闭时 classify_degraded 不影响路由（锁现状，降级静默走问答）"""
        from coursepilot.agent.routing import route_by_intent

        # 默认 flag=False：即使带 classify_degraded 标记，仍按 intent=question 走问答
        assert route_by_intent({
            "intent": "question", "classify_degraded": True,
        }) == "query_rag"

    def test_route_after_rag_snapshot(self):
        """query_rag / agentic_rag 后：practice/review 继续出题，其余收口"""
        from coursepilot.agent.routing import route_after_rag

        assert route_after_rag({"intent": "question"}) == "finalize"
        assert route_after_rag({"intent": "diagnose"}) == "finalize"
        assert route_after_rag({"intent": "practice"}) == "generate_quiz"
        assert route_after_rag({"intent": "review"}) == "generate_quiz"
        assert route_after_rag({}) == "finalize"

    def test_route_after_evaluate_snapshot(self):
        """evaluate 后：FAIL 重试（<3 次）、PASS 分支 practice/review、兜底 finalize"""
        from coursepilot.agent.routing import route_after_evaluate

        # FAIL + retry < 3 → 回 generate_quiz 重试
        assert route_after_evaluate({
            "eval_result": {"status": "FAIL"}, "retry_count": 0, "intent": "practice",
        }) == "generate_quiz"
        # FAIL + retry = 3（已达上限）→ 按 intent 走终点
        assert route_after_evaluate({
            "eval_result": {"status": "FAIL"}, "retry_count": 3, "intent": "review",
        }) == "review_plan"
        # PASS + review → review_plan
        assert route_after_evaluate({
            "eval_result": {"status": "PASS"}, "retry_count": 0, "intent": "review",
        }) == "review_plan"
        # PASS + practice → create_plan
        assert route_after_evaluate({
            "eval_result": {"status": "PASS"}, "retry_count": 0, "intent": "practice",
        }) == "create_plan"
        # 其他 → finalize
        assert route_after_evaluate({
            "eval_result": {"status": "PASS"}, "retry_count": 0, "intent": "question",
        }) == "finalize"

    def test_route_after_review_snapshot(self):
        """审批后：rejected → 收口；approved + practice/review → get_mastery"""
        from coursepilot.agent.routing import route_after_review

        assert route_after_review({
            "human_review_result": "rejected", "intent": "practice",
        }) == "finalize"
        assert route_after_review({
            "human_review_result": "approved", "intent": "practice",
        }) == "get_mastery"
        assert route_after_review({
            "human_review_result": "approved", "intent": "review",
        }) == "get_mastery"
        assert route_after_review({
            "human_review_result": "approved", "intent": "question",
        }) == "finalize"


# ═══════════════════════════════════════════════════════════════
# 2. 真实图拓扑快照
# ═══════════════════════════════════════════════════════════════

class TestGraphTopologySnapshot:
    """build_agent_graph() 编译后的节点集合快照"""

    @pytest.mark.asyncio
    async def test_real_graph_custom_nodes(self, real_graph):
        """真实图包含 13 个自定义节点（含 agentic_rag / human_review / fallback）"""
        custom_nodes = {n for n in real_graph.nodes if not str(n).startswith("__")}
        expected = {
            "build_context", "classify", "query_rag", "get_mastery",
            "generate_quiz", "evaluate_quiz", "create_plan", "diagnose",
            "review_plan", "finalize", "human_review", "agentic_rag",
            "fallback",
        }
        assert custom_nodes == expected

    @pytest.mark.asyncio
    async def test_real_graph_is_compiled_with_checkpointer(self, real_graph):
        """编译图必须带 checkpointer（生产使用 AsyncPostgresSaver，测试用 MemorySaver）"""
        assert getattr(real_graph, "checkpointer", None) is not None


# ═══════════════════════════════════════════════════════════════
# 3. E2E 行为快照（真实图 + 全 mock）
# ═══════════════════════════════════════════════════════════════

class TestE2EBehaviorSnapshot:
    """真实图端到端：节点执行序列 + 输出快照"""

    @pytest.mark.asyncio
    async def test_question_path(self, real_graph, base_state, mock_asf):
        """question+simple → build_context → classify → query_rag → finalize"""
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        events = []
        with _mock_external_deps(mock_asf):
            async for chunk in real_graph.astream(base_state, config, stream_mode="updates"):
                events.append(chunk)
            final = await real_graph.aget_state(config)

        sequence = _node_sequence(events)
        assert sequence == ["build_context", "classify", "query_rag", "finalize"]
        values = final.values
        assert values["intent"] == "question"
        assert "进程调度" in values["answer"]
        assert values["sources"] == [{"kp_path": "OS/进程管理/进程调度"}]
        assert values["token_count"] == 60  # classify 30 + query_rag 30
        assert values["error"] is None

    @pytest.mark.asyncio
    async def test_none_intent_path(self, real_graph, base_state, mock_asf):
        """none 意图当前静默走 query_rag（锁现状）"""
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        events = []
        classify = AsyncMock(return_value=("none", "simple", MOCK_TOKENS))
        with _mock_external_deps(mock_asf, classify_intent=classify):
            async for chunk in real_graph.astream(base_state, config, stream_mode="updates"):
                events.append(chunk)
            final = await real_graph.aget_state(config)

        sequence = _node_sequence(events)
        assert sequence == ["build_context", "classify", "query_rag", "finalize"]
        assert final.values["intent"] == "none"
        assert final.values["error"] is None

    @pytest.mark.asyncio
    async def test_none_intent_fallback_when_flag_on(self, real_graph, base_state, mock_asf):
        """flag on + none → fallback 收口（区别于 flag 关闭的静默 query_rag）"""
        from coursepilot.config import settings

        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        classify = AsyncMock(return_value=("none", "simple", MOCK_TOKENS))

        with patch.object(settings, "orch_route_fallback", True), \
                _mock_external_deps(mock_asf, classify_intent=classify):
            events = []
            async for chunk in real_graph.astream(base_state, config, stream_mode="updates"):
                events.append(chunk)
            final = await real_graph.aget_state(config)

        sequence = _node_sequence(events)
        assert sequence == ["build_context", "classify", "fallback", "finalize"]
        assert final.values["intent"] == "none"           # fallback_node 收敛
        assert final.values["fallback_reason"] == "none"
        assert "重新描述" in final.values["answer"]
        assert final.values["error"] is None

    @pytest.mark.asyncio
    async def test_unknown_intent_fallback_when_flag_on(self, real_graph, base_state, mock_asf):
        """flag on + 未知 intent（∉ VALID_INTENTS）→ fallback 收口"""
        from coursepilot.config import settings

        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        classify = AsyncMock(return_value=("do_my_homework", "simple", MOCK_TOKENS))

        with patch.object(settings, "orch_route_fallback", True), \
                _mock_external_deps(mock_asf, classify_intent=classify):
            events = []
            async for chunk in real_graph.astream(base_state, config, stream_mode="updates"):
                events.append(chunk)
            final = await real_graph.aget_state(config)

        sequence = _node_sequence(events)
        assert sequence == ["build_context", "classify", "fallback", "finalize"]
        assert final.values["fallback_reason"] == "unclassified"
        assert final.values["intent"] == "none"
        assert "重新描述" in final.values["answer"]
        assert final.values["error"] is None

    @pytest.mark.asyncio
    async def test_classify_degraded_fallback_when_flag_on(self, real_graph, base_state, mock_asf):
        """flag on + classify 节点异常 → classify_degraded=True → fallback 收口"""
        from coursepilot.config import settings

        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        classify = AsyncMock(side_effect=RuntimeError("classify 上游超时"))

        with patch.object(settings, "orch_route_fallback", True), \
                _mock_external_deps(mock_asf, classify_intent=classify):
            events = []
            async for chunk in real_graph.astream(base_state, config, stream_mode="updates"):
                events.append(chunk)
            final = await real_graph.aget_state(config)

        sequence = _node_sequence(events)
        assert sequence == ["build_context", "classify", "fallback", "finalize"]
        assert final.values["fallback_reason"] == "classify_degraded"
        assert final.values["intent"] == "none"
        assert "重新描述" in final.values["answer"]
        assert final.values["error"] is None

    @pytest.mark.asyncio
    async def test_diagnose_path_no_practice(self, real_graph, base_state, mock_asf):
        """diagnose：无练习记录时返回提示，路径为 diagnose → finalize"""
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        events = []
        classify = AsyncMock(return_value=("diagnose", "simple", MOCK_TOKENS))
        with _mock_external_deps(mock_asf, classify_intent=classify, extra_patches=[
            patch("coursepilot.agent.nodes.diagnose", AsyncMock(return_value={
                "summary": "学情概览",
                "total_practiced": 0,
                "overall_rate": 0.0,
                "kp_stats": {},
                "weak_kps": [],
            })),
        ]):
            async for chunk in real_graph.astream(base_state, config, stream_mode="updates"):
                events.append(chunk)
            final = await real_graph.aget_state(config)

        sequence = _node_sequence(events)
        assert sequence == ["build_context", "classify", "diagnose", "finalize"]
        assert "暂无练习记录" in final.values["answer"]
        assert final.values["error"] is None

    @pytest.mark.asyncio
    async def test_practice_path_stops_at_human_review(self, real_graph, base_state, mock_asf):
        """practice：学生请求在生成前被 human_review 拦截（锁盲审位置）"""
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        events = []
        classify = AsyncMock(return_value=("practice", "simple", MOCK_TOKENS))
        with _mock_external_deps(mock_asf, classify_intent=classify):
            async for chunk in real_graph.astream(base_state, config, stream_mode="updates"):
                events.append(chunk)
            snapshot = await real_graph.aget_state(config)

        sequence = _node_sequence(events)
        # 审批前：build_context → classify 后即挂起。
        # human_review 节点执行到 interrupt() 时被中断、不产出 update，
        # "停在 human_review"由下方 snapshot.next + interrupt payload 证明。
        assert sequence == ["build_context", "classify"]
        assert snapshot.next, "应因 interrupt 挂起在 human_review"
        # interrupt payload 携带 intent 与原始 query（前端展示用）
        interrupt_payload = snapshot.interrupts[0].value if snapshot.interrupts else None
        assert interrupt_payload is not None
        assert interrupt_payload["type"] == "human_review"
        assert interrupt_payload["intent"] == "practice"
        assert interrupt_payload["query"] == base_state["query"]

    @pytest.mark.asyncio
    async def test_practice_path_resume_after_approval(self, real_graph, base_state, mock_asf):
        """practice：审批通过后走 get_mastery → query_rag → generate_quiz → evaluate_quiz → create_plan → finalize"""
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        classify = AsyncMock(return_value=("practice", "simple", MOCK_TOKENS))
        get_mastery = AsyncMock(return_value={"mastery_level": {}, "weak_kps": []})
        generate_quiz = AsyncMock(return_value=({
            "questions": [
                {"question_text": "进程调度的目的是？",
                 "options": {"A": "提高吞吐", "B": "降低延迟"},
                 "kp_path": "OS/进程管理/进程调度"},
            ],
        }, MOCK_TOKENS))
        evaluate_quiz = AsyncMock(return_value=({"status": "PASS", "score": 0.9}, MOCK_TOKENS))

        with _mock_external_deps(
            mock_asf, classify_intent=classify, extra_patches=[
                patch("coursepilot.agent.nodes.get_mastery", get_mastery),
                patch("coursepilot.agent.nodes.generate_quiz", generate_quiz),
                patch("coursepilot.agent.nodes.evaluate_quiz", evaluate_quiz),
            ],
        ):
            # 第一次执行：停在 human_review
            async for _ in real_graph.astream(base_state, config, stream_mode="updates"):
                pass
            snapshot = await real_graph.aget_state(config)
            assert snapshot.next, "应挂起在 human_review"

            # 审批通过 → 继续执行
            events = []
            async for chunk in real_graph.astream(
                Command(resume={"approved": True}), config, stream_mode="updates"
            ):
                events.append(chunk)
            final = await real_graph.aget_state(config)

        sequence = _node_sequence(events)
        # resume 后 human_review 重跑消费 resume 值，然后继续主路径
        assert sequence == [
            "human_review", "get_mastery", "query_rag", "generate_quiz",
            "evaluate_quiz", "create_plan", "finalize",
        ]
        assert final.values["intent"] == "practice"
        assert "为你生成了 1 道练习题" in final.values["answer"]
        assert final.values["human_review_result"] == "approved"
        assert final.values["error"] is None

    @pytest.mark.asyncio
    async def test_review_path_stops_at_human_review(self, real_graph, base_state, mock_asf):
        """review：与 practice 一致，生成前被 human_review 拦截"""
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        events = []
        classify = AsyncMock(return_value=("review", "simple", MOCK_TOKENS))
        with _mock_external_deps(mock_asf, classify_intent=classify):
            async for chunk in real_graph.astream(base_state, config, stream_mode="updates"):
                events.append(chunk)
            snapshot = await real_graph.aget_state(config)

        sequence = _node_sequence(events)
        # 与 practice 一致：classify 后挂起，human_review 不产出 update，
        # 挂起位置由 snapshot.next + interrupt payload 证明。
        assert sequence == ["build_context", "classify"]
        assert snapshot.next, "应因 interrupt 挂起在 human_review"
        interrupt_payload = snapshot.interrupts[0].value if snapshot.interrupts else None
        assert interrupt_payload is not None
        assert interrupt_payload["intent"] == "review"


# ═══════════════════════════════════════════════════════════════
# 4. 状态分层契约（InputState / OutputState / AgentState）
# ═══════════════════════════════════════════════════════════════

class TestStateSchemaContract:
    """三层 schema 的运行时行为契约

    input_schema / output_schema 不是静态类型提示，而是在图的入口与出口
    真实生效的过滤（未声明的入参被静默丢弃、返回值只保留声明字段）。
    这里锁定这三条边界，避免后续改动无声改变它们。
    """

    def test_agent_state_superset_of_input_and_output(self):
        """类型层：AgentState 是 InputState + OutputState 的超集"""
        from typing import get_type_hints

        from coursepilot.agent.state import AgentState, InputState, OutputState

        hints = set(get_type_hints(AgentState))
        assert set(get_type_hints(InputState)) <= hints
        assert set(get_type_hints(OutputState)) <= hints
        # 中间字段只存在于内部状态，不进输入/输出契约
        for field in ("mastery", "quiz_data", "eval_result", "diagnosis", "review_plan"):
            assert field in hints

    @pytest.mark.asyncio
    async def test_output_scoped_to_output_schema(self, real_graph, base_state, mock_asf):
        """出口：ainvoke 返回值只含 OutputState 声明的字段"""
        from typing import get_type_hints

        from coursepilot.agent.state import OutputState

        config = {"configurable": {"thread_id": str(uuid4())}}

        with _mock_external_deps(mock_asf):
            result = await real_graph.ainvoke(base_state, config)

        output_fields = set(get_type_hints(OutputState))
        assert set(result) <= output_fields, f"图输出越界: {set(result) - output_fields}"
        # 产物字段必须在场
        assert {"answer", "sources", "token_count", "intent"} <= set(result)
        # 内部字段不得外泄
        assert "retrieved_metadata" not in result
        assert "context" not in result

    @pytest.mark.asyncio
    async def test_input_schema_drops_undeclared_keys(self, real_graph, base_state, mock_asf):
        """入口：未在 InputState 声明的 key 被静默丢弃（不报错）"""
        injected = {**base_state, "mastery": {"injected": True}}
        config = {"configurable": {"thread_id": str(uuid4())}}

        with _mock_external_deps(mock_asf):
            await real_graph.ainvoke(injected, config)
            final = await real_graph.aget_state(config)

        # question 路径不调用 get_mastery：若 mastery 出现，说明入口未做过滤
        assert "mastery" not in final.values

    @pytest.mark.asyncio
    async def test_checkpoint_keeps_internal_fields(self, real_graph, base_state, mock_asf):
        """持久化：checkpoint 保留 AgentState 全量，不受 output_schema 影响"""
        config = {"configurable": {"thread_id": str(uuid4())}}

        with _mock_external_deps(mock_asf):
            await real_graph.ainvoke(base_state, config)
            final = await real_graph.aget_state(config)

        for field in ("course_context", "context", "retrieved_metadata", "llm_calls"):
            assert field in final.values, f"内部字段 {field} 未进入 checkpoint"
