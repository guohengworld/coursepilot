"""Agentic RAG（P0.3/P0.4）单元测试。

覆盖范围：
    - EvidenceRegistry：多轮检索 ref_id 全局唯一、citation_map 合并、截断一致性
    - Guardrails：步数 / web 次数 / 重复 query / token 预算 / 参数校验
    - build_agent_messages：system 注入课程名与画像、历史与当前 query
    - 循环骨架：mock LLM 的 tool_calls 序列 → messages 结构 / tool_call_id 匹配 / 收敛 / forced_stop
    - 工具执行器：mock 底层技能，验证参数透传与返回格式
    - finalize_answer：闭环契约 6 字段

运行方式：
    .venv/Scripts/python -m pytest tests/unit/test_agent_rag.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import coursepilot.agent.rag_agent as rag_agent  # noqa: E402
from coursepilot.agent.rag_agent import (  # noqa: E402
    EvidenceRegistry,
    Guardrails,
    build_agent_messages,
    dispatch_tool,
)
from coursepilot.rag.config import config as rag_config  # noqa: E402

UID = "00000000-0000-0000-0000-000000000000"


def make_state(**overrides) -> dict:
    """构造 agentic_rag_node 的最小 state。"""
    state = {
        "query": "导数的定义是什么",
        "course_id": UID,
        "user_id": UID,
        "session_id": UID,
        "course_context": {"name": "高等数学", "chapters": ["第一章 函数与极限"]},
        "user_profile": {"level": "beginner"},
        "conversation": [],
        "rolling_summary": "",
        "llm_calls": [],
        "agent_steps": [],
        "tool_history": [],
    }
    state.update(overrides)
    return state


def make_tool_call(cid: str, name: str, arguments: str):
    """构造 OpenAI tool_call mock。"""
    call = MagicMock()
    call.id = cid
    call.type = "function"
    call.function.name = name
    call.function.arguments = arguments
    return call


def make_response(tool_calls=None, content=None):
    """构造 chat.completions.create 的响应 mock。

    message 用 SimpleNamespace 模拟 OpenAI 的 ChatCompletionMessage 对象：
    agentic_rag_node 把 message 对象原样 append 进 messages（协议要求回传），
    断言时对 assistant 消息用属性访问、对 tool 消息用 dict 访问。
    """
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    resp = MagicMock()
    resp.usage = usage
    resp.choices = [SimpleNamespace(message=message)]
    return resp


# ═══════════════════════════════════════════════════════════════
# P0.5.1 EvidenceRegistry
# ═══════════════════════════════════════════════════════════════

class TestEvidenceRegistry:
    """多轮检索后 ref_id 全局唯一、citation_map 正确（方案 §10 验证点 #6）。"""

    CTX1 = '<source id="1" path="/第一章/导数定义" pages="3" book="高数教材">导数定义...</source>'
    CTX2 = '<source id="1" path="/第二章/定积分" pages="10" book="高数教材">定积分定义...</source>'

    def test_register_assigns_global_unique_ref_ids(self):
        reg = EvidenceRegistry()
        reg.register(self.CTX1, {"citation_map": {"1": {"uuid": "u1", "kp_path": "/第一章/导数定义", "page_ref": "3"}}})
        reg.register(self.CTX2, {"citation_map": {"1": {"uuid": "u2", "kp_path": "/第二章/定积分", "page_ref": "10"}}})

        # 两轮检索的局部 id 都是 "1"，重写后必须是不同全局 id
        merged = reg.merged_context()
        assert '<source id="1"' in merged
        assert '<source id="2"' in merged
        # 两个全局 id 的引用映射都在
        assert set(reg.merged_citation_map()) == {"1", "2"}
        assert reg.merged_citation_map()["1"]["uuid"] == "u1"
        assert reg.merged_citation_map()["2"]["uuid"] == "u2"

    def test_register_falls_back_to_path_attr(self):
        """metadata 缺省（如 web 结果）时从 <source> path 属性回退取 kp_path。"""
        reg = EvidenceRegistry()
        reg.register('<source id="1" path="网络搜索/Sogou" pages="" book="网络搜索结果">内容</source>', {})
        cmap = reg.merged_citation_map()
        assert set(cmap) == {"1"}
        assert cmap["1"]["kp_path"] == "网络搜索/Sogou"

    def test_register_multi_block_keeps_index_sync(self):
        """一次 register 含多个 <source> 块时，_blocks 与 _block_ref_ids 必须一一对应，
        否则 _build_merged 会索引越界（回归：IndexError: list index out of range）。"""
        ctx = (
            '<source id="1" path="/第一章/导数定义" pages="3" book="高数教材">导数定义...</source>\n'
            '<source id="2" path="/第一章/极限" pages="5" book="高数教材">极限定义...</source>\n'
            '<source id="3" path="/第一章/连续" pages="7" book="高数教材">连续定义...</source>'
        )
        reg = EvidenceRegistry()
        reg.register(ctx, {})
        merged = reg.merged_context()  # 不应抛 IndexError
        assert '<source id="1"' in merged
        assert '<source id="2"' in merged
        assert '<source id="3"' in merged
        assert set(reg.merged_citation_map()) == {"1", "2", "3"}
        # 后续 register 仍保持不变量
        reg.register('<source id="1" path="/第二章/定积分" pages="10" book="高数教材">定积分定义...</source>', {})
        assert set(reg.merged_citation_map()) == {"1", "2", "3", "4"}

    def test_register_empty_context_noop(self):
        reg = EvidenceRegistry()
        reg.register("", {})
        reg.register("   \n", {})
        assert reg.merged_context() == ""
        assert reg.merged_citation_map() == {}

    def test_merged_truncation_keeps_xml_valid_and_syncs_map(self):
        """截断按整块进行：被截掉块的引用不会出现在 citation_map。"""
        with patch.object(rag_config, "context_max_chars", 30):
            reg = EvidenceRegistry()
            reg.register(self.CTX1, {"citation_map": {"1": {"uuid": "u1", "kp_path": "/k1", "page_ref": "3"}}})
            # 第二块加入会超过 30 字符，应整体截掉
            reg.register(self.CTX2, {"citation_map": {"1": {"uuid": "u2", "kp_path": "/k2", "page_ref": "10"}}})
            merged = reg.merged_context()
            cmap = reg.merged_citation_map()
        assert '<source id="1"' in merged
        assert '<source id="2"' not in merged
        assert set(cmap) == {"1"}

    def test_raw_blocks_for_audit(self):
        reg = EvidenceRegistry()
        reg.register(self.CTX1, {})
        reg.register(self.CTX2, {})
        assert len(reg.raw_blocks()) == 2
        assert 'id="1"' in reg.raw_blocks()[0]
        assert 'id="2"' in reg.raw_blocks()[1]


# ═══════════════════════════════════════════════════════════════
# P0.5.2 Guardrails
# ═══════════════════════════════════════════════════════════════

class TestGuardrails:
    """四类触发（步数 / web 次数 / 重复 query / token 预算）+ 参数校验。"""

    def test_step_limit_forced_stop(self):
        guard = Guardrails(max_steps=2, max_web_searches=2, token_budget=1000)
        assert guard.check() is False   # 第 1 步
        assert guard.check() is False   # 第 2 步
        assert guard.check() is True    # 第 3 步超限 → 强制停止

    def test_web_search_limit(self):
        guard = Guardrails(max_steps=10, max_web_searches=2, token_budget=1000)
        assert guard.before_tool("web_search", {"query": "q1"}, []) is None       # 第 1 次放行
        assert guard.before_tool("web_search", {"query": "q2"}, []) is None       # 第 2 次放行
        msg = guard.before_tool("web_search", {"query": "q3"}, [])                # 第 3 次拒绝
        assert msg is not None and "上限" in msg

    def test_duplicate_query_rejected(self):
        guard = Guardrails(max_steps=10, max_web_searches=2, token_budget=1000)
        history = [
            {"tool": "search_textbook", "args": {"query": "导数"}, "result_summary": "a"},
            {"tool": "search_textbook", "args": {"query": "导数"}, "result_summary": "b"},
        ]
        msg = guard.before_tool("search_textbook", {"query": "导数"}, history)
        assert msg is not None and "已" in msg and "改写" in msg
        # 不同的 query 不受影响
        assert guard.before_tool("search_textbook", {"query": "极限"}, history) is None

    def test_token_budget_forced_stop(self):
        guard = Guardrails(max_steps=10, max_web_searches=2, token_budget=100)
        guard.accrue_tokens(60)
        assert guard.check() is False
        guard.accrue_tokens(60)   # 累计 120 > 100
        assert guard.check() is True

    def test_missing_required_args_rejected(self):
        guard = Guardrails(max_steps=10, max_web_searches=2, token_budget=1000)
        msg = guard.before_tool("search_textbook", {}, [])
        assert msg is not None and "缺少必填参数" in msg
        msg2 = guard.before_tool("plan", {"query": "q"}, [])
        assert msg2 is not None and "sub_questions" in msg2
        assert guard.before_tool("search_textbook", {"query": "q"}, []) is None

    def test_unknown_tool_no_required_check(self):
        guard = Guardrails(max_steps=10, max_web_searches=2, token_budget=1000)
        assert guard.before_tool("unknown_tool", {}, []) is None


# ═══════════════════════════════════════════════════════════════
# P0.3.5 build_agent_messages
# ═══════════════════════════════════════════════════════════════

class TestBuildAgentMessages:
    """system 注入课程上下文与画像，历史轮次与当前 query 正确。"""

    def test_system_contains_course_and_profile(self):
        messages = build_agent_messages(make_state())
        system = messages[0]["content"]
        assert "高等数学" in system
        assert "第一章 函数与极限" in system
        assert "学生画像" in system

    def test_last_message_is_current_query(self):
        messages = build_agent_messages(make_state(query="当前问题"))
        assert messages[-1] == {"role": "user", "content": "当前问题"}

    def test_conversation_turns_injected(self):
        conv = [
            {"role": "user", "content": "之前的提问"},
            {"role": "assistant", "content": "之前的回答"},
        ]
        messages = build_agent_messages(make_state(conversation=conv))
        contents = [m["content"] for m in messages]
        assert "之前的提问" in contents
        assert "之前的回答" in contents

    def test_duplicate_trailing_user_removed(self):
        conv = [
            {"role": "user", "content": "问题A"},
            {"role": "assistant", "content": "答A"},
            {"role": "user", "content": "导数的定义是什么"},
        ]
        messages = build_agent_messages(make_state(conversation=conv))
        # 末尾重复的 user 轮应被去掉，只保留一次当前 query
        assert sum(1 for m in messages if m == {"role": "user", "content": "导数的定义是什么"}) == 1


# ═══════════════════════════════════════════════════════════════
# P0.5.3 循环骨架（mock LLM）
# ═══════════════════════════════════════════════════════════════

class TestAgenticRagLoop:
    """mock LLM 返回固定 tool_calls 序列 → 验证循环行为。"""

    @patch.object(rag_agent.settings, "llm_api_key", "test-key")
    @patch.object(rag_config, "agent_max_steps", 8)
    @patch.object(rag_agent, "dispatch_tool", new_callable=AsyncMock, return_value="工具结果")
    @patch.object(rag_agent, "finalize_answer", new_callable=AsyncMock)
    async def test_converges_when_no_more_tool_calls(
        self, mock_finalize, mock_dispatch, *_args,
    ):
        """LLM 先调一次工具，随后不再返回 tool_calls → 收敛停止，非降级。"""
        mock_finalize.return_value = {"answer": "最终答案"}
        responses = [
            make_response(tool_calls=[
                make_tool_call("call_1", "search_textbook", '{"query": "导数"}'),
            ]),
            make_response(content="证据足够，直接回答。"),
        ]
        with patch.object(
            rag_agent.AsyncOpenAI, "chat", create=True,
        ) as mock_chat, patch("coursepilot.agent.rag_agent.AsyncOpenAI") as mock_cls:
            client = mock_cls.return_value
            client.chat.completions.create = AsyncMock(side_effect=responses)

            result = await rag_agent.agentic_rag_node(make_state())

        # 收敛：只调用了 2 次 LLM
        assert client.chat.completions.create.await_count == 2
        # 工具被分发一次，参数透传（dispatch_tool(tool_name, args, state, evidence) 全位置参数）
        mock_dispatch.assert_awaited_once()
        args, kwargs = mock_dispatch.await_args
        assert kwargs == {}
        assert args[0] == "search_textbook"
        assert args[1] == {"query": "导数"}
        # 收敛 → 非降级
        mock_finalize.assert_awaited_once()
        assert mock_finalize.await_args.kwargs["degraded"] is False
        assert result["answer"] == "最终答案"

    @patch.object(rag_agent.settings, "llm_api_key", "test-key")
    @patch.object(rag_config, "agent_max_steps", 2)
    @patch.object(rag_agent, "dispatch_tool", new_callable=AsyncMock, return_value="工具结果")
    @patch.object(rag_agent, "finalize_answer", new_callable=AsyncMock)
    async def test_forced_stop_sets_degraded(
        self, mock_finalize, mock_dispatch, *_args,
    ):
        """LLM 永远返回同一个 tool_call → 步数超限强制停止 → degraded_mode=True。"""
        mock_finalize.return_value = {"answer": "降级答案", "degraded_mode": True}
        responses = [
            make_response(tool_calls=[
                make_tool_call("call_1", "search_textbook", '{"query": "导数"}'),
            ]),
            make_response(tool_calls=[
                make_tool_call("call_2", "search_textbook", '{"query": "导数"}'),
            ]),
        ]
        with patch("coursepilot.agent.rag_agent.AsyncOpenAI") as mock_cls:
            client = mock_cls.return_value
            client.chat.completions.create = AsyncMock(side_effect=responses)

            await rag_agent.agentic_rag_node(make_state())

        mock_finalize.assert_awaited_once()
        assert mock_finalize.await_args.kwargs["degraded"] is True

    @patch.object(rag_agent.settings, "llm_api_key", "test-key")
    @patch.object(rag_config, "agent_max_steps", 8)
    @patch.object(rag_agent, "dispatch_tool", new_callable=AsyncMock, return_value="工具结果")
    @patch.object(rag_agent, "finalize_answer", new_callable=AsyncMock)
    async def test_tool_call_id_matches_tool_role_message(
        self, mock_finalize, mock_dispatch, *_args,
    ):
        """assistant.tool_calls 与 role="tool" 结果成对回传，tool_call_id 必须匹配。"""
        mock_finalize.return_value = {"answer": "答案"}
        responses = [
            make_response(tool_calls=[
                make_tool_call("call_abc", "web_search", '{"query": "拉格朗日"}'),
            ]),
            make_response(content="好了。"),
        ]
        with patch("coursepilot.agent.rag_agent.AsyncOpenAI") as mock_cls:
            client = mock_cls.return_value
            client.chat.completions.create = AsyncMock(side_effect=responses)

            await rag_agent.agentic_rag_node(make_state())

        # 第二次 LLM 调用时，messages 中 assistant tool_calls 与 tool 结果成对
        second_messages = client.chat.completions.create.await_args_list[1].kwargs["messages"]
        # assistant 消息是对象（SimpleNamespace），用属性访问验证 tool_calls 被回传
        assistant_msgs = [m for m in second_messages
                          if isinstance(m, SimpleNamespace) and m.tool_calls]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].tool_calls[0].id == "call_abc"
        # tool 消息是 dict，tool_call_id 必须与 assistant 的 call id 匹配
        tool_msgs = [m for m in second_messages if isinstance(m, dict) and m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_abc"

    @patch.object(rag_agent.settings, "llm_api_key", "test-key")
    @patch.object(rag_config, "agent_max_steps", 8)
    @patch.object(rag_agent, "dispatch_tool", new_callable=AsyncMock, return_value="工具结果")
    @patch.object(rag_agent, "finalize_answer", new_callable=AsyncMock)
    async def test_guardrail_rejection_injected_to_llm(
        self, mock_finalize, mock_dispatch, *_args,
    ):
        """参数缺失被 Guardrails 拒绝时，拒绝文本以 tool 角色回传，且不执行工具。"""
        mock_finalize.return_value = {"answer": "答案"}
        responses = [
            make_response(tool_calls=[
                make_tool_call("call_1", "search_textbook", '{"top_k": 3}'),  # 缺 query
            ]),
            make_response(content="好的。"),
        ]
        with patch("coursepilot.agent.rag_agent.AsyncOpenAI") as mock_cls:
            client = mock_cls.return_value
            client.chat.completions.create = AsyncMock(side_effect=responses)

            await rag_agent.agentic_rag_node(make_state())

        # 参数缺失 → 工具不执行，拒绝文本回传
        mock_dispatch.assert_not_awaited()
        second_messages = client.chat.completions.create.await_args_list[1].kwargs["messages"]
        tool_msg = [m for m in second_messages if isinstance(m, dict) and m.get("role") == "tool"][0]
        assert "缺少必填参数" in tool_msg["content"]

    @patch.object(rag_agent.settings, "llm_api_key", "test-key")
    @patch.object(rag_config, "agent_max_steps", 8)
    @patch.object(rag_agent, "dispatch_tool", new_callable=AsyncMock)
    @patch.object(rag_agent, "finalize_answer", new_callable=AsyncMock)
    async def test_llm_error_returns_error_state(
        self, mock_finalize, mock_dispatch, *_args,
    ):
        """LLM 调用抛异常 → 显式返回 error（不静默）。"""
        with patch("coursepilot.agent.rag_agent.AsyncOpenAI") as mock_cls:
            client = mock_cls.return_value
            client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

            result = await rag_agent.agentic_rag_node(make_state())

        assert result.get("error") is not None
        assert "boom" in result["error"]
        mock_finalize.assert_not_awaited()

    def test_missing_api_key_returns_error(self):
        async def run():
            with patch.object(rag_agent.settings, "llm_api_key", ""):
                return await rag_agent.agentic_rag_node(make_state())
        result = asyncio_run(run())
        assert result.get("error") is not None

    def test_unknown_tool_error_returned(self):
        """dispatch_tool 对未知工具返回错误文本给 LLM。"""

        async def run():
            return await dispatch_tool("no_such_tool", {}, make_state(), EvidenceRegistry())

        result = asyncio_run(run())
        assert "未知工具" in result


# ═══════════════════════════════════════════════════════════════
# P0.5.4 工具执行器
# ═══════════════════════════════════════════════════════════════

def asyncio_run(coro):
    import asyncio
    return asyncio.new_event_loop().run_until_complete(coro)


class TestToolExecutors:
    """mock 底层技能，验证参数透传与返回格式。"""

    def test_search_textbook_wraps_retriever(self):
        ctx = '<source id="1" path="/k1" pages="3" book="b">导数定义</source>'
        metadata = {"citation_map": {"1": {"uuid": "u1", "kp_path": "/k1", "page_ref": "3"}}}

        async def run():
            with patch.object(rag_config, "rerank_top_k", 5):
                with patch("coursepilot.agent.rag_agent.async_session_factory") as mock_asf, \
                        patch("coursepilot.rag.retriever.Retriever") as MockRetriever:
                    session_cm = AsyncMock()
                    session_cm.__aenter__.return_value = MagicMock()
                    mock_asf.return_value = session_cm
                    retriever = MockRetriever.return_value
                    retriever.retrieve = AsyncMock(return_value=(ctx, metadata))

                    evidence = EvidenceRegistry()
                    result = await rag_agent._tool_search_textbook(
                        {"query": "导数", "top_k": 3}, make_state(), evidence,
                    )

            # 返回文本与证据登记
            assert "search_textbook 结果" in result
            assert evidence.merged_citation_map()["1"]["uuid"] == "u1"
            # 临时覆盖的 rerank_top_k 已恢复
            assert rag_config.rerank_top_k == 5
            return True

        assert asyncio_run(run())

    def test_plan_wraps_decompose_query(self):
        sub = [
            {"id": 1, "query": "什么是极限", "target_concept": "极限", "reason": "基础"},
            {"id": 2, "query": "极限的计算", "target_concept": "极限计算", "reason": "应用"},
        ]

        async def fake_retrieve(query, course_id, top_k=None):
            return (f'<source id="1" path="/k1" pages="3" book="b">{query}</source>',
                    {"citation_map": {"1": {"uuid": "u1", "kp_path": f"/{query}", "page_ref": "3"}}})

        async def run():
            with patch("coursepilot.agent.rag_agent.decompose_query",
                       new_callable=AsyncMock,
                       return_value={"sub_queries": sub, "decomposition_type": "sequential"}), \
                    patch("coursepilot.agent.rag_agent._retrieve_textbook", new=fake_retrieve):
                result = await rag_agent._tool_plan({"query": "q"}, make_state(), EvidenceRegistry())
            assert "plan 分解结果" in result
            assert "什么是极限" in result
            return True

        assert asyncio_run(run())

    def test_plan_parallel_retrieves_sub_queries(self):
        """plan 分解后子问题异步并行检索，全部收束后按顺序汇总并登记证据。"""
        sub = [
            {"id": 1, "query": "什么是极限", "target_concept": "极限", "reason": "基础"},
            {"id": 2, "query": "极限的计算", "target_concept": "极限计算", "reason": "应用"},
        ]

        async def fake_retrieve(query, course_id, top_k=None):
            return (f'<source id="1" path="/k1" pages="3" book="b">{query}</source>',
                    {"citation_map": {"1": {"uuid": "u1", "kp_path": f"/{query}", "page_ref": "3"}}})

        async def run():
            with patch("coursepilot.agent.rag_agent.decompose_query",
                       new_callable=AsyncMock,
                       return_value={"sub_queries": sub, "decomposition_type": "sequential"}), \
                    patch("coursepilot.agent.rag_agent._retrieve_textbook", new=fake_retrieve):
                evidence = EvidenceRegistry()
                result = await rag_agent._tool_plan({"query": "q"}, make_state(), evidence)

            # 收束：两个子问题都检索完成，结果按子问题顺序汇总
            assert "已并行检索" in result
            assert result.index("什么是极限") < result.index("极限的计算")
            assert "[子问题 1 检索结果]" in result
            assert "[子问题 2 检索结果]" in result
            # 两个子问题的证据都已登记（局部 id 重写为全局 1/2）
            assert set(evidence.merged_citation_map()) == {"1", "2"}
            return True

        assert asyncio_run(run())

    def test_plan_parallel_isolation_on_subquery_failure(self):
        """单个子问题检索失败/为空不阻塞其余子问题（收束后整体返回）。"""
        sub = [
            {"id": 1, "query": "A概念", "target_concept": "A", "reason": ""},
            {"id": 2, "query": "B概念", "target_concept": "B", "reason": ""},
        ]

        async def fake_retrieve(query, course_id, top_k=None):
            if query == "A概念":
                raise RuntimeError("检索管线异常")
            return "", {}

        async def run():
            with patch("coursepilot.agent.rag_agent.decompose_query",
                       new_callable=AsyncMock,
                       return_value={"sub_queries": sub, "decomposition_type": "compare"}), \
                    patch("coursepilot.agent.rag_agent._retrieve_textbook", new=fake_retrieve):
                evidence = EvidenceRegistry()
                result = await rag_agent._tool_plan({"query": "q"}, make_state(), evidence)

            assert "[子问题 1 检索失败]：检索管线异常" in result
            assert "[子问题 2 检索结果]：未检索到相关内容" in result
            # 空结果不登记证据
            assert evidence.merged_citation_map() == {}
            return True

        assert asyncio_run(run())

    def test_plan_prefers_llm_sub_questions(self):
        """LLM 传入 sub_questions 时优先使用，不再走 decompose_query。"""
        llm_sub = [
            {"question": "问题A", "target_concept": "A", "reason": ""},
            {"question": "问题B", "target_concept": "B", "reason": ""},
        ]

        async def fake_retrieve(query, course_id, top_k=None):
            return (f'<source id="1" path="/k1" pages="3" book="b">{query}</source>',
                    {"citation_map": {"1": {"uuid": "u1", "kp_path": f"/{query}", "page_ref": "3"}}})

        async def run():
            with patch("coursepilot.agent.rag_agent.decompose_query",
                       new_callable=AsyncMock) as mock_decompose, \
                    patch("coursepilot.agent.rag_agent._retrieve_textbook", new=fake_retrieve):
                evidence = EvidenceRegistry()
                result = await rag_agent._tool_plan(
                    {"query": "q", "sub_questions": llm_sub}, make_state(), evidence,
                )

            mock_decompose.assert_not_awaited()
            assert "问题A" in result
            assert "问题B" in result
            assert set(evidence.merged_citation_map()) == {"1", "2"}
            return True

        assert asyncio_run(run())

    def test_plan_empty_subqueries_hint(self):
        async def run():
            with patch("coursepilot.agent.rag_agent.decompose_query",
                       new_callable=AsyncMock,
                       return_value={"sub_queries": [], "decomposition_type": "single"}):
                result = await rag_agent._tool_plan({"query": "q"}, make_state(), EvidenceRegistry())
            assert "无需分解" in result
            return True

        assert asyncio_run(run())

    def test_web_search_wraps_and_registers(self):
        results = [{"title": "微积分教程", "snippet": "极限定义", "url": "http://x", "source": "Sogou"}]

        async def run():
            with patch("coursepilot.agent.rag_agent.web_search",
                       new_callable=AsyncMock, return_value=results), \
                    patch("coursepilot.agent.rag_agent.format_web_context",
                          return_value='<source id="1" path="网络搜索/Sogou" pages="" book="网络搜索结果">内容</source>'):
                evidence = EvidenceRegistry()
                result = await rag_agent._tool_web_search({"query": "q"}, make_state(), evidence)
            assert "web_search 结果" in result
            assert evidence.merged_citation_map()["1"]["kp_path"] == "网络搜索/Sogou"
            return True

        assert asyncio_run(run())

    def test_web_search_no_results(self):
        async def run():
            with patch("coursepilot.agent.rag_agent.web_search",
                       new_callable=AsyncMock, return_value=[]):
                evidence = EvidenceRegistry()
                result = await rag_agent._tool_web_search({"query": "q"}, make_state(), evidence)
            assert "未找到有效结果" in result
            assert evidence.merged_context() == ""
            return True

        assert asyncio_run(run())

    def test_memory_recall_wraps(self):
        records = [{"qa_id": "qa1", "query": "极限", "answer": "答案是...", "scores": {"score": 0.9}}]

        async def run():
            with patch("coursepilot.agent.rag_agent.async_session_factory") as mock_asf, \
                    patch("coursepilot.agent.memory.retriever.recall_memory_turns",
                          new_callable=AsyncMock, return_value=records):
                session_cm = AsyncMock()
                session_cm.__aenter__.return_value = MagicMock()
                mock_asf.return_value = session_cm
                result = await rag_agent._tool_memory_recall({"query": "极限"}, make_state(), EvidenceRegistry())
            assert "memory_recall 结果" in result
            assert "qa1" in result
            return True

        assert asyncio_run(run())

    def test_evaluate_context_wraps_check_sufficiency(self):
        async def run():
            evidence = EvidenceRegistry()
            evidence.register(
                '<source id="1" path="/k1" pages="3" book="b">内容</source>',
                {"citation_map": {"1": {"uuid": "u1", "kp_path": "/k1", "page_ref": "3"}}},
            )
            with patch("coursepilot.agent.rag_agent.check_sufficiency",
                       new_callable=AsyncMock,
                       return_value={"sufficient": False, "confidence": 0.3,
                                     "missing_info": "缺极限计算", "covered_aspects": [],
                                     "uncovered_aspects": ["极限计算"]}), \
                    patch("coursepilot.agent.rag_agent.evaluate_multidim",
                          new_callable=AsyncMock,
                          return_value={"coverage": 0.5, "consistency": 1.0,
                                        "timeliness": 1.0, "authority": 0.9,
                                        "completeness": 0.4,
                                        "weaknesses": ["缺少计算步骤"]}):
                result = await rag_agent._tool_evaluate_context(
                    {"question": "极限怎么算", "evidence": "已有证据"},
                    make_state(), evidence,
                )
            assert "sufficient=False" in result
            assert "缺极限计算" in result
            # P2.1: 五维评分输出
            assert "多维评分（P2.1）" in result
            assert "覆盖度" in result and "完整性" in result
            assert "缺少计算步骤" in result
            assert "继续检索" in result
            return True

        assert asyncio_run(run())

    def test_evaluate_context_no_evidence_yet(self):
        async def run():
            result = await rag_agent._tool_evaluate_context(
                {"question": "q"}, make_state(), EvidenceRegistry(),
            )
            assert "尚无任何已收集证据" in result
            return True

        assert asyncio_run(run())


# ═══════════════════════════════════════════════════════════════
# P0.3.7 finalize_answer 闭环契约
# ═══════════════════════════════════════════════════════════════

class TestFinalizeAnswer:
    """闭环契约 6 字段：answer / context / sources / citation_map / degraded_mode / llm_calls。"""

    def test_produces_contract_fields(self):
        async def run():
            evidence = EvidenceRegistry()
            evidence.register(
                '<source id="1" path="/k1" pages="3" book="b">导数定义</source>',
                {"citation_map": {"1": {"uuid": "u1", "kp_path": "/k1", "page_ref": "3"}}},
            )
            token_info = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                          "context_budget": {}}
            with patch("coursepilot.rag.generator.Generator") as MockGen:
                gen = MockGen.return_value
                gen.generate = AsyncMock(return_value=("生成答案", dict(token_info)))

                result = await rag_agent.finalize_answer(
                    make_state(),
                    evidence=evidence,
                    degraded=False,
                    llm_calls=[{"node": "agent_step", "total_tokens": 15}],
                    agent_steps=[{"tool": "search_textbook", "args": "{}"}],
                    tool_history=[],
                )

            assert result["answer"] == "生成答案"
            assert '<source id="1"' in result["context"]
            assert result["sources"] == [{"kp_path": "/k1"}]
            assert result["retrieved_metadata"]["citation_map"]["1"]["uuid"] == "u1"
            assert result["degraded_mode"] is False
            assert len(result["llm_calls"]) == 2  # agent_step + agent_finalize
            assert result["llm_calls"][-1]["node"] == "agent_finalize"
            return True

        assert asyncio_run(run())

    def test_degraded_adds_disclaimer_to_context(self):
        async def run():
            evidence = EvidenceRegistry()
            evidence.register('<source id="1" path="/k1" pages="3" book="b">内容</source>', {})
            with patch("coursepilot.rag.generator.Generator") as MockGen:
                gen = MockGen.return_value
                gen.generate = AsyncMock(return_value=("答案", {"total_tokens": 5}))

                await rag_agent.finalize_answer(
                    make_state(),
                    evidence=evidence,
                    degraded=True,
                    llm_calls=[],
                    agent_steps=[],
                    tool_history=[],
                )

            # 降级 → 生成器收到的 context 带免责声明前缀
            gen_context = gen.generate.await_args.kwargs["context"]
            assert gen_context.startswith("[注意]")
            assert rag_agent._DEGRADED_DISCLAIMER in gen_context
            return True

        assert asyncio_run(run())


# ═══════════════════════════════════════════════════════════════
# P2.1 五维评估 evaluate_multidim
# ═══════════════════════════════════════════════════════════════

class TestEvaluateMultidim:
    """五维评分：mock LLM JSON 解析 / 无 key 降级 / 解析失败降级。"""

    def test_parses_five_dimensions(self):
        async def run():
            resp = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                    "coverage": 0.8, "consistency": 1.0, "timeliness": 1.0,
                    "authority": 0.9, "completeness": 0.6,
                    "weaknesses": ["推导步骤缺失"],
                })))],
            )
            with patch.object(rag_agent.settings, "llm_api_key", "key"), \
                    patch("coursepilot.agent.rag_agent.AsyncOpenAI") as mock_cls:
                client = mock_cls.return_value
                client.chat.completions.create = AsyncMock(return_value=resp)
                result = await rag_agent.evaluate_multidim("问题", "证据")
            assert result["coverage"] == 0.8
            assert result["completeness"] == 0.6
            assert result["weaknesses"] == ["推导步骤缺失"]
            return True

        assert asyncio_run(run())

    def test_no_api_key_returns_empty(self):
        async def run():
            with patch.object(rag_agent.settings, "llm_api_key", ""):
                result = await rag_agent.evaluate_multidim("问题", "证据")
            assert result == {}
            return True

        assert asyncio_run(run())

    def test_bad_json_returns_empty(self):
        async def run():
            resp = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="不是JSON"))],
            )
            with patch.object(rag_agent.settings, "llm_api_key", "key"), \
                    patch("coursepilot.agent.rag_agent.AsyncOpenAI") as mock_cls:
                client = mock_cls.return_value
                client.chat.completions.create = AsyncMock(return_value=resp)
                result = await rag_agent.evaluate_multidim("问题", "证据")
            assert result == {}
            return True

        assert asyncio_run(run())


# ═══════════════════════════════════════════════════════════════
# P2.2 早期证据压缩（EvidenceRegistry.summarize_early / 工具 / 自动触发）
# ═══════════════════════════════════════════════════════════════

class TestSummarizeContext:
    """summarize_context 工具 + harness 自动压缩（P2.2）。"""

    CTX1 = '<source id="1" path="/k1" pages="3" book="b">导数定义...</source>'
    CTX2 = '<source id="2" path="/k2" pages="10" book="b">定积分定义...</source>'
    CTX3 = '<source id="3" path="/k3" pages="20" book="b">拉格朗日...</source>'

    def test_summarize_early_replaces_early_blocks(self):
        reg = EvidenceRegistry()
        reg.register(self.CTX1, {})
        reg.register(self.CTX2, {})
        reg.register(self.CTX3, {})

        compressed = reg.summarize_early(2, "早期证据摘要")
        assert compressed == 2
        assert reg.summarized is True
        blocks = reg.raw_blocks()
        assert len(blocks) == 2                 # 摘要块 + 剩余 1 块
        assert "<summary>早期证据摘要：早期证据摘要</summary>" in blocks[0]
        assert 'id="3"' in blocks[1]            # 未被压缩块保留
        # 引用映射只含未被压缩块
        assert set(reg.merged_citation_map()) == {"3"}

    def test_summarize_early_noop_when_already_summarized(self):
        reg = EvidenceRegistry()
        reg.register(self.CTX1, {})
        reg.register(self.CTX2, {})
        reg.summarize_early(1, "摘要")
        assert reg.summarize_early(1, "再次") == 0
        assert len(reg.raw_blocks()) == 2

    def test_can_summarize_requires_two_blocks(self):
        reg = EvidenceRegistry()
        assert reg.can_summarize() is False
        reg.register(self.CTX1, {})
        assert reg.can_summarize() is False
        reg.register(self.CTX2, {})
        assert reg.can_summarize() is True

    def test_tool_summarize_context_success(self):
        async def run():
            evidence = EvidenceRegistry()
            evidence.register(self.CTX1, {})
            evidence.register(self.CTX2, {})
            with patch("coursepilot.agent.rag_agent._summarize_evidence",
                       new_callable=AsyncMock, return_value="压缩后的摘要"):
                result = await rag_agent._tool_summarize_context(
                    {"reason": "证据过长"}, make_state(), evidence,
                )
            assert "已把前 1 块早期证据压缩为摘要" in result
            assert "压缩后的摘要" in result
            assert evidence.summarized is True
            return True

        assert asyncio_run(run())

    def test_tool_summarize_context_no_evidence(self):
        async def run():
            result = await rag_agent._tool_summarize_context(
                {}, make_state(), EvidenceRegistry(),
            )
            assert "无需压缩" in result
            return True

        assert asyncio_run(run())

    def test_auto_summarize_over_threshold(self):
        """token 用掉 ≥50% 预算且证据未压缩 → harness 自动压缩并注入 system 消息。"""
        async def run():
            guard = Guardrails(max_steps=10, max_web_searches=2, token_budget=100)
            guard.accrue_tokens(60)   # 60 >= 50
            evidence = EvidenceRegistry()
            evidence.register(self.CTX1, {})
            evidence.register(self.CTX2, {})
            messages = []
            with patch("coursepilot.agent.rag_agent._summarize_evidence",
                       new_callable=AsyncMock, return_value="摘要"):
                await rag_agent._maybe_auto_summarize(messages, guard, evidence)
            assert evidence.summarized is True
            sys_msg = [m for m in messages if m["role"] == "system"]
            assert len(sys_msg) == 1
            assert "已压缩为摘要" in sys_msg[0]["content"]
            return True

        assert asyncio_run(run())

    def test_auto_summarize_below_threshold(self):
        async def run():
            guard = Guardrails(max_steps=10, max_web_searches=2, token_budget=100)
            guard.accrue_tokens(30)   # 30 < 50
            evidence = EvidenceRegistry()
            evidence.register(self.CTX1, {})
            evidence.register(self.CTX2, {})
            messages = []
            await rag_agent._maybe_auto_summarize(messages, guard, evidence)
            assert evidence.summarized is False
            assert messages == []
            return True

        assert asyncio_run(run())

    def test_auto_summarize_skips_when_no_evidence(self):
        async def run():
            guard = Guardrails(max_steps=10, max_web_searches=2, token_budget=100)
            guard.accrue_tokens(80)
            evidence = EvidenceRegistry()
            messages = []
            await rag_agent._maybe_auto_summarize(messages, guard, evidence)
            assert messages == []
            return True

        assert asyncio_run(run())


# ═══════════════════════════════════════════════════════════════
# P2.3 并行工具执行（多 tool_calls 时 asyncio.gather 并发）
# ═══════════════════════════════════════════════════════════════

class TestParallelToolExecution:
    """一次返回多个 tool_calls → 并发执行、按顺序回传、tool_call_id 匹配。"""

    @patch.object(rag_agent.settings, "llm_api_key", "test-key")
    @patch.object(rag_config, "agent_max_steps", 8)
    @patch.object(rag_agent, "dispatch_tool", new_callable=AsyncMock,
                  side_effect=lambda *a, **k: f"结果:{a[0]}")
    @patch.object(rag_agent, "finalize_answer", new_callable=AsyncMock)
    async def test_parallel_tool_calls_executed_and_matched(
        self, mock_finalize, mock_dispatch, *_args,
    ):
        """2 个 tool_calls 并发执行，tool 消息按原始顺序回传且 tool_call_id 匹配。"""
        mock_finalize.return_value = {"answer": "最终答案"}
        responses = [
            make_response(tool_calls=[
                make_tool_call("call_a", "search_textbook", '{"query": "极限"}'),
                make_tool_call("call_b", "web_search", '{"query": "拉格朗日"}'),
            ]),
            make_response(content="够了。"),
        ]
        with patch("coursepilot.agent.rag_agent.AsyncOpenAI") as mock_cls:
            client = mock_cls.return_value
            client.chat.completions.create = AsyncMock(side_effect=responses)

            result = await rag_agent.agentic_rag_node(make_state())

        # 两个工具都被分发（并发）
        assert mock_dispatch.await_count == 2
        # 按原始顺序回传：call_a 先
        second_messages = client.chat.completions.create.await_args_list[1].kwargs["messages"]
        tool_msgs = [m for m in second_messages if isinstance(m, dict) and m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["call_a", "call_b"]
        assert [m["content"] for m in tool_msgs] == ["结果:search_textbook", "结果:web_search"]
        assert result["answer"] == "最终答案"

    @patch.object(rag_agent.settings, "llm_api_key", "test-key")
    @patch.object(rag_config, "agent_max_steps", 8)
    @patch.object(rag_agent, "dispatch_tool", new_callable=AsyncMock,
                  side_effect=lambda *a, **k: "结果")
    @patch.object(rag_agent, "finalize_answer", new_callable=AsyncMock)
    async def test_partial_rejection_does_not_block_valid_calls(
        self, mock_finalize, mock_dispatch, *_args,
    ):
        """一个调用参数缺失被拒，另一个合法调用仍并发执行并回传。"""
        mock_finalize.return_value = {"answer": "答案"}
        responses = [
            make_response(tool_calls=[
                make_tool_call("call_bad", "search_textbook", '{"top_k": 3}'),  # 缺 query
                make_tool_call("call_ok", "web_search", '{"query": "拉格朗日"}'),
            ]),
            make_response(content="好了。"),
        ]
        with patch("coursepilot.agent.rag_agent.AsyncOpenAI") as mock_cls:
            client = mock_cls.return_value
            client.chat.completions.create = AsyncMock(side_effect=responses)

            await rag_agent.agentic_rag_node(make_state())

        # 合法调用执行 1 次；被拒调用不执行
        assert mock_dispatch.await_count == 1
        assert mock_dispatch.await_args.args[0] == "web_search"
        second_messages = client.chat.completions.create.await_args_list[1].kwargs["messages"]
        tool_msgs = [m for m in second_messages if isinstance(m, dict) and m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["call_bad", "call_ok"]
        assert "缺少必填参数" in tool_msgs[0]["content"]
        assert tool_msgs[1]["content"] == "结果"
