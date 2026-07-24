"""测试上下文记忆管理层（ContextManager + Compactor）。

重点覆盖：
- token 估算不为零且对中文友好
- 预算装配时稳定内容在前、RAG 可被截断
- 滑动窗口保留最近轮次
- 滚动压缩把老轮次转入 summary
"""


from coursepilot.agent.memory import ContextManager, compact_conversation, micro_compact_turn


class TestEstimateTokens:
    def test_empty_is_zero(self):
        assert ContextManager._compact_text("", 100) == ""

    def test_chinese_greater_than_zero(self):
        text = "这是一个中文测试句子，包含一些常用字符。"
        from coursepilot.agent.memory.context_manager import estimate_tokens

        assert estimate_tokens(text) > 0

    def test_english_word_count(self):
        from coursepilot.agent.memory.context_manager import estimate_tokens

        text = "hello world this is a test"
        assert estimate_tokens(text) > 0


class TestContextManager:
    def test_build_view_basic_order(self):
        cm = ContextManager({"total_tokens": 10_000})
        view = cm.build_view(
            node="query_rag",
            system_prompt="系统提示。课程：{course_context}。教材：{sources}",
            course_context={"name": "高数", "textbook": "同济", "chapters": ["极限"]},
            user_profile={"mastery_level": {"1.1": 0.8}, "weak_kps": ["1.2"]},
            conversation=[
                {"role": "user", "content": "什么是导数？"},
                {"role": "assistant", "content": "导数是变化率。"},
            ],
            rolling_summary="之前讨论了极限。",
            current_query="求 f(x)=x^2 的导数",
            rag_context="<ref id=\"1\">导数定义...</ref>",
        )

        # system 在最前
        assert "系统提示" in view.system_prefix
        # 最近轮次保留
        assert len(view.recent_turns) == 2
        # RAG 上下文存在
        assert "导数定义" in view.rag_context
        # 预算报告有字段
        assert view.budget["node"] == "query_rag"

    def test_classify_node_drops_rag(self):
        cm = ContextManager({"total_tokens": 10_000})
        view = cm.build_view(
            node="classify",
            system_prompt="分类：{course_context}",
            course_context={"name": "高数"},
            user_profile=None,
            conversation=[{"role": "user", "content": "讲个笑话"}],
            rolling_summary="",
            current_query="继续",
            rag_context="不应出现",
        )
        assert view.rag_context == ""

    def test_rag_truncated_when_budget_tight(self):
        cm = ContextManager({
            "total_tokens": 2_000,
            "reserved_output": 500,
            "safety_margin": 100,
        })
        long_rag = "教材内容。" * 500
        view = cm.build_view(
            node="query_rag",
            system_prompt="系统。{course_context} {sources}",
            course_context={"name": "高数"},
            user_profile=None,
            conversation=[],
            rolling_summary="",
            current_query="问题",
            rag_context=long_rag,
        )
        assert view.budget["rag_truncated"] is True
        assert "已截断" in view.rag_context

    def test_recent_turns_respect_max(self):
        cm = ContextManager({"total_tokens": 10_000, "max_recent_turns": 2})
        view = cm.build_view(
            node="query_rag",
            system_prompt="系统。{course_context} {sources}",
            course_context={},
            user_profile=None,
            conversation=[
                {"role": "user", "content": "1"},
                {"role": "assistant", "content": "2"},
                {"role": "user", "content": "3"},
                {"role": "assistant", "content": "4"},
            ],
            rolling_summary="",
            current_query="5",
            rag_context="",
        )
        assert len(view.recent_turns) == 2
        assert view.recent_turns[0]["content"] == "3"
        assert view.recent_turns[-1]["content"] == "4"

    def test_latex_not_broken_by_truncation(self):
        text = "推导 $$E=mc^2$$ 的"
        result = ContextManager._compact_text(text, 5)
        assert result.count("$$") % 2 == 0


class TestCompactor:
    def test_micro_compact_preserves_kp_paths(self):
        compact = micro_compact_turn(
            query="什么是牛顿第二定律？",
            answer="F=ma，力等于质量乘以加速度。",
            sources=[{"kp_path": "1.2/3.4"}],
        )
        assert "1.2/3.4" in compact["kp_paths"]
        assert "F=ma" in compact["answer_summary"]

    async def test_compact_conversation_moves_old_turns(self):
        conversation = [
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答", "intent": "question", "query": "第一问"},
            {"role": "user", "content": "第二问"},
            {"role": "assistant", "content": "第二答", "intent": "question", "query": "第二问"},
            {"role": "user", "content": "第三问"},
            {"role": "assistant", "content": "第三答", "intent": "question", "query": "第三问"},
            {"role": "user", "content": "第四问"},
            {"role": "assistant", "content": "第四答", "intent": "question", "query": "第四问"},
        ]
        summary, count = await compact_conversation(conversation, max_summary_tokens=2_000)
        assert count == len(conversation) // 2
        assert "第一问" in summary or "第二问" in summary
        # 最近轮次应保留
        remaining = conversation[count:]
        assert remaining[-1]["content"] == "第四答"

    async def test_compact_conversation_empty(self):
        summary, count = await compact_conversation([], "已有摘要")
        assert summary == "已有摘要"
        assert count == 0
