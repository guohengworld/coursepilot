"""Phase 3 全方位测试：RBAC、Guardrails、Audit、Metrics

覆盖范围：
    - RBAC 权限矩阵（get_role_hierarchy / has_permission / filter_own_resources）
    - Guardrails 护栏（guard_answer / guard_kp_scope / guard_daily_limit / guard_token_limit）
    - Audit 审计日志（log_action 及便捷封装）
    - Metrics 指标聚合（get_course_stats / get_week_kp_summary / get_today_token_usage / get_daily_counts）

运行方式：
    .venv/Scripts/python -m pytest tests/unit/test_phase3.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

# 测试用有效 UUID
UID = "00000000-0000-0000-0000-000000000000"


# ═══════════════════════════════════════════════════════════════
# Shared Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_db():
    """异步 DB 会话 mock（同 Phase 1/2 模式）"""
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


class IterResult:
    """模拟 session.execute() 返回的可迭代结果"""

    def __init__(self, rows: list):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


# ═══════════════════════════════════════════════════════════════
# 1. RBAC 权限矩阵
# ═══════════════════════════════════════════════════════════════

class TestRBAC:
    """RoleHierarchy / has_permission / filter_own_resources"""

    def test_role_hierarchy_values(self):
        from coursepilot.governance.rbac import RoleHierarchy, get_role_hierarchy
        assert get_role_hierarchy("student") == 0
        assert get_role_hierarchy("teacher") == 1
        assert get_role_hierarchy("super") == 2

    def test_role_hierarchy_unknown_defaults_to_student(self):
        from coursepilot.governance.rbac import get_role_hierarchy
        assert get_role_hierarchy("unknown_role") == 0

    # ── has_permission ────────────────────────────────

    def test_student_can_read_course(self):
        from coursepilot.governance.rbac import has_permission
        assert has_permission("student", "course:read") is True

    def test_student_cannot_write_course(self):
        from coursepilot.governance.rbac import has_permission
        assert has_permission("student", "course:write") is False

    def test_teacher_can_write_course(self):
        from coursepilot.governance.rbac import has_permission
        assert has_permission("teacher", "course:write") is True

    def test_teacher_cannot_delete_course(self):
        from coursepilot.governance.rbac import has_permission
        assert has_permission("teacher", "course:delete") is False

    def test_super_can_delete_course(self):
        from coursepilot.governance.rbac import has_permission
        assert has_permission("super", "course:delete") is True

    def test_super_can_read_audit(self):
        from coursepilot.governance.rbac import has_permission
        assert has_permission("super", "audit:view") is True

    def test_student_cannot_read_audit(self):
        from coursepilot.governance.rbac import has_permission
        assert has_permission("student", "audit:view") is False

    def test_unknown_permission_returns_false(self):
        from coursepilot.governance.rbac import has_permission
        assert has_permission("super", "nonexistent:action") is False

    def test_student_can_chat(self):
        from coursepilot.governance.rbac import has_permission
        assert has_permission("student", "agent:chat") is True

    # ── filter_own_resources ─────────────────────────

    def test_student_list_is_own(self):
        from coursepilot.governance.rbac import filter_own_resources
        assert filter_own_resources("student", "agent:session:list") is True

    def test_teacher_list_is_not_own(self):
        from coursepilot.governance.rbac import filter_own_resources
        assert filter_own_resources("teacher", "agent:session:list") is False

    def test_super_list_is_not_own(self):
        from coursepilot.governance.rbac import filter_own_resources
        assert filter_own_resources("super", "agent:session:list") is False

    def test_non_list_permission_not_own(self):
        from coursepilot.governance.rbac import filter_own_resources
        assert filter_own_resources("student", "course:read") is False


# ═══════════════════════════════════════════════════════════════
# 2. Guardrails 护栏
# ═══════════════════════════════════════════════════════════════

class TestGuardrails:
    """guard_answer / guard_kp_scope / guard_daily_limit / guard_token_limit

    注意：_extract_keywords 对中文文本会返回整句（re.findall 不切分中文），
    因此 guard_answer 的关键词匹配仅对英文/混合文本有效。
    以下测试使用英文文本以获得确定性的关键词提取结果。
    """

    # ── guard_answer ─────────────────────────────────

    def test_empty_answer(self):
        from coursepilot.governance.guardrails import guard_answer
        issues = guard_answer("", "some context", [])
        assert "回答为空或过短" in issues

    def test_too_short_answer(self):
        from coursepilot.governance.guardrails import guard_answer
        issues = guard_answer("ok", "some context", [])
        assert "回答为空或过短" in issues

    def test_no_context_keywords_matched(self):
        """回答完全不包含 context 关键词 → 幻觉风险警告"""
        from coursepilot.governance.guardrails import guard_answer
        issues = guard_answer(
            "this answer is completely unrelated",
            "binary tree traversal preorder inorder postorder",
            []
        )
        kw_issues = [i for i in issues if "关键词" in i]
        assert kw_issues, f"应检测到关键词缺失，实际 issues={issues}"

    def test_context_keywords_matched_ok(self):
        from coursepilot.governance.guardrails import guard_answer
        issues = guard_answer(
            "the binary tree traversal uses preorder inorder postorder order",
            "binary tree traversal preorder inorder postorder",
            []
        )
        kw_issues = [i for i in issues if "关键词" in i]
        assert not kw_issues, f"关键词应匹配成功，实际 issues={issues}"

    def test_long_answer_missing_citation(self):
        """长回答（>100 字）缺少 [page:N] 引用标记"""
        from coursepilot.governance.guardrails import guard_answer
        answer = "binary tree is a fundamental data structure. " * 6  # > 100 chars
        assert len(answer) > 100
        issues = guard_answer(answer, "binary tree", [])
        assert "较长回答缺少引用标记" in issues

    def test_answer_with_citation_passes(self):
        from coursepilot.governance.guardrails import guard_answer
        issues = guard_answer(
            "binary tree is a data structure. [page:42]",
            "binary tree",
            []
        )
        assert "较长回答缺少引用标记" not in issues

    def test_direct_answer_detected(self):
        from coursepilot.governance.guardrails import guard_answer
        issues = guard_answer(
            "what is the answer? 答案是 B",
            "some context",
            []
        )
        assert any("直接给答案" in i for i in issues), f"应检测到直接给答案，实际 issues={issues}"

    def test_direct_answer_correct(self):
        from coursepilot.governance.guardrails import guard_answer
        issues = guard_answer(
            "正确答案是 C option is correct",
            "some context",
            []
        )
        assert any("直接给答案" in i for i in issues), f"应检测到直接给答案，实际 issues={issues}"

    def test_choice_pattern_detected(self):
        from coursepilot.governance.guardrails import guard_answer
        issues = guard_answer(
            "based on the analysis above, 选择 A ",
            "some context",
            []
        )
        assert any("直接给答案" in i for i in issues), f"应检测到直接给答案，实际 issues={issues}"

    def test_many_sources_short_answer(self):
        from coursepilot.governance.guardrails import guard_answer
        issues = guard_answer(
            "short answer.",
            "some context",
            [{"kp_path": "a"}, {"kp_path": "b"}, {"kp_path": "c"}, {"kp_path": "d"}]
        )
        assert "引用来源较多但回答过于简短" in issues

    def test_all_checks_pass(self):
        from coursepilot.governance.guardrails import guard_answer
        issues = guard_answer(
            "binary tree preorder traversal visits root then left then right. [page:42]",
            "binary tree preorder traversal",
            [{"kp_path": "tree"}]
        )
        assert issues == [], f"期望全部通过，实际 issues={issues}"

    # ── guard_kp_scope ───────────────────────────────

    def test_kp_all_in_scope(self):
        from coursepilot.governance.guardrails import guard_kp_scope
        course_tree = [{"kp_path": "OS/进程管理"}, {"kp_path": "OS/内存管理"}]
        out = guard_kp_scope(["OS/进程管理"], course_tree)
        assert out == []

    def test_kp_some_out_of_scope(self):
        from coursepilot.governance.guardrails import guard_kp_scope
        course_tree = [{"kp_path": "OS/进程管理"}]
        out = guard_kp_scope(["OS/进程管理", "OS/文件系统"], course_tree)
        assert out == ["OS/文件系统"]

    def test_kp_all_out_of_scope(self):
        from coursepilot.governance.guardrails import guard_kp_scope
        out = guard_kp_scope(["无关/知识点"], [])
        assert out == ["无关/知识点"]

    # ── guard_daily_limit ────────────────────────────

    def test_daily_limit_under(self):
        from coursepilot.governance.guardrails import guard_daily_limit
        assert guard_daily_limit(50, max_daily=200) is None

    def test_daily_limit_exceeded(self):
        from coursepilot.governance.guardrails import guard_daily_limit
        msg = guard_daily_limit(200, max_daily=200)
        assert msg is not None
        assert "每日上限" in msg

    def test_daily_limit_custom_max(self):
        from coursepilot.governance.guardrails import guard_daily_limit
        assert guard_daily_limit(5, max_daily=10) is None
        assert guard_daily_limit(10, max_daily=10) is not None

    # ── guard_token_limit ────────────────────────────

    def test_token_limit_all_pass(self):
        from coursepilot.governance.guardrails import guard_token_limit
        assert guard_token_limit(100, session_max=50000, daily_token=1000, daily_max=500000) is None

    def test_token_limit_session_over(self):
        from coursepilot.governance.guardrails import guard_token_limit
        msg = guard_token_limit(60000, session_max=50000)
        assert msg is not None
        assert "单次会话 token 超限" in msg

    def test_token_limit_daily_over(self):
        from coursepilot.governance.guardrails import guard_token_limit
        msg = guard_token_limit(100, session_max=50000, daily_token=600000, daily_max=500000)
        assert msg is not None
        assert "每日 token 超限" in msg


# ═══════════════════════════════════════════════════════════════
# 3. Audit 审计日志
# ═══════════════════════════════════════════════════════════════

class TestAudit:
    """log_action 及其便捷封装"""

    @pytest.mark.asyncio
    async def test_log_action_adds_audit_log(self, mock_asf):
        """log_action 应创建 AuditLog 并调用 session.add + flush"""
        added_objects = []

        def tracking_add(obj):
            added_objects.append(obj)

        mock_db = await mock_asf.__aenter__()
        mock_db.add.side_effect = tracking_add

        with patch("coursepilot.governance.audit.async_session_factory", return_value=mock_asf):
            from coursepilot.governance.audit import log_action
            await log_action(
                user_id=UID,
                action="test.action",
                resource_type="test",
                resource_id="res-1",
                details={"key": "value"},
                ip_address="192.168.1.1",
            )

        assert len(added_objects) == 1
        entry = added_objects[0]
        assert entry.action == "test.action"
        assert entry.resource_type == "test"
        assert entry.resource_id == "res-1"
        assert entry.details == {"key": "value"}
        mock_db.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_log_action_minimal_fields(self, mock_asf):
        """只传必填字段也能正常工作"""
        added_objects = []

        def tracking_add(obj):
            added_objects.append(obj)

        mock_db = await mock_asf.__aenter__()
        mock_db.add.side_effect = tracking_add

        with patch("coursepilot.governance.audit.async_session_factory", return_value=mock_asf):
            from coursepilot.governance.audit import log_action
            await log_action(user_id=UID, action="minimal.action")

        assert len(added_objects) == 1
        entry = added_objects[0]
        assert entry.action == "minimal.action"
        assert entry.resource_type is None
        assert entry.resource_id is None
        assert entry.details is None
        assert entry.ip_address is None

    @pytest.mark.asyncio
    async def test_log_action_exception_does_not_raise(self, mock_asf):
        """数据库异常应被捕获，不冒泡"""
        mock_db = await mock_asf.__aenter__()
        mock_db.flush.side_effect = Exception("DB 错误")

        with patch("coursepilot.governance.audit.async_session_factory", return_value=mock_asf):
            from coursepilot.governance.audit import log_action
            await log_action(user_id=UID, action="test.exception")

    @pytest.mark.asyncio
    async def test_log_agent_chat(self, mock_asf):
        """log_agent_chat 便捷封装"""
        added = []

        def track(obj):
            added.append(obj)

        mock_db = await mock_asf.__aenter__()
        mock_db.add.side_effect = track

        with patch("coursepilot.governance.audit.async_session_factory", return_value=mock_asf):
            from coursepilot.governance.audit import log_agent_chat
            await log_agent_chat(UID, "session-1", "question", "什么是二叉树")

        assert len(added) == 1
        entry = added[0]
        assert entry.action == "agent.chat"
        assert entry.resource_id == "session-1"
        assert entry.details["intent"] == "question"

    @pytest.mark.asyncio
    async def test_log_permission_denied(self, mock_asf):
        """log_permission_denied 便捷封装"""
        added = []

        def track(obj):
            added.append(obj)

        mock_db = await mock_asf.__aenter__()
        mock_db.add.side_effect = track

        with patch("coursepilot.governance.audit.async_session_factory", return_value=mock_asf):
            from coursepilot.governance.audit import log_permission_denied
            await log_permission_denied(UID, "course:write", "course")

        assert len(added) == 1
        entry = added[0]
        assert entry.action == "permission.denied"
        assert entry.details["permission"] == "course:write"

    @pytest.mark.asyncio
    async def test_log_quiz_generated(self, mock_asf):
        """log_quiz_generated 便捷封装"""
        added = []

        def track(obj):
            added.append(obj)

        mock_db = await mock_asf.__aenter__()
        mock_db.add.side_effect = track

        with patch("coursepilot.governance.audit.async_session_factory", return_value=mock_asf):
            from coursepilot.governance.audit import log_quiz_generated
            await log_quiz_generated(UID, "session-2", 3)

        assert len(added) == 1
        entry = added[0]
        assert entry.action == "quiz.generated"
        assert entry.details["question_count"] == 3

    @pytest.mark.asyncio
    async def test_log_guardrail_violation(self, mock_asf):
        """log_guardrail_violation 便捷封装"""
        added = []

        def track(obj):
            added.append(obj)

        mock_db = await mock_asf.__aenter__()
        mock_db.add.side_effect = track

        with patch("coursepilot.governance.audit.async_session_factory", return_value=mock_asf):
            from coursepilot.governance.audit import log_guardrail_violation
            await log_guardrail_violation(UID, "session-3", ["回答为空或过短"])

        assert len(added) == 1
        entry = added[0]
        assert entry.action == "guardrail.violation"
        assert entry.details["issues"] == ["回答为空或过短"]


# ═══════════════════════════════════════════════════════════════
# 4. Metrics 指标聚合
# ═══════════════════════════════════════════════════════════════

class TestMetrics:
    """get_course_stats / get_week_kp_summary / get_today_token_usage / get_daily_counts"""

    @pytest.mark.asyncio
    async def test_get_course_stats(self, mock_asf):
        """get_course_stats 聚合基础统计 + 分布"""
        mock_db = await mock_asf.__aenter__()
        mock_db.scalar = AsyncMock(side_effect=[10, 5000, 2.5])

        intent_rows = [("question", 8), ("practice", 2)]
        status_rows = [("completed", 9), ("failed", 1)]

        mock_db.execute = AsyncMock(side_effect=[
            IterResult(intent_rows),
            IterResult(status_rows),
        ])

        with patch("coursepilot.observability.metrics.async_session_factory", return_value=mock_asf):
            from coursepilot.observability.metrics import get_course_stats
            result = await get_course_stats(course_id=UID, days=30)

        assert result["total_sessions"] == 10
        assert result["total_tokens"] == 5000.0
        assert result["total_cost"] == 2.5
        assert result["intent_distribution"] == {"question": 8, "practice": 2}
        assert result["status_distribution"] == {"completed": 9, "failed": 1}
        assert result["course_id"] == UID
        assert result["period_days"] == 30

    @pytest.mark.asyncio
    async def test_get_course_stats_empty(self, mock_asf):
        """无数据时返回零值"""
        mock_db = await mock_asf.__aenter__()
        mock_db.scalar = AsyncMock(side_effect=[0, 0, 0])
        mock_db.execute = AsyncMock(side_effect=[
            IterResult([]),
            IterResult([]),
        ])

        with patch("coursepilot.observability.metrics.async_session_factory", return_value=mock_asf):
            from coursepilot.observability.metrics import get_course_stats
            result = await get_course_stats(course_id=UID, days=30)

        assert result["total_sessions"] == 0
        assert result["total_tokens"] == 0.0
        assert result["intent_distribution"] == {}
        assert result["status_distribution"] == {}

    @pytest.mark.asyncio
    async def test_get_week_kp_summary(self, mock_asf):
        """get_week_kp_summary 聚合薄弱知识点"""
        mock_db = await mock_asf.__aenter__()
        rows = [
            (["OS/进程同步", "OS/内存管理"], 0.40),
            (["OS/进程同步"], 0.50),
            (["OS/文件系统"], 0.60),
        ]
        mock_db.execute = AsyncMock(return_value=IterResult(rows))

        with patch("coursepilot.observability.metrics.async_session_factory", return_value=mock_asf):
            from coursepilot.observability.metrics import get_week_kp_summary
            result = await get_week_kp_summary(course_id=UID, days=30)

        assert len(result) == 3
        assert result[0]["kp_path"] == "OS/进程同步"
        assert result[0]["student_count"] == 2
        assert result[0]["avg_rate"] == pytest.approx(0.45, rel=0.01)

        paths = {r["kp_path"] for r in result}
        assert paths == {"OS/进程同步", "OS/内存管理", "OS/文件系统"}

    @pytest.mark.asyncio
    async def test_get_week_kp_summary_empty(self, mock_asf):
        """无薄弱知识点时返回空列表"""
        mock_db = await mock_asf.__aenter__()
        mock_db.execute = AsyncMock(return_value=IterResult([]))

        with patch("coursepilot.observability.metrics.async_session_factory", return_value=mock_asf):
            from coursepilot.observability.metrics import get_week_kp_summary
            result = await get_week_kp_summary(course_id=UID, days=30)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_today_token_usage(self, mock_asf):
        """get_today_token_usage 返回整型"""
        mock_db = await mock_asf.__aenter__()
        mock_db.scalar = AsyncMock(return_value=1500)

        with patch("coursepilot.observability.metrics.async_session_factory", return_value=mock_asf):
            from coursepilot.observability.metrics import get_today_token_usage
            result = await get_today_token_usage(user_id=UID)

        assert result == 1500
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_get_today_token_usage_zero(self, mock_asf):
        """无记录时返回 0"""
        mock_db = await mock_asf.__aenter__()
        mock_db.scalar = AsyncMock(return_value=None)

        with patch("coursepilot.observability.metrics.async_session_factory", return_value=mock_asf):
            from coursepilot.observability.metrics import get_today_token_usage
            result = await get_today_token_usage(user_id=UID)

        assert result == 0

    @pytest.mark.asyncio
    async def test_get_daily_counts(self, mock_asf):
        """get_daily_counts 返回日期列表"""
        mock_db = await mock_asf.__aenter__()

        from datetime import date
        mock_db.execute = AsyncMock(return_value=IterResult([
            (date(2026, 7, 1), 15),
            (date(2026, 7, 2), 22),
            (date(2026, 7, 3), 18),
        ]))

        with patch("coursepilot.observability.metrics.async_session_factory", return_value=mock_asf):
            from coursepilot.observability.metrics import get_daily_counts
            result = await get_daily_counts(course_id=UID, days=7)

        assert len(result) == 3
        assert result[0] == {"date": "2026-07-01", "count": 15}

    @pytest.mark.asyncio
    async def test_get_daily_counts_empty(self, mock_asf):
        """无会话时返回空列表"""
        mock_db = await mock_asf.__aenter__()
        mock_db.execute = AsyncMock(return_value=IterResult([]))

        with patch("coursepilot.observability.metrics.async_session_factory", return_value=mock_asf):
            from coursepilot.observability.metrics import get_daily_counts
            result = await get_daily_counts(course_id=UID, days=7)

        assert result == []
