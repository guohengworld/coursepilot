"""MCP Server 单元测试（已过时，整体跳过）

**状态：过时（2026-08-10 标记）**。本文件测试的是 MCP 重构前的旧 API：
位置参数签名（如 ``query_knowledge("query", "course-id")``）、已移除的函数
（``student_report`` / ``query_mastery`` / ``diagnose_weakness`` 等）与旧的
``server.py`` 单文件结构。

当前实现的对应物：
- 工具实现迁移至 ``src/coursepilot/mcp/tools/{tutor,practice,knowledge}.py``，
  统一为 ``params: XxxParams -> CallToolResult`` 签名；
- 协议级覆盖由 ``tests/test_mcp/``（test_gateway / test_stdio / test_schemas /
  test_validation / test_errors）提供，本文件的单元覆盖已并入其中。

处理建议：按新 API 重写（参照 ``docs/MCP_重构_TODO.md`` P1-T2 租户断言用例），
或直接删除。在此之前整体 skip，避免 CI 误报。
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.skip(
    reason="过时：测试 MCP 重构前的旧 API（位置参数签名 / 已删除函数），"
    "已由 tests/test_mcp/ 下的协议级测试取代；重写或删除见 MCP_重构_TODO.md"
)

ZERO_TOKENS = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@pytest.fixture
def mock_session():
    """mock DB session + chain: scalars → all → []"""
    session = AsyncMock()
    session.close = AsyncMock(return_value=None)

    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    result.scalars.return_value = result
    result.all.return_value = []
    # session.execute 是 AsyncMock，await session.execute() 返回 result（MagicMock）
    session.execute = AsyncMock(return_value=result)

    return session


# ── query_knowledge ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_knowledge_ok(mock_session):
    """正常 RAG 问答流程"""
    from coursepilot.mcp.server import query_knowledge

    with (
        patch("coursepilot.mcp.server._get_session", return_value=mock_session),
        patch(
            "coursepilot.mcp.server.build_course_context",
            return_value={"name": "OS", "textbook": "教材", "chapters": ["进程管理"]},
        ),
        patch("coursepilot.mcp.server.Retriever") as mock_ret_cls,
        patch("coursepilot.mcp.server.Generator") as mock_gen_cls,
    ):
        mock_ret = AsyncMock()
        mock_ret.retrieve.return_value = (
            "检索到的上下文",
            {"source_kp_paths": ["OS/进程管理"]},
        )
        mock_ret_cls.return_value = mock_ret

        mock_gen = AsyncMock()
        mock_gen.generate.return_value = ("进程调度是核心功能", ZERO_TOKENS)
        mock_gen_cls.return_value = mock_gen

        result = await query_knowledge("什么是进程调度？", "course-uuid")

    assert "进程调度" in result
    assert "Token 用量" in result


@pytest.mark.asyncio
async def test_query_knowledge_course_not_found(mock_session):
    """course_id 不存在"""
    from coursepilot.mcp.server import query_knowledge

    with (
        patch("coursepilot.mcp.server._get_session", return_value=mock_session),
        patch("coursepilot.mcp.server.build_course_context", return_value={}),
    ):
        result = await query_knowledge("test", "bad-uuid")

    assert "错误" in result


# ── get_kp_tree ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_kp_tree_no_root(mock_session):
    """课程没有知识点"""
    from coursepilot.mcp.server import get_kp_tree

    with patch("coursepilot.mcp.server._get_session", return_value=mock_session):
        result = await get_kp_tree("course-uuid")

    assert "暂无知识点" in result


@pytest.mark.asyncio
async def test_get_kp_tree_with_root(mock_session):
    """返回知识点树"""
    from coursepilot.mcp.server import get_kp_tree

    # mock 查到 root
    fake_root = MagicMock()
    fake_root.id = "root-uuid"
    mock_session.execute.return_value.scalar_one_or_none.return_value = fake_root

    # mock KPTree.get_subtree
    fake_node = MagicMock()
    fake_node.id = "root-uuid"
    fake_node.title = "操作系统"
    fake_node.kp_path = "OS"
    fake_node.children = []

    with (
        patch("coursepilot.mcp.server._get_session", return_value=mock_session),
        patch("coursepilot.mcp.server.KPTree") as mock_kp_cls,
    ):
        mock_tree = AsyncMock()
        mock_tree.get_subtree.return_value = fake_node
        mock_kp_cls.return_value = mock_tree

        result = await get_kp_tree("course-uuid")

    assert "操作系统" in result


# ── get_mastery ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_mastery_no_data(mock_session):
    """无练习记录"""
    from coursepilot.mcp.server import query_mastery

    with (
        patch("coursepilot.mcp.server._get_session", return_value=mock_session),
        patch(
            "coursepilot.mcp.server.get_mastery",
            return_value={"mastery_level": {}, "weak_kps": [], "avg_correct_rate": None},
        ),
    ):
        result = await query_mastery("user-uuid", "course-uuid")

    assert "暂无掌握度数据" in result


@pytest.mark.asyncio
async def test_query_mastery_with_data(mock_session):
    """有掌握度数据"""
    from coursepilot.mcp.server import query_mastery

    with (
        patch("coursepilot.mcp.server._get_session", return_value=mock_session),
        patch(
            "coursepilot.mcp.server.get_mastery",
            return_value={
                "mastery_level": {"OS/进程管理": 0.8},
                "weak_kps": ["OS/内存管理"],
                "avg_correct_rate": 0.75,
            },
        ),
    ):
        result = await query_mastery("user-uuid", "course-uuid")

    assert "75%" in result
    assert "内存管理" in result


# ── diagnose ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_diagnose_weakness(mock_session):
    """学情诊断"""
    from coursepilot.mcp.server import diagnose_weakness

    mock_result = {
        "weak_kps": ["OS/进程管理"],
        "kp_stats": {"OS/进程管理": {"total": 10, "correct": 3, "rate": 0.3}},
        "summary": "薄弱知识点：进程管理",
        "total_practiced": 10,
        "overall_rate": 0.3,
    }

    with (
        patch("coursepilot.mcp.server._get_session", return_value=mock_session),
        patch("coursepilot.mcp.server.diagnose", return_value=mock_result),
    ):
        result = await diagnose_weakness("user-uuid", "course-uuid")

    assert "3/10" in result or "30%" in result or "诊断报告" in result


# ── generate_practice ────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_practice_ok(mock_session):
    """生成练习题"""
    from coursepilot.mcp.server import generate_practice

    mock_quiz = {
        "questions": [
            {
                "question_text": "什么是进程？",
                "options": {"A": "程序", "B": "进程实例", "C": "文件"},
                "correct_answer": "B",
                "explanation": "进程是程序在执行过程中的实例",
            }
        ]
    }

    with (
        patch("coursepilot.mcp.server._get_session", return_value=mock_session),
        patch(
            "coursepilot.mcp.server.build_course_context",
            return_value={"name": "OS", "textbook": "教材", "chapters": []},
        ),
        patch("coursepilot.mcp.server.Retriever") as mock_ret_cls,
        patch("coursepilot.mcp.server.gen_quiz", return_value=(mock_quiz, ZERO_TOKENS)),
    ):
        mock_ret = AsyncMock()
        mock_ret.retrieve.return_value = ("上下文", {})
        mock_ret_cls.return_value = mock_ret

        result = await generate_practice("course-uuid")

    assert "练习题" in result
    assert "什么是进程" in result


# ── student_report ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_student_report_ok(mock_session):
    """综合报告"""
    from coursepilot.mcp.server import student_report

    mock_diag = {
        "weak_kps": ["OS/进程管理"],
        "kp_stats": {"OS/进程管理": {"total": 10, "correct": 3, "rate": 0.3}},
        "summary": "薄弱知识点：进程管理",
        "total_practiced": 10,
        "overall_rate": 0.3,
    }
    mock_mastery = {
        "mastery_level": {"OS/进程管理": 0.3},
        "weak_kps": ["OS/进程管理"],
        "avg_correct_rate": 0.3,
    }

    with (
        patch("coursepilot.mcp.server._get_session", return_value=mock_session),
        patch(
            "coursepilot.mcp.server.build_course_context",
            return_value={"name": "OS", "textbook": "教材", "chapters": []},
        ),
        patch("coursepilot.mcp.server.diagnose", return_value=mock_diag),
        patch("coursepilot.mcp.server.get_mastery", return_value=mock_mastery),
    ):
        result = await student_report("user-uuid", "course-uuid")

    assert "综合报告" in result
    assert "OS" in result
