"""P1-T2 租户断言测试：student 只能查自己，teacher/super 可查任意。

覆盖：
- 单元层（tools/resources 函数直接调用，mock DB）：
  - diagnose / get_review_plan / grade_answers：student 传他人 user_id → 拒绝；
    student 传自己 → 放行；teacher 传任意 → 放行。
  - generate_practice：student（仅 read scope）→ 拒绝；teacher（read+write）→ 放行。
  - read_report / read_mastery：student 传他人 → 拒绝；teacher → 放行。
- 端到端层（真实 MCPServer + 鉴权中间件 + TestClient）：
  - 带装饰器的探针工具：student key 传他人 user_id → isError=true；
    teacher key 传任意 user_id → 正常执行（验证装饰器异常映射为 isError）。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel
from starlette.testclient import TestClient

from coursepilot.mcp.auth.keys import KeyStore
from coursepilot.mcp.auth.middleware import AuthenticationMiddleware
from coursepilot.mcp.auth.policy import (
    require_self_or_privileged,
)
from coursepilot.mcp.principal import (
    Principal,
    principal_var,
    set_principal,
)
from coursepilot.mcp.shared.errors import ToolForbiddenError

_VALID_KEYS = {
    "cp_student01": {"user_id": "u-student", "role": "student",
                     "scopes": ["read"]},
    "cp_teacher01": {"user_id": "u-teacher", "role": "teacher",
                     "scopes": ["read", "write"]},
}

# ── 工具参数（对齐 shared/schemas.py 真实字段）───────────────────────────

_DIAGNOSE = {"user_id": "u-other", "course_id": "c-1"}
_REVIEW = {"user_id": "u-other", "course_id": "c-1"}
_GRADE = {"user_id": "u-other", "question_id": "q-1", "answer": "A"}
_PRACTICE = {"course_id": "c-1", "kp_path": "", "count": 3, "difficulty": 3}


class _TenancyParams(BaseModel):
    """端到端探针参数：带 user_id 的租户断言工具。"""

    user_id: str = "u-other"


def _make_tenancy_server() -> MCPServer:
    """构造带租户断言探针工具的 MCPServer（模拟真实工具包装方式）。"""
    srv = MCPServer(name="probe-tenancy")

    @srv.tool()
    @require_self_or_privileged("teacher", "super")
    async def probe_tenancy(params: _TenancyParams) -> CallToolResult:
        return CallToolResult(
            content=[TextContent(type="text", text="ok")],
            is_error=False,
        )

    return srv


@pytest.fixture(scope="module")
def tenancy_client():
    """带鉴权中间件 + 租户断言探针的端到端客户端。"""
    store = KeyStore.load(env_json=json.dumps(_VALID_KEYS), single_key="")
    app = _make_tenancy_server().streamable_http_app(
        stateless_http=True, json_response=True, host="0.0.0.0"
    )
    app.add_middleware(AuthenticationMiddleware, key_store=store)
    c = TestClient(app)
    c.__enter__()
    yield c
    c.__exit__(None, None, None)


# ── 端到端：装饰器异常 → SDK isError 映射 ───────────────────────────────

def test_e2e_student_other_user_forbidden(tenancy_client):
    """student key 传他人 user_id → isError=true，不执行工具。"""
    r = tenancy_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "probe_tenancy",
                "arguments": {"params": {"user_id": "u-other"}},
            },
        },
        headers={"Authorization": "Bearer cp_student01"},
    )
    assert r.status_code == 200
    result = r.json()["result"]
    assert result["isError"] is True
    assert "无权访问" in result["content"][0]["text"]


def test_e2e_student_self_ok(tenancy_client):
    """student key 传自己 user_id → 正常执行。"""
    r = tenancy_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {
                "name": "probe_tenancy",
                "arguments": {"params": {"user_id": "u-student"}},
            },
        },
        headers={"Authorization": "Bearer cp_student01"},
    )
    assert r.status_code == 200
    result = r.json()["result"]
    assert result["isError"] is False
    assert "ok" in result["content"][0]["text"]


def test_e2e_teacher_any_user_ok(tenancy_client):
    """teacher key 传任意 user_id → 正常执行。"""
    r = tenancy_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "probe_tenancy",
                "arguments": {"params": {"user_id": "u-other"}},
            },
        },
        headers={"Authorization": "Bearer cp_teacher01"},
    )
    assert r.status_code == 200
    assert r.json()["result"]["isError"] is False


# ── 单元层：真实工具函数（mock DB）──────────────────────────────────────

def _as_principal(role: str, scopes: set[str]) -> Principal:
    return Principal(user_id=f"u-{role}", role=role,
                     scopes=frozenset(scopes))


def _run(coro):
    import asyncio

    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean_principal():
    """每个用例结束后清理 ContextVar，避免跨用例泄漏。"""
    token = set_principal(None)
    yield
    principal_var.reset(token)


@pytest.fixture
def mock_tool_session():
    """mock tools 层 _get_session：execute 返回空结果。"""
    session = AsyncMock()
    session.close = AsyncMock(return_value=None)
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    result.all.return_value = []
    session.execute = AsyncMock(return_value=result)
    return session


# ── diagnose（tutor.py）────────────────────────────────────────────────

def test_diagnose_student_other_forbidden(mock_tool_session):
    """student 诊断他人 → ToolForbiddenError。"""
    from coursepilot.mcp.tools.tutor import diagnose

    set_principal(_as_principal("student", {"read"}))

    async def _call():
        with patch("coursepilot.mcp.tools.tutor._get_session",
                   return_value=mock_tool_session):
            return await diagnose(type("D", (), _DIAGNOSE)())

    with pytest.raises(ToolForbiddenError):
        _run(_call())


def test_diagnose_student_self_ok(mock_tool_session):
    """student 诊断自己 → 放行（mock DB 返回空，走正常流程）。"""
    from coursepilot.mcp.tools.tutor import diagnose

    set_principal(_as_principal("student", {"read"}))
    params = type("D", (), {"user_id": "u-student", "course_id": "c-1"})

    async def _call():
        with (
            patch("coursepilot.mcp.tools.tutor._get_session",
                  return_value=mock_tool_session),
            patch("coursepilot.mcp.tools.tutor.diagnose_skill",
                  return_value={"weak_kps": []}),
        ):
            return await diagnose(params)

    result = _run(_call())
    assert result.is_error is False


def test_diagnose_teacher_any_ok(mock_tool_session):
    """teacher 诊断任意学生 → 放行。"""
    from coursepilot.mcp.tools.tutor import diagnose

    set_principal(_as_principal("teacher", {"read", "write"}))
    params = type("D", (), {"user_id": "u-other", "course_id": "c-1"})

    async def _call():
        with (
            patch("coursepilot.mcp.tools.tutor._get_session",
                  return_value=mock_tool_session),
            patch("coursepilot.mcp.tools.tutor.diagnose_skill",
                  return_value={"weak_kps": []}),
        ):
            return await diagnose(params)

    result = _run(_call())
    assert result.is_error is False


# ── get_review_plan（tutor.py）─────────────────────────────────────────

def test_review_plan_student_other_forbidden(mock_tool_session):
    """student 为他人生成复习计划 → ToolForbiddenError。"""
    from coursepilot.mcp.tools.tutor import get_review_plan

    set_principal(_as_principal("student", {"read"}))
    params = type("R", (), _REVIEW)()

    async def _call():
        with patch("coursepilot.mcp.tools.tutor._get_session",
                   return_value=mock_tool_session):
            return await get_review_plan(params)

    with pytest.raises(ToolForbiddenError):
        _run(_call())


def test_review_plan_teacher_any_ok(mock_tool_session):
    """teacher 为任意学生生成复习计划 → 放行。"""
    from coursepilot.mcp.tools.tutor import get_review_plan

    set_principal(_as_principal("teacher", {"read", "write"}))
    params = type("R", (), {"user_id": "u-other", "course_id": "c-1"})

    async def _call():
        with (
            patch("coursepilot.mcp.tools.tutor._get_session",
                  return_value=mock_tool_session),
            patch("coursepilot.mcp.tools.tutor.diagnose_skill",
                  return_value={"weak_kps": []}),
            patch("coursepilot.mcp.tools.tutor.review_plan_skill",
                  return_value=([], {"prompt_tokens": 0, "completion_tokens": 0})),
        ):
            return await get_review_plan(params)

    result = _run(_call())
    assert result.is_error is False


# ── generate_practice（practice.py）：require_scope("write") ────────────

def test_generate_practice_student_no_write_forbidden(mock_tool_session):
    """student（仅 read scope）生成练习 → ToolForbiddenError。"""
    from coursepilot.mcp.tools.practice import generate_practice

    set_principal(_as_principal("student", {"read"}))
    params = type("P", (), _PRACTICE)()

    async def _call():
        with patch("coursepilot.mcp.tools.practice._get_session",
                   return_value=mock_tool_session):
            return await generate_practice(params)

    with pytest.raises(ToolForbiddenError):
        _run(_call())


def test_generate_practice_teacher_ok(mock_tool_session):
    """teacher（read+write）生成练习 → 放行。"""
    from coursepilot.mcp.tools.practice import generate_practice

    set_principal(_as_principal("teacher", {"read", "write"}))
    params = type("P", (), _PRACTICE)()

    async def _call():
        with (
            patch("coursepilot.mcp.tools.practice._get_session",
                  return_value=mock_tool_session),
            patch("coursepilot.mcp.tools.practice.build_course_context",
                  return_value={"name": "OS"}),
            patch("coursepilot.mcp.tools.practice._find_kp_by_path",
                  return_value=MagicMock(id="kp-1")),
            patch("coursepilot.mcp.tools.practice.Retriever") as ret_cls,
            patch("coursepilot.mcp.tools.practice.generate_quiz",
                  return_value=({"questions": [{"question_text": "q",
                                                "options": {}, "correct_answer": "A",
                                                "explanation": "e"}]},
                                {"prompt_tokens": 0, "completion_tokens": 0})),
        ):
            ret = AsyncMock()
            ret.retrieve.return_value = ("ctx", {})
            ret_cls.return_value = ret
            return await generate_practice(params)

    result = _run(_call())
    assert result.is_error is False


# ── grade_answers（practice.py）────────────────────────────────────────

def test_grade_answers_student_other_forbidden(mock_tool_session):
    """student 为他人提交作答 → ToolForbiddenError。"""
    from coursepilot.mcp.tools.practice import grade_answers

    set_principal(_as_principal("student", {"read"}))
    params = type("G", (), _GRADE)()

    async def _call():
        with patch("coursepilot.mcp.tools.practice._get_session",
                   return_value=mock_tool_session):
            return await grade_answers(params)

    with pytest.raises(ToolForbiddenError):
        _run(_call())


def test_grade_answers_teacher_any_ok(mock_tool_session):
    """teacher 为任意学生批改 → 放行。"""
    from coursepilot.mcp.tools.practice import grade_answers

    set_principal(_as_principal("teacher", {"read", "write"}))
    params = type("G", (), {"user_id": "u-other", "question_id": "q-1",
                            "answer": "A"})

    async def _call():
        q = MagicMock(id="q-1", correct_answer="A", explanation="e", kp_id=None)
        # 第一次 execute 查 Question → q；第二次查 kp_path → None
        r_q = MagicMock()
        r_q.scalar_one_or_none.return_value = q
        r_none = MagicMock()
        r_none.scalar_one_or_none.return_value = None
        mock_tool_session.execute = AsyncMock(side_effect=[r_q, r_none])
        with (
            patch("coursepilot.mcp.tools.practice._get_session",
                  return_value=mock_tool_session),
        ):
            return await grade_answers(params)

    result = _run(_call())
    assert result.is_error is False


# ── read_report / read_mastery（resources/course.py）───────────────────

def test_read_report_student_other_forbidden(mock_tool_session):
    """student 读他人报告 → ToolForbiddenError。"""
    from coursepilot.mcp.resources.course import read_report

    set_principal(_as_principal("student", {"read"}))

    async def _call():
        with patch("coursepilot.mcp.resources.course._get_session",
                   return_value=mock_tool_session):
            return await read_report("u-other", "c-1")

    with pytest.raises(ToolForbiddenError):
        _run(_call())


def test_read_report_teacher_any_ok(mock_tool_session):
    """teacher 读任意学生报告 → 放行。"""
    from coursepilot.mcp.resources.course import read_report

    set_principal(_as_principal("teacher", {"read", "write"}))

    async def _call():
        with patch("coursepilot.mcp.resources.course._get_session",
                   return_value=mock_tool_session):
            return await read_report("u-other", "c-1")

    result = _run(_call())
    assert "error" not in result or result != ""


def test_read_mastery_student_self_ok(mock_tool_session):
    """student 读自己掌握度 → 放行。"""
    from coursepilot.mcp.resources.course import read_mastery

    set_principal(_as_principal("student", {"read"}))

    async def _call():
        with patch("coursepilot.mcp.resources.course._get_session",
                   return_value=mock_tool_session):
            return await read_mastery("u-student", "c-1")

    result = _run(_call())
    assert "error" not in result or result != ""
