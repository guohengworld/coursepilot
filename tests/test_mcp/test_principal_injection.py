"""集成测试：Principal 注入 + ContextVar 透传。

覆盖：
1. 构造请求流经 AuthenticationMiddleware → principal_var.set → 工具 handler 内
   ``get_principal()`` 能读到身份（端到端真实请求，非 mock）。
2. 无效 / 缺失 Key → 401（-32001），不注入 Principal。
3. 未走网关（无中间件 set）直接调用工具 → 抛 ``UnauthenticatedError``（工具
   返回 isError，不匿名执行）。
4. ContextVar 透传跨 SDK 内部 task 边界成立（主路径，已实测）。

依据（mcp 2.0.0 实测）：
- ``streamable_http_app`` 返回的 Starlette 自带 lifespan，TestClient 可
  ``__enter__`` / ``__exit__`` 正常启停（无需 gateway/main.py 的 daemon hack）。
- 工具参数沿用项目现有风格 ``params: XxxParams``（SDK 会包装为顶层
  ``params`` 字段，调用时 arguments 需嵌套）。
"""

from __future__ import annotations

import json

import pytest
from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel
from starlette.testclient import TestClient

from coursepilot.mcp.auth.keys import KeyStore
from coursepilot.mcp.auth.middleware import AuthenticationMiddleware
from coursepilot.mcp.principal import (
    Principal,
    get_principal,
    principal_var,
    set_principal,
)
from coursepilot.mcp.shared.errors import (
    MCPErrorCode,
    ToolForbiddenError,
    UnauthenticatedError,
)

_VALID_KEYS = {
    "cp_student01": {"user_id": "u-student", "role": "student",
                     "scopes": ["read"]},
    "cp_teacher01": {"user_id": "u-teacher", "role": "teacher",
                     "scopes": ["read", "write"]},
}


class _ProbeParams(BaseModel):
    """测试工具参数（与项目 shared/schemas.py 风格一致）。"""

    echo: str = ""


def _make_server() -> MCPServer:
    """构造带探针工具的独立 MCPServer（不污染生产 server.py）。"""
    srv = MCPServer(name="probe-p1t1")

    @srv.tool()
    async def probe_principal(params: _ProbeParams) -> CallToolResult:
        """返回当前 Principal 身份；未认证时抛 UnauthenticatedError。"""
        p = get_principal()
        text = json.dumps({
            "user_id": p.user_id,
            "role": p.role,
            "scopes": sorted(p.scopes),
        }, ensure_ascii=False)
        return CallToolResult(
            content=[TextContent(type="text", text=text)],
            is_error=False,
        )

    return srv


def _build_app(*, with_auth: bool):
    """构造 MCP app；with_auth=True 挂 AuthenticationMiddleware。"""
    srv = _make_server()
    app = srv.streamable_http_app(
        stateless_http=True, json_response=True, host="0.0.0.0"
    )
    if with_auth:
        store = KeyStore.load(env_json=json.dumps(_VALID_KEYS), single_key="")
        app.add_middleware(AuthenticationMiddleware, key_store=store)
    return app


@pytest.fixture(scope="module")
def client():
    """带鉴权中间件的 MCP app 测试客户端（lifespan 正常启停）。"""
    c = TestClient(_build_app(with_auth=True))
    c.__enter__()
    yield c
    c.__exit__(None, None, None)


@pytest.fixture(scope="module")
def no_auth_client():
    """不带鉴权中间件的 MCP app（验证未认证直连被工具层拒绝）。"""
    c = TestClient(_build_app(with_auth=False))
    c.__enter__()
    yield c
    c.__exit__(None, None, None)


def _call_probe(c, *, key: str | None = None):
    """发出 tools/call 请求。"""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return c.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "probe_principal",
                "arguments": {"params": {"echo": "hi"}},
            },
        },
        headers=headers,
    )


def _result_json(r) -> dict:
    """提取工具返回的 JSON 文本。"""
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body, body
    return json.loads(body["result"]["content"][0]["text"])


# ── 1. 主路径：Principal 注入 + ContextVar 透传（端到端）────────────────

def test_principal_injected_and_read_in_tool(client):
    """合法 key 请求：工具 handler 内 get_principal() 读到网关注入的身份。"""
    r = _call_probe(client, key="cp_student01")
    assert _result_json(r) == {
        "user_id": "u-student", "role": "student", "scopes": ["read"],
    }


def test_principal_teacher_scopes(client):
    """teacher key：role 与 scopes（read+write）均正确注入。"""
    r = _call_probe(client, key="cp_teacher01")
    assert _result_json(r) == {
        "user_id": "u-teacher", "role": "teacher",
        "scopes": ["read", "write"],
    }


# ── 2. 认证失败：不注入 Principal ──────────────────────────────────────

def test_missing_key_401(client):
    """缺失 Authorization 头 → 401，-32001。"""
    r = _call_probe(client)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == MCPErrorCode.AUTHENTICATION_ERROR


def test_invalid_key_401(client):
    """无效 key → 401。"""
    r = _call_probe(client, key="cp_wrong99")
    assert r.status_code == 401


# ── 3. 未走网关直连：拒绝匿名执行 ──────────────────────────────────────

def test_unauthenticated_direct_call(no_auth_client):
    """绕过鉴权中间件直连工具 → 工具层抛 UnauthenticatedError（isError）。

    SDK 捕获工具异常后，错误文本格式为
    ``Error executing tool <name>: <异常消息>``（不含类型名），
    以 isError + "未认证" 消息为判定依据。
    """
    r = _call_probe(no_auth_client)
    assert r.status_code == 200  # 传输层放行，工具层拒绝
    body = r.json()
    result = body["result"]
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "Error executing tool probe_principal" in text
    assert "未认证" in text


def test_get_principal_raises_when_not_set():
    """principal_var 为 None 时 get_principal() 抛 UnauthenticatedError。"""
    token = set_principal(None)
    try:
        assert principal_var.get() is None
        with pytest.raises(UnauthenticatedError):
            get_principal()
    finally:
        principal_var.reset(token)


# ── 4. 备用路径（防御）：ctx.headers 可读性 ─────────────────────────────

def test_ctx_headers_property_exists():
    """若未来 ContextVar 透传失效，备用路径依赖 ctx.headers（实测存在）。"""
    import inspect

    from mcp.server.context import Context

    assert hasattr(Context, "headers")
    src = inspect.getsource(Context)
    assert "def headers" in src


# ── 5. 策略层基础行为（装饰器）────────────────────────────────────────

def _run_async(coro):
    import asyncio

    return asyncio.run(coro)


def test_policy_require_self_or_privileged():
    """租户断言：student 访问他人 → ToolForbiddenError；自身 / 特权角色放行。"""
    from coursepilot.mcp.auth.policy import require_self_or_privileged

    async def _target(params):
        return "ok"

    wrapped = require_self_or_privileged("teacher", "super")(_target)

    class _P:
        def __init__(self, user_id):
            self.user_id = user_id

    async def _run():
        token = None
        try:
            # student 查自己 → 放行
            token = set_principal(Principal(
                user_id="u-student", role="student", scopes=frozenset({"read"})))
            r1 = await wrapped(_P("u-student"))
            # student 查他人 → 拒绝
            set_principal(Principal(
                user_id="u-student", role="student", scopes=frozenset({"read"})))
            try:
                await wrapped(_P("u-other"))
                r2 = "no-error"
            except ToolForbiddenError:
                r2 = "forbidden"
            # teacher 查任意 → 放行
            set_principal(Principal(
                user_id="u-teacher", role="teacher",
                scopes=frozenset({"read", "write"})))
            r3 = await wrapped(_P("u-other"))
            return r1, r2, r3
        finally:
            if token is not None:
                principal_var.reset(token)

    r1, r2, r3 = _run_async(_run())
    assert r1 == "ok"
    assert r2 == "forbidden"
    assert r3 == "ok"


def test_policy_require_scope():
    """scope 断言：缺 write → ToolForbiddenError；teacher(read+write) 放行。"""
    from coursepilot.mcp.auth.policy import require_scope

    async def _target(params):
        return "ok"

    wrapped = require_scope("write")(_target)

    async def _run():
        token = None
        try:
            # student（仅 read）→ 拒绝
            token = set_principal(Principal(
                user_id="u-s", role="student", scopes=frozenset({"read"})))
            try:
                await wrapped(object())
                r1 = "no-error"
            except ToolForbiddenError:
                r1 = "forbidden"
            # teacher（read+write）→ 放行
            set_principal(Principal(
                user_id="u-t", role="teacher",
                scopes=frozenset({"read", "write"})))
            r2 = await wrapped(object())
            return r1, r2
        finally:
            if token is not None:
                principal_var.reset(token)

    r1, r2 = _run_async(_run())
    assert r1 == "forbidden"
    assert r2 == "ok"
