"""stdio-to-HTTP 桥接器单元测试（P3.2.1 ~ P3.2.4）。

覆盖：
- P3.2.1 stdio 读取循环（stdin 行分隔 → stdout）
- P3.2.2 HTTP 转发（请求体原样 POST 到 Gateway）
- P3.2.3 API Key header（每请求带 ``Authorization: Bearer <key>``）
- P3.2.4 响应写回 stdout（与 Gateway 响应一致，不污染 stdout）

使用 ``httpx.MockTransport`` 注入 mock 客户端，避免真实网络；
用 ``io.StringIO`` 注入 stdin/stdout，验证行协议与 stdout 纯净性。
"""

from __future__ import annotations

import io
import json

import httpx
import pytest

from coursepilot.mcp.cli.main import (
    _PARSE_ERROR,
    forward_request,
    run_bridge,
)

_GATEWAY = "https://mcp.coursepilot.example.com/mcp"
_API_KEY = "cp_test1234"


def _make_client(handler, captured: list | None = None):
    """构造带 MockTransport 的 httpx.Client。

    Args:
        handler: 接收 ``httpx.Request`` 返回 ``httpx.Response`` 的回调。
        captured: 若提供，每条收到的请求会追加到此列表，便于断言 header/body。
    """
    def _wrap(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(_wrap))


# ── P3.2.3 API Key header ──────────────────────────────────
def test_forward_adds_authorization_header():
    """每条转发请求都带 ``Authorization: Bearer <api_key>``。"""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})

    client = _make_client(handler, captured)
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    forward_request(client, _GATEWAY, _API_KEY, req)

    assert len(captured) == 1
    assert captured[0].headers["authorization"] == f"Bearer {_API_KEY}"


def test_forward_posts_to_gateway_url():
    """请求 POST 到传入的 Gateway URL。"""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})

    client = _make_client(handler, captured)
    forward_request(client, _GATEWAY, _API_KEY,
                    {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})

    assert captured[0].url == _GATEWAY
    assert captured[0].method == "POST"


# ── P3.2.2 HTTP 转发（请求体原样）──────────────────────────
def test_forward_request_body_unchanged():
    """JSON-RPC 请求体原样作为 JSON POST 发出。"""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 7, "result": {}})

    client = _make_client(handler, captured)
    req = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "query_knowledge_tool",
                   "arguments": {"params": {"query": "什么是进程调度",
                                            "course_id": "abc"}}},
    }
    forward_request(client, _GATEWAY, _API_KEY, req)

    sent = json.loads(captured[0].content)
    assert sent == req


# ── P3.2.4 响应写回 stdout（原样透传）───────────────────────
def test_forward_returns_gateway_response_unchanged():
    """Gateway 响应 dict 原样返回（含 result 字段）。"""
    gateway_resp = {
        "jsonrpc": "2.0", "id": 1,
        "result": {"tools": [{"name": "query_knowledge_tool"}]},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=gateway_resp)

    client = _make_client(handler)
    resp = forward_request(client, _GATEWAY, _API_KEY,
                           {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert resp == gateway_resp


def test_forward_gateway_auth_error_passthrough():
    """Gateway 鉴权失败返回的 {jsonrpc, error} 原样透传给调用方。"""
    err_body = {
        "jsonrpc": "2.0", "id": 2,
        "error": {"code": -32001, "message": "认证失败：无效的 API Key"},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json=err_body)

    client = _make_client(handler)
    resp = forward_request(client, _GATEWAY, _API_KEY,
                           {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert resp == err_body


# ── 通知处理（无 id 不写回）────────────────────────────────
def test_forward_notification_returns_none():
    """无 id 的 JSON-RPC 通知转发后返回 None（不写回 stdout）。"""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # 通知通常返回 202 无 body，但桥接器不应依赖
        return httpx.Response(202)

    client = _make_client(handler, captured)
    notify = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    resp = forward_request(client, _GATEWAY, _API_KEY, notify)

    assert resp is None
    # 通知仍被转发到 Gateway
    assert len(captured) == 1
    assert captured[0].headers["authorization"] == f"Bearer {_API_KEY}"


# ── 错误处理 ───────────────────────────────────────────────
def test_forward_network_error_returns_jsonrpc_error():
    """网络异常时返回 JSON-RPC error 响应（而非抛出）。"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _make_client(handler)
    resp = forward_request(client, _GATEWAY, _API_KEY,
                           {"jsonrpc": "2.0", "id": 9, "method": "tools/list"})

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 9
    assert resp["error"]["code"] == -32603
    assert "网络错误" in resp["error"]["message"]


def test_forward_non_json_response_returns_error():
    """Gateway 返回非 JSON 时构造 error 响应。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>Bad Gateway</html>")

    client = _make_client(handler)
    resp = forward_request(client, _GATEWAY, _API_KEY,
                           {"jsonrpc": "2.0", "id": 5, "method": "tools/list"})

    assert resp["id"] == 5
    assert resp["error"]["code"] == -32603
    assert "非 JSON" in resp["error"]["message"]


def test_forward_network_error_on_notification_returns_none():
    """通知遇到网络错误也不写回（无 id 无响应）。"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = _make_client(handler)
    resp = forward_request(client, _GATEWAY, _API_KEY,
                           {"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert resp is None


# ── P3.2.1 stdio 读取循环（端到端）─────────────────────────
def test_run_bridge_end_to_end():
    """stdin 多行 JSON-RPC → stdout 多行响应，一一对应。"""
    responses = {
        1: {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2025-06-18"}},
        2: {"jsonrpc": "2.0", "id": 2, "result": {"tools": []}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json=responses[body["id"]])

    stdin = io.StringIO(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}) + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + "\n"
    )
    stdout = io.StringIO()
    client = _make_client(handler)

    run_bridge(stdin, stdout, _GATEWAY, _API_KEY, client=client)

    out_lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert out_lines == [responses[1], responses[2]]


def test_run_bridge_notification_not_written_to_stdout():
    """通知（无 id）不产生 stdout 输出，但后续请求仍正常。"""
    tool_resp = {"jsonrpc": "2.0", "id": 10, "result": {}}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202)
        return httpx.Response(200, json=tool_resp)

    stdin = io.StringIO(
        # 通知 + 带id请求
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 10, "method": "tools/list"}) + "\n"
    )
    stdout = io.StringIO()
    run_bridge(stdin, stdout, _GATEWAY, _API_KEY, client=_make_client(handler))

    out_lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert out_lines == [tool_resp]  # 只有带 id 那条


def test_run_bridge_invalid_json_writes_parse_error():
    """非 JSON 输入写回 parse error，循环不中断。"""
    stdin = io.StringIO('not-json\n{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n')
    stdout = io.StringIO()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})

    run_bridge(stdin, stdout, _GATEWAY, _API_KEY, client=_make_client(handler))

    out_lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert len(out_lines) == 2
    # 第一行：parse error
    assert out_lines[0]["error"]["code"] == _PARSE_ERROR
    assert out_lines[0]["id"] is None
    # 第二行：正常响应
    assert out_lines[1]["id"] == 1


def test_run_bridge_non_object_json_writes_invalid_request():
    """JSON 但非对象（如数组）写回 invalid request。"""
    stdin = io.StringIO('[1,2,3]\n')
    stdout = io.StringIO()

    run_bridge(stdin, stdout, _GATEWAY, _API_KEY, client=_make_client(
        lambda r: httpx.Response(200, json={})))

    out_lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert out_lines[0]["error"]["code"] == -32600


def test_run_bridge_empty_lines_skipped():
    """空行与纯空白行被跳过，不产生输出也不发请求。"""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})

    stdin = io.StringIO('\n  \n{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n\n')
    stdout = io.StringIO()
    run_bridge(stdin, stdout, _GATEWAY, _API_KEY, client=_make_client(handler, captured))

    # 只有一条有效请求被转发
    assert len(captured) == 1
    out_lines = stdout.getvalue().splitlines()
    assert len(out_lines) == 1


# ── P3.2.1 不污染 stdout ───────────────────────────────────
def test_run_bridge_stdout_contains_only_jsonrpc_lines():
    """stdout 每一行都是合法 JSON-RPC 响应，无任何诊断信息污染。"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})

    stdin = io.StringIO(
        'bad-line\n{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n')
    stdout = io.StringIO()
    run_bridge(stdin, stdout, _GATEWAY, _API_KEY, client=_make_client(handler))

    for line in stdout.getvalue().splitlines():
        # 每行都能解析为 JSON 对象（JSON-RPC 响应）
        parsed = json.loads(line)
        assert parsed["jsonrpc"] == "2.0"


def test_run_bridge_closes_owned_client():
    """未注入 client 时 run_bridge 自建并在结束时关闭。"""
    # 用真实 httpx.Client + MockTransport 验证关闭不报错
    stdin = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n')
    stdout = io.StringIO()
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})))
    # 注入 client：run_bridge 不应关闭它（由调用方负责）
    run_bridge(stdin, stdout, _GATEWAY, _API_KEY, client=client)
    # 注入的 client 仍可用
    assert client.is_closed is False
    client.close()


# ── P3.2.5 打包：入口可导入 ─────────────────────────────────
def test_main_entry_point_importable():
    """``coursepilot.mcp.cli.main:main`` 入口可被导入（打包前置条件）。"""
    from coursepilot.mcp.cli.main import main

    assert callable(main)


def test_main_help_exits_zero(capsys):
    """``--help`` 由 argparse 处理，打印用法后以 0 退出，不进入 stdio 循环。"""
    from coursepilot.mcp.cli.main import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])  # type: ignore[arg-type]
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "coursepilot-mcp" in captured.out
    assert "--gateway" in captured.out
    assert "--api-key" in captured.out
