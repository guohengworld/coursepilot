"""MCP stdio 模式测试。"""

import json
import subprocess
import sys

import pytest


@pytest.fixture
def mcp_server():
    """启动 MCP Server 子进程。"""
    proc = subprocess.Popen(
        [sys.executable, "-m", "coursepilot.mcp.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        yield proc
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def _send(proc, request: dict) -> dict:
    """发送 JSON-RPC 请求并读取响应。"""
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("MCP Server 未返回响应")
    return json.loads(line)


def test_initialize(mcp_server):
    """stdio 模式能完成 initialize 握手。"""
    resp = _send(mcp_server, {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"},
        },
    })
    assert "error" not in resp
    assert resp["result"]["protocolVersion"] == "2025-06-18"
    assert resp["result"]["serverInfo"]["name"] == "coursepilot"


def test_tools_list(mcp_server):
    """tools/list 返回 P1 阶段的 3 个核心工具。"""
    _send(mcp_server, {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"},
        },
    })

    resp = _send(mcp_server, {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    })
    assert "error" not in resp
    tools = resp["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    assert "query_knowledge_tool" in tool_names
    assert "generate_practice_tool" in tool_names
    assert "grade_answers_tool" in tool_names
    assert "diagnose_tool" in tool_names
    assert "get_review_plan_tool" in tool_names
    assert "search_knowledge_units_tool" in tool_names
    assert "get_kp_tree_tool" in tool_names


def test_tools_have_annotations(mcp_server):
    """每个工具都包含 annotations 元数据。"""
    _send(mcp_server, {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"},
        },
    })

    resp = _send(mcp_server, {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    })
    tools = resp["result"]["tools"]
    for tool in tools:
        assert "annotations" in tool
        assert "readOnlyHint" in tool["annotations"]


def test_tool_call_invalid_course(mcp_server):
    """调用 query_knowledge 时传入不存在的 course_id 返回错误。"""
    _send(mcp_server, {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"},
        },
    })

    resp = _send(mcp_server, {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "query_knowledge_tool",
            "arguments": {
                "params": {
                    "query": "什么是进程调度",
                    "course_id": "550e8400-e29b-41d4-a716-446655440000",
                }
            },
        },
    })
    assert "error" not in resp
    content = resp["result"]["content"]
    assert len(content) == 1
    assert resp["result"]["isError"] is True
    assert "不存在" in content[0]["text"]
