"""MCP stdio 模式测试。

由于 server 启动需要加载大量模型（约 10 秒），本测试一次性启动一个 server
进程并复用，完成所有 stdio 协议验证。
"""

import json
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def server_process():
    """启动 MCP Server 子进程（不做握手，由测试自行完成）。"""
    proc = subprocess.Popen(
        [sys.executable, "-m", "coursepilot.mcp.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
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
    proc.stdin.write((json.dumps(request) + "\n").encode())
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("MCP Server 未返回响应")
    return json.loads(line.decode())


def _notify(proc, notification: dict) -> None:
    """发送 JSON-RPC 通知（无 id，不期待响应）。"""
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(notification) + "\n").encode())
    proc.stdin.flush()


def test_stdio_protocol(server_process):
    """stdio 模式下完整的协议验证（单次 server 启动）。"""
    server = server_process

    # 1. initialize
    resp = _send(server, {
        "jsonrpc": "2.0",
        "id": 10,
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

    # 1.1 发送 initialized 通知，完成握手（否则后续请求被拒绝）
    _notify(server, {"jsonrpc": "2.0", "method": "notifications/initialized"})

    # 2. tools/list
    resp = _send(server, {
        "jsonrpc": "2.0",
        "id": 20,
        "method": "tools/list",
        "params": {},
    })
    assert "error" not in resp
    tools = resp["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    expected_tools = {
        "query_knowledge_tool",
        "generate_practice_tool",
        "grade_answers_tool",
        "diagnose_tool",
        "get_review_plan_tool",
        "search_knowledge_units_tool",
        "get_kp_tree_tool",
    }
    assert expected_tools.issubset(tool_names)
    for tool in tools:
        assert "annotations" in tool
        assert "readOnlyHint" in tool["annotations"]

    # 3. tools/call with invalid course
    resp = _send(server, {
        "jsonrpc": "2.0",
        "id": 30,
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
    assert resp["result"]["isError"] is True
    assert "不存在" in resp["result"]["content"][0]["text"]

    # 4. resources/templates/list
    resp = _send(server, {
        "jsonrpc": "2.0",
        "id": 40,
        "method": "resources/templates/list",
        "params": {},
    })
    assert "error" not in resp
    templates = resp["result"]["resourceTemplates"]
    names = {t["name"] for t in templates}
    expected_resources = {
        "课程知识点树",
        "课程文档清单",
        "课程统计",
        "学生学情报告",
        "学生掌握度画像",
    }
    assert expected_resources.issubset(names)

    # 5. resources/read
    resp = _send(server, {
        "jsonrpc": "2.0",
        "id": 50,
        "method": "resources/read",
        "params": {
            "uri": "course://550e8400-e29b-41d4-a716-446655440000/kp-tree",
        },
    })
    assert "error" not in resp
    contents = resp["result"]["contents"]
    assert len(contents) == 1
    assert contents[0]["mimeType"] == "application/json"
    assert "uri" in contents[0]

    # 6. prompts/list
    resp = _send(server, {
        "jsonrpc": "2.0",
        "id": 60,
        "method": "prompts/list",
        "params": {},
    })
    assert "error" not in resp
    prompts = resp["result"]["prompts"]
    names = {p["name"] for p in prompts}
    expected_prompts = {"tutor_socratic", "quiz_blueprint", "diagnosis_report"}
    assert expected_prompts.issubset(names)

    # 7. prompts/get
    resp = _send(server, {
        "jsonrpc": "2.0",
        "id": 70,
        "method": "prompts/get",
        "params": {
            "name": "tutor_socratic",
            "arguments": {
                "course_id": "550e8400-e29b-41d4-a716-446655440000",
                "kp_path": "OS/进程管理/进程调度",
            },
        },
    })
    assert "error" not in resp
    assert "messages" in resp["result"]
    assert len(resp["result"]["messages"]) > 0
