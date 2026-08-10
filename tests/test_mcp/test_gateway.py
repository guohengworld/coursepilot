"""Gateway 端到端测试：/health、/mcp 鉴权与 tools/list、访问日志。

覆盖健康检查、Streamable HTTP /mcp、API Key 校验端到端、访问日志。

P1-T3 变更：移除 /sse 兼容端点相关用例（SSE 已按规范弃用并从网关删除）；
fixture 直接使用 SDK app 的自带 lifespan（TestClient 正常 enter/exit）。
"""

import json
import logging
import os

import pytest
from starlette.testclient import TestClient

_VALID_KEYS = {
    "cp_test1234": {"user_id": "u-test", "role": "super"},
    "cp_teacher01": {"user_id": "u-teacher", "role": "teacher"},
    "cp_student01": {"user_id": "u-student", "role": "student"},
}

_TOOLS_LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}


@pytest.fixture(scope="module")
def client():
    os.environ["COURSEPILOT_MCP_API_KEYS"] = json.dumps(_VALID_KEYS)
    from coursepilot.mcp.gateway.app import create_app

    c = TestClient(create_app())
    c.__enter__()
    yield c
    c.__exit__(None, None, None)


# ── 健康检查 ─────────────────────────────────────────────
def test_health_ok(client):
    """/health 返回 200 且无需 API Key。"""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── API Key 校验（端到端）──────────────────────────────
def test_mcp_missing_key_401(client):
    """缺失 Authorization 头 → 401。"""
    r = client.post("/mcp", json=_TOOLS_LIST)
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["code"] == -32001


def test_mcp_invalid_key_401(client):
    """无效 key → 401。"""
    r = client.post(
        "/mcp", json=_TOOLS_LIST, headers={"Authorization": "Bearer cp_wrong"}
    )
    assert r.status_code == 401


def test_mcp_wrong_scheme_401(client):
    """非 Bearer 方案 → 401。"""
    r = client.post(
        "/mcp", json=_TOOLS_LIST, headers={"Authorization": "Basic cp_test1234"}
    )
    assert r.status_code == 401


def test_mcp_three_valid_keys_pass(client):
    """三把不同有效 key 均通过校验（非 401）。"""
    for key in ("cp_test1234", "cp_teacher01", "cp_student01"):
        r = client.post(
            "/mcp", json=_TOOLS_LIST, headers={"Authorization": f"Bearer {key}"}
        )
        assert r.status_code == 200, f"{key} 应通过校验"


# ── Streamable HTTP /mcp ──────────────────────────────
def test_mcp_tools_list(client):
    """有效 key 调 tools/list 返回全部 7 个工具。"""
    r = client.post(
        "/mcp", json=_TOOLS_LIST, headers={"Authorization": "Bearer cp_test1234"}
    )
    assert r.status_code == 200
    names = {t["name"] for t in r.json()["result"]["tools"]}
    expected = {
        "query_knowledge_tool",
        "generate_practice_tool",
        "grade_answers_tool",
        "diagnose_tool",
        "get_review_plan_tool",
        "search_knowledge_units_tool",
        "get_kp_tree_tool",
    }
    assert expected.issubset(names)


@pytest.mark.skip(
    reason="TestClient 的 lifespan portal 事件循环与 asyncpg 不兼容，tools/call "
    "触发数据库查询会卡住；该工具执行路径已由 test_stdio.py（subprocess 独立 "
    "事件循环）覆盖。Gateway 传输层（鉴权/路由/序列化）由 tools/list 验证。"
)
def test_mcp_tools_call_invalid_course(client):
    """有效 key 调 tools/call（不存在的课程）→ isError=true。"""
    req = {
        "jsonrpc": "2.0",
        "id": 2,
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
    }
    r = client.post(
        "/mcp", json=req, headers={"Authorization": "Bearer cp_test1234"}
    )
    assert r.status_code == 200
    result = r.json()["result"]
    assert result["isError"] is True
    assert "不存在" in result["content"][0]["text"]


# ── 访问日志 ──────────────────────────────────────────
def test_access_log_emitted(client, caplog):
    """成功请求产生 access 日志，含 key 前缀、tool 名与状态。

    401 由 AuthenticationMiddleware（logger=coursepilot.mcp.auth）记录，
    gateway logger 仅记录鉴权通过的成功请求。
    """
    caplog.set_level(logging.INFO, logger="coursepilot.mcp.gateway")
    client.post(
        "/mcp",
        json=_TOOLS_LIST,
        headers={"Authorization": "Bearer cp_test1234"},
    )  # 200
    access_lines = [r.message for r in caplog.records if r.message.startswith("access")]
    assert len(access_lines) >= 1
    ok_line = next(line for line in access_lines if "cp_tes" in line)
    assert "tools/list" in ok_line
    assert "status=200" in ok_line
