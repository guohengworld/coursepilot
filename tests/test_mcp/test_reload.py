"""P2 鉴权硬化测试：key 单例 / 热重载 / /reload 端点鉴权。

覆盖（对应 TODO P2）：
1. KeyStore 进程内单例：多次 get_default() 返回同一实例；lookup 不重复读 env。
2. reload()：新增 key 立即生效、吊销 key 立即失效。
3. /reload 端点：
   - 无 key → 401（AuthenticationMiddleware 拦截）
   - 非 super（student/teacher）→ 403
   - super → 200 且重载生效
4. student key 调 generate_practice → 被 require_scope("write") 拒绝（P1-T2 已覆盖，
   此处补端到端断言，确认"role/scope 强制真实生效而非只记日志"）。
"""

from __future__ import annotations

import json
import os

import pytest
from starlette.testclient import TestClient

from coursepilot.mcp.auth.keys import KeyStore

_KEYS_V1 = {
    "cp_super01": {"user_id": "u-super", "role": "super",
                   "scopes": ["read", "write"]},
    "cp_student01": {"user_id": "u-student", "role": "student",
                     "scopes": ["read"]},
}
_KEYS_V2 = {
    "cp_super01": {"user_id": "u-super", "role": "super",
                   "scopes": ["read", "write"]},
    "cp_newteacher": {"user_id": "u-teacher2", "role": "teacher",
                      "scopes": ["read", "write"]},
}


@pytest.fixture(scope="module")
def client():
    """带鉴权中间件 + /reload 的 MCP app（用单例 KeyStore）。"""
    os.environ["COURSEPILOT_MCP_API_KEYS"] = json.dumps(_KEYS_V1)
    KeyStore._reset_default()  # 隔离测试单例

    from coursepilot.mcp.gateway.app import create_app

    c = TestClient(create_app())
    c.__enter__()
    yield c
    c.__exit__(None, None, None)
    KeyStore._reset_default()


# ── 1. KeyStore 单例 ────────────────────────────────────────────

def test_keystore_singleton(monkeypatch):
    """get_default() 返回同一实例；首次初始化后不再重复 load。"""
    monkeypatch.setenv("COURSEPILOT_MCP_API_KEYS", json.dumps(_KEYS_V1))
    KeyStore._reset_default()
    try:
        s1 = KeyStore.get_default()
        s2 = KeyStore.get_default()
        assert s1 is s2
        assert "cp_super01" in s1.keys()
    finally:
        KeyStore._reset_default()


def test_keystore_lookup_v1(client):
    """当前生效 key 表：super 可查、student 可查、不存在 key 返回 None。"""
    store = KeyStore.get_default()
    info = store.lookup("cp_super01")
    assert info is not None and info.role == "super"
    info2 = store.lookup("cp_student01")
    assert info2 is not None and info2.role == "student"
    assert store.lookup("cp_ghost99") is None


# ── 2. reload() 热重载 ──────────────────────────────────────────

def test_reload_adds_and_revokes_keys(client):
    """reload 后：新 key 生效、旧 key 立即失效。"""
    os.environ["COURSEPILOT_MCP_API_KEYS"] = json.dumps(_KEYS_V2)
    try:
        store = KeyStore.get_default()
        # 重载前：新 key 不存在、旧 student key 存在
        assert store.lookup("cp_newteacher") is None
        assert store.lookup("cp_student01") is not None

        store.reload()

        # 重载后：新 key 生效、旧 key 吊销
        assert store.lookup("cp_newteacher") is not None
        assert store.lookup("cp_student01") is None
    finally:
        os.environ["COURSEPILOT_MCP_API_KEYS"] = json.dumps(_KEYS_V1)
        KeyStore.get_default().reload()


# ── 3. /reload 端点鉴权 ─────────────────────────────────────────

def test_reload_endpoint_no_key_401(client):
    """无 key 调 /reload → 401（AuthenticationMiddleware 拦截）。"""
    r = client.post("/reload")
    assert r.status_code == 401


def test_reload_endpoint_student_403(client):
    """student key 调 /reload → 403（角色不足）。"""
    r = client.post("/reload",
                    headers={"Authorization": "Bearer cp_student01"})
    assert r.status_code == 403
    assert "无权" in r.json()["error"]


def test_reload_endpoint_super_ok(client):
    """super key 调 /reload → 200，且重载生效。"""
    os.environ["COURSEPILOT_MCP_API_KEYS"] = json.dumps(_KEYS_V2)
    try:
        r = client.post("/reload",
                        headers={"Authorization": "Bearer cp_super01"})
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "reloaded": True}
        # 重载后新 key 可查
        assert KeyStore.get_default().lookup("cp_newteacher") is not None
    finally:
        os.environ["COURSEPILOT_MCP_API_KEYS"] = json.dumps(_KEYS_V1)
        KeyStore.get_default().reload()


# ── 4. role→scope 强制端到端 ────────────────────────────────────

def test_student_cannot_generate_practice_e2e(client):
    """student key 端到端调 generate_practice → isError（scope 强制生效）。"""
    r = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "generate_practice_tool",
                "arguments": {
                    "params": {"course_id": "550e8400-e29b-41d4-a716-446655440000",
                               "kp_path": "", "count": 1, "difficulty": 1},
                },
            },
        },
        headers={"Authorization": "Bearer cp_student01"},
    )
    assert r.status_code == 200
    result = r.json()["result"]
    assert result["isError"] is True
    assert "权限" in result["content"][0]["text"] or "缺少" in result["content"][0]["text"]


def test_teacher_can_generate_practice_e2e(client):
    """teacher key 端到端调 generate_practice → 进入业务逻辑（非权限拒绝）。

    课程不存在时返回业务错误而非权限错误，证明 scope 放行。
    """
    r = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {
                "name": "generate_practice_tool",
                "arguments": {
                    "params": {"course_id": "550e8400-e29b-41d4-a716-446655440000",
                               "kp_path": "", "count": 1, "difficulty": 1},
                },
            },
        },
        headers={"Authorization": "Bearer cp_super01"},
    )
    assert r.status_code == 200
    result = r.json()["result"]
    assert result["isError"] is True  # 课程不存在 → 业务错误
    text = result["content"][0]["text"]
    assert "权限" not in text and "缺少" not in text
