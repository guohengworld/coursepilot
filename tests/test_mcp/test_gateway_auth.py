"""API Key 校验单元测试。

覆盖有效 / 无效 / 缺失 key 各 3 组，纯函数测试不启动 Gateway。
"""

import json

import pytest

from coursepilot.mcp.gateway.auth import (
    ApiKeyInfo,
    InvalidApiKeyError,
    verify_authorization,
    verify_authorization_from_headers,
)

# 三把有效 key，分别对应三种角色
_VALID_KEYS = {
    "cp_test1234": {"user_id": "u-test", "role": "super"},
    "cp_teacher01": {"user_id": "u-teacher", "role": "teacher"},
    "cp_student01": {"user_id": "u-student", "role": "student"},
}


@pytest.fixture(autouse=True)
def _set_keys(monkeypatch):
    """注入测试用 key 表，并清空单 key 配置避免干扰。"""
    monkeypatch.setenv("COURSEPILOT_MCP_API_KEYS", json.dumps(_VALID_KEYS))
    monkeypatch.setattr("coursepilot.config.settings.mcp_api_key", "")


# ── 有效 key（3 组）────────────────────────────────────────
@pytest.mark.parametrize(
    "key,expected_role",
    [
        ("cp_test1234", "super"),
        ("cp_teacher01", "teacher"),
        ("cp_student01", "student"),
    ],
    ids=["super", "teacher", "student"],
)
def test_valid_keys(key, expected_role):
    info = verify_authorization(f"Bearer {key}")
    assert isinstance(info, ApiKeyInfo)
    assert info.role == expected_role
    assert info.api_key_prefix == key[:6]
    assert info.user_id


# ── 无效 key（3 组）────────────────────────────────────────
@pytest.mark.parametrize(
    "key", ["cp_wrong", "cp_expired99", "random-no-prefix"], ids=["wrong", "expired", "no-prefix"]
)
def test_invalid_keys(key):
    with pytest.raises(InvalidApiKeyError):
        verify_authorization(f"Bearer {key}")


# ── 缺失 key（3 组）────────────────────────────────────────
def test_missing_key_none():
    with pytest.raises(InvalidApiKeyError):
        verify_authorization(None)


def test_missing_key_empty():
    with pytest.raises(InvalidApiKeyError):
        verify_authorization("")


def test_missing_key_wrong_scheme():
    """非 Bearer 方案应拒绝。"""
    with pytest.raises(InvalidApiKeyError):
        verify_authorization("Basic cp_test1234")


# ── 边界 ───────────────────────────────────────────────────
def test_bearer_no_token():
    with pytest.raises(InvalidApiKeyError):
        verify_authorization("Bearer")


def test_prefix_masking():
    """日志脱敏：只保留前 6 位，不含完整 key。"""
    info = verify_authorization("Bearer cp_test1234")
    assert info.api_key_prefix == "cp_tes"
    assert "1234" not in info.api_key_prefix


def test_verify_from_asgi_headers():
    """从 ASGI scope headers 列表提取并校验。"""
    headers = [(b"authorization", b"Bearer cp_test1234")]
    info = verify_authorization_from_headers(headers)
    assert info.role == "super"


def test_verify_from_asgi_headers_missing():
    headers = [(b"content-type", b"application/json")]
    with pytest.raises(InvalidApiKeyError):
        verify_authorization_from_headers(headers)
