"""MCP 错误码测试。"""

import pytest

from coursepilot.mcp.shared.errors import ERROR_MESSAGES, MCPErrorCode, make_rpc_error


def test_error_codes_unique():
    """所有错误码应唯一。"""
    codes = [
        MCPErrorCode.PARSE_ERROR,
        MCPErrorCode.INVALID_REQUEST,
        MCPErrorCode.METHOD_NOT_FOUND,
        MCPErrorCode.INVALID_PARAMS,
        MCPErrorCode.INTERNAL_ERROR,
        MCPErrorCode.SERVER_ERROR,
        MCPErrorCode.AUTHENTICATION_ERROR,
        MCPErrorCode.AUTHORIZATION_ERROR,
        MCPErrorCode.RATE_LIMIT_ERROR,
        MCPErrorCode.IDEMPOTENCY_ERROR,
        MCPErrorCode.TIMEOUT_ERROR,
        MCPErrorCode.VALIDATION_ERROR,
        MCPErrorCode.DOWNSTREAM_ERROR,
        MCPErrorCode.CONFIRMATION_REQUIRED,
    ]
    assert len(codes) == len(set(codes))


def test_error_messages_cover_all_codes():
    """每个错误码都有默认文案。"""
    for attr_name in dir(MCPErrorCode):
        if attr_name.startswith("_"):
            continue
        code = getattr(MCPErrorCode, attr_name)
        assert code in ERROR_MESSAGES


def test_make_rpc_error_default_message():
    """不指定 message 时使用默认文案。"""
    resp = make_rpc_error(MCPErrorCode.INVALID_PARAMS, request_id=1)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert resp["error"]["code"] == MCPErrorCode.INVALID_PARAMS
    assert "参数非法" in resp["error"]["message"]


def test_make_rpc_error_custom_message_and_data():
    """支持自定义 message 和 data。"""
    resp = make_rpc_error(
        MCPErrorCode.VALIDATION_ERROR,
        message="course_id 不存在",
        data={"field": "course_id"},
        request_id="abc",
    )
    assert resp["error"]["message"] == "course_id 不存在"
    assert resp["error"]["data"] == {"field": "course_id"}


@pytest.mark.parametrize(
    "code,expected_keyword",
    [
        (MCPErrorCode.AUTHENTICATION_ERROR, "认证失败"),
        (MCPErrorCode.RATE_LIMIT_ERROR, "频率"),
        (MCPErrorCode.TIMEOUT_ERROR, "超时"),
    ],
)
def test_error_message_keywords(code, expected_keyword):
    """关键错误文案包含预期关键字。"""
    resp = make_rpc_error(code, request_id=None)
    assert expected_keyword in resp["error"]["message"]
