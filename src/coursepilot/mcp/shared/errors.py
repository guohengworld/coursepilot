"""MCP JSON-RPC 错误码与错误响应工具。

错误码遵循 JSON-RPC 2.0 规范，并扩展了 MCP 场景专用错误。
所有错误文案需对 LLM 友好，指导模型如何修正。
"""

from typing import Any


class MCPErrorCode:
    """JSON-RPC 2.0 标准错误码 + MCP 扩展错误码。"""

    # 标准 JSON-RPC 错误码
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_ERROR = -32000

    # MCP 扩展错误码（-32001 ~ -32099 为服务端保留区间）
    AUTHENTICATION_ERROR = -32001
    AUTHORIZATION_ERROR = -32002
    RATE_LIMIT_ERROR = -32003
    IDEMPOTENCY_ERROR = -32004
    TIMEOUT_ERROR = -32005
    VALIDATION_ERROR = -32006
    DOWNSTREAM_ERROR = -32007
    CONFIRMATION_REQUIRED = -32008


class MCPError(Exception):
    """MCP 业务异常基类：携带稳定错误码，供工具层映射为 isError 响应。"""

    code: int = MCPErrorCode.INTERNAL_ERROR
    """稳定错误码（见 MCPErrorCode）。"""

    def __init__(self, message: str | None = None, *, code: int | None = None):
        super().__init__(message or ERROR_MESSAGES.get(self.code, "未知错误"))
        if code is not None:
            self.code = code


class UnauthenticatedError(MCPError):
    """未认证：请求未携带有效 Principal（未走网关 / Key 无效）。"""

    code = MCPErrorCode.AUTHENTICATION_ERROR


class ToolForbiddenError(MCPError):
    """已认证但无权执行：租户越权或缺少所需 scope。"""

    code = MCPErrorCode.AUTHORIZATION_ERROR


ERROR_MESSAGES: dict[int, str] = {
    MCPErrorCode.PARSE_ERROR: "请求 JSON 格式错误，请检查请求体是否为合法 JSON。",
    MCPErrorCode.INVALID_REQUEST: "请求不符合 JSON-RPC 2.0 格式，请检查 jsonrpc/version/id 字段。",
    MCPErrorCode.METHOD_NOT_FOUND: "调用的方法不存在，请使用 tools/list 查询可用方法。",
    MCPErrorCode.INVALID_PARAMS: "参数非法，请根据 inputSchema 修正参数类型和取值。",
    MCPErrorCode.INTERNAL_ERROR: "服务器内部错误，请稍后重试或联系管理员。",
    MCPErrorCode.SERVER_ERROR: "服务端通用错误，请稍后重试。",
    MCPErrorCode.AUTHENTICATION_ERROR: "认证失败，请检查 API Key 或重新完成 OAuth 授权流程。",
    MCPErrorCode.AUTHORIZATION_ERROR: "无权访问该资源或工具，请确认当前身份是否有对应 scope。",
    MCPErrorCode.RATE_LIMIT_ERROR: "调用频率超限或 Token 预算耗尽，请降低频率或申请提额。",
    MCPErrorCode.IDEMPOTENCY_ERROR: "幂等键处理失败，请更换 idempotency_key 后重试。",
    MCPErrorCode.TIMEOUT_ERROR: "后端处理超时，请稍后重试或检查任务状态。",
    MCPErrorCode.VALIDATION_ERROR: "业务校验失败，请根据错误信息修正参数。",
    MCPErrorCode.DOWNSTREAM_ERROR: "下游依赖（LLM/数据库/向量库）异常，已触发降级策略。",
    MCPErrorCode.CONFIRMATION_REQUIRED: "该操作需要二次确认，请先调用获取 confirmation_token。",
}


def make_rpc_error(
    code: int,
    message: str | None = None,
    data: dict[str, Any] | None = None,
    request_id: Any = None,
) -> dict[str, Any]:
    """构造标准 JSON-RPC error 响应。

    Args:
        code: JSON-RPC 错误码。
        message: 错误信息，缺省时使用内置文案。
        data: 额外错误数据（可包含 field、hint 等 LLM 可理解的信息）。
        request_id: 对应请求的 id。

    Returns:
        JSON-RPC error 响应字典。
    """
    if message is None:
        message = ERROR_MESSAGES.get(code, "未知错误")

    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data

    return {"jsonrpc": "2.0", "id": request_id, "error": error}
