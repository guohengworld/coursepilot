"""Gateway 可观测：访问日志中间件（P1-T3 拆分自 gateway/main.py）。

职责：
- 每个 HTTP 请求记录一行访问日志：时间 / key 前缀 / user / role / 路径 /
  工具名 / 延迟 / 状态，不记录参数与响应（R4.9 脱敏）。
- 工具名提取优先读 MCP 2.0 头路由（``MCP-Protocol-Version`` /
  ``Mcp-Method`` / ``Mcp-Name``），body 解析仅兜底——比"每次解析 body
  再重放"更稳更安全。

中间件栈（从外到内）：AuthenticationMiddleware → AccessLogMiddleware →
MCP 路由。鉴权失败由 AuthenticationMiddleware 直接返回 401 并记日志，
本中间件只记录鉴权通过后的成功请求。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from coursepilot.mcp.principal import principal_var

_LOGGER = logging.getLogger("coursepilot.mcp.gateway")

# MCP 2.0 头路由：优先从这些头提取方法/工具名（body 解析仅兜底）
_MCP_PROTOCOL_HEADER = b"mcp-protocol-version"
_MCP_METHOD_HEADER = b"mcp-method"
_MCP_NAME_HEADER = b"mcp-name"


def _header(headers: list[tuple[bytes, bytes]], name: bytes) -> str | None:
    """按字节小写匹配头名，返回首个值（去空白）。"""
    for k, v in headers:
        if k.lower() == name:
            return v.decode("latin-1").strip() or None
    return None


def _extract_tool(headers: list[tuple[bytes, bytes]],
                  body: bytes | None = None) -> str:
    """从 MCP 头路由提取工具名；头缺失时兜底解析 body。

    Returns:
        工具名 / 方法名；无法识别返回空串。
    """
    name = _header(headers, _MCP_NAME_HEADER)
    if name:
        return name
    method = _header(headers, _MCP_METHOD_HEADER)
    if method:
        return method
    if body:
        try:
            req = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return ""
        if isinstance(req, dict):
            m = req.get("method", "")
            if m == "tools/call":
                params = req.get("params") or {}
                if isinstance(params, dict):
                    return str(params.get("name") or m)
            return str(m or "")
    return ""


class AccessLogMiddleware:
    """ASGI 中间件：记录成功请求的访问日志（脱敏，不含参数/响应）。

    读 ``principal_var`` 拿身份（由外层 AuthenticationMiddleware 注入）。
    工具名提取：优先读 MCP 头路由（零开销）；头缺失时 capture body 提取
    并原样重放给下游（兼容旧客户端/测试）。
    """

    def __init__(self, app: Callable[..., Awaitable[None]]):
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope["path"]
        start = time.monotonic()
        status_holder = {"status": 200}
        headers = scope.get("headers", [])

        # 优先头路由；头缺失时 capture body（并重放），提取工具名
        tool = _extract_tool(headers)
        if not tool:
            body, replay_receive = await self._capture_body(receive)
            tool = _extract_tool(headers, body)
        else:
            replay_receive = receive

        async def send_wrapper(msg):
            if msg["type"] == "http.response.start":
                status_holder["status"] = msg.get("status", 200)
            await send(msg)

        try:
            await self.app(scope, replay_receive, send_wrapper)
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            p = principal_var.get()
            key_prefix = p.api_key_prefix if p else "-"
            user_id = p.user_id if p else "-"
            role = p.role if p else "-"
            _LOGGER.info(
                "access ts=%s key=%s user=%s role=%s path=%s tool=%s "
                "latency_ms=%.0f status=%d",
                datetime.now(UTC).isoformat(timespec="seconds"),
                key_prefix, user_id, role, path, tool or "-",
                latency_ms, status_holder["status"],
            )

    @staticmethod
    async def _capture_body(receive) -> tuple[bytes, Callable[[], Awaitable[dict]]]:
        """读取完整请求体，返回可重放的 receive（不破坏下游）。"""
        body = b""
        more = True
        while more:
            msg = await receive()
            if msg["type"] == "http.request":
                body += msg.get("body", b"")
                more = msg.get("more_body", False)
            else:
                break

        replayed = False

        async def replay_receive() -> dict:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        return body, replay_receive
