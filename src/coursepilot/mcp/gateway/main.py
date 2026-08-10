"""CoursePilot MCP Gateway（轻量）。

职责（见 MVP 设计第 4、5 节）：

- TLS 终结（生产由 Nginx 反代或 uvicorn ``--ssl-certfile`` 提供）
- API Key 校验（``Authorization: Bearer cp_xxx``）
- 请求路由：
    ``GET  /health``  健康检查（无需 API Key）
    ``POST /mcp``     Streamable HTTP 端点（主传输）
    ``GET  /sse``     SSE 兼容端点（旧版客户端）
    ``POST /messages/`` SSE 消息回传
- 最小访问日志（不记录参数与响应，避免隐私泄露）

架构：Gateway 是一个 FastAPI app，复用单体 Server 的 MCPServer 实例
（``coursepilot.mcp.server.mcp``），把其 ``streamable_http_app`` 与 ``sse_app``
的路由合并进来，外层用一个 ASGI 中间件统一做 API Key 校验与访问日志。
Streamable HTTP 采用无状态模式（``stateless_http=True``），每个请求独立，
契合 MCP 2026-07-28 无状态修订。

启动：

    PYTHONPATH=src uv run python -m coursepilot.mcp.gateway.main --port 8080
    # 生产 HTTPS（自签或正式证书）
    PYTHONPATH=src uv run python -m coursepilot.mcp.gateway.main \\
        --ssl-certfile cert.pem --ssl-keyfile key.pem

TLS/HTTPS：

  Gateway 只暴露 HTTPS，有两种方式：

  1. uvicorn 直启 TLS（适合单机/测试）——先生成自签证书：

         openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem \\
             -days 365 -nodes -subj "/CN=localhost"

     启动后验证：

         curl -k https://localhost:8080/health          # 应返回 {"status":"ok"}
         curl -k -H "Authorization: Bearer cp_xxx" \\
             https://localhost:8080/mcp -d @tools/list.json

  2. Nginx 反代 TLS（生产推荐）——Nginx 终结 TLS，回源到 Gateway 的 HTTP：

         server {
             listen 443 ssl;
             server_name mcp.coursepilot.example.com;
             ssl_certificate     /etc/ssl/certs/coursepilot.pem;
             ssl_certificate_key /etc/ssl/private/coursepilot.key;
             location / {
                 proxy_pass http://127.0.0.1:8080;
                 proxy_http_version 1.1;
                 proxy_set_header Host $host;
                 proxy_buffering off;          # SSE 流式不缓冲
                 proxy_read_timeout 300s;      # SSE 长连接
             }
         }

     此时 Gateway 用 HTTP 启动（--ssl-certfile 留空），只监听 127.0.0.1。
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Awaitable, Callable

from fastapi import FastAPI

from coursepilot.config import settings
from coursepilot.mcp.gateway.auth import (
    InvalidApiKeyError,
    verify_authorization_from_headers,
)
from coursepilot.mcp.server import mcp

_LOGGER = logging.getLogger("coursepilot.mcp.gateway")

# 无需 API Key 即可访问的公共路径
_PUBLIC_PATHS = {"/health"}


def _extract_tool(body: bytes) -> str:
    """从 JSON-RPC 请求体中提取工具名（用于访问日志）。

    tools/call 取 params.name；其它请求取 method；解析失败返回空串。
    """
    if not body:
        return ""
    try:
        req = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return ""
    if not isinstance(req, dict):
        return ""
    method = req.get("method", "")
    if method == "tools/call":
        params = req.get("params") or {}
        if isinstance(params, dict):
            return str(params.get("name") or method)
    return str(method or "")


class GatewayMiddleware:
    """ASGI 中间件：API Key 校验 + 最小访问日志。

    设计为纯 ASGI 中间件（而非 Starlette HTTPMiddleware），因为需要
    "偷看"请求体提取工具名后再原样重放给下游 MCPServer 路由——
    HTTPMiddleware 消费 body 后无法回灌给挂载的 ASGI 子应用。
    """

    def __init__(self, app: Callable[..., Awaitable[None]]):
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope["path"]

        # 公共路径直接放行并记录日志
        if path in _PUBLIC_PATHS:
            await self._proxy(scope, receive, send, key_prefix="-",
                              user_id="-", role="-", path=path)
            return

        # 1. API Key 校验（仅读 header，不碰 body）
        try:
            key_info = verify_authorization_from_headers(scope.get("headers", []))
        except InvalidApiKeyError as exc:
            await self._send_json(send, 401, {
                "jsonrpc": "2.0",
                "error": {"code": -32001, "message": f"认证失败：{exc}"},
            })
            self._log("-", "-", "-", path, "", 0.0, 401)
            return

        # 2. 偷看 body 提取工具名，再重放给下游
        body, replay_receive = await self._capture_body(receive)
        tool = _extract_tool(body)

        await self._proxy(scope, replay_receive, send,
                          key_prefix=key_info.api_key_prefix,
                          user_id=key_info.user_id,
                          role=key_info.role,
                          path=path, tool=tool)

    async def _proxy(self, scope, receive, send, *, key_prefix, user_id,
                     role, path, tool="") -> None:
        """转发给下游并记录访问日志。"""
        status_holder = {"status": 200}

        async def send_wrapper(msg):
            if msg["type"] == "http.response.start":
                status_holder["status"] = msg.get("status", 200)
            await send(msg)

        start = time.monotonic()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            self._log(key_prefix, user_id, role, path, tool,
                      latency_ms, status_holder["status"])

    @staticmethod
    async def _capture_body(receive) -> tuple[bytes, Callable[[], Awaitable[dict]]]:
        """读取完整请求体并返回一个可重放的 receive 可调用对象。"""
        body = b""
        more = True
        while more:
            msg = await receive()
            if msg["type"] == "http.request":
                body += msg.get("body", b"")
                more = msg.get("more_body", False)
            else:
                # http.disconnect 等非 body 消息：停止读取
                break

        replayed = False

        async def replay_receive() -> dict:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            # 下游继续读取时返回空 body + 结束标记，避免卡住
            return {"type": "http.request", "body": b"", "more_body": False}

        return body, replay_receive

    @staticmethod
    async def _send_json(send, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    def _log(key_prefix, user_id, role, path, tool, latency_ms, status) -> None:
        _LOGGER.info(
            "access ts=%s key=%s user=%s role=%s path=%s tool=%s "
            "latency_ms=%.0f status=%d",
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            key_prefix, user_id, role, path, tool or "-", latency_ms, status,
        )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """接入 Streamable HTTP 的 session manager（合并 routes 后必须手动启动）。

    [PRIVATE-API-MARK] 访问 ``mcp._lowlevel_server._session_manager`` 是
    双下划线私有依赖（待清理）。
    实测依据（mcp 2.0.0）：``_handle_request`` 首行即检查
    ``self._task_group``，无状态模式同样依赖 ``session_manager.run()`` 初始化
    task group；SDK 返回的 Starlette app 自带 ``lifespan=session_manager.run``，
    正确姿势是不合并 routes 直接挂载，从而删除本函数与私有访问。
    """
    session_manager = getattr(mcp._lowlevel_server, "_session_manager", None)
    if session_manager is not None:
        async with session_manager.run():
            yield
    else:
        yield


def create_app() -> FastAPI:
    """构建 Gateway FastAPI app（供 uvicorn 与测试复用）。"""
    # 复用单体 Server 的 MCPServer 实例，取其两种传输的路由。
    # 调用 streamable_http_app 会把 session_manager 挂到 lowlevel server，
    # 供 _lifespan 启动。
    # 传 host=settings.mcp_host（0.0.0.0）：SDK 仅在 localhost 时自动启用 DNS
    # rebinding protection（会校验 Host header），Gateway 面向远程/测试客户端，
    # 不应启用该限制，生产 Host 校验由前置 Nginx 负责。
    mcp_http = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,  # MVP 无状态：每个请求独立
        json_response=True,   # 返回纯 JSON 而非 SSE 流，契合无状态单次请求-响应
        host=settings.mcp_host,
    )
    # [DEPRECATED-MARK] SSE 传输已于 MCP 2026-07-28 规范弃用。
    # 当前仅因 test_gateway.py 的 /sse 用例保留，重构时删除本段与下方路由合并。
    mcp_sse = mcp.sse_app(
        sse_path="/sse", message_path="/messages/", host=settings.mcp_host
    )

    app = FastAPI(
        title="CoursePilot MCP Gateway",
        version="0.1.0",
        lifespan=_lifespan,
    )

    @app.get("/health")
    async def health() -> dict:
        """健康检查（无需 API Key）。"""
        return {"status": "ok"}

    # 合并 MCPServer 的两种传输路由到 Gateway
    app.router.routes.extend(mcp_http.routes)
    app.router.routes.extend(mcp_sse.routes)

    # 中间件在路由合并后添加：Starlette 中间件包裹整个 app，
    # 后添加的先执行（最外层）。GatewayMiddleware 需在最外层拦截。
    app.add_middleware(GatewayMiddleware)

    return app


def main() -> None:
    """Gateway 入口：用 uvicorn 启动。"""
    parser = argparse.ArgumentParser(description="CoursePilot MCP Gateway")
    parser.add_argument("--host", default=settings.mcp_host, help="监听地址")
    parser.add_argument("--port", type=int, default=settings.mcp_port, help="监听端口")
    parser.add_argument("--ssl-certfile", default=None, help="TLS 证书文件（启用 HTTPS）")
    parser.add_argument("--ssl-keyfile", default=None, help="TLS 私钥文件（启用 HTTPS）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import uvicorn

    scheme = "https" if args.ssl_certfile else "http"
    _LOGGER.info("启动 Gateway（%s://%s:%d）", scheme, args.host, args.port)
    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )


if __name__ == "__main__":
    main()
