"""MCP Gateway 应用构建（P1-T3 拆分自 gateway/main.py）。

正确姿势（2026-08-10 实测修正，见设计 §7）：
- 直接使用 ``mcp.streamable_http_app()`` 返回的 Starlette app——它**自带
  ``lifespan=session_manager.run``**，无状态模式同样依赖 task group；
- 在返回的 app 上追加自定义路由（``/health``）与中间件（鉴权 → 访问日志），
  不合并 routes 到 FastAPI、不手写 lifespan、不访问 SDK 私有属性。

启动：
    PYTHONPATH=src uv run python -m coursepilot.mcp.gateway.main --port 8080
"""

from __future__ import annotations

import logging

from starlette.responses import JSONResponse
from starlette.routing import Route

from coursepilot.config import settings
from coursepilot.mcp.auth.keys import KeyStore
from coursepilot.mcp.auth.middleware import AuthenticationMiddleware
from coursepilot.mcp.gateway.observability import AccessLogMiddleware
from coursepilot.mcp.server import mcp

_LOGGER = logging.getLogger("coursepilot.mcp.gateway")


async def _health(request) -> JSONResponse:
    """健康检查（无需 API Key，AuthenticationMiddleware 放行 /health）。"""
    return JSONResponse({"status": "ok"})


def create_app():
    """构建 MCP Gateway ASGI app（供 uvicorn 与测试复用）。

    返回 SDK 的 Starlette app 本身（自带 lifespan），仅追加 /health 路由
    与中间件栈：AuthenticationMiddleware（最外层）→ AccessLogMiddleware。
    """
    app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,  # 无状态：每个请求独立
        json_response=True,   # 返回纯 JSON 而非 SSE 流
        host=settings.mcp_host,
    )

    # 追加自定义路由（SDK 支持在返回的 app 上追加）
    app.router.routes.append(Route("/health", endpoint=_health, methods=["GET"]))

    # 中间件：先添加的先执行（最外层）。先加 AccessLog（内层，记录时
    # principal 仍有效），再加 Authentication（最外层，先鉴权后放行）。
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(AuthenticationMiddleware, key_store=KeyStore.load())

    return app
