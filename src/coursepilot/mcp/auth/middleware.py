"""AuthenticationMiddleware：解析 Bearer Key → 校验 → 注入 Principal。

设计为原生 ASGI 中间件，鉴权通过后调用 ``principal_var.set(Principal(...))``，
工具/资源在同一 task 上下文内经 ``get_principal()`` 读取。

- 认证失败返回 401（JSON-RPC error，-32001）。
- 访问日志脱敏：只记 key 前缀 + 请求路径 + 状态 + 延迟，不记参数/响应（R4.9）。
- 不解析 body：仅读 ``Authorization`` 头；成功请求的完整访问日志
  （含工具名）由 ``gateway/observability.py`` 的 AccessLogMiddleware 负责。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from coursepilot.mcp.auth.keys import ApiKeyInfo, KeyStore
from coursepilot.mcp.principal import Principal, principal_var, set_principal

_LOGGER = logging.getLogger("coursepilot.mcp.auth")

# 无需 API Key 即可访问的公共路径
_PUBLIC_PATHS: frozenset[str] = frozenset({"/health"})

_401_BODY = (
    '{"jsonrpc":"2.0","id":null,"error":'
    '{"code":-32001,"message":"认证失败：无效的 API Key"}}'
).encode()


def _extract_bearer(headers: list[tuple[bytes, bytes]]) -> str | None:
    """从 ASGI headers 提取 Bearer token；缺失/格式错误返回 None。"""
    for name, value in headers:
        if name.lower() == b"authorization":
            parts = value.decode("latin-1").split(" ", 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1].strip() or None
            return None
    return None


class AuthenticationMiddleware:
    """ASGI 中间件：API Key 认证 + Principal 注入。

    Usage::

        app = FastAPI(...)
        app.add_middleware(AuthenticationMiddleware, key_store=store)
    """

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        key_store: KeyStore | None = None,
    ) -> None:
        self.app = app
        self.key_store = key_store  # 允许延迟绑定（测试注入）

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope["path"]
        start = time.monotonic()

        # 公共路径直接放行（不注入 Principal；工具层直连时 get_principal 会拒绝）
        if path in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            self._log("-", "-", path, 0, "", None)
            return

        key = _extract_bearer(scope.get("headers", []))
        info: ApiKeyInfo | None = None
        if key is not None:
            # 未显式注入 store 时用进程内单例：首次初始化读 env，之后零 IO
            store = self.key_store or KeyStore.get_default()
            info = store.lookup(key)

        if info is None:
            await self._send_401(send)
            self._log("-", "-", path, 401, "", (time.monotonic() - start) * 1000)
            return

        # 注入 Principal：同一请求后续（工具 handler）可在同一 task 内读取
        token = set_principal(
            Principal(
                user_id=info.user_id,
                role=info.role,
                scopes=info.scopes,
                api_key_prefix=info.api_key_prefix,
            )
        )
        try:
            await self.app(scope, receive, send)
        finally:
            # 请求结束恢复 ContextVar（reset 到 set 前状态），避免泄漏到复用连接
            principal_var.reset(token)

    @staticmethod
    async def _send_401(send) -> None:
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(_401_BODY)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": _401_BODY})

    @staticmethod
    def _log(key_prefix: str, user_id: str, path: str, status: int,
             tool: str, latency_ms: float | None) -> None:
        _LOGGER.info(
            "access ts=%s key=%s user=%s path=%s tool=%s status=%d latency_ms=%s",
            datetime.now(UTC).isoformat(timespec="seconds"),
            key_prefix or "-", user_id or "-", path, tool or "-", status,
            f"{latency_ms:.0f}" if latency_ms is not None else "-",
        )
