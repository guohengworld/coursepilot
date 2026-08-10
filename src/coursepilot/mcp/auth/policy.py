"""AuthorizationPolicy：租户断言 / scope 断言装饰器。

装饰器统一做授权断言，避免 8 个工具各写一遍：

- ``require_self_or_privileged("teacher", "super")``：断言 ``params.user_id``
  等于当前 Principal 的 user_id，或角色在特权集内（学生只能查自己）。
- ``require_scope("write")``：断言当前 Principal 的 scopes 覆盖所需 scope。

断言失败抛 ``ToolForbiddenError``（错误码 -32002，供工具层映射 isError）。
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable

from coursepilot.mcp.principal import get_principal
from coursepilot.mcp.shared.errors import ToolForbiddenError

# 允许的"当前用户"字段名（参数对象上的 user_id 字段）
_USER_ID_FIELD = "user_id"


def _resolve_user_id(params) -> str | None:
    """从参数对象提取 user_id；支持 Pydantic 模型 / dataclass / dict。"""
    if params is None:
        return None
    if isinstance(params, dict):
        raw = params.get(_USER_ID_FIELD)
    else:
        raw = getattr(params, _USER_ID_FIELD, None)
    return str(raw) if raw is not None else None


def require_self_or_privileged(*privileged_roles: str) -> Callable:
    """租户断言：学生只能操作自己的数据，teacher/super 可操作任意。

    Args:
        *privileged_roles: 可访问任意 user_id 的角色（如 teacher、super）。

    Raises:
        ToolForbiddenError: 当前 Principal 角色不在特权集，且目标
            user_id 非空且不等于 Principal.user_id。
    """

    def deco(fn: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
        @functools.wraps(fn)
        async def wrapper(params, *args, **kwargs):
            p = get_principal()
            uid = _resolve_user_id(params)
            if p.role not in privileged_roles and uid and uid != p.user_id:
                raise ToolForbiddenError(
                    f"无权访问用户 {uid} 的数据（当前身份 {p.user_id}）"
                )
            return await fn(params, *args, **kwargs)

        return wrapper

    return deco


def require_scope(*required_scopes: str) -> Callable:
    """scope 断言：Principal.scopes 须覆盖全部所需 scope。

    Args:
        *required_scopes: 至少需要的 scope（如 write）。

    Raises:
        ToolForbiddenError: 缺少任一所需 scope。
    """

    def deco(fn: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
        @functools.wraps(fn)
        async def wrapper(params, *args, **kwargs):
            p = get_principal()
            missing = [s for s in required_scopes if s not in p.scopes]
            if missing:
                raise ToolForbiddenError(
                    f"缺少所需权限 {missing}（当前 scopes: {sorted(p.scopes)}）"
                )
            return await fn(params, *args, **kwargs)

        return wrapper

    return deco
