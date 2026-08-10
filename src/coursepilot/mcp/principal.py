"""Principal 注入基座。

把「调用方身份」作为一等公民：网关鉴权后将 ``Principal`` 写入
``principal_var``（ContextVar），工具/资源在处理时经 ``get_principal()``
读取并做租户断言。未走网关直连调用时 ``principal_var`` 为 None，
``get_principal()`` 抛 ``UnauthenticatedError``，杜绝匿名执行。

- 主路径：ContextVar 在同一 task 内透传（集成测试已验证）。
- 备用路径：若跨 SDK 内部 task 边界透传失败，改由网关注入
  ``X-Principal-User-Id`` / ``X-Principal-Role`` / ``X-Principal-Scopes``
  头，工具从 ``ctx.headers`` 读取。
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field

from coursepilot.mcp.shared.errors import UnauthenticatedError


@dataclass(frozen=True)
class Principal:
    """已认证调用方身份。

    Attributes:
        user_id: 调用方用户 ID（如 ``u-001``）。
        role: 角色，student / teacher / super。
        scopes: 授权 scope 集合（如 ``{"read"}`` / ``{"read", "write"}``）。
        api_key_prefix: 来源 Key 的脱敏前缀（仅用于日志，可空）。
    """

    user_id: str
    role: str  # student | teacher | super
    scopes: frozenset[str] = field(default_factory=frozenset)
    api_key_prefix: str = ""


principal_var: ContextVar[Principal | None] = ContextVar(
    "cp_principal", default=None
)


def get_principal() -> Principal:
    """读取当前请求的 Principal；未认证时抛 ``UnauthenticatedError``。

    Raises:
        UnauthenticatedError: ``principal_var`` 为 None（未走网关直连）。
    """
    p = principal_var.get()
    if p is None:
        raise UnauthenticatedError("未认证：请求缺少有效调用方身份")
    return p


def set_principal(p: Principal | None) -> object:
    """设置当前上下文 Principal，返回用于 reset 的 token。

    网关中间件在鉴权通过后调用；测试可注入/清空。
    """
    return principal_var.set(p)
