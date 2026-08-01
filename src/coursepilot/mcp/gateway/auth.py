"""API Key 校验（MVP 阶段轻量认证）。

协议：

    Authorization: Bearer cp_xxxxxxxx

Key 必须以 ``cp_`` 前缀开头，便于辨识与日志脱敏（只记录前 6 位）。

Key 来源（后者覆盖前者）：

1. 环境变量 ``COURSEPILOT_MCP_API_KEYS``，JSON 格式，支持多 Key 与角色映射::

       {"cp_abcdef1234": {"user_id": "u-001", "role": "student"},
        "cp_9999":       {"user_id": "u-002", "role": "teacher"}}

2. ``settings.mcp_api_key``，单 Key，映射到 ``user_id="local"``、``role="super"``，
   供本地开发与单用户部署使用。

Gateway 层只校验 Key 是否有效（见 MVP 设计 5.3），更细粒度的"学生只能查自己"
在 Server 层用 Pydantic 校验实现。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from coursepilot.config import settings


@dataclass(frozen=True)
class ApiKeyInfo:
    """校验通过后的 API Key 身份信息。"""

    api_key_prefix: str  # 脱敏前缀，如 cp_ab
    user_id: str
    role: str  # student / teacher / super


class InvalidApiKeyError(Exception):
    """API Key 缺失或无效。错误信息可直接回传给客户端。"""


def _prefix(key: str) -> str:
    """取脱敏前缀（前 6 位，足够定位又不会泄露完整 Key）。"""
    return key[:6] if len(key) >= 6 else key


def _load_keys() -> dict[str, ApiKeyInfo]:
    """加载 API Key 表。每次调用读取最新环境变量，便于测试注入。"""
    table: dict[str, ApiKeyInfo] = {}

    # 1. 多 Key（JSON 环境变量）
    raw = os.getenv("COURSEPILOT_MCP_API_KEYS", "")
    if raw.strip():
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InvalidApiKeyError(
                "COURSEPILOT_MCP_API_KEYS 不是合法 JSON"
            ) from exc
        if not isinstance(data, dict):
            raise InvalidApiKeyError(
                "COURSEPILOT_MCP_API_KEYS 必须是 JSON 对象"
            )
        for k, v in data.items():
            if not isinstance(k, str) or not k:
                continue
            info = v if isinstance(v, dict) else {}
            table[k] = ApiKeyInfo(
                api_key_prefix=_prefix(k),
                user_id=str(info.get("user_id", "")),
                role=str(info.get("role", "student")),
            )

    # 2. 单 Key（settings），不覆盖已配置的多 Key
    single = settings.mcp_api_key
    if single:
        table.setdefault(
            single,
            ApiKeyInfo(_prefix(single), "local", "super"),
        )

    return table


def verify_authorization(authorization: str | None) -> ApiKeyInfo:
    """校验 ``Authorization`` header 值，返回身份信息或抛出 InvalidApiKeyError。

    Args:
        authorization: ``Authorization`` header 原始值，可能为 None。
    """
    if not authorization:
        raise InvalidApiKeyError("缺失 Authorization 头")

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise InvalidApiKeyError("Authorization 格式错误，应为 Bearer <api_key>")

    key = parts[1].strip()
    if not key:
        raise InvalidApiKeyError("Authorization 中 API Key 为空")

    table = _load_keys()
    info = table.get(key)
    if info is None:
        raise InvalidApiKeyError("无效的 API Key")
    return info


def verify_authorization_from_headers(headers: list[tuple[bytes, bytes]]) -> ApiKeyInfo:
    """从 ASGI scope 的 headers 列表中提取并校验 Authorization。"""
    authorization: str | None = None
    for name, value in headers:
        if name.lower() == b"authorization":
            authorization = value.decode("latin-1")
            break
    return verify_authorization(authorization)
