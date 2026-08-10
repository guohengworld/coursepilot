"""KeyStore：API Key 启动载入 + 查找。

相比旧 ``gateway/auth.py`` 每次请求 ``os.getenv`` 解析 JSON 的做法，
本模块在进程启动时一次性载入 key 表，查找走进程内字典（支持
``/reload`` 触发热重载）。

Key 格式（环境变量 ``COURSEPILOT_MCP_API_KEYS``，JSON 对象）::

    {"cp_abcdef1234": {"user_id": "u-001", "role": "student", "scopes": ["read"]},
     "cp_teacher01":  {"user_id": "u-002", "role": "teacher",
                        "scopes": ["read", "write"]}}

role → 默认 scopes 映射（未显式声明 scopes 时使用）：
    super: {read, write}；teacher: {read, write}；student: {read}。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from coursepilot.config import settings

# 角色 → 默认 scopes（集中定义；后续如需按工具细化，扩展为 role → scopes → tools）
_ROLE_SCOPES: dict[str, frozenset[str]] = {
    "super": frozenset({"read", "write"}),
    "teacher": frozenset({"read", "write"}),
    "student": frozenset({"read"}),
}


class ApiKeyError(Exception):
    """API Key 缺失或无效。"""


@dataclass(frozen=True)
class ApiKeyInfo:
    """校验通过后的 API Key 身份信息。"""

    api_key_prefix: str  # 脱敏前缀，如 cp_ab
    user_id: str
    role: str  # student / teacher / super
    scopes: frozenset[str] = field(default_factory=frozenset)


def _prefix(key: str) -> str:
    """取脱敏前缀（前 6 位，足够定位又不会泄露完整 Key）。"""
    return key[:6] if len(key) >= 6 else key


def _resolve_scopes(role: str, declared: list[str] | None) -> frozenset[str]:
    """合并显式 scopes 与角色默认 scopes（显式优先，兜底默认）。"""
    if declared:
        return frozenset(declared)
    return _ROLE_SCOPES.get(role, frozenset({"read"}))


class KeyStore:
    """启动时载入的 API Key 表。

    Usage::

        store = KeyStore.load()          # 读环境变量 + settings
        info = store.lookup("cp_xxx")    # ApiKeyInfo | None
    """

    def __init__(self, table: dict[str, ApiKeyInfo]):
        self._table = table

    @classmethod
    def load(cls, *, env_json: str | None = None,
             single_key: str | None = None) -> KeyStore:
        """从环境变量 / settings 载入 key 表（可扩展 ``/reload`` 重载入口）。

        Args:
            env_json: 多 key JSON；默认读 ``COURSEPILOT_MCP_API_KEYS``。
            single_key: 单 key（settings.mcp_api_key 兜底），映射 local/super。
        """
        table: dict[str, ApiKeyInfo] = {}

        raw = env_json if env_json is not None else os.getenv(
            "COURSEPILOT_MCP_API_KEYS", "")
        if raw.strip():
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ApiKeyError(
                    "COURSEPILOT_MCP_API_KEYS 不是合法 JSON"
                ) from exc
            if not isinstance(data, dict):
                raise ApiKeyError("COURSEPILOT_MCP_API_KEYS 必须是 JSON 对象")
            for k, v in data.items():
                if not isinstance(k, str) or not k:
                    continue
                info = v if isinstance(v, dict) else {}
                table[k] = ApiKeyInfo(
                    api_key_prefix=_prefix(k),
                    user_id=str(info.get("user_id", "")),
                    role=str(info.get("role", "student")),
                    scopes=_resolve_scopes(
                        str(info.get("role", "student")),
                        info.get("scopes"),
                    ),
                )

        single = single_key if single_key is not None else settings.mcp_api_key
        if single:
            table.setdefault(
                single,
                ApiKeyInfo(_prefix(single), "local", "super",
                           frozenset({"read", "write"})),
            )

        return cls(table)

    def lookup(self, key: str) -> ApiKeyInfo | None:
        """按完整 key 查身份信息；不存在返回 None。"""
        return self._table.get(key)

    def keys(self) -> set[str]:
        """当前所有有效 key（供测试/运维）。"""
        return set(self._table.keys())
