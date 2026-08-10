"""MCP 本地 stdio-to-HTTP 桥接 CLI。

把本地宿主（WorkBuddy / Trae）通过 stdio 发来的 JSON-RPC 请求，
封装为 HTTPS POST 转发到远端 MCP Gateway，并把响应原样写回 stdout。
设计见 ``docs/mcp/MCP-MVP-Design.md`` §3.2。

数据流::

    本地 stdio 客户端（WorkBuddy / Trae）
        │  行分隔 JSON-RPC
        ▼
    ┌───────────────────────┐
    │ coursepilot-mcp CLI   │
    │ - 从 stdin 读 JSON-RPC │
    │ - 封装为 HTTPS POST    │
    │ - 带 API Key header    │
    │ - 响应写回 stdout      │
    └──────────┬────────────┘
               │ HTTPS
               ▼
        MCP Gateway（POST /mcp）

用法::

    PYTHONPATH=src uv run python -m coursepilot.mcp.cli
    # 或安装后：
    coursepilot-mcp --gateway https://mcp.coursepilot.example.com/mcp

环境变量（优先级：命令行参数 > 环境变量 > settings）::

    COURSEPILOT_MCP_GATEWAY  - 远端 Gateway URL（含 /mcp 路径）
    COURSEPILOT_MCP_API_KEY  - API Key（cp_ 前缀）

注意：stdout 仅供 JSON-RPC 响应使用，所有诊断/日志信息一律走 stderr，
以免污染与宿主之间的协议流。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import IO, Any

import httpx

from coursepilot.config import settings

# JSON-RPC 错误码（沿用规范约定）
_PARSE_ERROR = -32700      # 非 JSON 输入
_INVALID_REQUEST = -32600  # 非法请求结构
_INTERNAL_ERROR = -32603   # 桥接器内部/网络错误


def _resolve_gateway(cli_value: str | None) -> str:
    """解析远端 Gateway URL。

    优先级：命令行参数 > 环境变量 ``COURSEPILOT_MCP_GATEWAY`` > settings。
    """
    url = cli_value or os.getenv("COURSEPILOT_MCP_GATEWAY") or settings.mcp_gateway
    if not url:
        raise RuntimeError(
            "未配置 Gateway URL，请通过 --gateway 参数或环境变量 "
            "COURSEPILOT_MCP_GATEWAY 指定"
        )
    return url


def _resolve_api_key(cli_value: str | None) -> str:
    """解析 API Key。

    优先级：命令行参数 > 环境变量 ``COURSEPILOT_MCP_API_KEY`` > settings。
    """
    key = cli_value or os.getenv("COURSEPILOT_MCP_API_KEY") or settings.mcp_api_key
    if not key:
        raise RuntimeError(
            "未配置 API Key，请通过 --api-key 参数或环境变量 "
            "COURSEPILOT_MCP_API_KEY 指定"
        )
    return key


def _make_error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    """构造一条 JSON-RPC error 响应。"""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def _is_notification(request: dict[str, Any]) -> bool:
    """判断是否为 JSON-RPC 通知（无 id，无需响应）。"""
    return "id" not in request


def forward_request(
    client: httpx.Client,
    gateway_url: str,
    api_key: str,
    request: dict[str, Any],
) -> dict[str, Any] | None:
    """把单条 JSON-RPC 请求转发到 Gateway，返回响应 dict。

    - 请求带 ``Authorization: Bearer <api_key>`` 头。
    - 请求体原样作为 JSON POST 发出。
    - 通知（无 id）转发后返回 ``None``，不写回 stdout。
    - 网络错误或 Gateway 返回非 JSON 时，构造 JSON-RPC error 响应返回，
      保证调用方对每条带 id 的请求都能拿到一条响应。

    Args:
        client: 复用的 httpx 同步客户端。
        gateway_url: Gateway URL（含 /mcp 路径）。
        api_key: API Key。
        request: 已解析的 JSON-RPC 请求 dict。

    Returns:
        响应 dict；通知返回 ``None``。
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    is_notification = _is_notification(request)
    try:
        resp = client.post(gateway_url, json=request, headers=headers)
    except httpx.HTTPError as exc:
        # 通知不期待响应，网络失败也无需回写
        if is_notification:
            return None
        return _make_error(request.get("id"), _INTERNAL_ERROR,
                           f"桥接器网络错误: {exc}")

    # 通知：转发即可，不解析响应体
    if is_notification:
        return None

    # 解析响应体；Gateway 鉴权失败等情况返回 {jsonrpc, error}，原样透传
    try:
        body = resp.json()
    except (ValueError, json.JSONDecodeError):
        return _make_error(
            request.get("id"), _INTERNAL_ERROR,
            f"Gateway 返回非 JSON 响应（HTTP {resp.status_code}）",
        )
    return body


def _write_response(stdout: IO[str], payload: dict[str, Any]) -> None:
    """把响应以单行 JSON 写回 stdout 并立即 flush。

    行分隔格式与 test_stdio.py 的 stdio 协议一致，便于宿主逐行读取。
    """
    stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stdout.flush()


def run_bridge(
    stdin: IO[str],
    stdout: IO[str],
    gateway_url: str,
    api_key: str,
    *,
    client: httpx.Client | None = None,
) -> None:
    """stdio 桥接主循环。

    从 stdin 逐行读取 JSON-RPC，转发到 Gateway，响应写回 stdout。
    所有诊断信息走 stderr，绝不污染 stdout。

    Args:
        stdin: 输入流（行分隔 JSON-RPC）。
        stdout: 输出流（响应原样写回）。
        gateway_url: Gateway URL（含 /mcp）。
        api_key: API Key。
        client: 可选的 httpx.Client（测试注入）；为 ``None`` 则自建带超时的客户端。
    """
    own_client = client is None
    if own_client:
        # 60s 读超时兼容 LLM 长生成；10s 连接超时快速失败
        client = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
    try:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            # 1. 解析 stdin 行为 JSON-RPC
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                _write_response(stdout, _make_error(None, _PARSE_ERROR,
                                                    f"解析错误: {exc}"))
                continue
            if not isinstance(request, dict):
                _write_response(stdout, _make_error(
                    None, _INVALID_REQUEST, "请求必须是 JSON 对象"))
                continue
            # 2. 转发并写回响应
            response = forward_request(client, gateway_url, api_key, request)
            if response is not None:
                _write_response(stdout, response)
    finally:
        if own_client:
            client.close()


def main(argv: list[str] | None = None) -> None:
    """CLI 入口。

    ``--help`` 由 argparse 处理后即退出，不会进入 stdio 循环，
    因此 ``coursepilot-mcp --help`` 可正常输出用法。

    Args:
        argv: 命令行参数；为 ``None`` 时读 ``sys.argv[1:]``（正式 CLI 调用）。
    """
    # MCP stdio 协议要求 UTF-8；Windows 默认 cp936 会导致中文响应（工具描述、
    # 资源名等）写出时乱码。此处强制 stdin/stdout 走 UTF-8。
    # StringIO（测试注入）等无 reconfigure 方法的流会被跳过。
    for _stream in (sys.stdin, sys.stdout):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        prog="coursepilot-mcp",
        description="CoursePilot MCP stdio-to-HTTP 桥接器："
                    "把本地 stdio JSON-RPC 转发到远端 MCP Gateway。",
    )
    parser.add_argument(
        "--gateway",
        default=None,
        help="远端 Gateway URL（含 /mcp 路径）；"
             "默认读环境变量 COURSEPILOT_MCP_GATEWAY 或 settings。",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API Key（cp_ 前缀）；"
             "默认读环境变量 COURSEPILOT_MCP_API_KEY 或 settings。",
    )
    args = parser.parse_args(argv)

    gateway_url = _resolve_gateway(args.gateway)
    api_key = _resolve_api_key(args.api_key)

    # 诊断信息走 stderr，stdout 留给 JSON-RPC 响应
    print(f"[coursepilot-mcp] 桥接到 {gateway_url}", file=sys.stderr)
    run_bridge(sys.stdin, sys.stdout, gateway_url, api_key)


if __name__ == "__main__":
    main()
