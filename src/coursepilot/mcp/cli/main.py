"""MCP 本地 stdio-to-HTTP 桥接 CLI。

用法：
    PYTHONPATH=src uv run python -m coursepilot.mcp.cli

环境变量：
    COURSEPILOT_MCP_GATEWAY - 远端 Gateway URL
    COURSEPILOT_MCP_API_KEY - API Key
"""


def main() -> None:
    """CLI 入口（占位实现）。"""
    raise NotImplementedError("stdio 桥接器将在 P3 阶段实现")


if __name__ == "__main__":
    main()
