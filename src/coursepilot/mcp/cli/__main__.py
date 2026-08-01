"""支持 ``python -m coursepilot.mcp.cli`` 启动 stdio 桥接器。

与 ``coursepilot-mcp`` 命令入口（``coursepilot.mcp.cli.main:main``）等价，
供未安装为可执行命令的环境使用（如设计文档中的 WorkBuddy/Trae 配置示例）。
"""

from coursepilot.mcp.cli.main import main

if __name__ == "__main__":
    main()
