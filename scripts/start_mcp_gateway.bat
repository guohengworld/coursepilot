@echo off
REM CoursePilot MCP Gateway 启动脚本（独立窗口运行，勿关闭）
REM 多 worker 模式（--workers 2）：隔离 GET SSE 长连接，避免单进程事件循环
REM 被 WorkBuddy 的 GET /mcp 流占住导致整体卡死（22:17 实测确认）
cd /d F:\all-projs\coursepilot
".venv\Scripts\python.exe" -m uvicorn coursepilot.mcp.gateway.app:create_app --factory --host 0.0.0.0 --port 8080 --workers 2
