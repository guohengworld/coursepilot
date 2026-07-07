"""CoursePilot MCP Server — 将教学能力暴露为 MCP 工具和资源

MCP (Model Context Protocol) 是 Anthropic 推出的 AI 工具集成协议。
本模块将 CoursePilot 的核心能力（RAG 问答、学情诊断、练习生成等）
封装为标准 MCP Server，供 Claude Desktop、VS Code Copilot、
Cline 等任何兼容 MCP 的客户端调用。
"""
