"""CoursePilot MCP Server - v2 SDK 实现。

基于 mcp.server.MCPServer 将教学能力暴露为 Tools。

传输协议：
    stdio（默认，供 WorkBuddy / Trae 嵌入使用）
    streamable-http（远程模式，通过 --http 启用）

启动：
    PYTHONPATH=src uv run python -m coursepilot.mcp.server
    PYTHONPATH=src uv run python -m coursepilot.mcp.server --http --port 8080
"""

from __future__ import annotations

import argparse
import logging

from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from coursepilot.config import settings

from coursepilot.mcp.shared.schemas import (
    GeneratePracticeParams,
    GradeAnswersParams,
    QueryKnowledgeParams,
)
from coursepilot.mcp.tools.practice import generate_practice, grade_answers
from coursepilot.mcp.tools.tutor import query_knowledge

_LOGGER = logging.getLogger(__name__)

mcp = MCPServer(
    name="coursepilot",
    title="CoursePilot MCP Server",
    description="AI 教学助手：基于教材的 RAG 问答、练习题生成与批改",
    version="0.1.0",
)


def _make_error(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)


@mcp.tool(
    annotations=ToolAnnotations(
        title="课程知识问答",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )
)
async def query_knowledge_tool(params: QueryKnowledgeParams) -> CallToolResult:
    """[1-用途] 基于课程教材内容回答学生问题。
    [2-限制] query 长度不超过 2000 字符；仅查询指定 course_id 下的内容。
    [3-成本] 中，需要调用一次 LLM。
    [4-副作用] 无，只读工具。
    [5-输入格式] query: 问题文本；course_id: 课程 UUID；kp_path: 可选知识点范围。
    [6-输出格式] 返回答案、涉及知识点和 Token 用量。
    """
    return await query_knowledge(params)


@mcp.tool(
    annotations=ToolAnnotations(
        title="生成练习题",
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    )
)
async def generate_practice_tool(params: GeneratePracticeParams) -> CallToolResult:
    """[1-用途] 根据课程内容和知识点生成选择题。
    [2-限制] 一次最多生成 3 道题；course_id 必须存在。
    [3-成本] 中，需要调用一次 LLM。
    [4-副作用] 会写入 Question 表。
    [5-输入格式] course_id: 课程 UUID；kp_path: 目标知识点路径；count: 题目数量；difficulty: 难度 1-5。
    [6-输出格式] 返回题目列表（不含答案），每题包含 question_id、题干、选项、类型。
    """
    return await generate_practice(params)


@mcp.tool(
    annotations=ToolAnnotations(
        title="批改答案",
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    )
)
async def grade_answers_tool(params: GradeAnswersParams) -> CallToolResult:
    """[1-用途] 提交作答并批改。
    [2-限制] 必须传入有效 question_id 和单字符答案。
    [3-成本] 低，纯逻辑比对。
    [4-副作用] 会写入 PracticeRecord 表。
    [5-输入格式] user_id: 学生 UUID；question_id: 题目 UUID；answer: 学生答案。
    [6-输出格式] 返回批改结果、正确答案、解析和涉及知识点。
    """
    return await grade_answers(params)


def main() -> None:
    parser = argparse.ArgumentParser(description="CoursePilot MCP Server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="使用 Streamable HTTP 传输模式（默认 stdio）",
    )
    parser.add_argument("--host", default=settings.mcp_host, help="HTTP 监听地址")
    parser.add_argument("--port", type=int, default=settings.mcp_port, help="HTTP 监听端口")
    parser.add_argument("--path", default="/mcp", help="HTTP 端点路径")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.http:
        _LOGGER.info(
            "启动 CoursePilot MCP Server（streamable-http, %s:%d%s）",
            args.host,
            args.port,
            args.path,
        )
        mcp.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            streamable_http_path=args.path,
        )
    else:
        _LOGGER.info("启动 CoursePilot MCP Server（stdio）")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
