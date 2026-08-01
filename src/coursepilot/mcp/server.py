"""CoursePilot MCP Server - 基于 mcp v2 SDK（MCPServer）。

将教学能力暴露为 Tools、Resources 和 Prompts。

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
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    PromptMessage,
    TextContent,
    TextResourceContents,
    ToolAnnotations,
)

from coursepilot.config import settings

from coursepilot.mcp.prompts import (
    diagnosis_report,
    quiz_blueprint,
    tutor_socratic,
)
from coursepilot.mcp.resources.course import (
    read_documents,
    read_kp_tree,
    read_mastery,
    read_report,
    read_stats,
)
from coursepilot.mcp.shared.schemas import (
    DiagnoseParams,
    GeneratePracticeParams,
    GetKPTreeParams,
    GetReviewPlanParams,
    GradeAnswersParams,
    QueryKnowledgeParams,
    SearchKnowledgeUnitsParams,
)
from coursepilot.mcp.tools.knowledge import get_kp_tree, search_knowledge_units
from coursepilot.mcp.tools.practice import generate_practice, grade_answers
from coursepilot.mcp.tools.tutor import diagnose, get_review_plan, query_knowledge

_LOGGER = logging.getLogger(__name__)

mcp = MCPServer(
    name="coursepilot",
    title="CoursePilot MCP Server",
    description="AI 教学助手：基于教材的 RAG 问答、练习题生成与批改",
    version="0.1.0",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Resources
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.resource(
    uri="course://{course_id}/kp-tree",
    name="课程知识点树",
    description="列出指定课程的全部知识点路径",
    mime_type="application/json",
)
async def read_course_kp_tree_resource(course_id: str) -> str:
    """读取课程知识点树资源。"""
    return await read_kp_tree(course_id)


@mcp.resource(
    uri="course://{course_id}/documents",
    name="课程文档清单",
    description="列出指定课程的教材文档及处理状态",
    mime_type="application/json",
)
async def read_course_documents_resource(course_id: str) -> str:
    """读取课程文档清单资源。"""
    return await read_documents(course_id)


@mcp.resource(
    uri="course://{course_id}/stats",
    name="课程统计",
    description="统计指定课程的知识点、知识单元和文档数量",
    mime_type="application/json",
)
async def read_course_stats_resource(course_id: str) -> str:
    """读取课程统计资源。"""
    return await read_stats(course_id)


@mcp.resource(
    uri="student://{user_id}/{course_id}/report",
    name="学生学情报告",
    description="返回学生在某课程下的综合学情报告（MVP 简化版）",
    mime_type="application/json",
)
async def read_student_report_resource(user_id: str, course_id: str) -> str:
    """读取学生学情报告资源。"""
    return await read_report(user_id, course_id)


@mcp.resource(
    uri="student://{user_id}/{course_id}/mastery",
    name="学生掌握度画像",
    description="返回学生在某课程下的掌握度画像（MVP 简化版）",
    mime_type="application/json",
)
async def read_student_mastery_resource(user_id: str, course_id: str) -> str:
    """读取学生掌握度画像资源。"""
    return await read_mastery(user_id, course_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.prompt(
    name="tutor_socratic",
    description="苏格拉底式辅导系统提示。适用于引导学生自己推导出答案。",
)
async def tutor_socratic_prompt(course_id: str, kp_path: str) -> GetPromptResult:
    """返回苏格拉底式辅导 Prompt。"""
    return tutor_socratic.render(course_id, kp_path)


@mcp.prompt(
    name="quiz_blueprint",
    description="出题蓝图系统提示。适用于生成结构化练习题。",
)
async def quiz_blueprint_prompt(
    course_id: str,
    kp_path: str,
    count: int = 3,
    difficulty: int = 3,
) -> GetPromptResult:
    """返回出题蓝图 Prompt。"""
    return quiz_blueprint.render(course_id, kp_path, count, difficulty)


@mcp.prompt(
    name="diagnosis_report",
    description="诊断报告生成系统提示。适用于根据练习数据生成学情分析。",
)
async def diagnosis_report_prompt(user_id: str, course_id: str) -> GetPromptResult:
    """返回诊断报告 Prompt。"""
    return diagnosis_report.render(user_id, course_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════════════════

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


@mcp.tool(
    annotations=ToolAnnotations(
        title="学情诊断",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )
)
async def diagnose_tool(params: DiagnoseParams) -> CallToolResult:
    """[1-用途] 分析学生练习记录，识别薄弱知识点。
    [2-限制] 需要学生已有 PracticeRecord 数据。
    [3-成本] 低，仅聚合统计。
    [4-副作用] 无，只读工具。
    [5-输入格式] user_id: 学生 UUID；course_id: 课程 UUID。
    [6-输出格式] 返回薄弱知识点、各 KP 统计、总练习量和整体正确率。
    """
    return await diagnose(params)


@mcp.tool(
    annotations=ToolAnnotations(
        title="生成复习计划",
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    )
)
async def get_review_plan_tool(params: GetReviewPlanParams) -> CallToolResult:
    """[1-用途] 基于诊断结果生成并持久化复习计划。
    [2-限制] 需要先有练习记录。
    [3-成本] 中，需要调用一次 LLM。
    [4-副作用] 会写入 ReviewPlan 表。
    [5-输入格式] user_id: 学生 UUID；course_id: 课程 UUID。
    [6-输出格式] 返回复习计划项、总数、摘要和 plan_id。
    """
    return await get_review_plan(params)


@mcp.tool(
    annotations=ToolAnnotations(
        title="检索知识单元",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )
)
async def search_knowledge_units_tool(params: SearchKnowledgeUnitsParams) -> CallToolResult:
    """[1-用途] 从课程教材中检索相关知识单元。
    [2-限制] 仅检索指定 course_id 下的内容。
    [3-成本] 低，走检索管线，不调用 LLM。
    [4-副作用] 无，只读工具。
    [5-输入格式] query: 检索文本；course_id: 课程 UUID；top_k: 返回条数。
    [6-输出格式] 返回知识单元列表。
    """
    return await search_knowledge_units(params)


@mcp.tool(
    annotations=ToolAnnotations(
        title="获取知识点树",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )
)
async def get_kp_tree_tool(params: GetKPTreeParams) -> CallToolResult:
    """[1-用途] 列出课程的全部知识点路径。
    [2-限制] 仅返回指定 course_id 下的知识点。
    [3-成本] 低，仅查数据库。
    [4-副作用] 无，只读工具。
    [5-输入格式] course_id: 课程 UUID。
    [6-输出格式] 返回知识点路径列表。
    """
    return await get_kp_tree(params)


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
