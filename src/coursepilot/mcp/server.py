"""Coursepilot MCP Server - 核心实现

通过 FastMCP 将教学功能暴露为可被 LLM 调用的 Tools，以及可读取的 Resources

传输协议：stdio（默认，供 Claude Desktop / VS Code 嵌入使用）
        SSE（可选的远程模式，通过 --transport sse 启用）

Tools（由 LLM 主动调用）：
  - query_knowledge:    RAG 问答查询
  - get_kp_tree:        知识点树查询
  - get_mastery:        学生掌握度查询
  - diagnose:           学情诊断
  - generate_practice:  生成练习题
  - get_student_report: 学生综合报告（含诊断 + 掌握度 + 复习 + 计划建议）

Resources（由 LLM 读取的结构化数据）：
  - course://{course_id}/kp-tree        → 课程知识点树（JSON）
  - student://{user_id}/{course_id}/report → 学生综合报告（JSON）

用法：
  # stdio 模式（默认，用于 Claude Desktop / VS Code 的 mcp.json）
  PYTHONPATH=src uv run python -m coursepilot.mcp.server

  # SSE 模式（用于远程 MCP Client，如 Cline）
  PYTHONPATH=src uv run python -m coursepilot.mcp.server --sse --port 8080
"""

import json
import logging
import argparse
from typing import Any

# - MCP SDK
from mcp.server.fastmcp import FastMCP

# - 数据库
from sqlalchemy.ext.asyncio import AsyncSession
from coursepilot.db import async_session_factory

# - 教学组件
from coursepilot.knowledge.kp_tree import KPTree
from coursepilot.rag.retriever import Retriever
from coursepilot.rag.generator import Generator, build_course_context
from coursepilot.agent.skills.diagnose import diagnose
from coursepilot.agent.skills.get_mastery import get_mastery
from coursepilot.agent.skills.generate_quiz import generate_quiz as gen_quiz

_LOGGER = logging.getLogger(__name__)

# - FastMCP 实例
# 服务器名称和描述会暴露给 MCP Client，方便 LLM 理解这个 Server 的用途
mcp = FastMCP(
    "CoursePilot",
    instructions="AI 教学助手：基于教材的 RAG 问答、学情诊断、练习题生成、知识点树查询",
)

# 内部工具函数
async def _get_session() -> AsyncSession:
    """获取一个异步 Db 会话

    每个 MCP 工具调用都独立创建/关闭 session
    避免跨请求状态污染。调用方必须用 try/finally 确保 close
    """
    return async_session_factory()


# MCP Tools
# 每个函数上的 @mcp.tool() 将其注册为一个 MCP Tool，LLM 可以在对话中按需调用。函数签名 + docstring 会被MCP SDK 自动转为 tool schema 描述。
@mcp.tool(
    name="query_knowledge",
    description="基于课程教材内容回答学生问题。传入 query（问题）和 course_id（课程 UUID）。",
)
async def query_knowledge(query: str, course_id: str) -> str:
    """RAG 问答：检索教材 → LLM 生成回答

    流程：
        1. 从 DB 查课程上下文（名称 + 教材 + 章节列表）
        2. Retriever 五阶段检索（外加一次粗排和精排）：改写 → BGE-M3 编码 → Milvus 混合检索
            → bge-reranker-v2-m3 重排序 → KP 路径扩展
        3. Generator 组装 System Prompt + 检索结果 → 调用 DeepSeek回答

    Args:
        query: 学生的回答原文（如”什么是进程调度“）
        course_id: 课程 UUID

    Return:
        基于教材的回答文本（含 <ref> 引用标记）
    """
    session = await _get_session()
    try:
        # Step1: 获取课程上下文
        course_context = await build_course_context(session, course_id)
        if not course_context:
            return f"错误：course_id={course_id} 不存在"

        # Step2: 检索 + 生成
        retriever = Retriever()
        context, metadata = await retriever.retrieve(session, query, course_id)

        generator = Generator()
        answer, token_info = await generator.generate(query, context, course_context)

        # Step3: 组装返回（回答 + Token 用量 + 来源 KP 路径）
        source_kps = metadata.get("source_kp_paths", [])
        token_summary = (
            f"\n\n---\n"
            f"Token 用量：{token_info.get('total_tokens', 0)} "
            f"(输入 {token_info.get('prompt_tokens', 0)} / "
            f"输出 {token_info.get('completion_tokens', 0)})"
        )
        kp_summary = ""
        if source_kps:
            kp_summary = f"\n涉及知识点：{'、'.join(source_kps[:5])}"

        return f"{answer}{kp_summary}{token_summary}"
    finally:
        await session.close()

@mcp.tool(
    name="get_kp_tree",
    description="查询课程的知识点树（层级结构）。传入 course_id（课程 UUID）。返回 JSON 树。",
)
async def get_kp_tree(course_id: str) -> str:
    """知识点树查询

    用 KPTree 的递归 CTE 查出整棵以根为起点的知识点树，返回 JSON格式的嵌套结构

    Args:
        course_id: 课程 UUID

    Returns:
        JSON 字符串：{id, title, kp_path, level, children: [...]}
    """
    session = await _get_session()
    try:
        # 取课程下的根知识点（parent_id IS NULL）
        from sqlalchemy import select
        from coursepilot.models import KnowledgePoint

        result = await session.execute(
            select(KnowledgePoint)
            .where(
                KnowledgePoint.course_id == course_id,
                KnowledgePoint.parent_id.is_(None),
            )
            .limit(1)
        )
        root = result.scalar_one_or_none()
        if not root:
            return json.dumps({"error": "该课程暂无知识点"}, ensure_ascii=False)

        tree = KPTree(session)
        root_node = await tree.get_subtree(str(root.id))

        def _to_dict(node) -> dict:
            return {
                "id": node.id,
                "title": node.title,
                "kp_path": node.kp_path,
                "children": [_to_dict(c) for c in node.children],
            }

        return json.dumps(
            _to_dict(root_node), ensure_ascii=False, indent=2
        )
    finally:
        await session.close()

@mcp.tool(
    name="get_mastery",
    description="查询学生在某课程的知识点掌握度。传入 user_id 和 course_id",
)
async def query_mastery(user_id: str, course_id: str) -> str:
    """学生掌握度查询

    从 user_profiles 表读取预计算的掌握度数据
    该数据由 profile_updater 在每次 Agent 执行 finalize 后异步更新

    Args:
        user_id: 学生 UUID
        course_id: 课程 UUID

    Returns:
        掌握度 JSON：{mastery_level, weak_kps, avg_correct_rate}
    """
    session = await _get_session()
    try:
        mastery = await get_mastery(session, user_id, course_id)
        if mastery.get("avg_correct_rate") is None:
            return f"学生 {user_id} 在课程 {course_id} 尚无练习记录，暂无掌握度数据"

        weak = mastery.get("weak_kps", [])
        weak_text = f"\n薄弱知识点（{len(weak)} 个）：{'、'.join(weak[:8])}" if weak else "\n暂无薄弱知识点"

        return (
            f"平均正确率：{mastery['avg_correct_rate']:.0%}\n"
            f"已掌握知识点：{len(mastery.get('mastery_level', {}))} 个"
            f"{weak_text}"
        )
    finally:
        await session.close()

@mcp.tool(
    name="diagnose",
    description="学情诊断：分析学生练习记录，识别薄弱知识点。传入 user_id 和 course_id。",
)
async def diagnose_weakness(user_id: str, course_id: str) -> str:
    """学情诊断

    聚合 PracticeRecord → 关联 Question → 按 KP 分组计算正确率
    → 识别正确率 < threshold（默认 60%）的知识点。

    底层复用 coursepilot.agent.skills.diagnose.diagnose() 函数。

    Args:
        user_id:  学生 UUID
        course_id: 课程 UUID

    Returns:
        诊断报告文本：总题数、整体正确率、薄弱知识点列表及统计
    """
    session = await _get_session()
    try:
        result = await diagnose(session, user_id, course_id)

        parts = [
            f"📊 诊断报告",
            f"总练习量：{result['total_practiced']} 题",
            f"整体正确率：{result['overall_rate']:.0%}",
            f"涉及知识点：{len(result['kp_stats'])} 个",
        ]

        if result["weak_kps"]:
            weak_lines = []
            for kp in result["weak_kps"]:
                stat = result["kp_stats"].get(kp, {})
                weak_lines.append(
                    f"  - {kp}：{stat.get('correct', 0)}/{stat.get('total', 0)} "
                    f"正确率 {stat.get('rate', 0):.0%}"
                )
            parts.append(f"\n薄弱知识点（{len(result['weak_kps'])} 个）：")
            parts.extend(weak_lines[:10])  # 最多显示 10 个

        parts.append(f"\n{result['summary']}")
        return "\n".join(parts)
    finally:
        await session.close()

@mcp.tool(
    name="generate_practice",
    description="基于知识点和教材内容生成 3 道练习题。传入 course_id 和可选参数 kp_path。",
)
async def generate_practice(
    course_id: str,
    kp_path: str = "",
) -> str:
    """生成练习题

    流程：
        1. 检索教材中与 kp_path 相关的上下文
        2. LLM 根据教材内容生成 3 道难度递进的选择题
        3. 每道题含选项、正确答案、解析、关联知识点

    Args:
        course_id: 课程 UUID
        kp_path:   知识点路径（可选）。为空时检索课程全部内容。

    Returns:
        练习题 JSON 或文本格式
    """
    session = await _get_session()
    try:
        # 构建课程上下文
        course_context = await build_course_context(session, course_id)
        if not course_context:
            return f"错误：course_id={course_id} 不存在"

        # 根据 kp_path 或全课程检索教材内容
        retriever = Retriever()
        query = kp_path or course_context.get("name", "全部内容")
        context, metadata = await retriever.retrieve(session, query, course_id)

        # 调用 generate_quiz skill
        # mastery 传空字典，表示不针对薄弱点（外部工具无此上下文）
        quiz_data, token_info = await gen_quiz(context, course_context, {})

        if not quiz_data.get("questions"):
            return "生成失败：LLM 返回空结果，请检查 API Key 是否正确"

        # 格式化输出
        lines = [f"📝 {course_context.get('name', '')} - 练习题"]
        for i, q in enumerate(quiz_data["questions"], 1):
            opts = q.get("options", {})
            opt_lines = "\n".join(f"  {k}. {v}" for k, v in opts.items())
            lines.append(
                f"\n{i}. {q['question_text']}\n{opt_lines}\n"
                f"   答案：{q['correct_answer']}   解析：{q.get('explanation', '')}"
            )

        lines.append(
            f"\n---\nToken 用量：{token_info.get('total_tokens', 0)}"
        )
        return "\n".join(lines)
    finally:
        await session.close()

@mcp.tool(
    name="get_student_report",
    description="学生综合报告：掌握度 + 诊断 + 课程信息。传入 user_id 和 course_id。",
)
async def student_report(user_id: str, course_id: str) -> str:
    """生成学生综合报告

    组合 get_mastery + diagnose 的数据，外加课程基本信息，
    形成一份完整的学情快照，方便教师或 LLM 快速了解学生状态。

    Args:
        user_id:  学生 UUID
        course_id: 课程 UUID

    Returns:
        综合报告文本
    """
    session = await _get_session()
    try:
        # 课程信息
        course_ctx = await build_course_context(session, course_id)
        course_name = course_ctx.get("name", "未知")

        # 诊断
        diag = await diagnose(session, user_id, course_id)

        # 掌握度
        mastery = await get_mastery(session, user_id, course_id)

        # 组装报告
        parts = [
            f"═══ 学生综合报告 ═══",
            f"课程：{course_name}",
            f"学生：{user_id}",
            "",
            f"【练习统计】",
            f"总练习量：{diag['total_practiced']} 题",
            f"整体正确率：{diag['overall_rate']:.0%}",
            f"涉及知识点：{len(diag['kp_stats'])} 个",
        ]

        if mastery.get("avg_correct_rate") is not None:
            parts.append(
                f"画像平均正确率：{mastery['avg_correct_rate']:.0%}"
            )

        if diag["weak_kps"]:
            parts.extend([
                "",
                f"【薄弱知识点（共 {len(diag['weak_kps'])} 个）】",
            ])
            for kp in diag["weak_kps"][:8]:
                stat = diag["kp_stats"].get(kp, {})
                parts.append(
                    f"  🔴 {kp}：{stat.get('correct', 0)}/{stat.get('total', 0)} "
                    f"({stat.get('rate', 0):.0%})"
                )

        parts.append(f"\n{diag['summary']}")
        return "\n".join(parts)
    finally:
        await session.close()


# MCP Resources（可被 LLM 主动读取的结构化数据）
# Resources 是具名、带 URI 的数据源，LLM 可以在工具调用之外
# 主动"读取"它们。比 Tool 更接近"文件读取"的语义。
@mcp.resource(
    uri="course://{course_id}/kp-tree",
    name="知识点树",
    description="课程知识点树（JSON 嵌套结构）",
    mime_type="application/json",
)
async def kp_tree_resource(course_id: str) -> str:
    """知识点树 Resource

    和 get_kp_tree tool 共享相同逻辑，但以 resource 形式提供，
    方便 LLM 直接"读取"知识点树而无需手动调用 tool。
    """
    return await get_kp_tree(course_id)

@mcp.resource(
    uri="student://{user_id}/{course_id}/report",
    name="学生报告",
    description="学生综合学情报告（JSON）",
    mime_type="application/json",
)
async def student_report_resource(user_id: str, course_id: str) -> str:
    """学生综合报告 Resource"""
    return await student_report(user_id, course_id)


# 入口点
def main():
    """解析命令行参数并启动 MCP Server

    支持两种传输模式：
      - stdio（默认）：标准输入输出通信，适合 Claude Desktop / VS Code
      - SSE：HTTP Server-Sent Events，适合远程客户端
    """
    parser = argparse.ArgumentParser(description="Couesepilot MCP Server")
    parser.add_argument(
        "--sse", action="store_true",
        help="以 SSE 模式运行（默认 stdio）"
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="SSE 模式下的监听端口（默认 8080）"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="SSE 模式下的监听地址（默认 0.0.0.0）"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _LOGGER.info(
        "启动 CoursePilot MCP Server（transport=%s）",
        "sse" if args.sse else "stdio",
    )

    if args.sse:
        # SSE 模式：host/port 需在 FastMCP 构造函数中设置，
        # 这里直接覆盖 settings
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        # stdio 模式：通过标准输入输出通信
        # 此为 Claude Desktop / VS Code 的标准嵌入方式
        mcp.run(transport="stdio")


if __name__ == '__main__':
    main()
