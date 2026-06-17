"""Seed: 从教材文件提取目录大纲，构建知识点树。

用法（在项目根目录执行）：
    python -m scripts.seed_knowledge <教材路径> --course-name "高等数学"

示例：
    python -m scripts.seed_knowledge tests/fixtures/pdfs/同济高等数学·第八版 上册.pdf --course-name "高等数学"

支持 PDF / DOCX / MD 三种格式，从解析结果中提取标题行（text_level ≤ 4）
自动构建 kp_path 层级并写入 knowledge_points 表。

前提：需要 DATABASE_URL 环境变量已配置，且对应课程已存在
      （如课程不存在，请先通过 POST /api/v1/courses 创建）
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coursepilot.db import get_session_etx
from coursepilot.models import Course, KnowledgePoint


async def extract_headings(file_path: str) -> list[dict]:
    """解析教材文件，只提取标题行（text_level ≤ 4），保留 page_idx。"""
    ext = Path(file_path).suffix.lower().lstrip(".")

    if ext == "pdf":
        from coursepilot.ingestion.pdf_parser import parse_pdf
        result = await parse_pdf(file_path)
    elif ext == "docx":
        from coursepilot.ingestion.docx_parser import parse_docx
        result = await parse_docx(file_path)
    elif ext == "md":
        from coursepilot.ingestion.markdown_parser import parse_markdown
        result = parse_markdown(file_path)
    else:
        raise ValueError(f"不支持的格式: .{ext}，仅支持 pdf/docx/md")

    content_list = result.get("content_list", [])
    if not content_list:
        raise ValueError("解析结果为空，请检查文件是否可读")

    headings = []
    for item in content_list:
        level = item.get("text_level", 99)
        if level and level <= 4:
            headings.append({
                "title": item.get("text", "").strip(),
                "level": level,
                "page_idx": item.get("page_idx", 0),
            })
    return headings


def headings_to_syllabus(headings: list[dict], course_name: str) -> list[dict]:
    """将标题列表转换为知识点节点列表（含 kp_path + parent_title）。

    用栈维护层级关系，功能等价于 SyllabusParser.parse() + flatten()。
    """
    stack: list[dict] = []  # [{title, level, ...}]
    result: list[dict] = []
    counters: dict[int, int] = {}

    for h in headings:
        title = h["title"]
        level = h["level"]
        if not title:
            continue

        # 弹出栈中 level >= 当前 level 的兄弟
        while stack and stack[-1]["level"] >= level:
            stack.pop()

        counters[level] = counters.get(level, 0) + 1

        # 构建 kp_path
        if stack:
            parent = stack[-1]
            kp_path = parent["kp_path"] + "/" + title
            parent_title = parent["title"]
        else:
            kp_path = course_name + "/" + title
            parent_title = None

        node = {
            "title": title,
            "level": level,
            "kp_path": kp_path,
            "parent_title": parent_title,
            "sort_order": counters[level],
            "summary": "",
            "difficulty": 1,
            "source": "textbook",
        }
        result.append(node)
        stack.append(node)

    return result


async def main():
    parser = argparse.ArgumentParser(
        description="从教材提取大纲并写入 knowledge_points 表"
    )
    parser.add_argument("file", help="教材文件路径（pdf/docx/md）")
    parser.add_argument(
        "--course-name", required=True,
        help="课程名称（需与 courses 表中的 name 字段一致）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印提取结果，不写入数据库",
    )
    args = parser.parse_args()

    # 1. 提取标题
    print(f"解析文件: {args.file}")
    headings = await extract_headings(args.file)

    if not headings:
        print("未提取到任何标题（text_level ≤ 4），请检查文件内容")
        return

    print(f"提取到 {len(headings)} 个标题:")
    for h in headings:
        indent = "  " * (h["level"] - 1)
        print(f"  {indent}[L{h['level']}] {h['title']}  (page {h['page_idx']})")

    # 2. 构建知识点节点
    nodes = headings_to_syllabus(headings, args.course_name)
    print(f"\n构建 {len(nodes)} 个知识点节点:")
    for n in nodes:
        indent = "  " * (n["level"] - 1)
        print(f"  {indent}{n['kp_path']}")

    if args.dry_run:
        print("\n[dry-run] 未写入数据库")
        return

    # 3. 写入数据库
    async with get_session_etx() as session:
        from sqlalchemy import select, delete

        result = await session.execute(
            select(Course).where(Course.name == args.course_name)
        )
        course = result.scalar_one_or_none()
        if not course:
            print(f"\n未找到课程 '{args.course_name}'，请先通过 API 或 seed_course.py 创建课程")
            return

        # 幂等：先删旧知识点
        await session.execute(
            delete(KnowledgePoint).where(KnowledgePoint.course_id == course.id)
        )
        print(f"\n已清除课程 '{args.course_name}' 的旧知识点")

        # 逐节点插入，回填 parent_id
        title_to_id: dict[str, str] = {}
        for node in nodes:
            kp = KnowledgePoint(
                course_id=course.id,
                kp_path=node["kp_path"],
                title=node["title"],
                summary=node.get("summary", ""),
                difficulty=node.get("difficulty", 1),
                sort_order=node.get("sort_order", 0),
                source=node.get("source", "textbook"),
            )
            session.add(kp)
            await session.flush()
            title_to_id[node["title"]] = str(kp.id)

        # 回填 parent_id
        for node in nodes:
            if node["parent_title"] and node["parent_title"] in title_to_id:
                kid = title_to_id[node["title"]]
                pid = title_to_id[node["parent_title"]]
                kp = await session.get(KnowledgePoint, kid)
                if kp:
                    kp.parent_id = pid

        await session.flush()
        print(f"已写入 {len(nodes)} 个知识点到数据库")


if __name__ == "__main__":
    asyncio.run(main())
