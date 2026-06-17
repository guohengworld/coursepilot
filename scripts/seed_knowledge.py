"""Seed: 从教材文件提取目录大纲，构建知识点树。

用法（在项目根目录执行）：
    python -m scripts.seed_knowledge <教材路径> --course-name "高等数学"

示例：
    python -m scripts.seed_knowledge tests/fixtures/pdfs/同济高等数学·第八版 上册.pdf --course-name "高等数学" --ingest

支持 PDF / DOCX / MD 三种格式。
--ingest：除知识点树外，同时将全文解析为知识单元入库。
          一次 MinerU 解析，两阶段复用，避免重复 OCR。
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


async def parse_file(file_path: str) -> tuple[list[dict], list[dict]]:
    """解析教材文件，返回 (content_list, headings)。

    这是整个 seed 流程中唯一一次文件解析。
    """
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
    return content_list, headings


def headings_to_syllabus(headings: list[dict], course_name: str) -> list[dict]:
    """将标题列表转换为知识点节点列表（含 kp_path + parent_title）。

    用栈维护层级关系，功能等价于 SyllabusParser.parse() + flatten()。
    """
    stack: list[dict] = []
    result: list[dict] = []
    counters: dict[int, int] = {}

    for h in headings:
        title = h["title"]
        level = h["level"]
        if not title:
            continue

        while stack and stack[-1]["level"] >= level:
            stack.pop()

        counters[level] = counters.get(level, 0) + 1

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
    parser.add_argument(
        "--ingest", action="store_true",
        help="除知识点树外，同时将全文解析为知识单元入库（一次解析，不重复调用 MinerU）",
    )
    args = parser.parse_args()

    # ━━━━ 1. 解析文件（唯一一次 MinerU 调用）━━━━
    print(f"解析文件: {args.file}")
    content_list, headings = await parse_file(args.file)

    if not headings:
        print("未提取到任何标题（text_level ≤ 4），请检查文件内容")
        return

    print(f"提取到 {len(headings)} 个标题，content_list 共 {len(content_list)} 行:")
    for h in headings:
        indent = "  " * (h["level"] - 1)
        print(f"  {indent}[L{h['level']}] {h['title']}  (page {h['page_idx']})")

    # ━━━━ 2. 构建知识点节点 ━━━━
    nodes = headings_to_syllabus(headings, args.course_name)
    print(f"\n构建 {len(nodes)} 个知识点节点:")
    for n in nodes:
        indent = "  " * (n["level"] - 1)
        print(f"  {indent}{n['kp_path']}")

    if args.dry_run:
        print("\n[dry-run] 未写入数据库")
        return

    # ━━━━ 3. 写入数据库 ━━━━
    from coursepilot.db import get_session_etx
    from coursepilot.models import Course, KnowledgePoint

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

        # ━━━━ 4. --ingest：继续做知识单元入库 ━━━━
        if not args.ingest:
            return

        import uuid
        from coursepilot.models import Document, KnowledgeUnit, User
        from coursepilot.ingestion.parser_utils import extract_knowledge_units
        from coursepilot.knowledge.kp_splitter import KPSplitter

        print(f"\n── 开始 ingestion（复用已有解析结果，不重复调用 MinerU）──")

        # 4a. 创建 Document 记录
        r = await session.execute(
            select(User).where(User.role == "super").limit(1)
        )
        uploader = r.scalar_one()

        doc = Document(
            course_id=course.id,
            filename=Path(args.file).name,
            file_type=Path(args.file).suffix.lower().lstrip("."),
            file_size=Path(args.file).stat().st_size,
            file_path=args.file,  # 原始文件路径
            uploader_id=uploader.id,
            status="processing",
        )
        session.add(doc)
        await session.flush()
        await session.refresh(doc)
        print(f"  Document 创建: id={doc.id}, file={doc.filename}")

        # 4b. 切分为知识单元
        units = extract_knowledge_units(
            content_list,
            document_id=str(doc.id),
            kp_id="",
        )
        print(f"  切分得到 {len(units)} 个文本块")

        # 4c. 查知识点列表 + KPSplitter 分配
        kp_result = await session.execute(
            select(KnowledgePoint)
            .where(KnowledgePoint.course_id == course.id)
            .order_by(KnowledgePoint.sort_order)
        )
        kp_nodes = [
            {
                "id": str(kp.id), "title": kp.title,
                "kp_path": kp.kp_path,
                "level": len(kp.kp_path.split("/")),
            }
            for kp in kp_result.scalars()
        ]

        if kp_nodes:
            splitter = KPSplitter(kp_nodes, str(course.id))
            units = splitter.assign(units)

        # 4d. 批量插入 knowledge_units
        for u in units:
            ku = KnowledgeUnit(
                kp_id=uuid.UUID(u["kp_id"]) if u.get("kp_id") else None,
                document_id=doc.id,
                content=u["content"],
                summary=u.get("summary"),
                seq_order=u.get("seq_order", 0),
                page_ref=u.get("page_ref", ""),
                meta_data=u.get("meta_data", {}),
            )
            session.add(ku)

        doc.status = "ready"
        doc.page_count = len(units)
        await session.flush()

        # 统计覆盖
        kp_covered = len(set(u.get("kp_id") for u in units if u.get("kp_id")))
        print(f"  入库 {len(units)} 个知识单元，覆盖 {kp_covered}/{len(kp_nodes)} 个知识点")
        print(f"  Document 状态: {doc.status}")


if __name__ == "__main__":
    asyncio.run(main())
