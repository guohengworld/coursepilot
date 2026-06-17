"""批量入库：一次性处理 tests/fixtures/pdfs/ 下所有教材。

用法：
    cd f:/all-projs/coursepilot
    PYTHONPATH=src .venv/Scripts/python -m scripts.batch_ingest

特性：
- 每本 PDF 只解析一次（MinerU），同时产出知识点树 + 知识单元
- 同一课程的多卷教材（上/下册）共享知识点树
- 自动创建课程（如不存在）
- 单本失败不中断，最后打印汇总
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ═══════════════════════════════════════════════════════════
# 教材 → 课程映射
# ═══════════════════════════════════════════════════════════

PDF_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "pdfs"

# key = 课程名, value = [pdf文件名, ...]（按上→下顺序）
COURSE_PDFS: dict[str, list[str]] = {
    "高等数学": [
        "同济高等数学·第八版 上册.pdf",
        "同济高等数学·第八版 下册.pdf",
    ],
    "微积分": [
        "大学数学 微积分 上册.pdf",
        "大学数学 微积分 下册.pdf",
    ],
    "数学分析": [
        "数学分析 第五版上册.pdf",
        "数学分析 第五版下册.pdf",
    ],
    "概率论与数理统计": [
        "概率论与数理统计 第五版.pdf",
    ],
    "高等代数": [
        "高等代数 第五版.pdf",
    ],
}

# ═══════════════════════════════════════════════════════════
# 核心逻辑（复用 seed_knowledge 的函数）
# ═══════════════════════════════════════════════════════════

from scripts.seed_knowledge import parse_file, headings_to_syllabus


async def process_course(
    course_name: str,
    pdf_names: list[str],
    pdf_dir: Path,
) -> dict:
    """处理一个课程的所有 PDF，返回统计信息。

    同一课程的多卷教材：合并所有标题构建统一知识点树，
    每卷 PDF 各创建一个 Document + 各自的知识单元。
    """
    from coursepilot.db import get_session_etx
    from coursepilot.models import Course, KnowledgePoint, Document, KnowledgeUnit, User
    from sqlalchemy import select, delete

    total_start = time.time()
    stats = {"course": course_name, "pdfs": [], "total_units": 0, "total_kps": 0, "error": None}

    # ── 1. 解析所有 PDF，收集 content_list + headings ──
    all_headings: list[dict] = []
    pdf_data: list[dict] = []  # [{filename, content_list, headings}]

    for pdf_name in pdf_names:
        pdf_path = pdf_dir / pdf_name
        if not pdf_path.exists():
            print(f"  ⚠ 文件不存在，跳过: {pdf_path}")
            continue

        file_size_mb = pdf_path.stat().st_size / 1024 / 1024
        print(f"\n  📄 解析: {pdf_name} ({file_size_mb:.0f} MB)")
        t0 = time.time()

        try:
            content_list, headings = await parse_file(str(pdf_path))
        except Exception as e:
            print(f"  ❌ 解析失败: {e}")
            stats["pdfs"].append({"name": pdf_name, "status": "parse_failed", "error": str(e)})
            continue

        elapsed = time.time() - t0
        print(f"     ⏱ {elapsed:.0f}s | content_list={len(content_list)} 行, headings={len(headings)} 个")

        pdf_data.append({
            "filename": pdf_name,
            "file_path": str(pdf_path),
            "file_size": pdf_path.stat().st_size,
            "content_list": content_list,
            "headings": headings,
        })
        all_headings.extend(headings)

    if not pdf_data:
        stats["error"] = "没有成功解析任何 PDF"
        return stats

    # ── 2. 合并标题，构建知识点树 ──
    print(f"\n  🌳 构建知识点树（合并 {len(all_headings)} 个标题）...")
    nodes = headings_to_syllabus(all_headings, course_name)
    print(f"     {len(nodes)} 个知识点节点")

    # ── 3. 写入数据库 ──
    async with get_session_etx() as session:
        # 3a. 创建课程（如不存在）
        result = await session.execute(
            select(Course).where(Course.name == course_name)
        )
        course = result.scalar_one_or_none()
        if not course:
            r = await session.execute(
                select(User).where(User.role == "super").limit(1)
            )
            superuser = r.scalar_one()
            course = Course(
                name=course_name,
                description=f"{course_name} 教材知识库",
                created_by=superuser.id,
            )
            session.add(course)
            await session.flush()
            await session.refresh(course)
            print(f"     ✨ 创建课程: {course.name} (id={course.id})")
        else:
            print(f"     📚 课程已存在: {course.name} (id={course.id})")

        # 3b. 幂等：清除旧知识点
        await session.execute(
            delete(KnowledgePoint).where(KnowledgePoint.course_id == course.id)
        )
        await session.flush()

        # 3c. 逐节点插入知识点
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

        # 重新查出所有知识点（供 KPSplitter 用）
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
        stats["total_kps"] = len(kp_nodes)
        print(f"     ✅ 知识点树入库: {len(kp_nodes)} 个节点")

        # 3d. 为每卷 PDF 创建 Document + 知识单元
        import uuid
        from coursepilot.ingestion.parser_utils import extract_knowledge_units
        from coursepilot.knowledge.kp_splitter import KPSplitter

        r = await session.execute(
            select(User).where(User.role == "super").limit(1)
        )
        uploader = r.scalar_one()

        for pd in pdf_data:
            print(f"\n  📝 知识单元入库: {pd['filename']}")

            doc = Document(
                course_id=course.id,
                filename=pd["filename"],
                file_type="pdf",
                file_size=pd["file_size"],
                file_path=pd["file_path"],
                uploader_id=uploader.id,
                status="processing",
            )
            session.add(doc)
            await session.flush()
            await session.refresh(doc)

            units = extract_knowledge_units(
                pd["content_list"],
                document_id=str(doc.id),
                kp_id="",
            )

            if kp_nodes:
                splitter = KPSplitter(kp_nodes, str(course.id))
                units = splitter.assign(units)

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

            kp_covered = len(set(u.get("kp_id") for u in units if u.get("kp_id")))
            stats["total_units"] += len(units)
            stats["pdfs"].append({
                "name": pd["filename"],
                "status": "ok",
                "units": len(units),
                "kp_covered": kp_covered,
            })
            print(f"     ✅ {len(units)} 个知识单元，覆盖 {kp_covered}/{len(kp_nodes)} 个知识点")

    total_elapsed = time.time() - total_start
    stats["elapsed"] = total_elapsed
    print(f"\n  🏁 课程 '{course_name}' 完成 ({total_elapsed:.0f}s)")
    return stats


# ═══════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════

async def main():
    print("=" * 70)
    print("  CoursePilot 批量教材入库")
    print(f"  共 {len(COURSE_PDFS)} 门课程，{sum(len(v) for v in COURSE_PDFS.values())} 本 PDF")
    print("=" * 70)

    global_start = time.time()
    all_stats: list[dict] = []

    for course_name, pdf_names in COURSE_PDFS.items():
        print(f"\n{'─' * 70}")
        print(f"📚 课程: {course_name} ({len(pdf_names)} 卷)")
        print(f"{'─' * 70}")

        try:
            stats = await process_course(course_name, pdf_names, PDF_DIR)
        except Exception as e:
            stats = {"course": course_name, "error": str(e)}
        all_stats.append(stats)

    # ── 汇总 ──
    total_elapsed = time.time() - global_start
    print(f"\n{'=' * 70}")
    print(f"  批量入库完成")
    print(f"{'=' * 70}")
    print(f"  总耗时: {total_elapsed/60:.1f} 分钟 ({total_elapsed:.0f}s)")

    total_pdfs = 0
    total_ok = 0
    total_units = 0
    total_kps = 0
    for s in all_stats:
        if s.get("error"):
            print(f"\n  ❌ {s['course']}: {s['error']}")
        else:
            for p in s["pdfs"]:
                total_pdfs += 1
                if p["status"] == "ok":
                    total_ok += 1
                    print(f"  ✅ {p['name']}: {p['units']} 单元, 覆盖 {p['kp_covered']} KP")
                else:
                    print(f"  ❌ {p['name']}: {p.get('error', 'unknown')}")
            total_units += s.get("total_units", 0)
            total_kps += s.get("total_kps", 0)

    print(f"\n  📊 统计: {total_ok}/{total_pdfs} PDF 成功")
    print(f"  📊 知识单元总计: {total_units}")
    print(f"  📊 知识点总计: {total_kps}")


if __name__ == "__main__":
    asyncio.run(main())
