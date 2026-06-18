"""真实端到端集成测试：PDF → MinerU 解析 → 大纲提取 → 知识点树 → 全文入库

以《大学数学 微积分 下册》为例，展示两阶段管线的完整效果。

运行方式：
    cd f:/all-projs/coursepilot
    PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_real_pipeline.py -v -s

-s 参数：让 print 输出可见，便于观察每一步结果。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

os.environ["MINERU_MODEL_SOURCE"] = "local"
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PDF_PATH = Path(__file__).parent / "fixtures" / "pdfs" / "大学数学 微积分 下册.pdf"
COURSE_NAME = "微积分"
OUTPUT_DIR = "tests/output/real_pipeline_test"


# ═══════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════


def _section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _step(n: int, title: str):
    print(f"\n── 步骤 {n}: {title} ──")


def _json_preview(obj, max_items: int = 5):
    """打印列表的前 N 项预览"""
    s = json.dumps(obj[:max_items], ensure_ascii=False, indent=2)
    if len(obj) > max_items:
        s += f"\n  ... (共 {len(obj)} 项)"
    print(s)


# ═══════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.slow
async def test_real_pipeline():
    """真实管线：PDF 解析 → 大纲提取 → 知识点树 → 全文入库"""

    from coursepilot.db import get_session_etx
    from coursepilot.models import Course, Document, KnowledgePoint, KnowledgeUnit
    from sqlalchemy import select, text, delete

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 阶段 A：PDF 解析 + 大纲提取 + 知识点树入库
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    _section("阶段 A：从教材提取大纲，构建知识点树")

    # A1. 用 MinerU 解析 PDF（唯一一次，Phase B 复用此结果）
    _step(1, "MinerU 解析 PDF → content_list + markdown（唯一一次解析）")

    from coursepilot.ingestion.pdf_parser import parse_pdf

    # 只解析前 20 页（目录 + 前几章），避免 MinerU 对大文件崩溃
    pdf_result = await parse_pdf(
        str(PDF_PATH),
        output_dir=OUTPUT_DIR,
        start_page=0,
        end_page=20,
    )

    content_list = pdf_result["content_list"]
    markdown_text = pdf_result["markdown"]
    print(f"  解析完成：content_list 共 {len(content_list)} 行，markdown 共 {len(markdown_text)} 字符")
    if "_timings" in pdf_result:
        print(f"  [TIMINGS] {pdf_result['_timings']}")
    if "_config" in pdf_result:
        print(f"  [CONFIG] {pdf_result['_config']}")

    # 展示 markdown 前 1500 字符
    print("\n  ── Markdown 预览（前 1500 字符）──")
    print(markdown_text[:1500])
    if len(markdown_text) > 1500:
        print(f"  ... (共 {len(markdown_text)} 字符)")

    # A2. 提取标题行（text_level ≤ 4）
    _step(2, "从 content_list 提取标题行（text_level ≤ 4）")

    headings = []
    for item in content_list:
        level = item.get("text_level", 99)
        if level and level <= 4:
            headings.append({
                "title": item.get("text", "").strip(),
                "level": level,
                "page_idx": item.get("page_idx", 0),
                "type": item.get("type", ""),
            })

    print(f"  提取到 {len(headings)} 个标题行：")
    for h in headings[:30]:
        indent = "  " * (h["level"] - 1)
        page_label = h["page_idx"] + 1  # page_idx 从 0 开始
        print(f"  {indent}[L{h['level']}] {h['title'][:60]}  (p{page_label})")
    if len(headings) > 30:
        print(f"  ... (共 {len(headings)} 个，仅展示前 30)")

    # A3. 构建知识点节点（kp_path）
    _step(3, "构建知识点节点（title → kp_path 层级）")

    from scripts.seed_knowledge import headings_to_syllabus

    kp_nodes = headings_to_syllabus(headings, COURSE_NAME)
    print(f"  构建 {len(kp_nodes)} 个知识点节点：")
    for n in kp_nodes[:25]:
        indent = "  " * (n["level"] - 1)
        print(f"  {indent}{n['kp_path']}")
    if len(kp_nodes) > 25:
        print(f"  ... (共 {len(kp_nodes)} 个，仅展示前 25)")

    # A4. 写入数据库
    _step(4, "知识点树写入 knowledge_points 表")

    async with get_session_etx() as session:
        # 4a. 创建课程
        result = await session.execute(
            select(Course).where(Course.name == COURSE_NAME)
        )
        course = result.scalar_one_or_none()
        if not course:
            # 找 superuser
            from coursepilot.models import User
            r = await session.execute(
                select(User).where(User.role == "super").limit(1)
            )
            superuser = r.scalar_one()
            course = Course(
                name=COURSE_NAME,
                description="大学数学 微积分 下册",
                created_by=superuser.id,
            )
            session.add(course)
            await session.flush()
            await session.refresh(course)
            print(f"  创建课程: {course.name} (id={course.id})")
        else:
            print(f"  课程已存在: {course.name} (id={course.id})")

        # 4b. 清除旧知识点（幂等）
        await session.execute(
            delete(KnowledgePoint).where(KnowledgePoint.course_id == course.id)
        )
        await session.flush()
        print(f"  已清除课程 '{COURSE_NAME}' 的旧知识点")

        # 4c. 逐节点插入
        title_to_id: dict[str, str] = {}
        for node in kp_nodes:
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
        for node in kp_nodes:
            if node["parent_title"] and node["parent_title"] in title_to_id:
                kid = title_to_id[node["title"]]
                pid = title_to_id[node["parent_title"]]
                kp = await session.get(KnowledgePoint, kid)
                if kp:
                    kp.parent_id = pid
        await session.flush()

        # 验证
        result = await session.execute(
            select(KnowledgePoint).where(
                KnowledgePoint.course_id == course.id
            ).order_by(KnowledgePoint.sort_order)
        )
        kps = result.scalars().all()
        print(f"\n  入库完成：共 {len(kps)} 个知识点")
        root_kps = [kp for kp in kps if kp.parent_id is None]
        print(f"  根节点: {len(root_kps)} 个")
        for rkp in root_kps[:5]:
            print(f"    - {rkp.title}  (id={rkp.id})")

        course_id = str(course.id)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 阶段 B：模拟上传 + 全文 ingestion
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    _section("阶段 B：模拟上传教材，执行 ingestion 管线（复用阶段 A 的 content_list）")

    async with get_session_etx() as session:
        import uuid
        from coursepilot.ingestion.pipeline import run_ingestion
        from coursepilot.storage.file_store import FileStore

        # B1. 创建 Document 记录
        _step(5, "创建 Document 记录（status=pending）")

        # 获取 superuser
        from coursepilot.models import User
        r = await session.execute(
            select(User).where(User.role == "super").limit(1)
        )
        uploader = r.scalar_one()

        # 复制 PDF 到 uploads 目录（模拟文件上传）
        file_store = FileStore()
        file_info = file_store.save(
            PDF_PATH.read_bytes(),
            course_id,
            PDF_PATH.name,
        )
        print(f"  文件保存: {file_info['file_path']}")
        print(f"  文件大小: {file_info['file_size']} bytes")

        doc = Document(
            course_id=uuid.UUID(course_id),
            filename=PDF_PATH.name,
            file_type="pdf",
            file_size=file_info["file_size"],
            file_path=file_info["file_path"],
            uploader_id=uploader.id,
            status="pending",
        )
        session.add(doc)
        await session.flush()
        await session.refresh(doc)
        print(f"  Document 创建: id={doc.id}, status={doc.status}")

        doc_id = str(doc.id)

        # B2-B6. 执行 ingestion 管线（复用 A1 的 content_list，不再重复解析）
        _step(6, "执行 run_ingestion（跳过解析，直接切分 → KP 分配 → 入库）")

        await run_ingestion(
            session, doc_id,
            start_page=0, end_page=20,
            preparsed_content_list=content_list,
        )

        # 刷新看结果
        await session.refresh(doc)
        print(f"  Document 状态: {doc.status}")
        if doc.error_message:
            print(f"  错误信息: {doc.error_message}")
        print(f"  页数/单元数: {doc.page_count}")

        # B7. 查看 knowledge_units
        _step(7, "查看入库的 knowledge_units")

        result = await session.execute(
            select(KnowledgeUnit)
            .where(KnowledgeUnit.document_id == doc.id)
            .order_by(KnowledgeUnit.seq_order)
        )
        units = result.scalars().all()
        print(f"  共 {len(units)} 个知识单元：")

        for u in units[:15]:
            kp = await session.get(KnowledgePoint, u.kp_id) if u.kp_id else None
            kp_title = kp.title if kp else "—"
            content_preview = u.content[:80].replace("\n", " ")
            print(f"  [{u.seq_order:4d}] page={u.page_ref or '-':>6s}  KP={kp_title:<20s}  content={content_preview}...")
        if len(units) > 15:
            print(f"  ... (共 {len(units)} 个，仅展示前 15)")

        # B8. 统计 KP 覆盖情况
        _step(8, "知识点覆盖统计")

        result = await session.execute(text("""
            SELECT kp.title, kp.kp_path, COUNT(ku.id) as unit_count
            FROM knowledge_points kp
            LEFT JOIN knowledge_units ku ON ku.kp_id = kp.id
            WHERE kp.course_id = :cid
            GROUP BY kp.id, kp.title, kp.kp_path
            ORDER BY kp.sort_order
        """), {"cid": course_id})
        rows = result.fetchall()
        covered = sum(1 for r in rows if r[2] > 0)
        total = len(rows)
        print(f"  知识点覆盖: {covered}/{total} ({100*covered//total if total else 0}%)")

        # 展示有内容的知识点（前20个）
        print("\n  知识点 → 知识单元数:")
        shown = 0
        for r in rows:
            if r[2] > 0 and shown < 20:
                print(f"    {r[1]:<50s} → {r[2]:4d} 单元")
                shown += 1

    _section("完成")
    print(f"\n  课程「{COURSE_NAME}」已就绪")
    print(f"    知识点: {len(kps)} 个")
    print(f"    知识单元: {len(units)} 个")
    print(f"    Document 状态: {doc.status}")
    print(f"\n  接下来可以：")
    print(f"    - GET /api/v1/courses/{course_id}/knowledge-points  查看知识点树")
    print(f"    - GET /api/v1/courses/{course_id}/documents         查看资料列表")
