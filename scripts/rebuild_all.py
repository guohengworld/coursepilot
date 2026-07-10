"""一键重建全部教材数据（PostgreSQL + Milvus）

用法：
    cd f:/all-projs/coursepilot
    PYTHONPATH=src .venv/Scripts/python -m scripts.rebuild_all
    PYTHONPATH=src .venv/Scripts/python -m scripts.rebuild_all --yes       # 跳过确认
    PYTHONPATH=src .venv/Scripts/python -m scripts.rebuild_all --milvus-only  # 仅从 PG 重建 Milvus

流程：
  阶段 A：解析 PDF → 合并标题 → 构建知识点树 → PG 入库（KP 树，幂等去重）
  阶段 B：逐 PDF 调用 run_ingestion(preparsed_content_list=...)
          → 切分 → KP 分配 → SummaryBridge → BGE-M3 编码 → Milvus + PG 入库

特性：
  - 增量导入：已有 KP/Document/Unit 数据不受影响，_ensure_kp_tree 自动去重
  - 每本 PDF 只解析一次（MinerU），同一课程多卷教材共享知识点树
  - 支持 --milvus-only：从已有 PG 数据编码入库，无需重新解析 PDF
  - 单本失败不中断，最后打印汇总
"""

import argparse
import asyncio
import logging
import sys
import time
import uuid as _uuid
from pathlib import Path

# 确保 pipeline / SummaryBridge 的 logger.info() 可见
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("rebuild_all")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ═══════════════════════════════════════════════════════════════
# 教材 → 课程映射
# ═══════════════════════════════════════════════════════════════

PDF_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "pdfs"

COURSE_PDFS: dict[str, list[str]] = {
    # === 测试：仅 微积分 下册 ===
    "微积分": [
        "大学数学 微积分 下册.pdf",
    ],
    # "高等数学": [
    #     "同济高等数学·第八版 上册.pdf",
    #     "同济高等数学·第八版 下册.pdf",
    # ],
    # "数学分析": [
    #     "数学分析 第五版上册.pdf",
    #     "数学分析 第五版下册.pdf",
    # ],
    # "概率论与数理统计": [
    #     "概率论与数理统计 第五版.pdf",
    # ],
    # "高等代数": [
    #     "高等代数 第五版.pdf",
    # ],
}


# ═══════════════════════════════════════════════════════════════
# 辅助：解析文件（复用 seed_knowledge 的函数）
# ═══════════════════════════════════════════════════════════════

from scripts.seed_knowledge import parse_file


# ═══════════════════════════════════════════════════════════════
# 核心：处理一门课程
# ═══════════════════════════════════════════════════════════════

async def process_course_full(
    course_name: str,
    pdf_names: list[str],
    pdf_dir: Path,
    *,
    milvus_only: bool = False,
) -> dict:
    """增量处理一门课程：解析 → KP 树 → 切分 → SummaryBridge → 编码 → PG + Milvus"""
    from coursepilot.db import get_session_etx
    from coursepilot.models import Course, KnowledgePoint, Document, KnowledgeUnit, User
    from sqlalchemy import select

    total_start = time.time()
    stats = {
        "course": course_name,
        "pdfs": [],
        "total_units": 0,
        "total_kps": 0,
        "milvus_count": 0,
        "error": None,
    }

    # ── milvus_only 模式：直接从 PG 读取已有数据编码入库 ──
    if milvus_only:
        async with get_session_etx() as session:
            result = await session.execute(
                select(Course).where(Course.name == course_name)
            )
            course = result.scalar_one_or_none()
            if not course:
                stats["error"] = "课程不存在"
                return stats

            # 通过 Document 链查询课程下所有知识单元（含无 kp_id 的）
            result = await session.execute(
                select(KnowledgeUnit)
                .join(Document, KnowledgeUnit.document_id == Document.id)
                .where(Document.course_id == course.id)
                .order_by(KnowledgeUnit.id)
            )
            units = result.all()
            if not units:
                stats["error"] = "没有知识单元"
                return stats

            print(f"  从 PG 读取到 {len(units)} 个知识单元")
            ku_list = [
                {
                    "_unit_id": str(ku.id),
                    "kp_id": str(ku.kp_id) if ku.kp_id else "",
                    "content": ku.content,
                    "document_id": str(ku.document_id) if ku.document_id else "",
                    "kp_path": "",  # 下面从 KP 表查
                }
                for ku in units
            ]

            # 查 kp_path
            kp_ids = list({ku["kp_id"] for ku in ku_list if ku["kp_id"]})
            if kp_ids:
                kp_result = await session.execute(
                    select(KnowledgePoint).where(
                        KnowledgePoint.id.in_([_uuid.UUID(k) for k in kp_ids])
                    )
                )
                kp_map = {str(kp.id): kp.kp_path for kp in kp_result.scalars()}
                for ku in ku_list:
                    ku["kp_path"] = kp_map.get(ku["kp_id"], "")

            stats["total_units"] = len(ku_list)
            milvus_count = await _encode_and_insert(ku_list, str(course.id))
            stats["milvus_count"] = milvus_count
            stats["pdfs"].append({"name": "从 PG 重建", "status": "ok", "units": len(ku_list)})

        total_elapsed = time.time() - total_start
        stats["elapsed"] = total_elapsed
        print(f"  🏁 课程 '{course_name}' Milvus 重建完成 ({total_elapsed:.0f}s)")
        return stats

    # ── 完整模式：解析 PDF → 全套流程 ──

    # 1. 解析所有 PDF，收集 content_list
    pdf_data: list[dict] = []

    for pdf_name in pdf_names:
        pdf_path = pdf_dir / pdf_name
        if not pdf_path.exists():
            print(f"  ⚠ 文件不存在，跳过: {pdf_path}")
            continue

        file_size_mb = pdf_path.stat().st_size / 1024 / 1024
        print(f"\n  📄 解析: {pdf_name} ({file_size_mb:.0f} MB)")
        t0 = time.time()

        try:
            if pdf_path.suffix.lower() == ".pdf":
                content_list, _headings = await _parse_pdf_in_batches(str(pdf_path))
            else:
                # docx / md 文件小，直接解析
                content_list, _headings = await parse_file(str(pdf_path))
        except Exception as e:
            print(f"  ❌ 解析失败: {e}")
            stats["pdfs"].append({
                "name": pdf_name, "status": "parse_failed", "error": str(e),
            })
            continue

        elapsed = time.time() - t0
        print(f"     ⏱ {elapsed:.0f}s | {len(content_list)} 行")

        pdf_data.append({
            "filename": pdf_name,
            "file_path": str(pdf_path),
            "file_size": pdf_path.stat().st_size,
            "content_list": content_list,
        })

    if not pdf_data:
        stats["error"] = "没有成功解析任何 PDF"
        return stats

    # 3. 写入数据库
    async with get_session_etx() as session:
        # 3a. 创建课程
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

        # 3b. 查找 uploader
        r = await session.execute(
            select(User).where(User.role == "super").limit(1)
        )
        uploader = r.scalar_one()

        # 3c. 为每卷 PDF 创建 Document → 调用 run_ingestion（自动构建 KP 树）
        from coursepilot.ingestion.pipeline import run_ingestion

        for pd in pdf_data:
            print(f"\n  📝 处理: {pd['filename']}")

            doc = Document(
                course_id=course.id,
                filename=pd["filename"],
                file_type="pdf",
                file_size=pd["file_size"],
                file_path=pd["file_path"],
                uploader_id=uploader.id,
                status="pending",
            )
            session.add(doc)
            await session.flush()
            await session.refresh(doc)

            try:
                await run_ingestion(
                    session,
                    str(doc.id),
                    preparsed_content_list=pd["content_list"],
                )
                # run_ingestion 内部已设置 doc.status="ready" + doc.page_count
                await session.refresh(doc)
                units_count = doc.page_count or 0
                stats["total_units"] += units_count
                stats["pdfs"].append({
                    "name": pd["filename"],
                    "status": doc.status,
                    "units": units_count,
                })
                print(f"     ✅ 完成: {units_count} 单元, status={doc.status}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                stats["pdfs"].append({
                    "name": pd["filename"],
                    "status": "failed",
                    "error": str(e),
                })
                print(f"     ❌ 失败: {e}")

        # 统计 KP 数量（由 run_ingestion 自动构建）
        from coursepilot.models import KnowledgePoint
        kp_result = await session.execute(
            select(KnowledgePoint)
            .where(KnowledgePoint.course_id == course.id)
        )
        kp_count = len(kp_result.scalars().all())
        stats["total_kps"] = kp_count
        print(f"     ✅ 知识点树: {kp_count} 个节点（由管线自动构建）")

        # 统计 Milvus
        try:
            from coursepilot.rag.vector_store import VectorStore
            store = VectorStore()
            stats["milvus_count"] = store.count()
        except Exception:
            stats["milvus_count"] = 0

    total_elapsed = time.time() - total_start
    stats["elapsed"] = total_elapsed
    print(f"\n  🏁 课程 '{course_name}' 完成 ({total_elapsed:.0f}s)")
    return stats


# ═══════════════════════════════════════════════════════════════
# Milvus 编码 + 入库
# ═══════════════════════════════════════════════════════════════

async def _encode_and_insert(units: list[dict], course_id: str) -> int:
    """BGE-M3 编码 + Milvus 批量插入，返回成功插入条数"""
    print(f"\n  🔢 BGE-M3 编码 + Milvus 入库 ({len(units)} 条)...")

    try:
        from coursepilot.rag.encoder import Encoder
        from coursepilot.rag.vector_store import VectorStore
    except ImportError as e:
        print(f"  ⚠ 导入失败: {e}，跳过 Milvus 入库")
        return 0

    try:
        encoder = Encoder()
    except Exception as e:
        print(f"  ⚠ 加载 BGE-M3 失败: {e}，跳过 Milvus 入库")
        return 0

    try:
        store = VectorStore()
        store.create_collection()
    except Exception as e:
        print(f"  ⚠ 连接 Milvus 失败: {e}，跳过 Milvus 入库")
        return 0

    # 批量编码（每批 32 条）
    batch_size = 32
    total_inserted = 0

    for batch_start in range(0, len(units), batch_size):
        batch = units[batch_start:batch_start + batch_size]
        texts = [
            (u.get("summary") or "") + "\n" + u["content"]
            for u in batch
        ]

        try:
            vecs = encoder.encode(texts)
        except Exception as e:
            print(f"  ⚠ 编码批次 {batch_start} 失败: {e}")
            continue

        payloads = []
        for u, vec in zip(batch, vecs):
            payloads.append({
                "uuid": u["_unit_id"],
                "document_id": u.get("document_id", ""),
                "dense_vec": vec["dense"],
                "sparse_vec": vec["sparse"],
                "kp_id": u.get("kp_id", ""),
                "course_id": course_id,
                "kp_path": u.get("kp_path", ""),
                "content": u["content"][:8192],
            })

        try:
            store.insert(payloads)
            total_inserted += len(payloads)
        except Exception as e:
            print(f"  ⚠ Milvus 插入批次 {batch_start} 失败: {e}")

    print(f"     ✅ Milvus: {total_inserted}/{len(units)} 条入库")
    return total_inserted


# ═══════════════════════════════════════════════════════════════
# 分批 PDF 解析（避免 MinerU 一次加载全部页面的 MemoryError）
# ═══════════════════════════════════════════════════════════════

async def _parse_pdf_in_batches(
    pdf_path: str,
    batch_size: int = 15,
) -> tuple[list[dict], list[dict]]:
    """分批解析 PDF，每批 batch_size 页，合并 content_list + headings。

    MinerU 在 Windows 下用 multiprocessing 加载 PDF 图像，高分辨率
    扫描页的图像数据 pickle 序列化时容易 MemoryError。分批缩小每次
    加载的页数范围，绕过 IPC 内存瓶颈。
    """
    import fitz  # PyMuPDF — MinerU 的依赖，可直接用

    pdf_doc = fitz.open(pdf_path)
    total_pages = pdf_doc.page_count
    pdf_doc.close()

    logger.info("PDF 共 %d 页，分批解析（每批 %d 页）", total_pages, batch_size)

    from coursepilot.ingestion.pdf_parser import parse_pdf
    from coursepilot.knowledge.syllabus_parser import extract_headings

    all_content = []

    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages) - 1
        logger.info("  解析第 %d~%d 页...", batch_start + 1, batch_end + 1)

        try:
            result = await parse_pdf(
                pdf_path,
                start_page=batch_start,
                end_page=batch_end,
            )
        except Exception as e:
            logger.error("第 %d~%d 页解析失败: %s", batch_start + 1, batch_end + 1, e)
            raise

        batch_content = result.get("content_list", [])
        if not batch_content:
            logger.warning("第 %d~%d 页解析结果为空，跳过", batch_start + 1, batch_end + 1)
            continue

        # page_idx 校正：MinerU 返回的 page_idx 可能是 batch 内相对值
        # （从 0 开始），需要修正为绝对页码
        if batch_content and batch_content[0].get("page_idx", 0) < batch_start:
            for item in batch_content:
                if "page_idx" in item:
                    item["page_idx"] = item["page_idx"] + batch_start

        all_content.extend(batch_content)
        logger.info(
            "第 %d~%d 页完成: %d 条（累计 %d 条）",
            batch_start + 1, batch_end + 1,
            len(batch_content), len(all_content),
        )

    if not all_content:
        raise ValueError("解析结果为空，请检查文件是否可读")

    headings = extract_headings(all_content)
    logger.info("分批解析完成: 共 %d 条 content_list, %d 个标题", len(all_content), len(headings))
    return all_content, headings


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

async def main():
    # Windows 终端默认 GBK 编码无法输出 emoji，强行切换到 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="一键重建全部教材数据（PG + Milvus）")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认提示")
    parser.add_argument("--milvus-only", action="store_true",
                        help="仅从已有 PG 数据重建 Milvus（不重新解析 PDF）")
    parser.add_argument("--course", type=str, default=None,
                        help="只处理指定课程（用于测试单门课程）")
    args = parser.parse_args()

    courses = {args.course: COURSE_PDFS[args.course]} if args.course else COURSE_PDFS

    mode = "Milvus 重建（从 PG）" if args.milvus_only else "完整重建（PG + Milvus）"
    total_pdfs = sum(len(v) for v in courses.values())

    print("=" * 70)
    print(f"  CoursePilot 一键重建 — {mode}")
    print(f"  共 {len(courses)} 门课程，{total_pdfs} 本 PDF")
    print("=" * 70)

    if not args.milvus_only and not args.yes:
        print("\n⚠  此操作将解析 PDF 并增量导入课程（已有数据不受影响）。")
        print("   MinerU 解析 8 本教材可能需要 40 分钟 ~ 数小时。")
        resp = input("\n  确认继续？[y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("  已取消")
            return

    global_start = time.time()
    all_stats: list[dict] = []

    for course_name, pdf_names in courses.items():
        print(f"\n{'─' * 70}")
        print(f"📚 课程: {course_name} ({len(pdf_names)} 卷)")
        print(f"{'─' * 70}")

        try:
            stats = await process_course_full(
                course_name, pdf_names, PDF_DIR, milvus_only=args.milvus_only,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            stats = {"course": course_name, "error": str(e)}
        all_stats.append(stats)

    # ── 汇总 ──
    total_elapsed = time.time() - global_start
    print(f"\n{'=' * 70}")
    print(f"  导入完成")
    print(f"{'=' * 70}")
    print(f"  总耗时: {total_elapsed / 60:.1f} 分钟 ({total_elapsed:.0f}s)")

    total_ok = 0
    total_fail = 0
    total_units = 0
    total_kps = 0
    total_milvus = 0

    for s in all_stats:
        if s.get("error"):
            print(f"\n  ❌ {s['course']}: {s['error']}")
            total_fail += 1
        else:
            total_ok += 1
            for p in s["pdfs"]:
                if p["status"] == "ok":
                    print(f"  ✅ {s['course']} / {p['name']}: {p['units']} 单元")
            total_units += s.get("total_units", 0)
            total_kps += s.get("total_kps", 0)
            total_milvus += s.get("milvus_count", 0)

    print(f"\n  📊 课程: {total_ok} 成功, {total_fail} 失败")
    print(f"  📊 知识点总计: {total_kps}")
    print(f"  📊 知识单元总计: {total_units}")
    print(f"  📊 Milvus 向量总计: {total_milvus}")

    # 验证 Milvus
    if total_milvus > 0:
        try:
            from coursepilot.rag.vector_store import VectorStore
            store = VectorStore()
            actual = store.count()
            print(f"  📊 Milvus 实际行数: {actual}")
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
