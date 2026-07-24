"""PDF 解析器 — 支持文字版快速通道（PyMuPDF）与扫描件 MinerU OCR 解析。

工作流：
  1. 自动检测 PDF 类型（文字版 / 扫描件）
  2. 文字版 → PyMuPDF 快速提取文本、标题、content_list
  3. 扫描件 → MinerU OCR 引擎解析 → Markdown + content_list.json
  4. 从 content_list 提取标题层级结构与文本块
  5. 按标题层级 + kp_max_tokens 约束切分为 KnowledgeUnit

性能说明：
  - 文字版 300 页通常 < 30s（PyMuPDF 纯 CPU）
  - 扫描件需要 CUDA 版 PyTorch
  - method="auto" 时文字页跳过 OCR，仅扫描页走 PaddleOCR
  - 返回的 _timings 字段记录各步骤耗时，用于诊断瓶颈
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from coursepilot.config import settings
from coursepilot.ingestion.parser_utils import extract_knowledge_units

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# 计时工具
# ═══════════════════════════════════════════════════════════


@contextmanager
def _timed(label: str, timings: dict[str, float] | None = None):
    """记录代码块耗时。若传入 timings dict，结果会写入其中。"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        if timings is not None:
            timings[label] = round(elapsed, 2)
        logger.info("⏱ %s: %.2fs", label, elapsed)


# ═══════════════════════════════════════════════════════════
# 文字版 PDF 快速通道
# ═══════════════════════════════════════════════════════════


def detect_pdf_type(pdf_path: str, sample_pages: int = 5) -> str:
    """采样检测 PDF 是文字版还是扫描件。

    取前、中、后若干页，统计每页可提取文本字符数。
    若平均字符数超过阈值，则判定为文字版（text），否则走 MinerU（mineru）。
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF 未安装，无法检测 PDF 类型，默认走 MinerU")
        return "mineru"

    doc = fitz.open(pdf_path)
    try:
        total = len(doc)
        if total == 0:
            return "mineru"

        # 采样前、1/4、1/2、3/4、末页，避免只采封面/目录
        indices = sorted({
            0,
            max(0, total // 4 - 1),
            max(0, total // 2 - 1),
            max(0, 3 * total // 4 - 1),
            total - 1,
        })
        indices = [i for i in indices if i < total][:sample_pages]

        total_chars = 0
        for i in indices:
            text = doc[i].get_text()
            total_chars += len(text.strip())

        avg_chars = total_chars / len(indices) if indices else 0
        pdf_type = "text" if avg_chars >= settings.pdf_text_min_chars_per_page else "mineru"
        logger.info(
            "PDF 类型检测: %s, 采样页=%s, 平均字符数=%.0f, 类型=%s",
            Path(pdf_path).name,
            indices,
            avg_chars,
            pdf_type,
        )
        return pdf_type
    finally:
        doc.close()


def _heading_level_from_font(size: float, is_bold: bool, min_size: float) -> int:
    """根据字体大小和粗体判断标题层级。"""
    if size >= min_size + 6 or (size >= min_size + 4 and is_bold):
        return 1
    if size >= min_size + 3:
        return 2
    if size >= min_size + 1:
        return 3
    return 99


def parse_text_pdf(pdf_path: str) -> dict[str, Any]:
    """用 PyMuPDF 快速解析文字版 PDF，输出与 MinerU 兼容的 content_list。

    输出字段：
      - text: 文本内容
      - text_level: 标题层级（1/2/3/99），99 表示正文
      - page_idx: 0-based 页码
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    content_list: list[dict[str, Any]] = []
    md_lines: list[str] = []
    page_count = len(doc)

    try:
        # 先计算全文字体大小中位数，作为正文基准字号
        sizes: list[float] = []
        for page in doc:
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            sizes.append(span.get("size", 12))
        base_size = sorted(sizes)[len(sizes) // 2] if sizes else 12.0
        min_heading_size = max(base_size, float(settings.pdf_heading_font_min))

        for page_idx, page in enumerate(doc):
            blocks = page.get_text("dict").get("blocks", [])
            # 按纵坐标排序，保证阅读顺序
            blocks.sort(key=lambda b: b["bbox"][1])

            for block in blocks:
                if "lines" not in block:
                    continue

                block_texts: list[str] = []
                max_size = 0.0
                is_bold = False

                for line in block["lines"]:
                    line_texts: list[str] = []
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        line_texts.append(text)
                        size = span.get("size", 12)
                        flags = span.get("flags", 0)
                        max_size = max(max_size, size)
                        # flags 第 4 位表示粗体
                        if flags & (1 << 4):
                            is_bold = True

                    if line_texts:
                        block_texts.append("".join(line_texts))

                if not block_texts:
                    continue

                full_text = "\n".join(block_texts).strip()
                if not full_text:
                    continue

                level = _heading_level_from_font(max_size, is_bold, min_heading_size)

                content_list.append({
                    "text": full_text,
                    "text_level": level,
                    "page_idx": page_idx,
                })

                # 同步生成 markdown，便于调试和下游复用
                if level <= 3:
                    md_lines.append(f"{'#' * level} {full_text}")
                else:
                    md_lines.append(full_text)

    finally:
        doc.close()

    return {
        "markdown": "\n\n".join(md_lines),
        "content_list": content_list,
        "pages": page_count,
        "output_dir": "",
        "_timings": {"text_extract": 0.0},
        "_config": {"method": "text_fast_path", "backend": "pymupdf"},
    }


# ═══════════════════════════════════════════════════════════
# MinerU OCR 解析
# ═══════════════════════════════════════════════════════════


async def _parse_pdf_mineru(
    pdf_path: str,
    output_dir: str,
    *,
    start_page: int,
    end_page: int | None,
    method: str,
    backend: str,
    lang: str,
    formula: bool,
    table: bool,
    timings: dict[str, float],
) -> dict[str, Any]:
    """调用 MinerU 解析扫描件 PDF。"""
    pdf = Path(pdf_path)
    out = Path(output_dir)

    os.environ["MINERU_MODEL_SOURCE"] = settings.mineru_model_source

    page_range = f"p{start_page}-{end_page}" if end_page else "全部"
    logger.info(
        "MinerU 开始解析: %s (%s) method=%s backend=%s formula=%s table=%s",
        pdf.name, page_range, method, backend, formula, table,
    )

    with _timed("mineru_ocr", timings):
        from mineru.cli.client import run_orchestrated_cli
        await run_orchestrated_cli(
            input_path=pdf,
            output_dir=out,
            method=method,
            backend=backend,
            lang=lang,
            server_url=None,
            api_url=None,
            start_page_id=start_page,
            end_page_id=end_page,
            formula_enable=formula,
            table_enable=table,
        )

    with _timed("read_output", timings):
        stem = pdf.stem
        result_dir = out / stem / method
        md_file = result_dir / f"{stem}.md"
        cl_file = result_dir / f"{stem}_content_list.json"

    return {
        "markdown": md_file.read_text(encoding="utf-8") if md_file.exists() else "",
        "content_list": json.loads(cl_file.read_text(encoding="utf-8")) if cl_file.exists() else [],
        "pages": end_page - start_page if end_page else None,
        "output_dir": str(result_dir),
    }


async def parse_pdf(
    pdf_path: str,
    output_dir: str | None = None,
    *,
    start_page: int = 0,
    end_page: int | None = None,
    method: str | None = None,
    backend: str | None = None,
    lang: str | None = None,
    formula: bool | None = None,
    table: bool | None = None,
    force_mineru: bool = False,
) -> dict[str, Any]:
    """解析 PDF，自动选择文字版快速通道或 MinerU OCR。

    参数为 None 时使用 config.py 中的默认值。
    force_mineru=True 时跳过检测，强制使用 MinerU（用于需要 OCR 的场景）。
    """
    timings: dict[str, float] = {}

    output_dir = output_dir or settings.mineru_output_dir
    method = method or settings.mineru_method
    backend = backend or settings.mineru_backend
    lang = lang or settings.mineru_lang
    if formula is None:
        formula = settings.mineru_formula_enable
    if table is None:
        table = settings.mineru_table_enable

    pdf = Path(pdf_path)

    # ── 文字版快速通道 ─────────────────────────────────────
    if not force_mineru and settings.pdf_enable_text_fast_path:
        with _timed("detect_pdf_type", timings):
            pdf_type = await asyncio.to_thread(detect_pdf_type, pdf_path)

        if pdf_type == "text":
            logger.info("文字版 PDF，走 PyMuPDF 快速通道: %s", pdf.name)
            with _timed("parse_text_pdf", timings):
                result = await asyncio.to_thread(parse_text_pdf, pdf_path)
            result["_timings"] = timings
            return result

    # ── 扫描件 MinerU OCR ──────────────────────────────────
    result = await _parse_pdf_mineru(
        pdf_path,
        output_dir,
        start_page=start_page,
        end_page=end_page,
        method=method,
        backend=backend,
        lang=lang,
        formula=formula,
        table=table,
        timings=timings,
    )
    result["_timings"] = timings
    result["_config"] = {
        "method": method,
        "backend": backend,
        "formula": formula,
        "table": table,
    }
    return result


__all__ = ["parse_pdf", "parse_text_pdf", "detect_pdf_type", "extract_knowledge_units"]
