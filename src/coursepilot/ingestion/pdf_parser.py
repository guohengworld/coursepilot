"""PDF 解析器 — 基于 MinerU 实现扫描件/文本 PDF 的 OCR 解析与结构化切分。

工作流：
  1. 调用 MinerU OCR 引擎解析 PDF → Markdown + content_list.json
  2. 从 content_list 提取标题层级结构与文本块
  3. 按标题层级 + kp_max_tokens 约束切分为 KnowledgeUnit

性能说明：
  - 需要 CUDA 版 PyTorch（`uv pip install torch --index-url https://download.pytorch.org/whl/cu124`）
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
from typing import Any, Optional

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
) -> dict[str, Any]:
    """调用 MinerU 解析 PDF，返回结构化结果。

    参数为 None 时使用 config.py 中的默认值。
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

    os.environ["MINERU_MODEL_SOURCE"] = settings.mineru_model_source

    pdf = Path(pdf_path)
    out = Path(output_dir)

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
        # MinerU v3 输出到 <out>/<stem>/<method>/ 目录
        result_dir = out / stem / method
        md_file = result_dir / f"{stem}.md"
        cl_file = result_dir / f"{stem}_content_list.json"

    return {
        "markdown": md_file.read_text(encoding="utf-8") if md_file.exists() else "",
        "content_list": json.loads(cl_file.read_text(encoding="utf-8")) if cl_file.exists() else [],
        "pages": end_page - start_page if end_page else None,
        "output_dir": str(result_dir),
        "_timings": timings,
        "_config": {
            "method": method,
            "backend": backend,
            "formula": formula,
            "table": table,
        },
    }


__all__ = ["parse_pdf", "extract_knowledge_units"]



