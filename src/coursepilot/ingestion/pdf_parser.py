"""PDF 解析器 — 基于 MinerU 实现扫描件/文本 PDF 的 OCR 解析与结构化切分。

工作流：
  1. 调用 MinerU OCR 引擎解析 PDF → Markdown + content_list.json
  2. 从 content_list 提取标题层级结构与文本块
  3. 按标题层级 + kp_max_tokens 约束切分为 KnowledgeUnit
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from coursepilot.config import settings
from coursepilot.ingestion.parser_utils import extract_knowledge_units

logger = logging.getLogger(__name__)


async def parse_pdf(
    pdf_path: str,
    output_dir: str | None = None,
    *,
    start_page: int = 0,
    end_page: int | None = None,
    method: str | None = None,
    backend: str | None = None,
    lang: str | None = None,
    formula: bool = True,
    table: bool = True,
) -> dict[str, Any]:
    """调用 MinerU 解析 PDF，返回结构化结果。"""
    output_dir = output_dir or settings.mineru_output_dir
    method = method or settings.mineru_method
    backend = backend or settings.mineru_backend
    lang = lang or settings.mineru_lang

    os.environ["MINERU_MODEL_SOURCE"] = settings.mineru_model_source
    from mineru.cli.client import run_orchestrated_cli

    pdf = Path(pdf_path)
    out = Path(output_dir)

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

    stem = pdf.stem
    ocr_dir = out / stem / "ocr"
    md_file = ocr_dir / f"{stem}.md"
    cl_file = ocr_dir / f"{stem}_content_list.json"

    return {
        "markdown": md_file.read_text(encoding="utf-8") if md_file.exists() else "",
        "content_list": json.loads(cl_file.read_text(encoding="utf-8")) if cl_file.exists() else [],
        "pages": end_page - start_page if end_page else None,
        "output_dir": str(ocr_dir),
    }


__all__ = ["parse_pdf", "extract_knowledge_units"]



