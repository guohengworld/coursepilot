"""DOCX 解析器 — 基于 python-docx 按 Heading 样式提取结构化内容。

与 pdf_parser.py 的区别：
- DOCX 有原生 Heading 样式，无需 OCR
- 直接通过 python-docx 读取段落样式即可获得层级
- 同样输出 content_list 格式，复用 parser_utils.extract_knowledge_units
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docx import Document as DocxDocument
from docx.oxml.ns import qn

from coursepilot.ingestion.parser_utils import extract_knowledge_units


async def parse_docx(file_path: str) -> dict[str, Any]:
    """解析 DOCX 文件，返回结构化内容。

    返回格式（与 MinerU 的 content_list 兼容）:
    {
        "markdown": "## 第一章…",
        "content_list": [
            {"type": "text", "text": "第一章 概述", "text_level": 2, "page_idx": 0},
            {"type": "text", "text": "本节介绍…", "text_level": 99, "page_idx": 0},
        ],
    }
    """
    doc = DocxDocument(file_path)
    items: list[dict] = []
    # DOCX 没有页码概念，统一标 0
    page_idx = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        level = _get_heading_level(para)
        items.append({
            "type": "text",
            "text": text,
            "text_level": level if level else 99,
            "page_idx": page_idx,
        })

    # 构建 markdown 概览
    md_lines = []
    for item in items:
        level = item["text_level"]
        prefix = "#" * level + " " if level <= 6 else ""
        md_lines.append(f"{prefix}{item['text']}")

    return {
        "markdown": "\n\n".join(md_lines),
        "content_list": items,
    }


def _get_heading_level(para) -> int | None:
    """从段落样式中提取标题层级。"""
    style = para.style
    if style is None:
        return None
    style_name = style.name or ""

    # Heading 1 → 1, Heading 2 → 2, ...
    if style_name.startswith("Heading"):
        try:
            return int(style_name.split()[-1])
        except (ValueError, IndexError):
            return 1

    # 中文字号样式（如 "标题 1"、"heading 1"）
    if "标题" in style_name or "heading" in style_name.lower():
        for ch in style_name:
            if ch.isdigit():
                return int(ch)

    return None


__all__ = ["parse_docx", "extract_knowledge_units"]
