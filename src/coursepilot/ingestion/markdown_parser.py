"""Markdown 解析器 — 按 # 层级提取结构化内容。

与 pdf_parser / docx_parser 的区别：
- MD 文件本身就是纯文本，无需 OCR 或提取
- # 数量直接对应标题层级
- 同样输出 content_list 格式，复用 parser_utils.extract_knowledge_units
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def parse_markdown(file_path: str) -> dict[str, Any]:
    """解析 Markdown 文件，返回结构化内容。

    返回格式（与 MinerU 的 content_list 兼容）:
    {
        "markdown": "原始全文",
        "content_list": [
            {"type": "text", "text": "# 第一章", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "这是正文。", "text_level": 99, "page_idx": 0},
        ],
    }
    """
    text = Path(file_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    items: list[dict] = []

    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # 检测代码块 fence（``` 或 ~~~）
        if re.match(r'^(`{3,}|~{3,})', stripped):
            in_code_block = not in_code_block
            continue

        # 代码块内部：全部当正文，不解析 heading
        if in_code_block:
            items.append({"type": "text", "text": stripped, "text_level": 99, "page_idx": 0})
            continue

        # 标题行：# 数量 = text_level，最多 4 级
        heading_match = re.match(r'^(#{1,6})\s+(.+)', stripped)
        if heading_match:
            level = min(len(heading_match.group(1)), 4)
            title = heading_match.group(2).strip()
            items.append({
                "type": "text",
                "text": title,
                "text_level": level,
                "page_idx": 0,
            })
        else:
            items.append({
                "type": "text",
                "text": stripped,
                "text_level": 99,
                "page_idx": 0,
            })

    return {
        "markdown": text,
        "content_list": items,
    }


__all__ = ["parse_markdown"]
