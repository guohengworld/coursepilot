"""解析器共享工具 —— 内容切片、标题分块、页码格式化

阶段 A 改造：
  - A1: _split_by_headings 追踪当前标题文本，写入 meta_data["heading"]
  - A2: _filter_garbage 垃圾过滤（CIP、封面、目录页）
  - A3: _split_text_v2 数学块感知 + 段落边界优先
"""

from __future__ import annotations

import re

# ── A2: 垃圾过滤 ───────────────────────────────────────────

GARBAGE_PATTERNS: list[str] = [
    r"^[A-Z]{3,}\s.*CIP",  # CIP 数据（英文）
    r"^图书在版编目",  # CIP 数据（中文）
    r"^内容简介|^本书.*编写",  # 出版信息
    r"^封面|^扉页|^版权",  # 页面类型标记
    # 页眉页脚噪声（页码、书名重复、章名重复）
    r"^\d{1,3}\s*$",  # 纯页码行
    r"^[-—]+\s*\d+\s*[-—]+$",  # "-- 123 --" 样式页码
    r"^习题\s*$",  # 单独的 "习题" 标题（不含编号，容易被误判）
    # 纯分隔线
    r"^[-—=_*]{3,}$",
]

GARBAGE_PAGE_RANGE: tuple[int, int] = (0, 2)  # 前 3 页（封面+目录）整体跳过


def _filter_garbage(content_list: list[dict]) -> list[dict]:
    """过滤掉垃圾内容：封面、目录、CIP 数据等。

    三步过滤：
      1. 跳过 GARBAGE_PAGE_RANGE 范围内的页面（仅当文档有超出范围的正文页时）
      2. 跳过匹配 GARBAGE_PATTERNS 的文本行
      3. 跳过空文本
    """
    min_page, max_page = GARBAGE_PAGE_RANGE
    garbage_re = re.compile("|".join(GARBAGE_PATTERNS))

    # 仅当文档存在超出垃圾页范围的正文时才启用页码过滤，
    # 避免 DOCX 等无页码文件被整本过滤掉
    all_pages = {item.get("page_idx", 0) for item in content_list}
    has_content_pages = any(p > max_page for p in all_pages)
    skip_front_matter = has_content_pages

    filtered: list[dict] = []
    for item in content_list:
        page = item.get("page_idx", 0)
        text = item.get("text", "").strip()

        if not text:
            continue

        # 跳过封面/目录页（仅在文档有正文页时生效）
        if skip_front_matter and min_page <= page <= max_page:
            continue

        # 跳过 CIP 等出版信息
        if garbage_re.match(text):
            continue

        filtered.append(item)

    return filtered


# ── A1: 标题分块（追踪 heading） ────────────────────────────


def _split_by_headings(content_list: list[dict]) -> list[dict]:
    """按标题层级将 content_list 聚合为文本块。

    text_level ≤ 4 视为标题边界，触发新 block。
    追踪当前标题文本，写入 meta_data["heading"]。
    """
    blocks: list[dict] = []
    current: list[str] = []
    current_heading: str = "未知章节"
    current_text_level: int = 99
    current_pages: set[int] = set()
    block_has_body: bool = False       # 追踪当前 block 是否包含正文

    for item in content_list:
        text = item.get("text", "").strip()
        if not text:
            continue

        level = item.get("text_level", 99)
        page = item.get("page_idx", 0)

        # text_level ≤ 4 触发新 block
        if level <= 4 and current:
            # 纯标题 block 保留 heading 的 text_level，让下游 KPSplitter 能更新上下文
            final_level = current_text_level if not block_has_body else 99
            blocks.append(
                {
                    "text": "\n".join(current),
                    "page_ref": _format_page_ref(sorted(current_pages)),
                    "meta_data": {
                        "text_level": final_level,
                        "heading": current_heading,
                    },
                }
            )
            current = []
            current_pages = set()
            block_has_body = False

        # 更新当前标题
        if level <= 4:
            current_heading = text
        else:
            block_has_body = True

        current.append(text)
        current_text_level = level
        current_pages.add(page)

    # 最后一个 block
    if current:
        final_level = current_text_level if not block_has_body else 99
        blocks.append(
            {
                "text": "\n".join(current),
                "page_ref": _format_page_ref(sorted(current_pages)),
                "meta_data": {
                    "text_level": final_level,
                    "heading": current_heading,
                },
            }
        )

    return blocks


# ── A3: 数学块感知的文本切分 ────────────────────────────────


def _find_math_ranges(text: str) -> list[tuple[int, int]]:
    """找到所有 $$...$$ 数学块的范围（不可切分区域）。"""
    ranges: list[tuple[int, int]] = []
    i = 0
    while True:
        start = text.find("$$", i)
        if start == -1:
            break
        end = text.find("$$", start + 2)
        if end == -1:
            # 未闭合的 $$，从 start 到末尾
            ranges.append((start, len(text)))
            break
        ranges.append((start, end + 2))
        i = end + 2
    return ranges


def _find_inline_math_ranges(text: str) -> list[tuple[int, int]]:
    """找到所有 $...$ 内联公式的范围（跳过 $$）。"""
    ranges: list[tuple[int, int]] = []
    i = 0
    while True:
        start = text.find("$", i)
        if start == -1:
            break
        if text[start : start + 2] == "$$":
            i = start + 2
            continue
        end = text.find("$", start + 1)
        if end == -1:
            break
        ranges.append((start, end + 1))
        i = end + 1
    return ranges


def _is_inside_protected(pos: int, ranges: list[tuple[int, int]]) -> bool:
    """检查位置是否在受保护的数学范围内（不含边界）。"""
    for s, e in ranges:
        if s < pos < e:
            return True
    return False


def _split_text_v2(
    text: str,
    target_chars: int = 800,
    hard_lower: int = 400,
    hard_upper: int = 1200,
) -> list[str]:
    """数学块感知的文本切分器。

    策略（按优先级）：
      1. 优先在段落边界（双换行 \\n\\n）切分
      2. 次优在句边界（。！？后跟换行）切分
      3. 兜底在句号处切分
      4. 强制在 hard_upper 切分（避开数学块内部）

    目标 ~800 字符/unit，硬下限 400，硬上限 1200。
    """
    if len(text) <= hard_upper:
        return [text]

    math_ranges = _find_math_ranges(text)
    inline_ranges = _find_inline_math_ranges(text)
    all_protected = math_ranges + inline_ranges

    chunks: list[str] = []
    pos = 0

    while pos < len(text):
        remaining = len(text) - pos
        if remaining <= hard_upper:
            chunks.append(text[pos:])
            break

        search_start = pos + hard_lower
        search_end = min(pos + hard_upper, remaining + pos)
        best_split: int | None = None

        # 优先级 1：段落边界（双换行）
        for i in range(search_start, search_end - 1):
            if text[i : i + 2] == "\n\n" and not _is_inside_protected(i, all_protected):
                best_split = i + 1  # 保留一个换行在当前 chunk 末尾
                break

        # 优先级 2：句边界（。！？后跟换行）
        if best_split is None:
            for i in range(search_start, search_end):
                if text[i] in "。！？" and not _is_inside_protected(i, all_protected):
                    if i + 1 >= len(text) or text[i + 1] in "\n":
                        best_split = i + 1
                        break

        # 优先级 3：任意句号（即使在行中）
        if best_split is None:
            for i in range(search_start, search_end):
                if text[i] in "。！？" and not _is_inside_protected(i, all_protected):
                    best_split = i + 1
                    break

        # 优先级 4：强制切分（避开数学块内部）
        if best_split is None:
            best_split = search_end
            while _is_inside_protected(best_split, all_protected) and best_split > pos:
                best_split -= 1

        # 兜底：防止全数学区域导致 best_split ≤ pos 引发死循环
        if best_split <= pos:
            best_split = min(pos + hard_upper, len(text))

        chunks.append(text[pos:best_split].strip())
        pos = best_split

    return chunks


# ── 页码格式化 ─────────────────────────────────────────────


def _format_page_ref(pages: list[int]) -> str:
    """将页码列表格式化为人类可读的字符串，如 'p1' 或 'p1-3'"""
    if not pages:
        return ""
    if len(pages) == 1:
        return f"p{pages[0] + 1}"
    return f"p{pages[0] + 1}-{pages[-1] + 1}"


# ── 主入口 ─────────────────────────────────────────────────


def extract_knowledge_units(
    content_list: list[dict],
    document_id: str,
    kp_id: str,
    *,
    target_chars: int = 800,
    hard_lower: int = 400,
    hard_upper: int = 1200,
) -> list[dict]:
    """按标题层级分为 KnowledgeUnits。

    流程：
      1. _filter_garbage       ← A2: 垃圾过滤
      2. _split_by_headings    ← A1: 标题分块 + heading 追踪
      3. _split_text_v2        ← A3: 数学块感知切分

    :param content_list: MinerU / DOCX / MD 解析器的结构化输出
    :param document_id: Document UUID
    :param kp_id: KnowledgePoint UUID（默认用根节点，后续 kp_splitter 再分配）
    :param target_chars: 每 unit 目标字符数
    :param hard_lower: 切分硬下限
    :param hard_upper: 切分硬上限
    :return: List[dict]，可直接用于 KnowledgeUnit INSERT。
    """
    import logging, time
    _log = logging.getLogger(__name__)

    t0 = time.time()

    # A2: 垃圾过滤
    filtered = _filter_garbage(content_list)
    _log.info("  _filter_garbage: %d → %d 行 (%.1fs)", len(content_list), len(filtered), time.time() - t0)

    # A1: 标题分块（带 heading 追踪）
    t1 = time.time()
    blocks = _split_by_headings(filtered)
    _log.info("  _split_by_headings: %d blocks (%.1fs)", len(blocks), time.time() - t1)

    # A3: 数学块感知切分
    t2 = time.time()
    units: list[dict] = []
    seq = 0
    for block in blocks:
        chunks = _split_text_v2(
            block["text"],
            target_chars=target_chars,
            hard_lower=hard_lower,
            hard_upper=hard_upper,
        )
        for chunk in chunks:
            seq += 1
            units.append(
                {
                    "kp_id": kp_id,
                    "document_id": document_id,
                    "content": chunk,
                    "summary": None,
                    "seq_order": seq,
                    "page_ref": block.get("page_ref", ""),
                    "meta_data": block.get("meta_data", {}),
                }
            )
    _log.info("  _split_text_v2: %d units (%.1fs)", len(units), time.time() - t2)

    return units
