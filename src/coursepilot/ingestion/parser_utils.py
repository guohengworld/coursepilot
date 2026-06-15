"""解析器共享工具 —— 内容切片、标题分块、页码格式化"""

from __future__ import annotations




def extract_knowledge_units(
        content_list: list[dict],
        document_id: str,
        kp_id: str,
        *,
        max_tokens: int = 512,
        overlap: int = 50,
) -> list[dict]:
    """按标题层级分为 KnowledgeUnits

    :param content_list: MinerU 或 DOCX 解析器的结构化输出
    :param document_id: Document UUID
    :param kp_id:  KnowledgePoint UUID（默认用根节点，后续 kp_splitter 再分配）
    :param max_tokens: 每单元最大 token 数
    :param overlap: 切分重叠字数
    :return: List[dict]，可直接用于 KnowledgeUnit INSERT。
    """
    blocks = _split_by_headings(content_list)
    units: list[dict] = []
    seq = 0
    for block in blocks:
        chunks = _split_text(block["text"], max_tokens=max_tokens, overlap=overlap)
        for chunk in chunks:
            seq += 1
            units.append({
                "kp_id": kp_id,
                "document_id": document_id,
                "content": chunk,
                "summary": None,
                "seq_order": seq,
                "page_ref": block.get("page_ref", ""),
                "meta_data": block.get("meta_data", {}),
            })

    return units


def _split_by_headings(content_list: list[dict]) -> list[dict]:
    """按标题层级将 content_list 聚合为文本块"""
    blocks: list[dict] = []
    current: list[str] = []
    current_meta: dict = {"text_level": 99} # 当前块最近的元数据（用于记录标题层级）
    current_pages: set[int] = set()         # 当前块涉及的页码集合（用于去重记录页码范围）

    for item in content_list:
        text = item.get("text", "").strip()
        if not text:
            continue    # 跳过空行
        level = item.get("text_level", 99)  # 若无层级，默认为最底层99
        page = item.get("page_idx", 0)      # MinerU 输出的页码从 0 开始

        # 遇到标题（level ≤9 均为标题），且当前缓存区有内容，则打包成一个 block
        # level 99 或无 level = 正文，不触发切分
        if level <= 9 and current:
            blocks.append({
                "text": "\n".join(current),
                "page_ref": _format_page_ref(sorted(current_pages)),
                "meta_data": {"text_level": current_meta.get("text_level", 99)}
            })
            current = []
            current_pages = set()

        # 将当前行加入缓存区
        current.append(text)
        current_meta = item
        current_pages.add(page)

    # 处理循环结束后，缓冲区中剩余的最后一个块
    if current:
        blocks.append({
            "text": "\n".join(current),
            "page_ref": _format_page_ref(sorted(current_pages)),
            "meta_data": {"text_level": current_meta.get("text_level")}
        })

    return blocks

def _split_text(text: str, * , max_tokens: int, overlap: int) -> list[str]:
    """按token数切分文本，带重叠"""
    chunk_size = int(max_tokens * 1.5)
    step = chunk_size - overlap
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks


def _format_page_ref(pages: list[int]) -> str:
    """将页码列表格式化为人类可读的字符串，如 'p1' 或 'p1-3'"""
    if not pages:
        return ""
    if len(pages) == 1:
        return f"p{pages[0] + 1}"

    # 连续页码展示起止页
    return f"p{pages[0] + 1}-{pages[-1] + 1}"


