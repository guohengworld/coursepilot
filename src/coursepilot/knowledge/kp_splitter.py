"""知识单元分配器：将解析后的文本块分配到对应知识点。

输入: 文本块列表 [{content, meta_data, ...}, ...] + 知识点列表 [{id, title, kp_path, level}]
输出: 每个文本块的 kp_id 和 kp_path 被填充。

MVP 策略: 标题前缀匹配 + 内容关键词匹配（后续可用 LLM 做语义匹配）。

输入：
  1. 文本块列表（来自 parser_utils.extract_knowledge_units）
     [{content: "FIFO 算法按页面...", meta_data: {text_level: 99}}, ...]
  2. 知识点列表（来自数据库 knowledge_points 表）
     [{id: "...", title: "FIFO", kp_path: "OS/.../FIFO", level: 4}, ...]

匹配策略：
  1. 维护 current_heading（"当前读到了哪一节的标题"）
  2. 遇到标题行 → 更新 current_heading，用标题去匹配 KP
  3. 遇到正文行 → 用 current_heading 匹配 KP
     - 精确匹配 "FIFO" == "FIFO"
     - 去编号匹配 "四、FIFO 页面置换" → "FIFO 页面置换" → 包含 "FIFO"
     - 内容关键词匹配 正文里出现 "Belady 异常" → 匹配到 KP "Belady 异常"
     - 兜底 → 分配给根 KP
输出：每个文本块的 kp_id 被填上


"""

from __future__ import annotations

import re
from collections.abc import Sequence


class KPSplitter:
    """知识单元分配器。

    splitter = KPSplitter(kp_flat_list, course_id)
    assigned = splitter.assign(parsed_blocks)
    """

    def __init__(self, kp_nodes: Sequence[dict], course_id: str):
        self.kp_nodes = list(kp_nodes)           # 展平的知识点列表
        self.course_id = course_id

        # title → [kp, ...] 索引，按 level 降序（深层优先）
        self._title_map: dict[str, list[dict]] = {}
        for kp in sorted(self.kp_nodes, key=lambda x: x.get("level", 99), reverse=True):
            t = kp.get("title", "")
            if t:
                self._title_map.setdefault(t, []).append(kp)

        # 根 KP（兜底用）
        self._root = next(
            (kp for kp in self.kp_nodes if kp.get("level") == 1),
            self.kp_nodes[0] if self.kp_nodes else None,
        )

    def assign(self, blocks: list[dict]) -> list[dict]:
        """为每个文本块填充 kp_id 和 kp_path。"""
        current_heading: str | None = None

        for block in blocks:
            text = block.get("content", "")
            text_level = block.get("meta_data", {}).get("text_level", 99)

            if text_level <= 4:       # 标题行 → 更新上下文
                current_heading = text
                kp_id = self._match_by_heading(text)
                block["kp_id"] = kp_id
                block["kp_path"] = self._lookup_path(kp_id)
                continue

            # 正文行：当前标题匹配 > 内容关键词匹配 > 根 KP 兜底
            kp_id = self._match_by_heading(current_heading) if current_heading else None
            if not kp_id:
                kp_id = self._match_by_content(text)
            if not kp_id and self._root:
                kp_id = self._root["id"]

            block["kp_id"] = kp_id
            block["kp_path"] = self._lookup_path(kp_id)

        return blocks

    # ── 匹配逻辑 ─────────────────────────────────────────

    def _match_by_heading(self, heading: str | None) -> str | None:
        if not heading:
            return None
        # 精确
        if heading in self._title_map:
            return self._title_map[heading][0]["id"]
        # 去编号
        cleaned = self._clean(heading)
        if cleaned and cleaned in self._title_map:
            return self._title_map[cleaned][0]["id"]
        # 模糊包含
        for kp in self.kp_nodes:
            t = kp.get("title", "")
            if t and (t in heading or heading in t):
                return kp["id"]
        return None

    def _match_by_content(self, text: str) -> str | None:
        """正文关键词匹配：KP title 至少 3 字且在正文中出现。"""
        for kp in self.kp_nodes:
            t = kp.get("title", "")
            if t and len(t) >= 3 and t in text:
                return kp["id"]
        return None

    def _lookup_path(self, kp_id: str | None) -> str:
        if not kp_id:
            return ""
        for kp in self.kp_nodes:
            if kp.get("id") == kp_id:
                return kp.get("kp_path", "")
        return ""

    @staticmethod
    def _clean(heading: str) -> str:
        return re.sub(
            r'^[\s]*(?:第[一二三四五六七八九十百千]+[章节]\s*'
            r'|[一二三四五六七八九十]+[、．.\s]\s*'
            r'|\d+[\.\、)\s]\s*'
            r'|\([一二三四五六七八九十\d]+\)\s*)',
            '', heading
        ).strip()
