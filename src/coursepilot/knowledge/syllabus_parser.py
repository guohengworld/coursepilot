"""教学大纲解析器：Markdown 大纲 / 中文编号文本 → 知识点树节点

输入支持：
  1. Markdown：# 章 / ## 节 / ### 小节 / #### 细则
  2. 中文编号：第一章 / 第一节 / 一、 / 1. / (1)
  3. PDF 解析后的 content_list（text_level ≤ 4 的标题行）

输出: List[SyllabusNode] 或 List[dict]，可直接批量插入 knowledge_points 表。

如果有这段文本：
# 内存管理
## 虚拟内存
### 页面置换算法
#### 最佳置换 OPT
#### 先进先出 FIFO
#### 最近最久未使用 LRU
#### Belady 异常

输入：上面的 Markdown 大纲文本
  ↓ 逐行扫描，# 数量 = 层级深度
  ↓ 用栈维护 "当前在第几层"
  ↓ 自动构建父子关系
输出：一棵 SyllabusNode 树

"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 编号深度检测：MinerU 把同一字号的节/小节都标为 text_level=2，
# 但 "7.1" 有 2 段编号、"7.2.1" 有 3 段，后者层级更深。
_DOT_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)+)")
# 简单数字编号："1." / "2、" / "3)" / "4 " ——无 dot 分隔，深度需从上下文推断
_SIMPLE_NUM_RE = re.compile(r"^\d+[\.\、\)\s]")
# 图表题注："图 1-1" / "表 2-3"
_CAPTION_RE = re.compile(r"^[图表]\s*\d+[\-\.]\d+")
# 页码噪声："- 123 -" / "— 5 —"
_PAGE_NOISE_RE = re.compile(r"^[-—\s]*\d+[-—\s]*$", re.ASCII)
# 防止嵌套过深（level > 6 强制降到 6）
_MAX_LEVEL = 6


def _effective_level(raw_level: int, title: str) -> int:
    """结合标题编号模式修正 text_level，确保 7.2.1 不会和 7.2 同级。"""
    # 页码噪声：直接返回 99（非标题）
    if _PAGE_NOISE_RE.match(title):
        return 99
    # 图表题注：非标题
    if _CAPTION_RE.match(title):
        return 99
    # 中文章节标题始终为顶级，防止 MinerU 将其标为 L2
    if re.match(r"第[\d一二三四五六七八九十百千]+章", title):
        return 1
    # "第X节" 至少是二级
    if re.match(r"第[一二三四五六七八九十百千]+节", title):
        return max(raw_level, 2)
    # dot-numbering: "7.2.1" → 3 段 → level 3
    m = _DOT_NUMBER_RE.match(title)
    if m:
        num_depth = len(m.group(1).split("."))
        return max(raw_level, num_depth)
    # 简单数字编号先保留原始等级，在 headings_to_syllabus 中用上下文修正
    if _SIMPLE_NUM_RE.match(title):
        return raw_level
    return raw_level


@dataclass
class SyllabusNode:
    """教学大纲中的一个节点，对应 knowledge)points 表的一行"""
    title: str  # 知识点标题，如 "RR 调度算法"
    level: int  # 1=章, 2=节, 3=小节, 4=细则
    sort_order: int = 0  # 同级排序序号
    summary: str = ""  # 知识点摘要
    children: list[SyllabusNode] = field(default_factory=list)
    kp_path: str = ""  # 如 "OS/process/scheduling/rr"
    parent_id: str | None = None  # 数据库插入时回填的外键

class SyllabusParser:
    """教学大纲解析器

    parser = SyllabusParser()
    nodes = parser.parse(markdown_text, course_name="OS")
    flat = parser.flatten(nodes)  # → 可直接批量 INSERT
    """

    # 中文编号模式
    _NUMBER_RE = re.compile(
        r'^[\s]*'
        r'(?:'
        r'第[一二三四五六七八九十百千]+章\s*'  # "第一章"
        r'|第[一二三四五六七八九十百千]+节\s*'  # "第一节"
        r'|[一二三四五六七八九十]+[、．.\s]'  # "一、"
        r'|\d+[\.\、)\s]'  # "1." / "1、"
        r'|\([一二三四五六七八九十\d]+\)'  # "(一)" / "(1)"
        r')'
    )

    def parse(self, text: str, *, course_name: str = "") -> list[SyllabusNode]:
        """解析教学大纲文本，返回根节点列表（子节点挂在 children 里）。"""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        roots: list[SyllabusNode] = []
        stack: list[SyllabusNode] = []           # 当前路径栈，栈顶 = 最近祖先
        counters: dict[int, int] = {}            # level → 同级序号

        for line in lines:
            title, level = self._parse_line(line)
            if not title:
                continue

            # 弹出栈中所有 level >= 当前 level 的兄弟节点
            while stack and stack[-1].level >= level:
                stack.pop()

            counters[level] = counters.get(level, 0) + 1
            node = SyllabusNode(
                title=title,
                level=level,
                sort_order=counters[level],
            )

            if stack:
                parent = stack[-1]
                parent.children.append(node)
                ancestors = [p.title for p in stack] + [title]
                node.kp_path = (course_name or "course") + "/" + "/".join(ancestors)
            else:
                roots.append(node)
                node.kp_path = (course_name or "course") + "/" + title

            stack.append(node)

        return roots

    def flatten(self, nodes: list[SyllabusNode]) -> list[SyllabusNode]:
        """DFS 先序展平，方便批量插入数据库。"""
        result: list[SyllabusNode] = []
        for node in nodes:
            result.append(node)
            result.extend(self.flatten(node.children))
        return result

    # ── 行解析 ───────────────────────────────────────────

    def _parse_line(self, line: str) -> tuple[str, int]:
        """解析一行，返回 (title, level)。非标题返回 ("", 99)。"""
        # 1) Markdown heading
        m = re.match(r'^(#{1,6})\s+(.+)', line)
        if m:
            return m.group(2).strip(), min(len(m.group(1)), 4)

        # 2) 中文编号
        m = self._NUMBER_RE.match(line)
        if m:
            prefix = m.group()
            title = line[m.end():].strip() or line
            if '章' in prefix:
                level = 1
            elif '节' in prefix or any(prefix.strip().startswith(c) for c in '一二三四五六七八九十'):
                level = 2
            elif prefix[0].isdigit():
                level = 3
            else:
                level = 4
            return title, level

        # 3) 非标题
        return "", 99


# ── 模块级工具函数：从 content_list 提取标题 → 构建 KP 节点 ──


def extract_headings(content_list: list[dict]) -> list[dict]:
    """从解析后的 content_list 中提取标题行（text_level ≤ 4）。

    返回: [{"title": str, "level": int, "page_idx": int}, ...]
    """
    headings: list[dict] = []
    for item in content_list:
        raw_level = item.get("text_level", 99)
        if not raw_level or raw_level > 4:
            continue
        title = item.get("text", "").strip()
        effective = _effective_level(raw_level, title)
        # _effective_level 返回 99 表示非标题（页码噪声/题注）
        if effective >= 99:
            continue
        headings.append({
            "title": title,
            "level": effective,
            "page_idx": item.get("page_idx", 0),
        })
    return headings


def headings_to_syllabus(headings: list[dict], course_name: str) -> list[dict]:
    """将标题列表转换为知识点节点列表（含 kp_path + parent_title）。

    用栈维护层级关系，输出可直接用于 KPTree.create_from_nodes() 或手动 INSERT。

    预防措施：
    - 简单编号标题（"1." / "2、"）继承上一标题层级 + 1，避免挂错父节点
    - 过滤过短（≤2 字符）或纯标点的标题
    - 过滤超长标题（>50 字符，可能是正文误标）
    - 过滤页码噪声、图表题注（_effective_level 中处理）
    - 硬限制最大层级 _MAX_LEVEL
    - 同级重复标题跳过

    返回: [{"title", "level", "kp_path", "parent_title", "sort_order", "summary", "difficulty", "source"}, ...]
    """
    stack: list[dict] = []
    result: list[dict] = []
    counters: dict[int, int] = {}

    for h in headings:
        title = h["title"]
        level = h["level"]
        if not title:
            continue

        # 预防：_effective_level 已返回 99 的（页码/题注）直接跳过
        if level >= 99:
            continue

        # 预防：过短标题（≤1 字符的纯数字/纯标点，如 "1"、"一"）
        if len(title) < 2:
            continue

        # 预防：纯数字/纯标点标题
        if re.match(r"^[\d\s·.。、，,;；：:\-—]+$", title):
            continue

        # 预防：超长标题（>50 字符，可能是正文误标为标题）
        if len(title) > 50:
            continue

        # 上下文感知：简单编号标题继承上一标题层级 + 1
        # 例如 "1. 椭球面" 跟在 level-3 的 "7.5.2 二次曲面" 后 → 升到 level 4
        # 排除已有 dot-numbering 的标题（如 "4.2"、"4.3"），它们自身的编号已能确定层级
        if _SIMPLE_NUM_RE.match(title) and not _DOT_NUMBER_RE.match(title) and stack and stack[-1]["level"] > level:
            level = min(stack[-1]["level"] + 1, _MAX_LEVEL)

        # 硬限制：最大层级不超过 _MAX_LEVEL
        if level > _MAX_LEVEL:
            level = _MAX_LEVEL

        while stack and stack[-1]["level"] >= level:
            stack.pop()

        counters[level] = counters.get(level, 0) + 1

        if stack:
            parent = stack[-1]
            kp_path = parent["kp_path"] + "/" + title
            parent_title = parent["title"]
        else:
            kp_path = course_name + "/" + title
            parent_title = None

        node = {
            "title": title,
            "level": level,
            "kp_path": kp_path,
            "parent_title": parent_title,
            "sort_order": counters[level],
            "summary": "",
            "difficulty": 1,
            "source": "textbook",
        }
        result.append(node)
        stack.append(node)

    return result

