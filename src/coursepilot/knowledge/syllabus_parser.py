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
        level = item.get("text_level", 99)
        if level and level <= 4:
            headings.append({
                "title": item.get("text", "").strip(),
                "level": level,
                "page_idx": item.get("page_idx", 0),
            })
    return headings


def headings_to_syllabus(headings: list[dict], course_name: str) -> list[dict]:
    """将标题列表转换为知识点节点列表（含 kp_path + parent_title）。

    用栈维护层级关系，输出可直接用于 KPTree.create_from_nodes() 或手动 INSERT。

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

