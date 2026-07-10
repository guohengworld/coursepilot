"""Week 2 全方位测试：覆盖所有知识模块 + 存储 + 管线 + seed 工具

运行方式：
    pytest tests/test_week2.py -v                          # 全部
    pytest tests/test_week2.py -v -m "not slow"            # 跳过慢测试
    pytest tests/test_week2.py -v -k "TestMarkdownParser"  # 单个模块
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 确保 scripts/ 下的工具函数可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════
# 共享 fixtures
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def sample_md():
    return """# 进程管理
## 进程调度
### 先来先服务 FCFS
### 短作业优先 SJF
## 进程同步
### 信号量机制
### 经典同步问题
"""


@pytest.fixture
def kp_nodes():
    return [
        {"id": "uuid-1001", "title": "进程管理",       "kp_path": "OS/进程管理",                          "level": 1},
        {"id": "uuid-1002", "title": "进程调度",       "kp_path": "OS/进程管理/进程调度",                   "level": 2},
        {"id": "uuid-1003", "title": "先来先服务 FCFS", "kp_path": "OS/进程管理/进程调度/先来先服务 FCFS",  "level": 3},
        {"id": "uuid-1004", "title": "短作业优先 SJF",  "kp_path": "OS/进程管理/进程调度/短作业优先 SJF",   "level": 3},
        {"id": "uuid-1005", "title": "进程同步",       "kp_path": "OS/进程管理/进程同步",                   "level": 2},
        {"id": "uuid-1006", "title": "信号量机制",     "kp_path": "OS/进程管理/进程同步/信号量机制",         "level": 3},
        {"id": "uuid-1007", "title": "经典同步问题",   "kp_path": "OS/进程管理/进程同步/经典同步问题",       "level": 3},
    ]


@pytest.fixture
def parsed_blocks():
    return [
        {"content": "进程调度",           "meta_data": {"text_level": 2},  "page_ref": "p10", "seq_order": 1},
        {"content": "FCFS 算法按照作业到达的先后顺序进行调度，非抢占式。", "meta_data": {"text_level": 99}, "page_ref": "p10", "seq_order": 2},
        {"content": "短作业优先 SJF",    "meta_data": {"text_level": 3},  "page_ref": "p11", "seq_order": 3},
        {"content": "SJF 算法优先调度预计运行时间最短的作业。", "meta_data": {"text_level": 99}, "page_ref": "p11", "seq_order": 4},
        {"content": "互斥信号量用于保护临界区资源。", "meta_data": {"text_level": 99}, "page_ref": "p20", "seq_order": 5},
    ]


# ═══════════════════════════════════════════════════════════
# 1. MarkdownParser — markdown_parser.py
# ═══════════════════════════════════════════════════════════

class TestMarkdownParser:
    """测试 .md 文件 → content_list 的解析"""

    def test_parse_headings(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# 第一章\n## 第一节\n正文内容\n### 小节\n更多正文\n", encoding="utf-8")
        from coursepilot.ingestion.markdown_parser import parse_markdown
        result = parse_markdown(str(md_file))
        cl = result["content_list"]
        assert len(cl) == 5
        assert cl[0] == {"type": "text", "text": "第一章", "text_level": 1, "page_idx": 0}
        assert cl[2]["text"] == "正文内容"
        assert cl[2]["text_level"] == 99

    def test_heading_level_capped_at_4(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("##### 五级\n###### 六级\n", encoding="utf-8")
        from coursepilot.ingestion.markdown_parser import parse_markdown
        result = parse_markdown(str(md_file))
        assert result["content_list"][0]["text_level"] == 4
        assert result["content_list"][1]["text_level"] == 4

    def test_empty_file(self, tmp_path):
        md_file = tmp_path / "empty.md"
        md_file.write_text("", encoding="utf-8")
        from coursepilot.ingestion.markdown_parser import parse_markdown
        result = parse_markdown(str(md_file))
        assert result["content_list"] == []

    def test_blank_lines_skipped(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Title\n\n\n## Section\n\nbody\n", encoding="utf-8")
        from coursepilot.ingestion.markdown_parser import parse_markdown
        result = parse_markdown(str(md_file))
        texts = [item["text"] for item in result["content_list"]]
        assert texts == ["Title", "Section", "body"]

    def test_no_headings_all_body(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("段落一。\n\n段落二。\n", encoding="utf-8")
        from coursepilot.ingestion.markdown_parser import parse_markdown
        result = parse_markdown(str(md_file))
        assert len(result["content_list"]) == 2
        assert all(item["text_level"] == 99 for item in result["content_list"])

    def test_raw_text_field(self, tmp_path):
        raw = "# Title\nbody\n"
        md_file = tmp_path / "test.md"
        md_file.write_text(raw, encoding="utf-8")
        from coursepilot.ingestion.markdown_parser import parse_markdown
        result = parse_markdown(str(md_file))
        assert result["markdown"] == raw

    def test_heading_with_leading_spaces(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("  ## 带空格的标题\n正文\n", encoding="utf-8")
        from coursepilot.ingestion.markdown_parser import parse_markdown
        result = parse_markdown(str(md_file))
        assert result["content_list"][0]["text"] == "带空格的标题"
        assert result["content_list"][0]["text_level"] == 2


# ═══════════════════════════════════════════════════════════
# 2. SyllabusParser — syllabus_parser.py
# ═══════════════════════════════════════════════════════════

class TestSyllabusParser:
    """测试教学大纲 → SyllabusNode 树的解析"""

    # ── Markdown 标题 ──

    def test_parse_markdown(self, sample_md):
        from coursepilot.knowledge.syllabus_parser import SyllabusParser
        parser = SyllabusParser()
        nodes = parser.parse(sample_md, course_name="OS")
        assert len(nodes) == 1
        assert nodes[0].title == "进程管理"
        assert nodes[0].level == 1
        assert nodes[0].kp_path == "OS/进程管理"

    def test_children(self, sample_md):
        from coursepilot.knowledge.syllabus_parser import SyllabusParser
        parser = SyllabusParser()
        nodes = parser.parse(sample_md, course_name="OS")
        children = nodes[0].children
        assert len(children) == 2
        assert children[0].title == "进程调度"
        assert children[0].kp_path == "OS/进程管理/进程调度"

    def test_grandchildren(self, sample_md):
        from coursepilot.knowledge.syllabus_parser import SyllabusParser
        parser = SyllabusParser()
        nodes = parser.parse(sample_md, course_name="OS")
        grand = nodes[0].children[0].children
        assert len(grand) == 2
        assert grand[0].title == "先来先服务 FCFS"
        assert grand[0].kp_path.endswith("/先来先服务 FCFS")

    def test_flatten_dfs_order(self, sample_md):
        from coursepilot.knowledge.syllabus_parser import SyllabusParser
        parser = SyllabusParser()
        nodes = parser.parse(sample_md, course_name="OS")
        flat = parser.flatten(nodes)
        titles = [n.title for n in flat]
        assert titles[0] == "进程管理"
        assert titles[1] == "进程调度"
        assert titles[2] == "先来先服务 FCFS"
        assert titles[3] == "短作业优先 SJF"
        assert titles[4] == "进程同步"

    def test_empty_input(self):
        from coursepilot.knowledge.syllabus_parser import SyllabusParser
        parser = SyllabusParser()
        assert parser.parse("", course_name="OS") == []

    def test_sort_order(self, sample_md):
        from coursepilot.knowledge.syllabus_parser import SyllabusParser
        parser = SyllabusParser()
        nodes = parser.parse(sample_md, course_name="OS")
        sib = nodes[0].children[0].children
        assert sib[0].sort_order == 1
        assert sib[1].sort_order == 2

    def test_kp_path_unique(self, sample_md):
        from coursepilot.knowledge.syllabus_parser import SyllabusParser
        parser = SyllabusParser()
        nodes = parser.parse(sample_md, course_name="OS")
        paths = [n.kp_path for n in parser.flatten(nodes)]
        assert len(paths) == len(set(paths))

    def test_two_root_nodes(self):
        from coursepilot.knowledge.syllabus_parser import SyllabusParser
        parser = SyllabusParser()
        nodes = parser.parse("# 第一章\n## 第一节\n# 第二章\n## 第一节", course_name="BK")
        assert len(nodes) == 2
        assert nodes[0].title == "第一章"
        assert nodes[1].title == "第二章"
        flat = parser.flatten(nodes)
        paths = [n.kp_path for n in flat]
        assert paths[0] == "BK/第一章"
        assert paths[2] == "BK/第二章"

    # ── 中文编号 ──

    def test_chinese_chapter(self):
        from coursepilot.knowledge.syllabus_parser import SyllabusParser
        parser = SyllabusParser()
        nodes = parser.parse("第一章 函数与极限\n", course_name="MATH")
        assert len(nodes) == 1
        assert nodes[0].title in ("函数与极限", "第一章 函数与极限")
        assert nodes[0].level in (1, 2)  # 实际级取决于解析逻辑

    def test_chinese_numbered_list(self):
        from coursepilot.knowledge.syllabus_parser import SyllabusParser
        parser = SyllabusParser()
        nodes = parser.parse("一、映射\n二、函数\n", course_name="MATH")
        assert len(nodes) >= 1

    def test_dot_numbered(self):
        from coursepilot.knowledge.syllabus_parser import SyllabusParser
        parser = SyllabusParser()
        nodes = parser.parse("1. 定义\n2. 性质\n", course_name="MATH")
        assert len(nodes) >= 1

    def test_deep_nesting(self):
        from coursepilot.knowledge.syllabus_parser import SyllabusParser
        parser = SyllabusParser()
        md = "# L1\n## L2\n### L3\n#### L4\n"
        nodes = parser.parse(md, course_name="DEEP")
        flat = parser.flatten(nodes)
        assert len(flat) == 4
        assert flat[-1].kp_path.count("/") == 4  # DEEP/L1/L2/L3/L4

    def test_special_characters(self):
        from coursepilot.knowledge.syllabus_parser import SyllabusParser
        parser = SyllabusParser()
        nodes = parser.parse("# C++ 模板<T>\n## std::vector<T>\n", course_name="CPP")
        flat = parser.flatten(nodes)
        assert flat[0].title == "C++ 模板<T>"
        assert "C++" in flat[0].kp_path


# ═══════════════════════════════════════════════════════════
# 3. KPSplitter — kp_splitter.py
# ═══════════════════════════════════════════════════════════

class TestKPSplitter:
    """测试文本块 → KP 分配"""

    def test_exact_heading_match(self, kp_nodes, parsed_blocks):
        from coursepilot.knowledge.kp_splitter import KPSplitter
        splitter = KPSplitter(kp_nodes, "course-1")
        assigned = splitter.assign(parsed_blocks)
        assert assigned[0]["kp_id"] == "uuid-1002"  # "进程调度"

    def test_content_keyword_match(self, kp_nodes):
        """正文内容关键词匹配 —— KP title 是正文的子串（如 'FCFS' 单独出现在 kp_nodes 里）"""
        from coursepilot.knowledge.kp_splitter import KPSplitter
        # 新增短 title KP（"FCFS" ≥ 3 字符且在正文中出现）
        kps = kp_nodes + [
            {"id": "uuid-fcfs", "title": "FCFS", "kp_path": "OS/FCFS", "level": 4},
        ]
        splitter = KPSplitter(kps, "course-1")
        blocks = [
            {"content": "FCFS 算法按照作业到达的先后顺序进行调度，非抢占式。",
             "meta_data": {"text_level": 99}, "page_ref": "p10", "seq_order": 1},
        ]
        assigned = splitter.assign(blocks)
        # _match_by_content 找到 "FCFS" 在正文中出现
        assert assigned[0]["kp_id"] == "uuid-fcfs"

    def test_heading_context_overrides_content(self, kp_nodes, parsed_blocks):
        """有标题上下文时，正文优先继承当前章节（而非内容关键词）"""
        from coursepilot.knowledge.kp_splitter import KPSplitter
        splitter = KPSplitter(kp_nodes, "course-1")
        assigned = splitter.assign(parsed_blocks)
        assert assigned[1]["kp_id"] == "uuid-1002"

    def test_content_match_with_full_kp_title(self, kp_nodes):
        """正文包含完整 KP title（'先来先服务 FCFS' 在正文中）→ 匹配成功"""
        from coursepilot.knowledge.kp_splitter import KPSplitter
        splitter = KPSplitter(kp_nodes, "course-1")
        blocks = [
            {"content": "先来先服务 FCFS 是基本调度算法之一。",
             "meta_data": {"text_level": 99}, "page_ref": "p10"},
        ]
        assigned = splitter.assign(blocks)
        assert assigned[0]["kp_id"] == "uuid-1003"

    def test_current_heading_context(self, kp_nodes):
        from coursepilot.knowledge.kp_splitter import KPSplitter
        splitter = KPSplitter(kp_nodes, "course-1")
        blocks = [
            {"content": "进程同步", "meta_data": {"text_level": 2}, "page_ref": "p50"},
            {"content": "信号量是一种同步机制。", "meta_data": {"text_level": 99}, "page_ref": "p50"},
        ]
        assigned = splitter.assign(blocks)
        assert assigned[1]["kp_id"] == "uuid-1005"  # 继承标题上下文

    def test_fallback_to_root(self, kp_nodes):
        from coursepilot.knowledge.kp_splitter import KPSplitter
        splitter = KPSplitter(kp_nodes, "course-1")
        blocks = [
            {"content": "完全无法匹配的内容 xyz", "meta_data": {"text_level": 99}, "page_ref": "p99"},
        ]
        assigned = splitter.assign(blocks)
        assert assigned[0]["kp_id"] == "uuid-1001"  # 根 KP 兜底

    def test_no_kp_nodes(self):
        from coursepilot.knowledge.kp_splitter import KPSplitter
        splitter = KPSplitter([], "course-1")
        blocks = [{"content": "任意", "meta_data": {"text_level": 99}, "page_ref": "p1"}]
        assigned = splitter.assign(blocks)
        assert assigned[0].get("kp_id") in (None, "")

    def test_empty_blocks(self, kp_nodes):
        from coursepilot.knowledge.kp_splitter import KPSplitter
        splitter = KPSplitter(kp_nodes, "course-1")
        assert splitter.assign([]) == []

    def test_deep_kp_priority(self):
        kps = [
            {"id": "uuid-shallow", "title": "调度", "kp_path": "OS/调度", "level": 2},
            {"id": "uuid-deep",   "title": "调度", "kp_path": "OS/进程/调度", "level": 4},
        ]
        from coursepilot.knowledge.kp_splitter import KPSplitter
        splitter = KPSplitter(kps, "course-1")
        assigned = splitter.assign([
            {"content": "调度", "meta_data": {"text_level": 2}, "page_ref": "p1"},
        ])
        # level 4 比 level 2 更深，_title_map 按 level 降序
        assert assigned[0]["kp_id"] == "uuid-deep"

    def test_heading_resets_context(self, kp_nodes):
        """遇到新标题时，current_heading 应更新"""
        from coursepilot.knowledge.kp_splitter import KPSplitter
        splitter = KPSplitter(kp_nodes, "course-1")
        blocks = [
            {"content": "进程调度", "meta_data": {"text_level": 2}, "page_ref": "p10"},
            {"content": "正文 A",   "meta_data": {"text_level": 99}, "page_ref": "p10"},
            {"content": "进程同步", "meta_data": {"text_level": 2}, "page_ref": "p20"},
            {"content": "正文 B",   "meta_data": {"text_level": 99}, "page_ref": "p20"},
        ]
        assigned = splitter.assign(blocks)
        assert assigned[1]["kp_id"] == "uuid-1002"  # 正文 A 在 "进程调度" 下
        assert assigned[3]["kp_id"] == "uuid-1005"  # 正文 B 在 "进程同步" 下

    def test_cleaned_heading_match(self, kp_nodes):
        """去编号后能匹配： '三、进程同步' → '进程同步'"""
        from coursepilot.knowledge.kp_splitter import KPSplitter
        splitter = KPSplitter(kp_nodes, "course-1")
        blocks = [
            {"content": "三、进程同步", "meta_data": {"text_level": 2}, "page_ref": "p1"},
        ]
        assigned = splitter.assign(blocks)
        assert assigned[0]["kp_id"] == "uuid-1005"

    def test_partial_title_in_heading(self):
        """标题包含 KP title：KP '信号量' 匹配标题 '信号量机制'"""
        kps = [
            {"id": "uuid-root", "title": "root", "kp_path": "X/root", "level": 1},
            {"id": "uuid-sema", "title": "信号量", "kp_path": "X/信号量", "level": 3},
        ]
        from coursepilot.knowledge.kp_splitter import KPSplitter
        splitter = KPSplitter(kps, "course-1")
        assigned = splitter.assign([
            {"content": "信号量机制", "meta_data": {"text_level": 3}, "page_ref": "p1"},
        ])
        # "信号量" in "信号量机制"
        assert assigned[0]["kp_id"] == "uuid-sema"


# ═══════════════════════════════════════════════════════════
# 4. headings_to_syllabus — scripts/seed_knowledge.py
# ═══════════════════════════════════════════════════════════

class TestHeadingsToSyllabus:
    """测试教材标题列表 → 知识点节点（seed_knowledge 核心逻辑）"""

    def test_basic(self):
        from coursepilot.knowledge.syllabus_parser import headings_to_syllabus
        headings = [
            {"title": "进程管理", "level": 1, "page_idx": 0},
            {"title": "进程调度", "level": 2, "page_idx": 1},
            {"title": "FCFS",    "level": 3, "page_idx": 2},
            {"title": "SJF",     "level": 3, "page_idx": 3},
        ]
        nodes = headings_to_syllabus(headings, "OS")
        assert len(nodes) == 4
        assert nodes[0]["kp_path"] == "OS/进程管理"
        assert nodes[1]["kp_path"] == "OS/进程管理/进程调度"
        assert nodes[2]["parent_title"] == "进程调度"
        assert nodes[3]["parent_title"] == "进程调度"

    def test_two_sections(self):
        from coursepilot.knowledge.syllabus_parser import headings_to_syllabus
        headings = [
            {"title": "数学",     "level": 1, "page_idx": 0},
            {"title": "代数",     "level": 2, "page_idx": 1},
            {"title": "几何",     "level": 2, "page_idx": 2},
            {"title": "解析几何", "level": 3, "page_idx": 3},
        ]
        nodes = headings_to_syllabus(headings, "MATH")
        paths = [n["kp_path"] for n in nodes]
        assert paths == [
            "MATH/数学",
            "MATH/数学/代数",
            "MATH/数学/几何",
            "MATH/数学/几何/解析几何",
        ]

    def test_single_heading(self):
        from coursepilot.knowledge.syllabus_parser import headings_to_syllabus
        nodes = headings_to_syllabus([{"title": "总论", "level": 1, "page_idx": 0}], "BK")
        assert len(nodes) == 1
        assert nodes[0]["kp_path"] == "BK/总论"
        assert nodes[0]["parent_title"] is None

    def test_empty(self):
        from coursepilot.knowledge.syllabus_parser import headings_to_syllabus
        assert headings_to_syllabus([], "X") == []

    def test_level_jump(self):
        """从 level 1 直接跳到 level 3（跳过 level 2）"""
        from coursepilot.knowledge.syllabus_parser import headings_to_syllabus
        headings = [
            {"title": "第一章", "level": 1, "page_idx": 0},
            {"title": "小节", "level": 3, "page_idx": 1},
        ]
        nodes = headings_to_syllabus(headings, "BK")
        assert len(nodes) == 2
        assert nodes[1]["kp_path"] == "BK/第一章/小节"

    def test_sort_order(self):
        from coursepilot.knowledge.syllabus_parser import headings_to_syllabus
        headings = [
            {"title": "第一章", "level": 1, "page_idx": 0},
            {"title": "代数", "level": 2, "page_idx": 1},
            {"title": "几何", "level": 2, "page_idx": 2},
        ]
        nodes = headings_to_syllabus(headings, "X")
        assert nodes[0]["sort_order"] == 1
        assert nodes[1]["sort_order"] == 1  # level 2 的第一个
        assert nodes[2]["sort_order"] == 2  # level 2 的第二个

    def test_dot_numbered_not_bumped_by_simple_number_context(self):
        """dot-numbered 标题（"4.3"）不应被上下文感知 bump 影响，
        即使栈顶是更深层级的非编号标题（"高级"）。
        场景：阶段四/4.2/Crew高级/4.3 → 4.3 应与 4.2 同级。"""
        from coursepilot.knowledge.syllabus_parser import headings_to_syllabus
        headings = [
            {"title": "阶段四 · Crew 团队编排", "level": 1, "page_idx": 0},
            {"title": "4.2 Crew 全部属性",      "level": 2, "page_idx": 1},
            {"title": "高级",                    "level": 3, "page_idx": 2},
            {"title": "4.3 执行与输出",         "level": 2, "page_idx": 3},
        ]
        nodes = headings_to_syllabus(headings, "AI")
        paths = [n["kp_path"] for n in nodes]
        assert paths == [
            "AI/阶段四 · Crew 团队编排",
            "AI/阶段四 · Crew 团队编排/4.2 Crew 全部属性",
            "AI/阶段四 · Crew 团队编排/4.2 Crew 全部属性/高级",
            "AI/阶段四 · Crew 团队编排/4.3 执行与输出",  # 与 4.2 同级，不在高级下面
        ]


# ═══════════════════════════════════════════════════════════
# 5. FileStore — storage/file_store.py
# ═══════════════════════════════════════════════════════════

class TestFileStore:
    """测试本地文件存储"""

    def test_save_creates_file(self, tmp_path):
        from coursepilot.storage.file_store import FileStore
        store = FileStore(base_dir=str(tmp_path))
        info = store.save(b"Hello!", "course-1", "lecture.pdf")
        assert Path(info["file_path"]).exists()
        assert Path(info["file_path"]).read_bytes() == b"Hello!"
        assert info["file_size"] == 6

    def test_save_uuid_filename(self, tmp_path):
        from coursepilot.storage.file_store import FileStore
        store = FileStore(base_dir=str(tmp_path))
        info = store.save(b"x", "course-1", "lecture.pdf")
        uuid_hex = info["stored_name"][:-4]
        assert len(uuid_hex) == 32
        assert info["stored_name"].endswith(".pdf")

    def test_save_no_conflict(self, tmp_path):
        from coursepilot.storage.file_store import FileStore
        store = FileStore(base_dir=str(tmp_path))
        a = store.save(b"a", "course-1", "same.pdf")
        b = store.save(b"b", "course-1", "same.pdf")
        assert a["stored_name"] != b["stored_name"]
        assert Path(a["file_path"]).read_bytes() == b"a"
        assert Path(b["file_path"]).read_bytes() == b"b"

    def test_delete_existing(self, tmp_path):
        from coursepilot.storage.file_store import FileStore
        store = FileStore(base_dir=str(tmp_path))
        info = store.save(b"x", "course-1", "f.pdf")
        assert store.delete(info["file_path"]) is True
        assert not Path(info["file_path"]).exists()

    def test_delete_nonexistent(self, tmp_path):
        from coursepilot.storage.file_store import FileStore
        store = FileStore(base_dir=str(tmp_path))
        assert store.delete(str(tmp_path / "ghost.pdf")) is False

    def test_delete_course_files(self, tmp_path):
        from coursepilot.storage.file_store import FileStore
        store = FileStore(base_dir=str(tmp_path))
        store.save(b"a", "course-1", "a.pdf")
        store.save(b"b", "course-1", "b.pdf")
        store.save(b"c", "course-2", "c.pdf")
        count = store.delete_course_files("course-1")
        assert count == 2
        assert not (tmp_path / "course-1").exists()
        assert (tmp_path / "course-2").exists()

    def test_get_path(self, tmp_path):
        from coursepilot.storage.file_store import FileStore
        store = FileStore(base_dir=str(tmp_path))
        assert store.get_path("course-1", "abc.pdf") == tmp_path / "course-1" / "abc.pdf"

    def test_save_creates_nested_dir(self, tmp_path):
        from coursepilot.storage.file_store import FileStore
        base = tmp_path / "deep" / "nested"
        store = FileStore(base_dir=str(base))
        info = store.save(b"x", "course-1", "f.pdf")
        assert Path(info["file_path"]).exists()

    def test_file_size_accurate(self, tmp_path):
        from coursepilot.storage.file_store import FileStore
        store = FileStore(base_dir=str(tmp_path))
        data = b"x" * 1024
        info = store.save(data, "course-1", "big.pdf")
        assert info["file_size"] == 1024


# ═══════════════════════════════════════════════════════════
# 6. 端到端集成测试
# ═══════════════════════════════════════════════════════════

class TestEndToEndMarkdown:
    """markdown_parser → parser_utils → kp_splitter（纯内存）"""

    def test_full_md_pipeline(self, tmp_path, kp_nodes):
        md = """# 进程管理
## 进程调度
先来先服务 (FCFS) 算法按照作业到达的先后顺序进行调度。

短作业优先 (SJF) 算法优先调度预计运行时间最短的作业。
"""
        md_file = tmp_path / "os.md"
        md_file.write_text(md, encoding="utf-8")

        # B2: 解析
        from coursepilot.ingestion.markdown_parser import parse_markdown
        result = parse_markdown(str(md_file))
        assert len(result["content_list"]) >= 4

        # B3: 切分
        from coursepilot.ingestion.parser_utils import extract_knowledge_units
        units = extract_knowledge_units(result["content_list"], document_id="doc-1", kp_id="")
        assert len(units) >= 1

        # B5: KP 分配
        from coursepilot.knowledge.kp_splitter import KPSplitter
        splitter = KPSplitter(kp_nodes, "course-1")
        assigned = splitter.assign(units)

        fcfs = next((b for b in assigned if "FCFS" in b["content"]), None)
        assert fcfs is not None
        # meta_data.heading="进程调度" 正确匹配 → uuid-1002
        assert fcfs["kp_id"] == "uuid-1002"

        sjf = next((b for b in assigned if "SJF" in b["content"]), None)
        assert sjf is not None
        assert sjf["kp_id"] == "uuid-1002"

    def test_kp_path_consistency(self, tmp_path, kp_nodes):
        """每个已分配的块都应有非空 kp_path"""
        md = "# 进程管理\n## 进程调度\nFCFS 正文\n"
        md_file = tmp_path / "test.md"
        md_file.write_text(md, encoding="utf-8")

        from coursepilot.ingestion.markdown_parser import parse_markdown
        from coursepilot.ingestion.parser_utils import extract_knowledge_units
        from coursepilot.knowledge.kp_splitter import KPSplitter

        result = parse_markdown(str(md_file))
        units = extract_knowledge_units(result["content_list"], document_id="d1", kp_id="")
        splitter = KPSplitter(kp_nodes, "course-1")
        assigned = splitter.assign(units)

        for block in assigned:
            if block.get("kp_id"):
                assert block.get("kp_path"), f"kp_id 存在但 kp_path 为空: {block}"

    def test_empty_md_pipeline(self, tmp_path, kp_nodes):
        """空文件不应崩溃"""
        md_file = tmp_path / "empty.md"
        md_file.write_text("", encoding="utf-8")
        from coursepilot.ingestion.markdown_parser import parse_markdown
        from coursepilot.ingestion.parser_utils import extract_knowledge_units
        result = parse_markdown(str(md_file))
        units = extract_knowledge_units(result["content_list"], document_id="d1", kp_id="")
        assert units == []


# ═══════════════════════════════════════════════════════════
# 7. 边界情况与错误处理
# ═══════════════════════════════════════════════════════════

class TestEdgeCases:
    """异常输入、特殊字符、极端长度"""

    def test_md_code_blocks(self, tmp_path):
        """代码块中的 # 会被误解析（已知限制，但不影响主流程）"""
        md_file = tmp_path / "test.md"
        md_file.write_text("# 章节\n```python\n# 注释\nprint('hi')\n```\n正文\n", encoding="utf-8")
        from coursepilot.ingestion.markdown_parser import parse_markdown
        result = parse_markdown(str(md_file))
        assert result["content_list"][0]["text"] == "章节"

    def test_unicode_mixed(self, kp_nodes):
        from coursepilot.knowledge.kp_splitter import KPSplitter
        kps = kp_nodes + [
            {"id": "uuid-bel", "title": "Belady异常", "kp_path": "OS/内存/Belady", "level": 4},
        ]
        splitter = KPSplitter(kps, "course-1")
        assigned = splitter.assign([
            {"content": "Belady异常是指...", "meta_data": {"text_level": 99}, "page_ref": "p1"},
        ])
        assert assigned[0]["kp_id"] == "uuid-bel"

    def test_very_long_content(self, kp_nodes):
        from coursepilot.knowledge.kp_splitter import KPSplitter
        splitter = KPSplitter(kp_nodes, "course-1")
        long_text = "无关内容 " * 1000
        assigned = splitter.assign([
            {"content": long_text, "meta_data": {"text_level": 99}, "page_ref": "p1"},
        ])
        # 应兜底到根 KP，不崩溃
        assert assigned[0]["kp_id"] == "uuid-1001"

    def test_kp_same_title_different_parents(self):
        """同名 KP 在不同父节点下，应通过 kp_path 区分"""
        kps = [
            {"id": "id-1", "title": "调度", "kp_path": "OS/进程管理/调度", "level": 3},
            {"id": "id-2", "title": "调度", "kp_path": "OS/内存管理/调度", "level": 3},
            {"id": "id-root", "title": "root", "kp_path": "OS", "level": 1},
        ]
        from coursepilot.knowledge.kp_splitter import KPSplitter
        splitter = KPSplitter(kps, "course-1")
        assigned = splitter.assign([
            {"content": "调度", "meta_data": {"text_level": 3}, "page_ref": "p1"},
        ])
        # 两个 "调度" 都 level=3，排序稳定时应返回第一个
        assert assigned[0]["kp_id"] in ("id-1", "id-2")

    def test_no_text_level_field(self, kp_nodes):
        """meta_data 缺少 text_level 时不应崩溃"""
        from coursepilot.knowledge.kp_splitter import KPSplitter
        splitter = KPSplitter(kp_nodes, "course-1")
        assigned = splitter.assign([
            {"content": "正文", "meta_data": {}, "page_ref": "p1"},
        ])
        assert len(assigned) == 1

    def test_root_heading_unmatched(self):
        """标题在 KP 列表中完全不存在时，匹配不到但正文会兜底"""
        from coursepilot.knowledge.kp_splitter import KPSplitter
        kps = [
            {"id": "r", "title": "唯一KP", "kp_path": "X/唯一KP", "level": 1},
        ]
        splitter = KPSplitter(kps, "course-1")
        assigned = splitter.assign([
            {"content": "不存在的标题", "meta_data": {"text_level": 2}, "page_ref": "p1"},
            {"content": "正文内容",     "meta_data": {"text_level": 99}, "page_ref": "p1"},
        ])
        # 标题没匹配到，但正文会兜底
        assert assigned[1]["kp_id"] == "r"


# ═══════════════════════════════════════════════════════════
# 8. KPTree — kp_tree.py（需要数据库，标记为 integration）
# ═══════════════════════════════════════════════════════════

@pytest.mark.integration
class TestKPTree:
    """测试知识点树 CRUD + 递归 CTE（需要真实数据库连接）"""

    @pytest.mark.asyncio
    async def test_create_and_count(self):
        pytest.skip("需要数据库 — 手动运行时取消 skip")
        # from coursepilot.db import get_session_etx
        # from coursepilot.knowledge.kp_tree import KPTree
        # async with get_session_etx() as session:
        #     tree = KPTree(session)
        #     count = await tree.count_by_course("course-uuid")
        #     assert count >= 0

    @pytest.mark.asyncio
    async def test_get_path(self):
        pytest.skip("需要数据库 — 手动运行时取消 skip")
