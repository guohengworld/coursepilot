"""kp_path 验证脚本：解析 PDF 前10页（目录）+ 117-127页 → 检查标题层级是否与 kp_path 一致

全程不写数据库，测试完清理 MinerU 输出。
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path

# Windows GBK 终端强制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coursepilot.ingestion.pdf_parser import parse_pdf
from coursepilot.knowledge.syllabus_parser import extract_headings, headings_to_syllabus


async def parse_page_range(pdf_path: str, start: int, end: int, output_dir: str) -> tuple[list[dict], list[dict], list[dict]]:
    """解析指定页码范围，返回 (content_list, headings, kp_nodes)"""
    print(f"\n{'='*60}")
    print(f"解析: {Path(pdf_path).name}  页码范围: {start}-{end}")
    print(f"{'='*60}")

    result = await parse_pdf(pdf_path, start_page=start, end_page=end, output_dir=output_dir)
    content_list = result.get("content_list", [])
    print(f"  content_list: {len(content_list)} 行")

    headings = extract_headings(content_list)
    print(f"  标题行 (text_level<=4): {len(headings)} 个")

    for h in headings:
        print(f"    L{h['level']}: {h['title'][:80]}")

    nodes = headings_to_syllabus(headings, "微积分")
    print(f"\n  KP 树 ({len(nodes)} 个节点):")
    for n in nodes:
        indent = "  " * (n["level"] - 1)
        print(f"    {indent}{n['kp_path']}")

    return content_list, headings, nodes


async def main():
    pdf_path = "tests/fixtures/pdfs/大学数学 微积分 下册.pdf"
    pdf_abs = Path(pdf_path).resolve()

    if not pdf_abs.exists():
        print(f"文件不存在: {pdf_abs}")
        return

    # ━━━━ 1. 解析目录页（前 10 页，0-indexed: 0-9）━━━━
    toc_dir = "tests/output/kp_verify_toc"
    toc_cl, toc_headings, toc_nodes = await parse_page_range(str(pdf_abs), 0, 9, toc_dir)

    # ━━━━ 2. 解析 117-127 页 (0-indexed: 116-126) ━━━━
    body_dir = "tests/output/kp_verify_body"
    body_cl, body_headings, body_nodes = await parse_page_range(str(pdf_abs), 116, 126, body_dir)

    # ━━━━ 3. 对比分析 ━━━━
    print(f"\n{'='*60}")
    print("对比分析")
    print(f"{'='*60}")

    print("\n--- 目录 (前10页) KP 树结构 ---")
    for n in toc_nodes:
        indent = "  " * (n["level"] - 1)
        print(f"  {indent}[L{n['level']}] {n['title']}")

    print("\n--- 正文 (117-127页) KP 树结构 ---")
    for n in body_nodes:
        indent = "  " * (n["level"] - 1)
        print(f"  {indent}[L{n['level']}] {n['title']}")

    print("\n--- 层级验证 ---")
    errors = []
    for n in body_nodes:
        path_segments = n["kp_path"].split("/")
        parent_title = n.get("parent_title")
        title = n["title"]

        # 验证1：有 parent_title 的节点，kp_path 必须包含 parent_title
        if parent_title:
            if parent_title not in n["kp_path"]:
                errors.append(f"parent缺失: title='{title}' parent='{parent_title}' path='{n['kp_path']}'")

        # 验证2：title 本身包含编号 ("9.1.1")，则 kp_path 中 title 前的父级应匹配编号层级
        # 例如 "9.1.1" (3段编号) 应挂在 "9.1" (2段编号) 下
        import re as _re
        m = _re.match(r"^(\d+(?:\.\d+)+)", title)
        if m:
            num_parts = m.group(1).split(".")
            # 该节点的直接父标题应包含前 n-1 段编号
            if len(num_parts) >= 2 and parent_title:
                parent_prefix = ".".join(num_parts[:-1])
                if parent_prefix not in parent_title:
                    errors.append(f"编号嵌套错: title='{title}' parent='{parent_title}' 期望父级含 '{parent_prefix}'")

        status = "[OK]" if not (parent_title and parent_title not in n["kp_path"]) else "[ERR]"
        print(f"  {status} {n['kp_path']}")

    if errors:
        print(f"\n{len(errors)} 个层级错误:")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print(f"\n所有 kp_path 层级正确")

    # ━━━━ 4. 清理 MinerU 输出 ━━━━
    print(f"\n{'='*60}")
    print("清理测试输出...")
    for d in [toc_dir, body_dir]:
        if Path(d).exists():
            shutil.rmtree(d)
            print(f"  已删除: {d}")
    print("完成")


if __name__ == "__main__":
    asyncio.run(main())
