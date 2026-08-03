"""引用验证 —— 解析 LLM 回答中的 <ref id="N" /> 标签，检测幻觉引用

用法：
    ok, hallucinated = validate_citations(answer, {1, 2, 3})
"""

from __future__ import annotations

import re

# 引用格式契约（唯一来源）：LLM prompt、guardrails 检查、API 解析共用此格式。
# 修改引用格式时必须同步修改 CITATION_SYNTAX 与 CITATION_PATTERN。
CITATION_SYNTAX = '<ref id="N" />'  # 展示/提示用，N 为 source id
CITATION_PATTERN = re.compile(r'<ref\s+id="(\d+)"\s*/>')  # 解析用


def validate_citations(
    answer: str,
    valid_source_ids: set[int],
) -> tuple[bool, set[int]]:
    """验证回答中的引用是否都来自合法的 source id。

    返回: (is_valid, hallucinated_ids)
      - is_valid: True 表示所有引用 id 都在 valid_source_ids 中
      - hallucinated_ids: 不在合法集合中的 id
    """
    cited = {int(m.group(1)) for m in CITATION_PATTERN.finditer(answer)}
    hallucinated = cited - valid_source_ids
    return len(hallucinated) == 0, hallucinated


def extract_citations(answer: str) -> list[int]:
    """提取回答中所有引用 id（按出现顺序，去重）。"""
    seen: set[int] = set()
    result: list[int] = []
    for m in CITATION_PATTERN.finditer(answer):
        ref_id = int(m.group(1))
        if ref_id not in seen:
            seen.add(ref_id)
            result.append(ref_id)
    return result


def count_citations(answer: str) -> int:
    """统计回答中的引用数量。"""
    return len(CITATION_PATTERN.findall(answer))
