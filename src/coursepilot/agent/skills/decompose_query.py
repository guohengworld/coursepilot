"""复杂查询分解 Skill：将复杂问题拆解为多个独立可检索的子问题。

规则优先（对比类问题直接拆概念），否则走 LLM 分解。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from coursepilot.config import settings

logger = logging.getLogger(__name__)

# 对比类匹配：如 "极限和连续的区别"、"Fermat定理与Lagrange中值定理的比较"
_COMPARISON_PAT = re.compile(
    r"(.{2,30}?)(?:与|和|跟|同)(.{2,30}?)(?:(?:有什么)?(?:区别|联系|不同|相同|差异|比较|对比))",
    re.UNICODE,
)

DECOMPOSE_SYSTEM = """你是一个教学问题分解专家。将复杂的高等数学问题拆解为多个独立的子问题。
每个子问题应能通过单次教材检索找到答案，且不重叠。

## 何时需要分解
- 问题涉及多个概念比较（"A和B有什么区别"）
- 问题包含多步推理（"先...然后..."）
- 问题跨多个章节或知识点
- 问题包含假设条件和结论两部分

## 无需分解
- 问题只问单一概念、定义、定理
- 问题只问一个公式或方法
- 返回 empty sub_queries 列表

## 输出格式
必须严格输出 JSON，不要包含其他内容：
{
  "decomposition_type": "compare" 或 "sequential" 或 "single",
  "sub_queries": [
    {"id": 1, "query": "第一个子问题", "target_concept": "相关知识点", "reason": "为什么拆这个"},
    {"id": 2, "query": "第二个子问题", "target_concept": "相关知识点", "reason": "为什么拆这个"}
  ]
}

如果无需分解，返回 {"decomposition_type": "single", "sub_queries": []}"""


async def decompose_query(
    query: str,
    course_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将复杂查询分解为可独立检索的子问题。

    Returns:
        {"sub_queries": list[dict], "decomposition_type": str}
        sub_queries 为空列表表示无需分解。
    """
    # ── 规则优先：对比类问题 ──
    m = _COMPARISON_PAT.search(query)
    if m:
        concept_a, concept_b = m.group(1).strip(), m.group(2).strip()
        sub_queries = [
            {
                "id": 1,
                "query": concept_a,
                "target_concept": concept_a,
                "reason": f"需要先了解「{concept_a}」的定义和性质",
            },
            {
                "id": 2,
                "query": concept_b,
                "target_concept": concept_b,
                "reason": f"需要先了解「{concept_b}」的定义和性质",
            },
        ]
        logger.info("decompose_query 规则匹配: %s ↔ %s", concept_a, concept_b)
        return {"sub_queries": sub_queries, "decomposition_type": "compare"}

    # ── LLM 分解 ──
    if not settings.llm_api_key:
        return {"sub_queries": [], "decomposition_type": "single"}

    try:
        # 拼上下文：课程名和章节信息辅助分解
        ctx_hint = ""
        if course_context:
            name = course_context.get("name", "")
            chapters = course_context.get("chapters", [])
            if chapters:
                ch_names = ", ".join(c[:20] for c in chapters[:5])
                ctx_hint = f"课程：{name}，主要章节：{ch_names}"

        parts = [f"用户问题：{query}"]
        if ctx_hint:
            parts.append(ctx_hint)

        client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM},
                {"role": "user", "content": "\n".join(parts)},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        raw = response.choices[0].message.content
        if not raw:
            return {"sub_queries": [], "decomposition_type": "single"}

        result = json.loads(raw.strip())
        sub_queries = result.get("sub_queries", [])
        dtype = result.get("decomposition_type", "single")

        # 校验 sub_queries 格式
        validated = []
        for sq in sub_queries:
            if isinstance(sq, dict) and sq.get("query"):
                validated.append(sq)
        return {"sub_queries": validated, "decomposition_type": dtype}

    except (json.JSONDecodeError, Exception) as e:
        logger.warning("decompose_query LLM 异常: %s，退化到单查询", e)
        return {"sub_queries": [], "decomposition_type": "single"}
