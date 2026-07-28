"""上下文充足性质检 Skill：在生成答案前，判断检索到的教材内容是否足够回答问题。

质检流程：
1. 快速规则过滤（空上下文、对比类问题缺概念）
2. LLM 判断：教材内容是否覆盖了问题的所有方面
3. 返回结构化结果，供 check_sufficiency_node 路由决策
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from coursepilot.config import settings
from coursepilot.rag.config import config as rag_config

logger = logging.getLogger(__name__)

# 对比类问题匹配（与 retriever.py 保持一致）
_COMPARISON_PAT = re.compile(
    r"(.{2,30}?)(?:与|和|跟|同)(.{2,30}?)(?:(?:有什么)?(?:区别|联系|不同|相同|差异|比较|对比))",
    re.UNICODE,
)

CHECK_SYSTEM = """你是一个 RAG 质量检验员。
你的任务不是回答问题，而是判断提供的【教材内容】是否足够回答【用户问题】。

判断标准：
1. 教材内容中是否包含回答该问题所需的**关键事实、定义、定理或方法**
2. 对于比较类问题，是否包含了**所有被比较对象**的相关内容
3. 内容是否足够**具体**，而非仅涉及边缘或背景信息
4. 对于需要推导或证明的问题，教材是否提供了**足够的推导步骤或逻辑依据**

输出 JSON，不要包含其他内容：
{
  "sufficient": true 或 false,
  "confidence": 0.0~1.0,
  "missing_info": "如果不足，描述缺少什么信息；如果充足则为空字符串",
  "covered_aspects": ["已覆盖的方面1", "已覆盖的方面2"],
  "uncovered_aspects": ["未覆盖的方面1"]
}"""


async def check_sufficiency(
    query: str,
    context: str,
    kp_paths: list[str] | None = None,
) -> dict[str, Any]:
    """判断检索到的 context 是否足够回答 query。

    Returns:
        {"sufficient": bool, "confidence": float, "missing_info": str,
         "covered_aspects": list[str], "uncovered_aspects": list[str]}
    """
    # ── 快速规则过滤 ──
    if not context or not context.strip():
        return {
            "sufficient": False,
            "confidence": 0.0,
            "missing_info": "未检索到任何教材内容",
            "covered_aspects": [],
            "uncovered_aspects": ["需要检索相关教材内容"],
        }

    # 对比类问题：检查是否两个概念都有覆盖
    m = _COMPARISON_PAT.search(query)
    if m and kp_paths:
        concept_a, concept_b = m.group(1).strip(), m.group(2).strip()
        has_a = any(concept_a[:4] in kp for kp in kp_paths)
        has_b = any(concept_b[:4] in kp for kp in kp_paths)
        if not has_a or not has_b:
            missing = []
            if not has_a:
                missing.append(concept_a)
            if not has_b:
                missing.append(concept_b)
            return {
                "sufficient": False,
                "confidence": 0.3,
                "missing_info": f"缺少概念「{'」「'.join(missing)}」的相关内容",
                "covered_aspects": [],
                "uncovered_aspects": [f"缺少{m}的内容" for m in missing],
            }

    # 对比类阈值：低于阈值直接判不足
    threshold = rag_config.context_sufficiency_threshold

    # ── LLM 判断 ──
    if not settings.llm_api_key:
        logger.warning("LLM API key 未配置，默认视为充足")
        return {
            "sufficient": True,
            "confidence": 0.7,
            "missing_info": "",
            "covered_aspects": [],
            "uncovered_aspects": [],
        }

    try:
        client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url
        )
        # 如果上下文太长，截断到 3000 字符
        context_truncated = context[:3000] + ("..." if len(context) > 3000 else "")

        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": CHECK_SYSTEM},
                {"role": "user", "content": f"【用户问题】\n{query}"
                 f"\n\n【教材内容】\n{context_truncated}"},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        raw = response.choices[0].message.content
        if not raw:
            logger.warning("check_sufficiency 返回空，默认视为充足")
            return _default_sufficient()

        result = json.loads(raw.strip())
        if not isinstance(result, dict):
            return _default_sufficient()

        # 标准化输出
        sufficient = bool(result.get("sufficient", True))
        confidence = float(result.get("confidence", 0.7))
        missing_info = (result.get("missing_info") or "").strip()
        covered = result.get("covered_aspects", [])
        uncovered = result.get("uncovered_aspects", [])

        # 如果 confidence 低于阈值且 judged sufficient，降级
        if sufficient and confidence < threshold:
            logger.info("check_sufficiency: confidence=%.2f 低于阈值=%.2f，降级为不足",
                        confidence, threshold)
            sufficient = False
            if not missing_info:
                missing_info = "检索内容置信度不足，需要更多信息"

        return {
            "sufficient": sufficient,
            "confidence": confidence,
            "missing_info": missing_info if not sufficient else "",
            "covered_aspects": covered if isinstance(covered, list) else [],
            "uncovered_aspects": uncovered if isinstance(uncovered, list) else [],
        }

    except (json.JSONDecodeError, Exception) as e:
        logger.warning("check_sufficiency LLM 判断异常: %s，默认视为充足", e)
        return _default_sufficient()


def _default_sufficient() -> dict[str, Any]:
    return {
        "sufficient": True,
        "confidence": 0.6,
        "missing_info": "",
        "covered_aspects": [],
        "uncovered_aspects": [],
    }
