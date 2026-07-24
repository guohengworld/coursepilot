"""对话压缩器：LLM 驱动的滚动摘要。

职责：
1. compact_conversation：当 L1 滑动窗口溢出时，使用 LLM 增量生成滚动摘要
2. micro_compact_turn：（兼容保留）单轮格式化辅助函数

改用 LLM 压缩替代原有规则压缩，提供更高质量的上下文记忆。
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from coursepilot.config import settings
from coursepilot.agent.memory.context_manager import estimate_tokens

logger = logging.getLogger(__name__)


def micro_compact_turn(
    query: str,
    answer: str,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """（兼容保留）格式化单轮问答为压缩输入，返回结构化摘要。

    Args:
        query: 用户问题
        answer: 助教回答
        sources: 引用来源（含 kp_path）

    Returns:
        {
            "query": 截断后的问题,
            "answer_summary": 答案截断,
            "key_formulas": 关键公式,
            "kp_paths": 知识点路径列表,
            "intent": 意图（始终为 'chat' 标记）,
        }
    """
    import re

    kp_paths = []
    if sources:
        for src in sources:
            if isinstance(src, dict) and src.get("kp_path"):
                kp_paths.append(src["kp_path"])
            if isinstance(src, str):
                kp_paths.append(src)

    # 提取公式
    formulas = re.findall(r"\$\$.*?\$\$|\$.*?\$", answer, re.DOTALL)
    key_formulas = " ".join(formulas[:3]) if formulas else ""

    return {
        "query": query[:200],
        "answer_summary": answer[:400],
        "key_formulas": key_formulas,
        "kp_paths": list(dict.fromkeys(kp_paths)),
        "intent": "chat",
    }


def _fmt_turns_for_llm(turns: list[dict[str, Any]], existing_summary: str) -> str:
    """把待压缩的轮次格式化为 LLM 可读的文本。"""
    parts = []
    if existing_summary.strip():
        parts.append(f"【已有摘要】\n{existing_summary.strip()}\n")
    parts.append("【待压缩对话】")
    for i, turn in enumerate(turns, 1):
        role = turn.get("role", "user")
        content = turn.get("content", "")
        intent = turn.get("intent", "")
        query = turn.get("query", "")
        if role == "user":
            parts.append(f"第{i}轮 用户: {content[:300]}")
        else:
            parts.append(
                f"第{i}轮 助教(intent={intent}, query={query[:80]}): {content[:600]}"
            )
    return "\n".join(parts)


_COMPACTION_SYSTEM = """你是一个课程辅导对话压缩专家。你的任务是把一段多轮课程辅导对话压缩成结构化的滚动摘要。

## 输入

你会收到两段内容：
1. 【已有摘要】：之前已经压缩好的历史摘要
2. 【待压缩对话】：本轮需要压缩的新对话轮次

## 输出要求

输出纯文本，格式如下：
```
[用户问题摘要] → [答案核心结论]
关键公式: (如有)
知识点路径: (如有)
---
[用户问题摘要] → [答案核心结论]
...
```

## 压缩原则

1. **保留关键信息**：每轮问答的核心结论、关键公式/定理、知识点路径
2. **去除冗余**：客套话、问候语、重复解释可省略
3. **保持连贯**：摘要读起来应该连贯，能支撑后续追问（如"刚才那道题第二步怎么算的"）
4. **公式保留**：关键数学公式用 LaTeX 保留（$...$ 或 $$...$$）
5. **知识点路径**：如果涉及知识点，在末尾标注 (kp: 路径)
6. **已有摘要保留**：如果有已有摘要，先写已有摘要，再写新压缩的内容
7. **总输出不超过 {max_tokens} token**，如果超长，优先保留最近的轮次
8. **不要添加无关内容**：只压缩输入中的内容，不要自由发挥

输出纯文本，不要用markdown代码块包裹。"""


async def _llm_compact(
    turns: list[dict[str, Any]],
    existing_summary: str = "",
    max_summary_tokens: int = 1_500,
) -> str:
    """调用 LLM 压缩对话为滚动摘要。"""
    if not settings.llm_api_key:
        logger.warning("LLM API key 未配置，回退到简单截断摘要")
        return _fallback_summary(turns, existing_summary)

    prompt_payload = _fmt_turns_for_llm(turns, existing_summary)
    system_prompt = _COMPACTION_SYSTEM.format(max_tokens=max_summary_tokens)

    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_payload},
            ],
            temperature=0.3,
            max_tokens=max_summary_tokens,
        )
        content = response.choices[0].message.content or ""
        logger.info(
            "LLM 压缩完成：输入 %d 轮，输出 %d 字符，prompt_tokens=%d",
            len(turns),
            len(content),
            response.usage.prompt_tokens if response.usage else 0,
        )
        return content.strip()
    except Exception:
        logger.exception("LLM 压缩失败，回退到简单截断摘要")
        return _fallback_summary(turns, existing_summary)


def _fallback_summary(turns: list[dict[str, Any]], existing_summary: str) -> str:
    """LLM 压缩失败时的回退方案：简单截断拼接。"""
    parts = [existing_summary.strip()] if existing_summary.strip() else []
    for turn in turns:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        intent = turn.get("intent", "")
        if role == "assistant" and content:
            q = turn.get("query", "")
            parts.append(
                f"[{intent or 'chat'}] {q[:80]} -> {content[:200]}"
            )
        elif content:
            parts.append(f"[user] {content[:120]}")
    return "\n".join(parts)


async def compact_conversation(
    conversation: list[dict[str, Any]],
    existing_summary: str = "",
    max_summary_tokens: int = 1_500,
) -> tuple[str, int]:
    """增量生成滚动摘要：把老轮次交给 LLM 压缩，保留最近轮次作为 L1。

    Args:
        conversation: 完整对话轮次列表（将被分为老轮次和保留轮次）
        existing_summary: 已有的滚动摘要
        max_summary_tokens: 摘要最大 token 预算

    Returns:
        (new_summary, compacted_count)
            new_summary: 压缩后的新摘要
            compacted_count: 被压缩的老轮次数
    """
    if not conversation:
        return existing_summary, 0

    # 前 50% 作为老轮次压缩进摘要，后 50% 保留为 L1
    cutoff = len(conversation) // 2
    if cutoff == 0:
        return existing_summary, 0

    older_turns = conversation[:cutoff]

    # 调用 LLM 压缩
    new_summary = await _llm_compact(
        older_turns,
        existing_summary=existing_summary,
        max_summary_tokens=max_summary_tokens,
    )

    # 确保不超预算
    if estimate_tokens(new_summary) > max_summary_tokens:
        lines = new_summary.splitlines()
        while estimate_tokens("\n".join(lines)) > max_summary_tokens and len(lines) > 3:
            lines.pop(0)
        new_summary = "\n".join(lines)

    logger.info(
        "对话压缩完成：%d 轮 -> summary (%d tokens)", cutoff, estimate_tokens(new_summary)
    )
    return new_summary, cutoff


async def compact_with_llm(
    conversation: list[dict[str, Any]],
    existing_summary: str = "",
    max_summary_tokens: int = 1_500,
) -> str:
    """LLM 驱动的对话压缩入口（直接调用 LLM 生成摘要）。

    这是 compact_conversation 的简化版本，只返回摘要不返回压缩轮数。
    """
    summary, _ = await compact_conversation(
        conversation, existing_summary, max_summary_tokens
    )
    return summary
