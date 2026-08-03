"""LLM 生成器 —— DeepSeek 调用 + System Prompt 组装

用法（新版，支持历史消息）：
    generator = Generator()
    answer, token_info = await generator.generate(
        query=query,
        context=context,
        course_context=course_context,
        conversation=conversation,
        rolling_summary=rolling_summary,
        user_profile=user_profile,
    )

上下文装配由 ContextManager 按预算完成，稳定内容（system + 课程上下文）
放在 messages 最前面，以提升 DeepSeek prompt caching 命中率。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID

import openai
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.agent.memory import ContextManager, ContextView
from coursepilot.config import settings
from coursepilot.models import Course, Document, KnowledgePoint
from coursepilot.rag.citation import CITATION_SYNTAX

logger = logging.getLogger(__name__)

# 引用格式契约来自 rag/citation.py 的 CITATION_SYNTAX（唯一来源），
# guardrails 的引用检查与之一致。
SYSTEM_PROMPT = f"""你是 CoursePilot 课程助教，为大学生解答数学问题。

## 回答规则

1. **必须基于教材**：回答必须严格依据下面 <sources> 中提供的教材内容，
   不得编造教材中没有的事实、公式或定理
2. **引用格式**：涉及教材内容时使用 {CITATION_SYNTAX} 引用，N 为 source id
3. **公式正确**：所有数学公式使用 LaTeX 语法，行内 $...$，独立行 $$...$$
4. **超出范围拒绝回答**：如果问题涉及的内容完全不在 <sources> 中，
   必须明确回答"教材未涉及此内容，无法回答"，不得自行编造
5. **启发思考**：先给出关键思路，再展示详细步骤，鼓励学生自己先尝试
6. **概念串联**：主动关联相关知识点，帮助学生建立知识网络
7. **语言风格**：简洁清晰，不啰嗦

## 当前课程

{{course_context}}

## 参考教材内容

{{sources}}
"""


class Generator:
    """
    DeepSeek LLM 调用封装

    支持普通生成和 SSE 流式生成两种模式
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None
    ):
        self.model = model or settings.llm_model
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key

    async def generate(
        self,
        query: str,
        context: str,
        course_context: dict,
        *,
        temperature: float | None = None,
        conversation: list[dict[str, Any]] | None = None,
        rolling_summary: str = "",
        user_profile: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        """生成回答，支持多轮历史与预算制上下文。

        :returns: (answer, token_info)
            token_info: {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N,
                         "context_budget": {...}}
        """
        if not self.api_key:
            return "错误：LLM API Key 未配置，无法生成回答", {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "context_budget": {},
            }

        view = ContextManager(settings.llm_context_budget).build_view(
            node="query_rag",
            system_prompt=SYSTEM_PROMPT,
            course_context=course_context,
            user_profile=user_profile,
            conversation=conversation,
            rolling_summary=rolling_summary,
            current_query=query,
            rag_context=context,
        )

        messages = _build_messages(view)

        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else settings.llm_temperature,
        )
        content = response.choices[0].message.content
        usage = response.usage
        token_info = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "context_budget": view.budget,
            "layer_tokens": view.layer_tokens,
            "cache_hit_estimated": self._estimate_cache_hit(view, usage.prompt_tokens if usage else 0),
        }
        return content, token_info

    @staticmethod
    def _estimate_cache_hit(view: ContextView, actual_prompt_tokens: int) -> dict[str, Any]:
        """估算缓存命中情况（P5 可观测）。

        DeepSeek 不返回缓存命中标记，因此用启发式：
        - 稳定前缀（system_prefix）占比越高，命中概率越大
        - 用估算 token 与实际 prompt_tokens 的比例作为辅助信号
        """
        stable_tokens = view.layer_tokens.get("system_prefix", 0)
        total_estimated = view.budget.get("used", 1)
        stable_ratio = stable_tokens / total_estimated if total_estimated > 0 else 0.0
        # 经验：稳定内容 >60% 时认为缓存命中率较高
        estimated_hit_rate = min(0.95, max(0.1, stable_ratio * 1.2))
        return {
            "estimated_hit_rate": round(estimated_hit_rate, 2),
            "stable_tokens": stable_tokens,
            "actual_prompt_tokens": actual_prompt_tokens,
            "note": "基于稳定前缀占比启发式估算，非 DeepSeek 官方缓存标记",
        }

    async def generate_stream(
        self,
        query: str,
        context: str,
        course_context: dict,
        *,
        temperature: float | None = None,
        max_tokens: int = 2000,
        conversation: list[dict[str, Any]] | None = None,
        rolling_summary: str = "",
        user_profile: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """SSE 流式生成，供 FastAPI StreamingResponse 消费，支持多轮历史。"""
        if not self.api_key:
            yield "错误：LLM API Key 未配置"
            return

        view = ContextManager(settings.llm_context_budget).build_view(
            node="query_rag",
            system_prompt=SYSTEM_PROMPT,
            course_context=course_context,
            user_profile=user_profile,
            conversation=conversation,
            rolling_summary=rolling_summary,
            current_query=query,
            rag_context=context,
        )
        messages = _build_messages(view)

        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        stream = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_tokens=max_tokens,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def _build_messages(view: ContextView) -> list[dict[str, str]]:
    """把 ContextView 组装成 LLM messages。

    顺序：system -> rolling_summary -> recent_turns -> current_query。
    system 中已经包含课程上下文与学生画像；RAG sources 在此处替换占位符。
    """
    system_content = view.system_prefix.replace("{sources}", view.rag_context)
    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

    if view.rolling_summary:
        messages.append({
            "role": "system",
            "content": f"以下是对话历史摘要：\n{view.rolling_summary}",
        })

    messages.extend(view.recent_turns)
    messages.append({"role": "user", "content": view.current_query})
    return messages


def _format_course(ctx: dict | None) -> str:
    """（兼容旧调用）已迁移至 ContextManager._fmt_course_context。"""
    if not ctx:
        return "（未指定课程）"
    chapters = "、".join(ctx.get("chapters", []))
    return (
        f"课程：{ctx.get('name', '未知')}\n"
        f"教材：{ctx.get('textbook', '未知')}\n"
        f"已学章节：{chapters or '暂无'}"
    )


async def build_course_context(
    session: AsyncSession,
    course_id: UUID | str,
) -> dict:
    """查询课程的 KP 树，构建层级大纲供 System Prompt 使用"""
    course = await session.get(Course, course_id)
    if not course:
        return {}

    result = await session.execute(
        select(KnowledgePoint)
        .where(KnowledgePoint.course_id == course_id)
        .order_by(KnowledgePoint.kp_path)
    )
    kp_list = result.scalars().all()

    chapters = [kp.title for kp in kp_list if kp.parent_id is None]

    # 从 Document 表取教材名称（Course 模型无 textbook 字段）
    doc_result = await session.execute(
        select(Document.filename)
        .where(Document.course_id == course_id)
        .limit(1)
    )
    textbook = doc_result.scalar_one_or_none() or "未知教材"

    return {
        "name": course.name,
        "textbook": textbook,
        "chapters": chapters,
    }
