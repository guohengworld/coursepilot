"""LLM 生成器 —— DeepSeek 调用 + System Prompt 组装

用法：
    generator = Generator()
    answer, token_info = await generator.generate(query, context, course_context)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from uuid import UUID

import openai
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.config import settings
from coursepilot.models import Course, KnowledgePoint, Document

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是 CoursePilot 课程助教，为大学生解答数学问题。

## 回答规则

1. **基于教材**：回答必须依据下面 <sources> 中提供的教材内容
2. **引用格式**：涉及教材内容时使用 <ref id="N" /> 引用，N 为 source id
3. **公式正确**：所有数学公式使用 LaTeX 语法，行内 $...$，独立行 $$...$$
4. **区分边界**：教材中有的内容正常回答；超出教材范围的，明确说"教材未涉及此内容"并提供已知的相关知识点
5. **启发思考**：先给出关键思路，再展示详细步骤，鼓励学生自己先尝试
6. **概念串联**：主动关联相关知识点，帮助学生建立知识网络
7. **语言风格**：简洁清晰，不啰嗦

## 当前课程

{course_context}

## 参考教材内容

{sources}
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
    ) -> tuple[str, dict]:
        """生成回答

        :returns: (answer, token_info)
            token_info: {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
        """
        if not self.api_key:
            return "错误：LLM API Key 未配置，无法生成回答", {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            }

        system = SYSTEM_PROMPT.format(
            course_context=_format_course(course_context),
            sources=context
        )

        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            temperature=temperature if temperature is not None else settings.llm_temperature,
            # max_tokens=max_tokens
        )
        content = response.choices[0].message.content
        usage = response.usage
        token_info = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }
        return content, token_info

    async def generate_stream(
        self,
        query: str,
        context: str,
        course_context: dict,
        *,
        temperature: float | None = None,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        """SSE 流式生成，供 FastAPI StreamingResponse 消费"""
        if not self.api_key:
            yield "错误：LLM API Key 未配置"
            return

        system = SYSTEM_PROMPT.format(
            course_context=_format_course(course_context),
            sources=context
        )

        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        stream = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_tokens=max_tokens,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def _format_course(ctx: dict | None) -> str:
    if not ctx:
        return "（未指定课程）"
    chapters = "、".join(ctx.get("chapters", []))
    return f"课程：{ctx.get('name', '未知')}\n教材：{ctx.get('textbook', '未知')}\n已学章节：{chapters or '暂无'}"


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
