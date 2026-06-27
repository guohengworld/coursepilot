"""
查询改写 —— 将学生口语化查询转为适合检索的标准化表述

用法：
    rewriter = QueryWriter()
    rewritten = await rewriter.rewrite("泰勒展开咋用来求极限")
"""

from __future__ import annotations

import logging

import openai

from coursepilot.config import settings

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """你是课程助教。将学生的口语化问题改写为适合检索的表述。

规则：
- 补充学科关键术语，但不要编造问题中没提到的内容
- 简单明确的问题保持原样，不过度扩展
- 消解指代词（"它"→具体概念名）
- 只输出改写后的问题，不加任何解释

学生问题：{query}
改写后："""


class QueryRewriter:
    """
    DeepSeek 查询改写器

    temperature = 0 保证确定性输出，max_tokens=200 限制改写长度
    API key 未配置时降级为直接返回原始 query
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

    async def rewrite(self, query: str) -> str:
        print(f"[rewrite] 开始改写: {query[:60]}...")
        if not self.api_key:
            logger.debug("LLM_API_KEY 未配置，降级为直接返回原问题")
            print("[rewrite] API key 未配置，跳过")
            return query

        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        prompt = REWRITE_PROMPT.format(query=query)

        try:
            print("[rewrite] 调用 DeepSeek...")
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            rewritten = response.choices[0].message.content.strip()
            print(f"[rewrite] 完成: '{query[:40]}...' → '{rewritten[:40]}...'")
            logger.debug("查询改写：'%s' → '%s'", query, rewritten)
            return rewritten
        except Exception as e:
            print(f"[rewrite] 失败: {e}")
            logger.warning("LLM 改写失败：%s", e)
            return query




