"""SummaryBridge — 为 KnowledgeUnit 调用 DeepSeek 生成中文摘要。

设计参考：docs/RAG_Engine_Design_v1.0.md §2.3

为什么需要：LaTeX 公式和自然语言在不同嵌入空间，纯 content 检索对公式不敏感。
LLM 生成的自然语言摘要将公式"翻译"为可检索的描述性文本。

index_text = {summary} + {content[:200]}
"""

from __future__ import annotations

import logging

from coursepilot.config import settings

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """用一句话（不超过80字）概括以下教材片段的核心内容。
如果是公式/定理，说明名称和作用；如果是例题，说明题型和用到的定理。

教材内容：
{content}

摘要："""


class SummaryBridge:
    """为 KnowledgeUnit 列表生成中文摘要。

    用法:
        bridge = SummaryBridge()
        units = await bridge.run(units)  # 原地更新 units[i]["summary"]
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model or settings.llm_model
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key

    async def run(self, units: list[dict]) -> list[dict]:
        """为每条 unit 生成摘要，写入 unit["summary"]。

        跳过已有摘要的 unit（幂等）。
        """
        import openai

        if not self.api_key:
            logger.warning("LLM_API_KEY 未配置，跳过摘要生成")
            return units

        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        total = len(units)
        for i, unit in enumerate(units):
            if unit.get("summary"):
                continue

            content = unit.get("content", "")
            if len(content) < 10:  # 跳过过短的文本
                unit["summary"] = ""
                continue

            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": SUMMARY_PROMPT.format(content=content[:500]),
                        }
                    ],
                    temperature=0,
                    max_tokens=120,
                    timeout=30.0,
                )
                summary = response.choices[0].message.content.strip()
                unit["summary"] = summary
                if (i + 1) % 20 == 0 or i == 0:
                    print(f"     📝 SummaryBridge [{i+1}/{total}]: {summary[:60]}...")
            except Exception:
                if (i + 1) % 20 == 0 or i == 0:
                    print(f"     ⚠ SummaryBridge [{i+1}/{total}] 失败，继续...")
                unit["summary"] = ""

        return units
