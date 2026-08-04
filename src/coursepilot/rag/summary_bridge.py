"""SummaryBridge — 为 KnowledgeUnit 调用 DeepSeek 生成中文摘要。

设计参考：docs/RAG_Engine_Design_v1.0.md §2.3

为什么需要：LaTeX 公式和自然语言在不同嵌入空间，纯 content 检索对公式不敏感。
LLM 生成的自然语言摘要将公式"翻译"为可检索的描述性文本。

index_text = {summary} + "\n" + {content}
"""

from __future__ import annotations

import asyncio
import logging
import time

from coursepilot.config import settings

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """用一句话（不超过80字）概括以下核心内容的摘要。
如果是公式/定理，必须用汉字描述公式的运算关系，说明名称和作用；如果是例题，说明题型和用到的定理。
【重要】只输出这一句话，不要包含任何解释、评价、前缀或问候语。"""

# 并发控制
MAX_CONCURRENT = 20


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
        max_concurrent: int = MAX_CONCURRENT,
    ):
        self.model = model or settings.llm_model
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key
        self.max_concurrent = max_concurrent

    async def run(self, units: list[dict]) -> list[dict]:
        """为每条 unit 生成摘要，写入 unit["summary"]。

        跳过已有摘要的 unit（幂等），跳过过短文本。
        并发调用 LLM API，加速处理。
        """
        import openai

        if not self.api_key:
            logger.warning("LLM_API_KEY 未配置，跳过摘要生成")
            return units

        # 找出需要生成摘要的 unit 索引
        pending: list[tuple[int, dict]] = []
        skip_short = 0
        for i, unit in enumerate(units):
            if unit.get("summary"):
                continue
            content = unit.get("content", "")
            if len(content) < 10:
                unit["summary"] = ""
                skip_short += 1
                continue
            pending.append((i, unit))

        total = len(units)
        pending_count = len(pending)
        if pending_count == 0:
            logger.info("SummaryBridge: 所有 unit 已有摘要 (skipped_short=%d)", skip_short)
            return units

        logger.info(
            "SummaryBridge: %d/%d 待生成摘要 (skipped_short=%d, concurrency=%d)",
            pending_count, total, skip_short, self.max_concurrent,
        )

        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed = 0
        failed = 0
        t0 = time.time()

        async def generate_one(idx: int, unit: dict) -> None:
            nonlocal completed, failed
            async with semaphore:
                content = unit["content"]
                try:
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SUMMARY_PROMPT},
                            {"role": "user", "content": content},
                        ],
                        temperature=0.3,
                        max_tokens=200,
                        timeout=60.0,
                        extra_body={"thinking": {"type": "disabled"}},
                    )
                    summary = response.choices[0].message.content
                    if summary:
                        unit["summary"] = summary.strip()
                    else:
                        unit["summary"] = ""
                        failed += 1
                except Exception:
                    unit["summary"] = ""
                    failed += 1

                completed += 1
                if completed % 50 == 0 or completed == pending_count:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (pending_count - completed) / rate if rate > 0 else 0
                    logger.info(
                        "SummaryBridge [%d/%d] %.1f/s ETA:%.0fs fail:%d",
                        completed, pending_count, rate, eta, failed,
                    )

        await asyncio.gather(*[generate_one(idx, unit) for idx, unit in pending])

        total_elapsed = time.time() - t0
        success = pending_count - failed
        logger.info(
            "SummaryBridge done: %d/%d 成功, %d 失败 (%.0fs)",
            success, pending_count, failed, total_elapsed,
        )

        return units
