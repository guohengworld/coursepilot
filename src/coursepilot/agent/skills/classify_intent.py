"""意图分类 Skill：基于 LLM 的意图识别

返回 question / practice / diagnose / review / code_help 之一
"""
import logging

from openai import AsyncOpenAI

from coursepilot.config import settings

logger = logging.getLogger(__name__)


CLASSIFY_SYSTEM = """你是一个教学系统意图分类器。根据用户的问题和对话上下文，判断用户最可能的意图。

分类选项：
- question: 用户问"什么是X"、"解释X"、"X是什么意思"、"X怎么用"等知识性提问，或问一个具体知识点。**不包括请求出题、练习、分析学习情况**
- practice: 用户明确要求"出几道题"、"出题"、"练习"、"练习题"、"做几道题"、"考考我"、"给我题目"、"来点题目"等练习请求
- diagnose: 想了解自己的学习情况、学习效果、薄弱环节，如"我哪里掌握得不好"、"帮我分析一下我的学习情况"、"帮我诊断"、"分析我的学习"、"我学得怎么样"、"我的掌握情况"
- review: 想复习，如"帮我复习一下"、"总结本章重点"、"制定复习计划"、"帮我总结"
- code_help: 代码相关问题，如"这个代码为什么报错"、"帮我调试"、"代码有bug"

只输出分类名称，不要输出其他内容。"""


async def classify_intent(
    query: str,
    course_context: dict | None = None,
    recent_qa: list[dict] | None = None
) -> tuple[str, dict]:
    """返回 (意图名称, token用量)。纯 LLM 分类。"""
    if not settings.llm_api_key:
        logger.warning("LLM API key 未配置，默认返回 question")
        return "question", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # LLM 分类
    parts = [f"用户问题：{query}"]
    if recent_qa:
        parts.append("最近回答：")
        for qa in recent_qa[-3:]:
            parts.append(f"  Q: {qa['query']}")

    prompt = "\n".join(parts)
    logger.info("classify_intent LLM prompt:\n%s", prompt)

    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url
    )
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=50,
    )
    raw = response.choices[0].message.content
    finish = response.choices[0].finish_reason
    logger.info("classify_intent LLM response finish=%s raw=%r", finish, raw)
    if not raw:
        logger.warning("classify_intent 模型返回空内容 finish=%s", finish)
        return "question", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    intent = raw.strip().lower()

    usage = response.usage
    token_info = {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }
    valid = {"question", "practice", "diagnose", "review", "code_help"}
    result = intent if intent in valid else "question"
    if intent != result:
        logger.warning("classify_intent 返回无效值 %r，已降级为 question", raw)
    return result, token_info
