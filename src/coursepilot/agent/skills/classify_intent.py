"""意图分类 Skill：用 DeepSeek 判断用户意图

返回 question / practice / diagnose / review / code_help 之一
"""
from openai import AsyncOpenAI

from coursepilot.config import settings

CLASSIFY_SYSTEM = """你是一个教学系统意图分类器。根据用户的问题和对话上下文，判断用户最可能的意图。

分类选项：
- question: 提问知识点问题（最常见），如"什么是二叉树"、"解释一下递归"
- practice: 想做练习题，如"出几道题"、"练习一下"
- diagnose: 想了解自己的学习情况，如"我哪里掌握得不好"、"帮我诊断"
- review: 想复习，如"帮我复习一下"、"总结本章重点"
- code_help: 代码相关问题，如"这个代码为什么报错"、"帮我调试"

只输出分类名称，不要输出其他内容。"""

async def classify_intent(
    query: str,
    course_context: dict | None = None,
    recent_qa: list[dict] | None = None
) -> str:
    """返回意图名称"""
    if not settings.llm_api_key:
        return "question"

    parts = [f"用户问题：{query}"]
    if recent_qa:
        parts.append("最近回答：")
        for qa in recent_qa[-3:]:
            parts.append(f"  Q: {qa['query']}")

    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url
    )
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": "\n".join(parts)}
        ],
        temperature=0.1,
        max_tokens=20,
    )
    intent = response.choices[0].message.content.strip().lower()
    valid = {"question", "practice", "diagnose", "review", "code_help"}
    return intent if intent in valid else "question"
