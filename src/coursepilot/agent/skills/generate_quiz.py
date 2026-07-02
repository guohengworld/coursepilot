"""生成练习题 Skill

基于知识点掌握度和 RAG 检索内容，生成 3 道选择题
generate_quiz_node 调用，结果存入 state["quiz_data"]
"""
import json
import logging
from openai import AsyncOpenAI
from coursepilot.config import settings

logger = logging.getLogger(__name__)

GENERATE_SYSTEM = """你是一个教学出题系统。根据以下知识点和教材内容，生成 3 道选择题。

每道题要求：
- 四选一（choice_4）
- 难度循序渐进（基础→进阶→综合）
- 选项清晰，无歧义
- 附正确答案和详细解析，关联到具体知识点

输出 JSON：
{
  "questions": [
    {
      "question_text": "...",
      "question_type": "choice_4",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "correct_answer": "A",
      "explanation": "...",
      "kp_path": "OS/xxx"
    }
  ]
}
"""

async def generate_quiz(
    context: str,
    course_context: dict,
    mastery: dict,
) -> dict:
    """生成 3 道练习题

    Returns:
        {"question": [...]}，空列表表示生成失败
    """
    if not settings.llm_api_key:
        return {"question": []}

    weak_kps = mastery.get("weak_kps", [])
    focus_hint = ""
    if weak_kps:
        focus_hint = f"重点关注以下薄弱知识点：{'、'.join(weak_kps[:3])}"

    prompt_parts = [
        f"课程：{course_context.get('name', '未知')}",
        f"已学章节：{'、'.join(course_context.get('chapters', [])[:5])}",
        focus_hint,
        f"\n教材内容：\n{context[:3000]}",
    ]

    client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": GENERATE_SYSTEM},
            {"role": "user", "content": "\n".join(prompt_parts)}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        logger.warning("generate_quiz: LLM 返回非 JSON，回退空结果")
        return {"question": []}



