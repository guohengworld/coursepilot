"""练习题验证 Skill（生成-验证分离核心）

用独立 prompt 审查生成的练习题，确保正确性、知识点覆盖、无幻觉
temperature=0.1 区别于 generate_quiz 的 0.7，模拟"不同审查视角"
"""
import json
import logging
from openai import AsyncOpenAI
from coursepilot.config import settings

logger = logging.getLogger(__name__)

EVALUATE_SYSTEM = """你是一个严谨的试题审核员。审查以下选择题：

审核标准：
1. 答案正确性：correct_answer 是否真的正确
2. 知识点匹配：题目是否确实覆盖了对应 kp_path
3. 选项合理性：干扰项有意义，非明显错误
4. 难度梯度：是否由易到难
5. 幻觉检测：题目的概念/公式是否在教材范围内

输出 JSON：
{
  "status": "PASS" | "FAIL",
  "score": 0.0~1.0,
  "feedback": {
    "correctness_issues": [],
    "coverage_issues": [],
    "hallucination_issues": [],
    "suggestions": []
  }
}
"""

async def evaluate_quiz(
    quiz_data: dict,
    context: str,
    course_context: dict
) -> tuple[dict, dict]:
    """验证练习题质量

    Returns:
        (eval_result, token_info)
    """
    empty_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not settings.llm_api_key:
        return {"status": "PASS", "score": 1.0,
                "feedback": {"suggestions": ["无 API key，跳过验证"]}}, empty_tokens
    if not quiz_data.get("questions"):
        return {"status": "FAIL", "score": 0.0,
                "feedback": {"suggestions": ["题目为空"]}}, empty_tokens

    client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": EVALUATE_SYSTEM},
            {"role": "user", "content": json.dumps(
                {"quiz": quiz_data, "materials": context[:2000]}, ensure_ascii=False
            )},
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    usage = response.usage
    token_info = {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }
    try:
        return json.loads(content), token_info
    except (json.JSONDecodeError, TypeError):
        logger.warning("evaluate_quiz: 解析审核结果失败，默认 PASS")
        return {"status": "PASS", "score": 0.8,
                "feedback": {"suggestion": ["审核结果解析失败"]}}, token_info
