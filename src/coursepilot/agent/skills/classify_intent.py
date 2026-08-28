"""意图分类 Skill：基于 LLM 的意图识别 + 复杂度判断

返回结构化分类结果，包含：
  - intent: question / practice / diagnose / review / none
  - complexity: simple / complex
"""
import json
import logging

from openai import AsyncOpenAI

from coursepilot.config import settings

logger = logging.getLogger(__name__)


CLASSIFY_SYSTEM = """你是一个教学系统意图分类器。
根据用户的问题、对话上下文和【当前课程】，判断用户最可能的意图和问题复杂度。

## 意图分类
- question: 问"什么是X"、"解释X"、"X是什么意思"等知识性提问，或问具体知识点。
  **不包括请求出题、练习、分析学习情况**
- practice: 明确要求"出题"、"练习"、"做几道题"、"考考我"、"来点题目"等练习请求
- diagnose: 想了解自己的学习情况、薄弱环节，如"帮我分析"、"我学得怎么样"、"掌握情况"
- review: 想复习，如"帮我复习"、"总结本章重点"、"制定复习计划"
- none: 结合上下文与【当前课程】后，话语仍未对课程学习内容表达任何具体学习意图。包括：
  (a) 寒暄/礼貌开场："你好"、"谢谢"、"在吗"、"能帮帮我吗"；
  (b) 欠指定请求：表达了"想被帮助"但未说明学什么，如"你好能帮帮我吗"、"不懂的能讲讲吗"（无上下文指向具体知识点时）；
  (c) 非学术/非学习类请求（即使以问句形态出现）："写英语作文"、"翻译这段"、"讲个笑话"、"今天股市如何"。
  注意：判定"离题/拒识"应以"是否为学术学习意图"为准，而非"是否在课程章节列表内"——数学范围内的问题（如矩阵特征值、条件概率、贝叶斯）无论属于哪个数学分支，仍属 question，不要判 none。"当前课程"字段用于辅助判断 borderline 请求是否在本课程范围内，但不应据此拒掉明确的学术提问。
  反向约束：只要话语点名了具体知识点或明确学习动作，即使以请求句表达也归对应意图，不要归 none。
  例："能帮我讲一下导数吗"→question（点名导数）；"你好能帮帮我吗"→none（未点名任何学习内容）；"写英语作文"→none（非学术任务）。

## 复杂度判断
分析问题是否需要多源信息拼接、跨知识点推理或对比分析：
- simple: 单知识点、事实性、可直接从教材一段内容回答。
  如"什么是极限"、"罗尔定理的条件是什么"
- complex: 多知识点比较（"A和B有什么区别"）、需要多步推理、
  信息跨多个章节、需要多次检索才能回答
（注：none 意图无需判断复杂度，可输出 simple）

## 输出格式
必须严格输出 JSON 格式，意图只能取 question/practice/diagnose/review/none 之一，不要包含其他内容：
{"intent": "分类名称", "complexity": "simple或complex", "reasoning": "简要判断理由"}"""


async def classify_intent(
    query: str,
    course_context: dict | None = None,
    recent_qa: list[dict] | None = None
) -> tuple[str, str, dict]:
    """返回 (intent, complexity, token_info)。

    Returns:
        intent: question / practice / diagnose / review / none
        complexity: simple / complex
        token_info: {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
    """
    if not settings.llm_api_key:
        logger.warning("LLM API key 未配置，默认返回 question+simple")
        return "question", "simple", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # LLM 分类
    parts = [f"用户问题：{query}"]
    # 注入当前课程上下文，供 none/离题 判定锚定基准（由调用方传入，可能为空）
    if course_context:
        cc_lines = []
        course_name = course_context.get("name")
        if course_name:
            cc_lines.append(f"当前课程：{course_name}")
        chapters = course_context.get("chapters") or []
        if chapters:
            cc_lines.append("课程章节范围：" + "、".join(chapters[:10]))
        if cc_lines:
            parts.append("\n".join(cc_lines))
    if recent_qa:
        parts.append("最近回答：")
        for qa in recent_qa[-3:]:
            # 兼容两种历史格式：数据库 QARecord(query/answer) 和 conversation(role/content)
            if isinstance(qa, dict) and "query" in qa:
                q_text = qa["query"]
            elif isinstance(qa, dict) and qa.get("role") == "user" and "content" in qa:
                q_text = qa["content"]
            else:
                continue
            parts.append(f"  Q: {q_text}")

    prompt = "\n".join(parts)
    logger.info("classify_intent LLM prompt:\n%s", prompt)

    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout,
    )
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800,
    )
    raw = response.choices[0].message.content
    finish = response.choices[0].finish_reason
    logger.info("classify_intent LLM response finish=%s raw=%r", finish, raw)

    usage = response.usage
    token_info = {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }

    # 降级默认值
    intent = "question"
    complexity = "simple"

    if not raw:
        logger.warning("classify_intent 模型返回空内容 finish=%s", finish)
        return intent, complexity, token_info

    # 尝试解析 JSON
    try:
        result = json.loads(raw.strip())
        if isinstance(result, dict):
            raw_intent = result.get("intent", "").strip().lower()
            raw_complexity = result.get("complexity", "").strip().lower()
            raw_reasoning = result.get("reasoning", "")

            valid_intents = {"question", "practice", "diagnose", "review", "none"}
            valid_complexities = {"simple", "complex"}

            if raw_intent in valid_intents:
                intent = raw_intent
            else:
                logger.warning("classify_intent 无效意图 %r，降级为 question", raw_intent)

            if raw_complexity in valid_complexities:
                complexity = raw_complexity
            else:
                logger.warning("classify_intent 无效复杂度 %r，降级为 simple", raw_complexity)

            if raw_reasoning:
                logger.info("classify_intent 推理: %s", raw_reasoning)
    except (json.JSONDecodeError, AttributeError) as e:
        # 兜底：尝试直接按文本解析（兼容旧格式）
        cleaned = raw.strip().lower().split("\n")[0].strip()
        valid_intents = {"question", "practice", "diagnose", "review"}
        if cleaned in valid_intents:
            intent = cleaned
            logger.info("classify_intent 回退文本解析: intent=%s", intent)
        else:
            logger.warning("classify_intent JSON 解析失败: %s，原始=%r", e, raw[:100])

    return intent, complexity, token_info
