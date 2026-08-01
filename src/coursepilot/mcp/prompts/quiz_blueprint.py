"""出题蓝图 Prompt。"""

from mcp.types import GetPromptResult, PromptMessage, TextContent


def render(course_id: str, kp_path: str, count: int, difficulty: int) -> GetPromptResult:
    """生成出题蓝图系统提示。

    用于指导 LLM 生成结构化的练习题。
    """
    system_prompt = f"""你是 CoursePilot 的出题专家。

当前课程：{course_id}
目标知识点：{kp_path}
题目数量：{count}
难度等级：{difficulty}（1 最简单，5 最难）

出题要求：
1. 每道题必须明确对应目标知识点的一个子点。
2. 题目难度分布围绕 {difficulty}，可上下浮动 1 级。
3. 优先使用选择题（4 个选项），干扰项必须来自常见 misconception。
4. 题干简洁，避免歧义。
5. 每道题必须包含：question_text、options（A/B/C/D）、correct_answer、explanation、kp_path。
6. 解释中引用教材原文或定理，说明为什么正确答案是正确的。
7. 输出为纯 JSON，不要 Markdown 包裹。

输出格式：
{{
  "questions": [
    {{
      "question_text": "...",
      "question_type": "choice_4",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct_answer": "A",
      "explanation": "...",
      "kp_path": "{kp_path}"
    }}
  ]
}}
"""

    return GetPromptResult(
        description=f"出题蓝图：{kp_path}",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=system_prompt),
            )
        ],
    )
