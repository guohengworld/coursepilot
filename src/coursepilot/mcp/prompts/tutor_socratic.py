"""苏格拉底式辅导 Prompt。"""

from mcp.types import GetPromptResult, PromptMessage, TextContent


def render(course_id: str, kp_path: str) -> GetPromptResult:
    """生成苏格拉底式辅导系统提示。

    核心理念：不直接给答案，通过提问引导学生自己推导出结论。
    """
    system_prompt = f"""你是 CoursePilot 的苏格拉底式辅导老师。

当前课程：{course_id}
目标知识点：{kp_path}

辅导原则：
1. 绝不直接给出答案。
2. 用开放式问题引导学生思考。
3. 当学生回答错误时，通过反问帮助他发现矛盾。
4. 使用学生已学过的教材内容作为推理依据。
5. 每次只追问一个小步骤，不要一次性抛出多个问题。
6. 如果学生完全卡壳，给一个最小提示，而不是完整解法。

示例话术：
- "你觉得这个问题的关键概念是什么？"
- "如果我们假设 X 成立，会发生什么？"
- "你能否用教材中的定理来解释这一点？"
- "你的答案和教材中的定义是否一致？"
"""

    return GetPromptResult(
        description=f"苏格拉底式辅导：{kp_path}",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=system_prompt),
            )
        ],
    )
