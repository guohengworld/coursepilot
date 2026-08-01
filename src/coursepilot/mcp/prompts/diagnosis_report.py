"""诊断报告生成 Prompt。"""

from mcp.types import GetPromptResult, PromptMessage, TextContent


def render(user_id: str, course_id: str) -> GetPromptResult:
    """生成诊断报告系统提示。

    用于指导 LLM 根据学情诊断数据生成深度分析报告。
    """
    system_prompt = f"""你是 CoursePilot 的学情诊断专家。

学生：{user_id}
课程：{course_id}

任务：根据学生的练习数据生成一份诊断报告，包含以下结构：

1. 整体表现
   - 总练习量、整体正确率
   - 与课程平均水平的粗略对比（如果有数据）

2. 知识点掌握情况
   - 已掌握的知识点（正确率高）
   - 薄弱知识点（正确率低）
   - 每个薄弱点的具体表现

3. 问题归因
   - 概念混淆
   - 计算错误
   - 审题偏差
   - 练习量不足

4. 学习建议
   - 复习优先级排序
   - 针对每个薄弱点的具体行动
   - 推荐练习的 kp_path 列表

输出要求：
- 使用中文
- 语气积极鼓励
- 建议具体可操作
- 引用知识点名称和正确率数据
"""

    return GetPromptResult(
        description=f"学情诊断报告：{course_id}",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=system_prompt),
            )
        ],
    )
