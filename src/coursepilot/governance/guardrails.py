"""确定性护栏：回答质量、知识点范围、引用完整性和资源上限检查。

所有函数都是纯同步的，不依赖外部服务。

Usage:
    from coursepilot.governance.guardrails import guard_answer

    # 在 finalize_node 返回前调用
    issues = guard_answer(answer, context, sources)
    if issues:
        logger.warning(f"Guardrail 警告: {issues}")
"""
import re
from typing import Any


def guard_answer(answer: str, context: str, sources: list[dict]) -> list[str]:
    """检查 LLM 生成的答案质量

    检查项：
    1. 空回答
    2. 未引用教材内容（幻觉风险）
    3. 直接给答案而非引导
    4. 引用标记完整性

    Returns:
        违规描述列表，为空表示全部通过
    """
    issues: list[str] = []

    if not answer or len(answer.strip()) < 10:
        issues.append("回答为空或过短")
        return issues

    # 1. 未引用教材内容：如果回答中完全不包含 context 中的关键词
    #    简单启发式：检查答案是否包含教材特有的术语（取 context 前 200 字的关键词）
    context_keywords = _extract_keywords(context, max_words=5)
    if context_keywords:
        matched = sum(1 for kw in context_keywords if kw in answer)
        if matched < 1:
            issues.append(f"回答可能未引用教材内容（匹配到 {matched}/{len(context_keywords)} 个关键词）")

    # 2. 引用标记完整性：检查 [page:N] 格式的标记
    citation_count = len(re.findall(r'\[page:\d+\]', answer))
    if citation_count == 0 and len(answer) > 100:
        issues.append("较长回答缺少引用标记")

    # 3. 直接给答案检测：包含"答案是""正确答案是"等短语且无引导说明
    direct_answer_patterns = [
        r"答案是\s*[A-Z]",
        r"正确答案是\s*[A-Z]",
        r"选择\s*[A-Z]\s*$",
    ]
    for pattern in direct_answer_patterns:
        if re.search(pattern, answer, re.MULTILINE):
            issues.append("存在直接给答案的表述，建议改为引导式回答")
            break

    # 4. source_kp_paths 完整性
    if sources and len(sources) > 3:
        # 引用来源超过 3 个时，答案应足够详细
        if len(answer) < 50:
            issues.append("引用来源较多但回答过于简短")

    return issues


def guard_kp_scope(kp_paths: list[str], course_kp_tree: list[dict]) -> list[str]:
    """检查知识点路径是否在课程范围内

    Args:
        kp_paths: 检索返回的 kp_path 列表
        course_kp_tree: 课程知识点树（扁平列表）

    Returns:
        越界的知识点路径列表
    """
    valid_paths = {kp["kp_path"] for kp in course_kp_tree}
    out_of_scope = [p for p in kp_paths if p not in valid_paths]
    return out_of_scope


def guard_daily_limit(
        daily_count: int, max_daily: int = 200
) -> str | None:
    """检查日调用次数是否超限

    Returns:
        错误消息或 None（通过）
    """
    if daily_count >= max_daily:
        return f"已超过每日上限 {max_daily} 次"
    return None


def guard_token_limit(
        session_token: int, session_max: int = 50000,
        daily_token: int = 0, daily_max: int = 500000,
) -> str | None:
    """检查 token 用量是否超限

    Args:
        session_token: 本次会话已用 token
        session_max: 单会话上限
        daily_token: 今日已用 token
        daily_max: 每日上限

    Returns:
        错误消息或 None（通过）
    """
    if session_token >= session_max:
        return f"单次会话 token 超限 ({session_token}/{session_max})"
    if daily_token >= daily_max:
        return f"每日 token 超限 ({daily_token}/{daily_max})"
    return None


def _extract_keywords(text: str, max_words: int = 5) -> list[str]:
    """从文本中提取代表性关键词（简单实现）

    取最长的不含停用词的词作为关键词。
    """
    stop_words = {"的", "了", "是", "在", "和", "就", "都", "而", "及", "与",
                  "着", "或", "一个", "没有", "我们", "你们", "他们", "这个",
                  "那个", "什么", "如何", "可以", "进行", "使用", "通过", "需要"}
    # 取前 200 字，按标点分割取最长片段
    segment = text[:200]
    # 简单 split，取非停用词的中文/英文词
    words = re.findall(r'[\w\u4e00-\u9fff]+', segment)
    words = [w for w in words if w not in stop_words and len(w) > 1]
    # 按长度降序取 top
    words.sort(key=len, reverse=True)
    return words[:max_words]
