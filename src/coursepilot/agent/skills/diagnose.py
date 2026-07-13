"""学情诊断 Skill

聚合 PracticeRecord → 按 KP 计算正确率 → 识别薄弱点
→ LLM 生成深度分析和学习建议
支持可配时间窗口、可配阈值、超时防护。
"""
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import TypedDict
from uuid import UUID

from openai import AsyncOpenAI
from sqlalchemy import Integer, and_, func as sa_func, select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.config import settings
from coursepilot.models import KnowledgePoint, PracticeRecord, Question

logger = logging.getLogger(__name__)


class KpStatItem(TypedDict):
    """单个知识点的统计"""
    total: int
    correct: int
    rate: float


class DiagnosisResult(TypedDict):
    """学情诊断结果"""
    weak_kps: list[str]
    kp_stats: dict[str, KpStatItem]
    summary: str
    total_practiced: int
    overall_rate: float


MAX_KP_LIMIT = 200


async def diagnose(
    session: AsyncSession,
    user_id: str,
    course_id: str,
    *,
    weak_threshold: float | None = None,
    lookback_days: int | None = None,
) -> DiagnosisResult:
    """执行学情诊断

    聚合 PracticeRecord → 关联 Question → 获取 KP → 按 KP 路径计算正确率
    → 识别薄弱知识点（正确率低于 weak_threshold）

    Args:
        session: 数据库会话
        user_id: 用户 UUID 字符串
        course_id: 课程 UUID 字符串
        weak_threshold: 薄弱阈值，默认来自 settings.diagnose_weak_threshold (0.6)
            传入 None 使用默认值，传入 0.0 禁用薄弱标记
        lookback_days: 分析最近 N 天的记录
            默认来自 settings.diagnose_lookback_days (90)
            传入 None 使用默认值，传入 0 禁用时间过滤（分析全量）

    Returns:
        DiagnosisResult

    Raises:
        ValueError: user_id / course_id 格式无效
    """
    # 1. 输入校验
    try:
        uid = UUID(user_id)
        cid = UUID(course_id)
    except ValueError as e:
        logger.warning("diagnose: 无效 UUID — user=%s course=%s", user_id, course_id)
        raise ValueError(f"无效的 UUID 参数: {e}") from e

    _threshold = (
        settings.diagnose_weak_threshold if weak_threshold is None else weak_threshold
    )
    _lookback = (
        settings.diagnose_lookback_days if lookback_days is None else lookback_days
    )

    # 2. 构建过滤条件
    filters = [
        PracticeRecord.user_id == uid,
        KnowledgePoint.course_id == cid,
        PracticeRecord.correct_flag.isnot(None),
    ]
    if _lookback:
        since = datetime.now(timezone.utc) - timedelta(days=_lookback)
        filters.append(PracticeRecord.answered_at >= since)

    t0 = time.perf_counter()

    # 3. 执行聚合查询
    try:
        result = await session.execute(
            select(
                KnowledgePoint.kp_path,
                sa_func.count(PracticeRecord.id).label("total"),
                sa_func.sum(
                    sa_func.cast(PracticeRecord.correct_flag, Integer())
                ).label("correct"),
            )
            .select_from(PracticeRecord)
            .join(Question, PracticeRecord.question_id == Question.id)
            .join(KnowledgePoint, Question.kp_id == KnowledgePoint.id)
            .where(and_(*filters))
            .group_by(KnowledgePoint.kp_path)
            .limit(MAX_KP_LIMIT)
        )
        rows = result.all()
    except Exception:
        logger.exception("diagnose: SQL 聚合查询失败")
        raise

    elapsed = time.perf_counter() - t0

    # 4. 计算统计
    kp_stats: dict[str, KpStatItem] = {}
    weak_kps: list[str] = []
    total_answered = 0
    total_correct = 0

    for kp_path, count, correct_sum in rows:
        total = int(count)
        correct = int(correct_sum or 0)
        rate = correct / total if total > 0 else 0.0
        kp_stats[kp_path] = KpStatItem(
            total=total, correct=correct, rate=round(rate, 2),
        )
        total_answered += total
        total_correct += correct
        if _threshold and rate < _threshold:
            weak_kps.append(kp_path)

    overall_rate = total_correct / total_answered if total_answered > 0 else 0.0

    # 5. 构建摘要
    summary_parts = [f"共练习 {total_answered} 题，正确率 {overall_rate:.0%}。"]
    if weak_kps:
        top_weak = weak_kps[:5]
        threshold_pct = f"{_threshold:.0%}"
        summary_parts.append(
            f"薄弱知识点（正确率<{threshold_pct}）：{'、'.join(top_weak)}"
        )
        if len(weak_kps) > 5:
            summary_parts.append(f"等 {len(weak_kps)} 个")

    # 6. 日志（可观测性）
    logger.info(
        "diagnose: user=%s elapsed=%.2fs kps=%d weak=%d total=%d rate=%.2f",
        user_id, elapsed, len(kp_stats), len(weak_kps),
        total_answered, overall_rate,
    )

    return DiagnosisResult(
        weak_kps=weak_kps,
        kp_stats=kp_stats,
        summary="".join(summary_parts),
        total_practiced=total_answered,
        overall_rate=round(overall_rate, 2),
    )


DIAGNOSE_ANALYSIS_SYSTEM = """你是一个学情诊断分析专家。根据学生的练习题统计结果，生成诊断分析报告。

请输出 JSON 格式（不要 markdown 包裹，纯 JSON）：
{
  "analysis": "详细的学情分析，包括：1) 整体表现评价 2) 各知识点掌握情况分析 3) 薄弱知识点的问题定位（概念混淆、计算错误、审题偏差等）",
  "recommendations": "分步学习建议，包括：1) 复习优先级和顺序 2) 针对每个薄弱点的具体复习方法 3) 后续练习建议"
}

要求：
- 分析要具体，引用知识点名称和正确率数据
- 建议要可操作，给出具体的学习方法（如：回顾教材第X章、做针对性练习等）
- 语气积极鼓励，帮助学生建立信心
- 使用中文
- 只输出 JSON，不要额外文字"""


async def generate_llm_analysis(
    diagnosis: DiagnosisResult,
    user_query: str = "",
    topic_kp_path: str = "",
) -> tuple[str, str, dict]:
    """调用 LLM 生成深度学情分析和学习建议

    Args:
        diagnosis: 诊断结果
        user_query: 用户原始查询（用于针对性分析）
        topic_kp_path: 用户问到的特定知识点路径（若有）

    Returns:
        (analysis, recommendations, token_info)
    """
    if not settings.llm_api_key or diagnosis["total_practiced"] == 0:
        return "", "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    kp_lines = []
    for kp_path, stat in sorted(
        diagnosis["kp_stats"].items(), key=lambda x: x[1]["rate"],
    ):
        kp_lines.append(
            f"- {kp_path}: 正确率 {stat['rate']:.0%} ({stat['correct']}/{stat['total']})"
        )

    weak_text = "、".join(diagnosis["weak_kps"]) if diagnosis["weak_kps"] else "无"
    prompt_parts = [
        f"学生总练习量：{diagnosis['total_practiced']} 题",
        f"总正确率：{diagnosis['overall_rate']:.0%}",
        f"薄弱知识点：{weak_text}",
        "\n各知识点详情：",
        "\n".join(kp_lines),
    ]

    # 如果用户问了特定知识点，添加到 prompt 顶部
    if user_query:
        prompt_parts.insert(0, f"学生的问题：{user_query}")
        prompt_parts.insert(0, "请针对学生的具体问题进行针对性分析。如果学生问的知识点不在上述数据中，请如实告知尚无练习记录。")

    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": DIAGNOSE_ANALYSIS_SYSTEM},
                {"role": "user", "content": "\n".join(prompt_parts)},
            ],
            temperature=0.5,
            response_format={"type": "json_object"},
            max_tokens=2000,
        )
        content = response.choices[0].message.content
        usage = response.usage
        token_info = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }

        if not content:
            logger.warning("generate_llm_analysis: 模型返回空")
            return "", "", token_info

        result = json.loads(content)
        return (
            result.get("analysis", ""),
            result.get("recommendations", ""),
            token_info,
        )
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("generate_llm_analysis 失败: %s", e)
        return "", "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
