"""复习计划 Skill

基于薄弱知识点生成分天复习计划，写入 review_plan 表
被 review_plan_node 调用，结果存入 state["review_plan"]
"""
import json
import logging
from uuid import UUID
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from coursepilot.config import settings
from coursepilot.models import ReviewPlan

logger = logging.getLogger(__name__)

REVIEW_SYSTEM = """你是一个学习规划师。根据学生的薄弱知识点和课程内容，制定复习计划。

输出 JSON：
{
  "items": [
    {"kp_path": "OS/进程管理/进程调度", "priority": 1,
     "reason": "正确率 30%", "status": "pending"}
  ],
  "total_count": 5,
  "plan_summary": "分 3 天复习，重点突破进程调度..."
}
priority: 1(最薄弱) ~ 5(已掌握)
"""

async def review_plan(
    session: AsyncSession,
    user_id: str,
    course_id: str,
    diagnosis: dict
) -> tuple[dict, dict]:
    """生成复习计划并持久化到 review_plans 表

    Returns:
        (plan_data, token_info)
        plan_data: {"items": [...], "total_count": int, "plan_summary": str, "plan_id": str}
    """
    empty_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    weak_kps = diagnosis.get("weak_kps", [])
    kp_stats = diagnosis.get("kp_stats", {})

    if not weak_kps:
        return {"items": [], "total_count": 0,
                "plan_summary": "暂无薄弱知识点，无需复习计划", "plan_id": ""}, empty_tokens

    if settings.llm_api_key:
        client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
        stats_text = "\n".join(
            f"{kp}: {s['correct']}/{s['total']} 正确率 {s['rate']:.0%}"
            for kp, s in kp_stats.items() if kp in weak_kps
        )
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": REVIEW_SYSTEM},
                {"role": "user",
                 "content": f"薄弱知识点：\n{stats_text}\n\n课程：{diagnosis.get('summary', '')}"},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        plan_data = json.loads(response.choices[0].message.content)
        usage = response.usage
        token_info = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }
    else:
        plan_data = {
            "items": [
                {"kp_path": kp, "priority": 1,
                 "reason": "薄弱知识点", "status": "pending"}
                for kp in weak_kps
            ],
            "total_count": len(weak_kps),
            "plan_summary": f"共 {len(weak_kps)} 个薄弱知识点需复习",
        }
        token_info = empty_tokens

    plan = ReviewPlan(
        user_id=UUID(user_id),
        course_id=UUID(course_id),
        items=plan_data.get("items", []),
        total_count=plan_data.get("total_count", len(weak_kps)),
        reviewed_count=0,
    )
    session.add(plan)
    await session.flush()
    plan_data["plan_id"] = str(plan.id)
    return plan_data, token_info
