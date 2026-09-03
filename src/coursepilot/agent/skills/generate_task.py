"""给指定学生生成 AI 任务草稿（⑤ 教师发布任务）

纯生成器：不做持久化，返回可直接落库的四层结构 task_data。

- 诊断依据（diagnosis）= 纯 DB 聚合：UserProfile（掌握度/薄弱点/常见错误/平均正确率）
  + 班级位置（同课程学生 avg_correct_rate 百分位），无 LLM 参与
- 目标 / 题组 / 验收（goal/groups/acceptance）= 一次 LLM 结构化输出
  （prompt 输入诊断依据 + 课程知识点候选，约束输出严格 JSON）
- LLM 缺失 / 解析失败 / 结构非法 → 确定性 fallback：基于学生薄弱点或课程
  核心知识点构造可布置的入门任务，保证无 key 环境也能产出草稿

被 api/tasks.py POST /tasks/draft 调用（同步等待）。与诊断/复习 skill 同层，
可独立单测（mock LLM 或 mock DB）。
"""
import json
import logging
from uuid import UUID

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.config import settings
from coursepilot.models import Course, KnowledgePoint, UserProfile

logger = logging.getLogger(__name__)

TASK_SYSTEM = """你是一名有经验的任课教师，负责给指定学生布置个性化 AI 训练任务。

依据学生的学情诊断（掌握度/薄弱点/常见错误/正确率/班级位置）与课程知识点清单，
设计一份结构化任务。必须严格输出 JSON，不要输出任何多余文字，格式：

{
  "goal": {
    "metric": "目标指标，如 practice_correct_rate / practice_completion",
    "description": "一句话可度量目标，如：两周内将「二叉树遍历」正确率从 40% 提升到 75%"
  },
  "groups": [
    {
      "kp_path": "知识点路径（必须从候选清单里选）",
      "kp_name": "知识点名称",
      "question_count": 5,
      "difficulty": 2,
      "source": "出处，如 课程讲义/真题 或留空",
      "reason": "布置理由，如：该知识点正确率仅 45%"
    }
  ],
  "time_limit_minutes": 60,
  "acceptance": {
    "pass_condition": "验收阈值，如：题组平均正确率 ≥ 70%",
    "fallback_action": "未达标时的后续动作，如：系统追加一轮同知识点巩固练习"
  }
}

规则：
1. groups 聚焦学生的薄弱知识点（diagnosis.weak_kps）；学生无数据时聚焦课程核心知识点。
2. question_count 每题组 3~15；difficulty 取值 1(易)~5(难)，优先 2~4。
3. groups 建议 2~4 组，总量控制在 10~30 题，别贪多。
4. goal 必须可度量、可事后验证，禁止空泛表述。
5. kp_path 只能从候选清单选择，不要编造。
"""

# fallback 的题组上限（防 LLM 乱给数量导致任务失控）
MAX_GROUPS = 8
MAX_PER_GROUP = 20
MAX_TOTAL = 60


async def generate_task(
    session: AsyncSession,
    course_id: str,
    student_id: str,
) -> tuple[dict, dict]:
    """生成四层任务结构（不落库）。

    Args:
        session: 数据库会话
        course_id: 课程 UUID 字符串
        student_id: 目标学生 UUID 字符串

    Returns:
        (task_data, token_info)
        task_data: {
            "diagnosis": {...},          # 四层-1（DB 聚合，始终非空 dict）
            "goal": {...},               # 四层-2
            "groups": [...],             # 四层-3
            "total_count": int,
            "time_limit_minutes": int | None,
            "acceptance": {...},         # 四层-4
            "has_profile": bool,         # 是否有画像依据（供 API 展示/落库参考）
        }
        token_info: {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
    """
    empty_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # 1. 课程上下文 + 知识点候选（任务"出处"的合法集）
    course = await _load_course(session, course_id)
    course_kps = await _load_course_kps(session, course_id, limit=40)
    kp_candidates = [
        {"kp_path": kp.kp_path, "kp_name": kp.title} for kp in course_kps
    ]

    # 2. 诊断依据（纯 DB 聚合，无 LLM）
    diagnosis, has_profile = await _collect_diagnosis(session, student_id, course_id)

    # 3. 目标/题组/验收：优先 LLM，失败回退确定性构造
    if settings.llm_api_key and (has_profile or kp_candidates):
        llm_task, token_info = await _generate_via_llm(
            course_name=course.name if course else "",
            diagnosis=diagnosis,
            has_profile=has_profile,
            kp_candidates=kp_candidates,
        )
        if llm_task is not None:
            task_data = {**llm_task, "diagnosis": diagnosis, "has_profile": has_profile}
            return task_data, token_info
        logger.warning("generate_task: LLM 输出非法/解析失败，回退确定性构造")

    fallback_task = _make_fallback(diagnosis, kp_candidates, course_kps)
    fallback_task["diagnosis"] = diagnosis
    fallback_task["has_profile"] = has_profile
    return fallback_task, empty_tokens


# ── 诊断依据：DB 聚合 ─────────────────────────────────────────


async def _load_course(session: AsyncSession, course_id: str) -> Course | None:
    try:
        result = await session.execute(
            select(Course).where(Course.id == UUID(course_id))
        )
        return result.scalar_one_or_none()
    except Exception:
        logger.exception("generate_task: 加载课程失败 course=%s", course_id)
        return None


async def _load_course_kps(
    session: AsyncSession, course_id: str, limit: int = 40,
) -> list[KnowledgePoint]:
    """按 sort_order 取课程知识点（任务题组的候选池）"""
    try:
        result = await session.execute(
            select(KnowledgePoint)
            .where(KnowledgePoint.course_id == UUID(course_id))
            .order_by(KnowledgePoint.sort_order)
            .limit(limit)
        )
        return list(result.scalars().all())
    except Exception:
        logger.exception("generate_task: 加载课程知识点失败 course=%s", course_id)
        return []


async def _collect_diagnosis(
    session: AsyncSession, student_id: str, course_id: str,
) -> tuple[dict, bool]:
    """聚合学生画像 + 班级位置。

    Returns:
        (diagnosis, has_profile)：无画像时返回空结构 + False，
        由调用方决定基于课程 KP 生成任务（新生也能被布置）。
    """
    base = {
        "mastery_level": {},
        "weak_kps": [],
        "common_mistakes": [],
        "avg_correct_rate": None,
        "class_rank": None,
    }
    try:
        result = await session.execute(
            select(UserProfile).where(
                UserProfile.user_id == UUID(student_id),
                UserProfile.course_id == UUID(course_id),
            )
        )
        profile = result.scalar_one_or_none()
    except Exception:
        logger.exception("generate_task: 加载学生画像失败 student=%s", student_id)
        return base, False
    if profile is None:
        return base, False

    avg = float(profile.avg_correct_rate) if profile.avg_correct_rate else None
    diagnosis = {
        "mastery_level": profile.mastery_level or {},
        "weak_kps": profile.weak_kps or [],
        "common_mistakes": profile.common_mistakes or [],
        "avg_correct_rate": avg,
        "class_rank": None,
    }
    if avg is not None:
        diagnosis["class_rank"] = await _class_rank(session, course_id, avg)
    return diagnosis, True


async def _class_rank(
    session: AsyncSession, course_id: str, my_avg: float,
) -> str | None:
    """班级位置：同课程已计算画像学生的 avg_correct_rate 分布百分位"""
    try:
        result = await session.execute(
            select(UserProfile.avg_correct_rate).where(
                UserProfile.course_id == UUID(course_id),
                UserProfile.avg_correct_rate.is_not(None),
            )
        )
        values = [float(v) for v in result.scalars().all() if v is not None]
        if len(values) < 3:
            return None
        above = sum(1 for v in values if v > my_avg)
        # 高于当前学生的比例；50% 以下说明高于多数同学
        pct_above = above / len(values)
        if pct_above <= 0.5:
            return f"高于 {round((1 - pct_above) * 100)}% 的同学"
        return f"低于 {round(pct_above * 100)}% 的同学"
    except Exception:
        logger.exception("generate_task: 班级位置计算失败")
        return None


# ── LLM 生成 ─────────────────────────────────────────────────


async def _generate_via_llm(
    course_name: str,
    diagnosis: dict,
    has_profile: bool,
    kp_candidates: list[dict],
) -> tuple[dict | None, dict]:
    """一次 LLM 调用生成 goal/groups/acceptance；结构非法返回 (None, tokens)。"""
    empty_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout,
    )

    if has_profile:
        stats_lines = [
            f"平均正确率：{diagnosis['avg_correct_rate']:.0%}"
            if diagnosis.get("avg_correct_rate") is not None else "平均正确率：暂无",
            f"班级位置：{diagnosis.get('class_rank') or '暂无'}",
        ]
        weak = "\n".join(diagnosis.get("weak_kps", [])) or "（无，可布置巩固拔高任务）"
        mistakes = json.dumps(diagnosis.get("common_mistakes", []), ensure_ascii=False)
        student_desc = (
            "学生学情：\n" + "\n".join(stats_lines)
            + f"\n薄弱知识点：\n{weak}"
            + f"\n常见错误模式：\n{mistakes}"
        )
    else:
        student_desc = "该学生尚无练习数据（新生），请基于课程核心知识点设计入门巩固任务。"

    kp_text = "\n".join(
        f"- {kp['kp_path']}（{kp['kp_name']}）" for kp in kp_candidates
    ) or "（课程暂无知识点）"
    user_content = (
        f"课程：{course_name or '未知'}\n\n{student_desc}\n\n"
        f"课程知识点候选清单：\n{kp_text}\n\n请输出任务 JSON。"
    )

    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": TASK_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        usage = response.usage
        token_info = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }
        raw = json.loads(response.choices[0].message.content)
    except Exception:
        logger.exception("generate_task: LLM 调用/解析异常")
        return None, empty_tokens

    cleaned = _sanitize_llm_output(raw, kp_candidates)
    if cleaned is None:
        logger.warning("generate_task: LLM 输出结构非法，丢弃")
        return None, token_info
    return cleaned, token_info


def _sanitize_llm_output(raw: dict, kp_candidates: list[dict]) -> dict | None:
    """结构白名单校验 + 数量封顶。非法返回 None（调用方回退确定性构造）。"""
    valid_kp_paths = {kp["kp_path"] for kp in kp_candidates}

    # goal
    goal = raw.get("goal")
    if not isinstance(goal, dict) or not goal.get("description"):
        return None
    goal = {
        "metric": str(goal.get("metric") or "practice_correct_rate"),
        "description": str(goal["description"]).strip()[:500],
    }

    # groups
    raw_groups = raw.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        return None
    groups: list[dict] = []
    for g in raw_groups[:MAX_GROUPS]:
        if not isinstance(g, dict):
            continue
        kp_path = str(g.get("kp_path") or "").strip()
        if not kp_path or (valid_kp_paths and kp_path not in valid_kp_paths):
            continue  # LLM 编造课程外知识点 → 丢组（至少留一个才有效）
        try:
            qty = max(1, min(int(g.get("question_count") or 3), MAX_PER_GROUP))
            diff = max(1, min(int(g.get("difficulty") or 3), 5))
        except (TypeError, ValueError):
            qty, diff = 3, 3
        groups.append({
            "kp_path": kp_path,
            "kp_name": str(g.get("kp_name") or kp_path.split("/")[-1])[:256],
            "question_count": qty,
            "difficulty": diff,
            "source": str(g.get("source") or "")[:64] or None,
            "reason": str(g.get("reason") or "")[:300] or None,
        })
    if not groups:
        return None
    total = min(sum(g["question_count"] for g in groups), MAX_TOTAL)

    # acceptance
    acc = raw.get("acceptance")
    if not isinstance(acc, dict) or not acc.get("pass_condition"):
        return None
    acceptance = {
        "pass_condition": str(acc["pass_condition"]).strip()[:300],
        "fallback_action": str(acc.get("fallback_action") or "").strip()[:300] or None,
    }

    # time limit（可选，非法置 None）
    try:
        t = int(raw.get("time_limit_minutes"))
        time_limit = t if 5 <= t <= 600 else None
    except (TypeError, ValueError):
        time_limit = None

    return {
        "goal": goal,
        "groups": groups,
        "total_count": total,
        "time_limit_minutes": time_limit,
        "acceptance": acceptance,
    }


# ── 确定性 fallback（无 LLM / 失败兜底） ─────────────────────


def _make_fallback(
    diagnosis: dict, kp_candidates: list[dict], course_kps: list[KnowledgePoint],
) -> dict:
    """基于学生薄弱点（或课程核心 KP）确定性构造可布置任务。"""
    weak_kps = diagnosis.get("weak_kps") or []
    if not weak_kps and course_kps:
        # 无薄弱数据：取课程前几个核心 KP 做入门巩固
        chosen = [
            {"kp_path": kp.kp_path, "kp_name": kp.title}
            for kp in course_kps[:5]
        ]
        reason = "课程核心知识点入门巩固"
    else:
        weak_set = set(weak_kps)
        # 优先取画像薄弱点；找不到对应课程 KP 时也允许直接以 kp_path 出组
        chosen = [
            {"kp_path": p, "kp_name": p.split("/")[-1]}
            for p in weak_kps[:5] if p in weak_set
        ]
        reason = "薄弱知识点专项突破"

    groups = []
    for kp in chosen:
        groups.append({
            "kp_path": kp["kp_path"],
            "kp_name": kp["kp_name"],
            "question_count": 5,
            "difficulty": 3,
            "source": "课程资料",
            "reason": reason,
        })
    total = sum(g["question_count"] for g in groups)
    kp_names = "、".join(g["kp_name"] for g in groups[:2])
    return {
        "goal": {
            "metric": "practice_correct_rate",
            "description": (
                f"完成 {kp_names} 专项训练并达到平均正确率 ≥ 60%"
                if kp_names else "完成课程基础训练并达到平均正确率 ≥ 60%"
            ),
        },
        "groups": groups,
        "total_count": total,
        "time_limit_minutes": None,
        "acceptance": {
            "pass_condition": "平均正确率 ≥ 60%",
            "fallback_action": "薄弱组重复一轮后重新诊断",
        },
    }
