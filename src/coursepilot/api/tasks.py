"""Tasks API：教师给指定学生发布 AI 任务（⑤）

两阶段（无 interrupt）：POST /draft 同步生成草稿落库 → 教师审核/编辑 → publish。
学生端只能看到发布给自己的任务（published），归属校验以 enrollments 为判据
（C3 选课表，teacher=课程创建者/student=候选名单）。

权限（rbac）：task:assign(teacher) 生成草稿与选学生候选；task:publish(teacher)
发布；task:view(student+) 查看。课程归属双判据兼容：enrollments role=teacher
或 course.created_by（防老库 enrollments 未回填时误拦）。
"""
import logging
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.api.deps import get_current_user
from coursepilot.db import get_session
from coursepilot.models import Course, Enrollment, Task, User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tasks", tags=["tasks"])

# 允许学生端读取的状态（草稿对学生不可见）
STUDENT_VISIBLE_STATUSES = ("published",)


# ── Request / Response Models ─────────────────────────────────

class TaskGroup(BaseModel):
    """题组编辑单元（PUT 时校验结构）"""
    kp_path: str = Field(..., min_length=1, max_length=512)
    kp_name: str = Field("", max_length=256)
    question_count: int = Field(..., ge=1, le=50)
    difficulty: int = Field(default=3, ge=1, le=5)
    source: str | None = Field(None, max_length=64)
    reason: str | None = Field(None, max_length=300)

class DraftRequest(BaseModel):
    course_id: str = Field(..., description="课程 UUID")
    student_id: str = Field(..., description="目标学生 UUID")

class DraftUpdate(BaseModel):
    """字段级编辑：收到哪层改哪层（exclude_unset 语义）。

    time_limit_minutes 传 null 表示清除时限。
    """
    goal: dict | None = None
    groups: list[TaskGroup] | None = None
    time_limit_minutes: int | None = Field(default=None, ge=5, le=600)
    acceptance: dict | None = None

class TaskCandidate(BaseModel):
    user_id: str
    username: str

class TaskListItem(BaseModel):
    id: str
    course_id: str
    student_id: str
    status: str
    goal: dict
    total_count: int
    time_limit_minutes: int | None
    created_at: str
    updated_at: str
    published_at: str | None

class TaskDetail(TaskListItem):
    created_by: str
    diagnosis: dict
    groups: list[dict]
    acceptance: dict


# ── Helpers ────────────────────────────────────────────────────

def _parse_uuid(raw: str, what: str) -> UUID:
    try:
        return UUID(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{what} 不是合法的 UUID") from None


async def _is_course_teacher(
    session: AsyncSession, user: User, course_id: UUID,
) -> bool:
    """课程归属判据：super 或 enrollments role=teacher 或 course.created_by。

    后两者取或：enrollments 是 C3 起的权威判据，created_by 兜底兼容
    尚未回填 enrollments 的旧库。
    """
    if user.role == "super":
        return True
    enr = await session.execute(
        select(Enrollment).where(
            Enrollment.user_id == user.id,
            Enrollment.course_id == course_id,
            Enrollment.role == "teacher",
        )
    )
    if enr.scalar_one_or_none() is not None:
        return True
    course = await session.execute(
        select(Course).where(Course.id == course_id)
    )
    c = course.scalar_one_or_none()
    return bool(c and c.created_by == user.id)


async def _is_course_student(
    session: AsyncSession, user_id: UUID, course_id: UUID,
) -> bool:
    """目标学生是否属于该课程（enrollments role=student 名单）"""
    enr = await session.execute(
        select(Enrollment).where(
            Enrollment.user_id == user_id,
            Enrollment.course_id == course_id,
            Enrollment.role == "student",
        )
    )
    return enr.scalar_one_or_none() is not None


async def _load_task_for_manage(
    db_session: AsyncSession, current_user: User, task_id: str,
) -> Task:
    """加载任务并校验管理权（创建者本人或 super）。"""
    tid = _parse_uuid(task_id, "任务 ID")
    result = await db_session.execute(select(Task).where(Task.id == tid))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.created_by != current_user.id and current_user.role != "super":
        raise HTTPException(status_code=403, detail="只能管理自己创建的任务")
    return task


def _group_to_dict(g: TaskGroup) -> dict:
    return {
        "kp_path": g.kp_path,
        "kp_name": g.kp_name or g.kp_path.split("/")[-1],
        "question_count": g.question_count,
        "difficulty": g.difficulty,
        "source": g.source,
        "reason": g.reason,
    }


def _to_detail(task: Task) -> TaskDetail:
    base = {
        "id": str(task.id),
        "course_id": str(task.course_id),
        "student_id": str(task.student_id),
        "status": task.status,
        "goal": task.goal or {},
        "total_count": task.total_count,
        "time_limit_minutes": task.time_limit_minutes,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "published_at": task.published_at.isoformat() if task.published_at else None,
    }
    return TaskDetail(
        **base,
        created_by=str(task.created_by),
        diagnosis=task.diagnosis or {},
        groups=task.groups or [],
        acceptance=task.acceptance or {},
    )


def _to_list_item(task: Task) -> TaskListItem:
    return TaskListItem(
        id=str(task.id),
        course_id=str(task.course_id),
        student_id=str(task.student_id),
        status=task.status,
        goal=task.goal or {},
        total_count=task.total_count,
        time_limit_minutes=task.time_limit_minutes,
        created_at=task.created_at.isoformat(),
        updated_at=task.updated_at.isoformat(),
        published_at=task.published_at.isoformat() if task.published_at else None,
    )


# ── Endpoints（路径声明顺序：静态段先于 /{task_id}） ─────────

@router.get("")
async def list_tasks(
    course_id: str | None = None,
    status_filter: str | None = None,
    current_user: User = Depends(get_current_user),
    db_session: AsyncSession = Depends(get_session),
) -> list[TaskListItem]:
    """任务列表（角色分流）：

    - 教师（task:view + 非 super）：自己创建的任务（含草稿）
    - super：全部
    - 学生：发布给自己的任务（published）
    可选 course_id / status 过滤。
    """
    from coursepilot.governance.rbac import has_permission
    if not has_permission(current_user.role, "task:view"):
        raise HTTPException(status_code=403, detail="无权限")

    query = select(Task)
    if current_user.role == "super":
        pass  # 全部课程任务
    elif current_user.role in ("teacher",):
        query = query.where(Task.created_by == current_user.id)
    else:  # student
        query = query.where(
            Task.student_id == current_user.id,
            Task.status.in_(STUDENT_VISIBLE_STATUSES),
        )

    if course_id:
        cid = _parse_uuid(course_id, "课程 ID")
        # 学生带 course_id 时也是自己的任务过滤条件，不构成越权
        query = query.where(Task.course_id == cid)
    if status_filter:
        if status_filter not in ("draft", "published"):
            raise HTTPException(status_code=400, detail="status 仅支持 draft/published")
        query = query.where(Task.status == status_filter)

    result = await db_session.execute(
        query.order_by(Task.updated_at.desc()).limit(100)
    )
    return [_to_list_item(t) for t in result.scalars().all()]


@router.get("/candidates")
async def list_candidates(
    course_id: str,
    current_user: User = Depends(get_current_user),
    db_session: AsyncSession = Depends(get_session),
) -> list[TaskCandidate]:
    """教师选人名单：该课程 enrollments role=student 的学生。

    相比从 UserProfile 反查，这里能覆盖"没做过练习的新生"——
    正是 C3 建 enrollments 的原因。
    """
    from coursepilot.governance.rbac import has_permission
    if not has_permission(current_user.role, "task:assign"):
        raise HTTPException(status_code=403, detail="无权限")

    cid = _parse_uuid(course_id, "课程 ID")
    if not await _is_course_teacher(db_session, current_user, cid):
        raise HTTPException(status_code=403, detail="您不是该课程的教师")

    result = await db_session.execute(
        select(Enrollment, User.username)
        .join(User, User.id == Enrollment.user_id)
        .where(
            Enrollment.course_id == cid,
            Enrollment.role == "student",
        )
        .order_by(User.username)
    )
    return [
        TaskCandidate(user_id=str(enr.user_id), username=username)
        for enr, username in result.all()
    ]


@router.post("/draft", status_code=201)
async def create_draft(
    request: DraftRequest,
    current_user: User = Depends(get_current_user),
    db_session: AsyncSession = Depends(get_session),
) -> TaskDetail:
    """教师给指定学生生成任务草稿（同步：等待生成完成后返回卡片）。"""
    from coursepilot.governance.rbac import has_permission
    if not has_permission(current_user.role, "task:assign"):
        raise HTTPException(status_code=403, detail="无权限")

    cid = _parse_uuid(request.course_id, "课程 ID")
    sid = _parse_uuid(request.student_id, "学生 ID")
    if not await _is_course_teacher(db_session, current_user, cid):
        raise HTTPException(status_code=403, detail="您不是该课程的教师")
    if not await _is_course_student(db_session, sid, cid):
        raise HTTPException(
            status_code=400, detail="目标学生不属于该课程（enrollments 无 student 记录）"
        )

    from coursepilot.agent.skills.generate_task import generate_task

    task_data, _token_info = await generate_task(
        session=db_session,
        course_id=str(cid),
        student_id=str(sid),
    )

    task = Task(
        course_id=cid,
        student_id=sid,
        created_by=current_user.id,
        status="draft",
        diagnosis=task_data.get("diagnosis") or {},
        goal=task_data.get("goal") or {},
        groups=task_data.get("groups") or [],
        total_count=task_data.get("total_count", 0),
        time_limit_minutes=task_data.get("time_limit_minutes"),
        acceptance=task_data.get("acceptance") or {},
    )
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)
    return _to_detail(task)


@router.get("/{task_id}")
async def get_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db_session: AsyncSession = Depends(get_session),
) -> TaskDetail:
    """任务详情。

    - 教师：自己创建的（含 draft）；super：任意
    - 学生：只读发布给自己的（published）
    """
    from coursepilot.governance.rbac import has_permission
    if not has_permission(current_user.role, "task:view"):
        raise HTTPException(status_code=403, detail="无权限")

    tid = _parse_uuid(task_id, "任务 ID")
    result = await db_session.execute(select(Task).where(Task.id == tid))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    if current_user.role == "super":
        return _to_detail(task)

    if current_user.role == "teacher":
        if task.created_by != current_user.id:
            raise HTTPException(status_code=404, detail="任务不存在")
        return _to_detail(task)

    # student：仅本人 + 已发布
    if task.student_id != current_user.id or task.status not in STUDENT_VISIBLE_STATUSES:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _to_detail(task)


@router.put("/{task_id}")
async def update_draft(
    task_id: str,
    request: DraftUpdate,
    current_user: User = Depends(get_current_user),
    db_session: AsyncSession = Depends(get_session),
) -> TaskDetail:
    """字段级编辑草稿（诊断依据不可改——它是教师判断 AI 是否瞎编的依据）。

    提供哪些字段改哪些（exclude_unset）；groups 更新后重算 total_count；
    time_limit_minutes 传 null 清除时限。仅 draft 状态可编辑。
    """
    from coursepilot.governance.rbac import has_permission
    if not has_permission(current_user.role, "task:assign"):
        raise HTTPException(status_code=403, detail="无权限")

    task = await _load_task_for_manage(db_session, current_user, task_id)
    if task.status != "draft":
        raise HTTPException(status_code=400, detail="仅草稿状态可编辑")

    payload = request.model_dump(exclude_unset=True)
    if "goal" in payload:
        goal = payload["goal"]
        if not isinstance(goal, dict) or not goal.get("description"):
            raise HTTPException(status_code=422, detail="goal 需含 metric 与 description")
        task.goal = {
            "metric": str(goal.get("metric") or "practice_correct_rate")[:64],
            "description": str(goal["description"]).strip()[:500],
        }
    if "groups" in payload:
        # 取模型实例列表而非 model_dump 出的 dict（避免 dict 无 kp_path 属性）
        groups = request.groups
        if not groups:
            raise HTTPException(status_code=422, detail="groups 至少保留一个题组")
        task.groups = [_group_to_dict(g) for g in groups]
        task.total_count = sum(g.question_count for g in groups)
    if "time_limit_minutes" in payload:
        task.time_limit_minutes = payload["time_limit_minutes"]
    if "acceptance" in payload:
        acc = payload["acceptance"]
        if not isinstance(acc, dict) or not acc.get("pass_condition"):
            raise HTTPException(status_code=422, detail="acceptance 需含 pass_condition")
        task.acceptance = {
            "pass_condition": str(acc["pass_condition"]).strip()[:300],
            "fallback_action": str(acc.get("fallback_action") or "").strip()[:300] or None,
        }

    await db_session.commit()
    await db_session.refresh(task)
    return _to_detail(task)


@router.post("/{task_id}/publish")
async def publish_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db_session: AsyncSession = Depends(get_session),
) -> TaskDetail:
    """发布草稿 → status=published，学生端立即可见。"""
    from coursepilot.governance.rbac import has_permission
    if not has_permission(current_user.role, "task:publish"):
        raise HTTPException(status_code=403, detail="无权限")

    task = await _load_task_for_manage(db_session, current_user, task_id)
    if task.status != "draft":
        raise HTTPException(status_code=400, detail="仅草稿状态可发布")
    if not task.groups:
        raise HTTPException(status_code=400, detail="任务无题组，无法发布")

    task.status = "published"
    task.published_at = datetime.now(UTC)
    await db_session.commit()
    await db_session.refresh(task)
    return _to_detail(task)
