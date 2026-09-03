"""回填 enrollments 选课表（幂等，可重复执行）

来源与角色：
  - teacher：courses.created_by（课程创建者）
  - student：user_profiles 按 (user_id, course_id) 去重（做过问答/练习即视为选课学生）

冲突语义（同一 user×course 双来源命中时 teacher 优先）：
  - teacher 行：ON CONFLICT DO UPDATE SET role='teacher' —— 可将已存在的
    student 行升级为 teacher，重复跑无副作用；
  - student 行：ON CONFLICT DO NOTHING —— 绝不覆盖 teacher 行。

用法：
    PYTHONPATH=src .venv/Scripts/python -m scripts.backfill_enrollments
        # dry-run：只统计不落库
    PYTHONPATH=src .venv/Scripts/python -m scripts.backfill_enrollments --execute
        # 真实回填（执行前后打印行数对比）
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from coursepilot.db import get_session_etx
from coursepilot.models import Course, Enrollment, UserProfile


async def _count(session, table) -> int:
    r = await session.execute(select(func.count()).select_from(table))
    return r.scalar() or 0


async def _gather_stats(session) -> dict:
    """统计回填来源与 enrollments 现状（只读）"""
    # student 来源：user_profiles 去重
    r = await session.execute(select(UserProfile.user_id, UserProfile.course_id).distinct())
    student_rows = [tuple(row) for row in r.fetchall()]
    # teacher 来源：courses.created_by（过滤 created_by 指向真实存在的用户）
    # 元组统一为 (user_id, course_id) 顺序，便于与 student 来源做交集冲突统计
    r = await session.execute(
        select(Course.id, Course.created_by)
        .where(Course.created_by.is_not(None))
    )
    teacher_rows = [(row[1], row[0]) for row in r.fetchall()]

    enrolled = await _count(session, Enrollment)
    r = await session.execute(
        select(Enrollment.role, func.count()).group_by(Enrollment.role)
    )
    role_dist = {row[0]: row[1] for row in r.fetchall()}

    return {
        "students": len(student_rows),
        "teachers": len(teacher_rows),
        "conflicts": len(set(student_rows) & set(teacher_rows)),
        "enrolled": enrolled,
        "role_dist": role_dist,
    }


async def _backfill(session) -> dict:
    """执行回填：teacher 先插（可升级），student 后插（不覆盖）。返回各类写入行数。"""
    # teacher 来源
    r = await session.execute(
        select(Course.id, Course.created_by).where(Course.created_by.is_not(None))
    )
    teacher_rows = [tuple(row) for row in r.fetchall()]
    if teacher_rows:
        stmt = pg_insert(Enrollment).values(
            [{"user_id": u, "course_id": c, "role": "teacher"} for c, u in teacher_rows]
        ).on_conflict_do_update(
            index_elements=[Enrollment.user_id, Enrollment.course_id],
            set_={"role": "teacher"},
        )
        res = await session.execute(stmt)
        teacher_count = res.rowcount if res.rowcount != -1 else len(teacher_rows)
    else:
        teacher_count = 0

    # student 来源（去重）
    r = await session.execute(select(UserProfile.user_id, UserProfile.course_id).distinct())
    student_rows = [tuple(row) for row in r.fetchall()]
    if student_rows:
        stmt = pg_insert(Enrollment).values(
            [{"user_id": u, "course_id": c, "role": "student"} for u, c in student_rows]
        ).on_conflict_do_nothing(
            index_elements=[Enrollment.user_id, Enrollment.course_id],
        )
        res = await session.execute(stmt)
        student_count = res.rowcount if res.rowcount != -1 else len(student_rows)
    else:
        student_count = 0

    await session.commit()
    return {"teacher": teacher_count, "student": student_count}


async def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="回填 enrollments 选课表（幂等）")
    parser.add_argument("--execute", action="store_true", help="真正回填；缺省仅 dry-run 统计")
    args = parser.parse_args()

    async with get_session_etx() as session:
        before = await _gather_stats(session)
        print("=" * 60)
        print("  enrollments 回填 — 回填前统计")
        print("=" * 60)
        print(f"  student 来源（user_profiles 去重）: {before['students']}")
        print(f"  teacher 来源（courses.created_by） : {before['teachers']}")
        print(f"  双来源冲突（teacher 优先）         : {before['conflicts']}")
        print(f"  enrollments 现有行数               : {before['enrolled']}")
        print(f"  现有 role 分布                     : {before['role_dist'] or {}}")

        if not args.execute:
            print("\n  [dry-run] 未落库。确认后加 --execute 执行。")
            return

        written = await _backfill(session)
        after = await _gather_stats(session)
        print(f"  ✅ 回填完成：teacher {written['teacher']} 行 / student {written['student']} 行")
        print(f"  enrollments 回填后行数: {after['enrolled']}（前 {before['enrolled']}）")
        print(f"  role 分布: {after['role_dist']}")
        print("  （后续重复执行幂等，行数不会继续增长）")


if __name__ == "__main__":
    asyncio.run(main())
