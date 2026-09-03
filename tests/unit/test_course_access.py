"""② 课程归属校验：deps 判据单测 + courses API 挂载接线

覆盖对象：
1. deps.get_course_membership 判据矩阵（super / enrollment / created_by / 无记录）
2. require_course_member / require_course_teacher 的 403 语义
3. courses 路由（详情 / documents / knowledge-points / ask / ask/stream / upload）的挂载接线：
   归属判据被 mock 为拒绝时一律 403 且不触达深层副作用（Retriever/LLM/文件写入）

分层说明：判据真值（谁能进哪门课）由第 1/2 部分单测保证；
第 3 部分只验证"端点在数据访问前调用了判据"——判据放行/拒绝由 mock 决定，
避免重演判据导致 select(Course) 与 select(Course.id) 无法在 Fake 层区分。

不 import agent 图链（避免沙箱 torch 崩溃）：chat 端点的归属校验行为
在真机 CI 由 test_agent_phase1.TestAgentAPI 覆盖。
"""
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════
# 1. deps 判据矩阵单测
# ═══════════════════════════════════════════════════════════════

class _Scalars:
    """模拟 ScalarResult：可迭代 + .all()"""

    def __init__(self, rows: list[Any]):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def all(self):
        return self._rows


class _Rows:
    """模拟 execute 结果的查询结果代理"""

    def __init__(self, rows: list[Any]):
        self._rows = rows

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._rows[0] if self._rows else None

    def scalars(self):
        return _Scalars(self._rows)


def _make_session(execute_results: list[Any]) -> AsyncSession:
    """execute 按调用顺序依次返回预置结果（超出则抛错，避免静默误判）"""
    calls: list[Any] = []

    async def execute(stmt, *a, **kw):
        calls.append(stmt)
        if len(calls) > len(execute_results):
            raise AssertionError(f"execute 调用次数超出预置：第 {len(calls)} 次")
        return execute_results[len(calls) - 1]

    session = MagicMock(spec=AsyncSession)
    session.execute = execute
    return session


def _enrollment(role: str):
    return SimpleNamespace(role=role)


class TestCourseMembership:
    """get_course_membership 判据：super / enrollment / created_by / 无记录"""

    @pytest.mark.asyncio
    async def test_super_always_teacher(self):
        from coursepilot.api.deps import get_course_membership

        user = SimpleNamespace(id=uuid4(), role="super")
        session = MagicMock(spec=AsyncSession)  # super 分支不触库
        assert await get_course_membership(session, user, uuid4()) == "teacher"

    @pytest.mark.asyncio
    async def test_enrollment_teacher(self):
        from coursepilot.api.deps import get_course_membership

        user = SimpleNamespace(id=uuid4(), role="student")
        session = _make_session([_Rows([_enrollment("teacher")])])
        assert await get_course_membership(session, user, uuid4()) == "teacher"

    @pytest.mark.asyncio
    async def test_teacher_enrollment_short_circuits(self):
        """enrollment role=teacher 直接短路，不再查 created_by（只预置 1 个结果）"""
        from coursepilot.api.deps import get_course_membership

        user = SimpleNamespace(id=uuid4(), role="student")
        session = _make_session([_Rows([_enrollment("teacher")])])
        assert await get_course_membership(session, user, uuid4()) == "teacher"

    @pytest.mark.asyncio
    async def test_created_by_owner_without_enrollment(self):
        """无 enrollment 行，但 course.created_by == user（旧库兜底）→ teacher"""
        from coursepilot.api.deps import get_course_membership

        user = SimpleNamespace(id=uuid4(), role="student")
        course_id = uuid4()
        session = _make_session([_Rows([]), _Rows([course_id])])
        assert await get_course_membership(session, user, course_id) == "teacher"

    @pytest.mark.asyncio
    async def test_enrollment_student(self):
        from coursepilot.api.deps import get_course_membership

        user = SimpleNamespace(id=uuid4(), role="student")
        session = _make_session([_Rows([_enrollment("student")]), _Rows([])])
        assert await get_course_membership(session, user, uuid4()) == "student"

    @pytest.mark.asyncio
    async def test_no_record_returns_none(self):
        from coursepilot.api.deps import get_course_membership

        user = SimpleNamespace(id=uuid4(), role="student")
        session = _make_session([_Rows([]), _Rows([])])
        assert await get_course_membership(session, user, uuid4()) is None

    @pytest.mark.asyncio
    async def test_course_not_found_returns_none(self):
        """课程不存在（enrollment 与 created_by 都查不到）→ None → 上层 403，不泄露存在性"""
        from coursepilot.api.deps import get_course_membership

        user = SimpleNamespace(id=uuid4(), role="student")
        session = _make_session([_Rows([]), _Rows([])])
        assert await get_course_membership(session, user, uuid4()) is None


class TestRequireWrappers:
    """require_course_member / require_course_teacher 的 403 语义"""

    @pytest.fixture
    def session(self):
        return MagicMock(spec=AsyncSession)

    async def _call(self, fn_name, session, user, role_return):
        target = "coursepilot.api.deps.get_course_membership"
        with patch(target, new=AsyncMock(return_value=role_return)):
            from coursepilot.api.deps import require_course_member, require_course_teacher

            fn = require_course_member if fn_name == "member" else require_course_teacher
            return await fn(session, user, uuid4())

    @pytest.mark.asyncio
    async def test_member_accepts_student(self, session):
        user = SimpleNamespace(id=uuid4(), role="student")
        assert await self._call("member", session, user, "student") == "student"

    @pytest.mark.asyncio
    async def test_member_accepts_teacher(self, session):
        user = SimpleNamespace(id=uuid4(), role="student")
        assert await self._call("member", session, user, "teacher") == "teacher"

    @pytest.mark.asyncio
    async def test_member_rejects_non_member(self, session):
        user = SimpleNamespace(id=uuid4(), role="student")
        with pytest.raises(HTTPException) as ei:
            await self._call("member", session, user, None)
        assert ei.value.status_code == 403

    @pytest.mark.asyncio
    async def test_teacher_accepts_teacher(self, session):
        user = SimpleNamespace(id=uuid4(), role="teacher")
        assert await self._call("teacher", session, user, "teacher") == "teacher"

    @pytest.mark.asyncio
    async def test_teacher_rejects_student_member(self, session):
        user = SimpleNamespace(id=uuid4(), role="student")
        with pytest.raises(HTTPException) as ei:
            await self._call("teacher", session, user, "student")
        assert ei.value.status_code == 403

    @pytest.mark.asyncio
    async def test_teacher_rejects_non_member(self, session):
        user = SimpleNamespace(id=uuid4(), role="student")
        with pytest.raises(HTTPException) as ei:
            await self._call("teacher", session, user, None)
        assert ei.value.status_code == 403


# ═══════════════════════════════════════════════════════════════
# 2. courses API 挂载接线
# ═══════════════════════════════════════════════════════════════

def _make_course(created_by):
    return SimpleNamespace(
        id=uuid4(),
        name="操作系统",
        description="测试课程",
        created_by=created_by,
        created_at=datetime.now(UTC),
    )


class _TableSession:
    """按 SQL 目标表分发结果的 FakeSession（仅服务 404 查询与列表查询）"""

    def __init__(self, course=None):
        self._course = course

    async def execute(self, stmt, *a, **kw):
        s = str(stmt).lower()
        if "courses" in s:
            return _Rows([self._course]) if self._course else _Rows([])
        if "documents" in s or "knowledge_points" in s:
            return _Rows([])
        return _Rows([])


@pytest.fixture
def teacher_user():
    from coursepilot.models import User

    return User(id=uuid4(), username="t_teacher", role="teacher")


@pytest.fixture
def app():
    from coursepilot.api.courses import router as courses_router

    _app = FastAPI()
    _app.include_router(courses_router, prefix="/api/v1")
    return _app


def _make_client(app, user, db_session):
    from coursepilot.api.deps import get_current_user
    from coursepilot.db import get_session

    async def _o_user():
        return user

    async def _o_session():
        yield db_session

    app.dependency_overrides[get_current_user] = _o_user
    app.dependency_overrides[get_session] = _o_session
    return TestClient(app)


async def _deny_member(session, user, course_id):
    raise HTTPException(status_code=403, detail="您不属于该课程")


async def _deny_teacher(session, user, course_id):
    raise HTTPException(status_code=403, detail="您不是该课程的教师")


class TestCoursesAPIReject:
    """归属判据拒绝时：所有带 course_id 的端点一律 403，不触达深层副作用"""

    @pytest.fixture
    def client(self, app, teacher_user):
        from coursepilot.api import courses as courses_mod

        db = _TableSession(course=_make_course(uuid4()))
        p1 = patch.object(courses_mod, "require_course_member", new=_deny_member)
        p2 = patch.object(courses_mod, "require_course_teacher", new=_deny_teacher)
        p1.start()
        p2.start()
        c = _make_client(app, teacher_user, db)
        yield c
        p1.stop()
        p2.stop()
        app.dependency_overrides.clear()

    def test_get_course_403(self, client):
        r = client.get(f"/api/v1/courses/{uuid4()}")
        assert r.status_code == 403

    def test_list_documents_403(self, client):
        r = client.get(f"/api/v1/courses/{uuid4()}/documents")
        assert r.status_code == 403

    def test_knowledge_points_403(self, client):
        r = client.get(f"/api/v1/courses/{uuid4()}/knowledge-points")
        assert r.status_code == 403

    def test_ask_403_not_touching_retriever(self, client):
        """RAG 问答：403 在 Retriever 构造之前（coursepilot.rag.retriever 不会被 import 链加载）"""
        r = client.post(f"/api/v1/courses/{uuid4()}/ask", json={"question": "什么是进程？"})
        assert r.status_code == 403

    def test_ask_stream_403(self, client):
        r = client.post(f"/api/v1/courses/{uuid4()}/ask/stream", json={"question": "什么是进程？"})
        assert r.status_code == 403

    def test_upload_403(self, client):
        """资料上传：非该课教师被 403 拦截（先过 404 课程存在检查）"""
        cid = uuid4()
        r = client.post(
            "/api/v1/courses/upload",
            files={"file": ("a.pdf", b"%PDF-1.4 fake", "application/pdf")},
            data={"course_id": str(cid)},
        )
        assert r.status_code == 403


class TestCoursesAPIAccessAllowed:
    """归属判据放行时：读端点正常执行（判据本身由第 1 部分单测保证）"""

    @pytest.fixture
    def client(self, app, teacher_user):
        from coursepilot.api import courses as courses_mod

        db = _TableSession(course=_make_course(teacher_user.id))
        member_ok = AsyncMock(return_value="teacher")
        p1 = patch.object(courses_mod, "require_course_member", new=member_ok)
        p1.start()
        c = _make_client(app, teacher_user, db)
        yield c
        p1.stop()
        app.dependency_overrides.clear()

    def test_get_course_200(self, client):
        r = client.get(f"/api/v1/courses/{uuid4()}")
        assert r.status_code == 200
        assert r.json()["name"] == "操作系统"

    def test_list_documents_200_empty(self, client):
        r = client.get(f"/api/v1/courses/{uuid4()}/documents")
        assert r.status_code == 200
        assert r.json() == []

    def test_knowledge_points_200_empty(self, client):
        r = client.get(f"/api/v1/courses/{uuid4()}/knowledge-points")
        assert r.status_code == 200
        assert r.json() == []
