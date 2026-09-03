"""⑤ 教师发布任务：task 生成器 + /tasks API 权限与状态机

测试策略：
- generator 纯函数（_sanitize_llm_output / _make_fallback）直接测
- generate_task 主流程：patch 三个 DB helper（_load_course/_load_course_kps/
  _collect_diagnosis），不走真库；LLM 走 patch AsyncOpenAI（对齐既有 skill 测试模式）
- API：TestClient + 仅挂 tasks router 的小 FastAPI app（不 import main 的
  agent 图链，规避沙箱 Torch import 崩溃面），override get_session/get_current_user；
  DB 交互用 FakeSession（python 层的可见性/状态机/权限断言有效，
  SQL 层过滤语义由代码审查保证——unit 层不连真库）
"""
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from coursepilot.api.deps import get_current_user, get_session
from coursepilot.api.tasks import router as tasks_router
from coursepilot.models import Task, User

# ═══════════════════════════════════════════════════════════
# Fakes
# ═══════════════════════════════════════════════════════════


class FakeResult:
    def __init__(self, scalar=None, rows=None):
        self._scalar = scalar
        self._rows = rows

    def scalar_one_or_none(self):
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        if self._rows is not None:
            return self._rows
        return [self._scalar] if self._scalar is not None else []

    def first(self):
        return self._scalar


class FakeSession:
    """execute 恒返回预设 result；add/commit/refresh 记录调用。"""

    def __init__(self, result: FakeResult | None = None):
        self._result = result or FakeResult()
        self.added: list = []
        self.commits = 0

    async def execute(self, *args, **kwargs):
        return self._result

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1

    async def refresh(self, obj):
        # 模拟 DB server_default：落库后 created_at/updated_at 才被填充
        now = datetime.now(UTC)
        for attr in ("created_at", "updated_at"):
            if getattr(obj, attr, None) is None:
                setattr(obj, attr, now)

    async def flush(self):
        pass


def make_user(role: str, uid: uuid.UUID | None = None) -> User:
    return User(
        id=uid or uuid.uuid4(),
        username=f"user_{role}",
        role=role,
        password_hash="x",
    )


def make_task(**overrides) -> Task:
    now = datetime.now(UTC)
    base = {
        "id": uuid.uuid4(),
        "course_id": uuid.uuid4(),
        "student_id": uuid.uuid4(),
        "created_by": uuid.uuid4(),
        "status": "draft",
        "diagnosis": {"weak_kps": ["OS/进程"]},
        "goal": {"metric": "practice_correct_rate", "description": "正确率提升到 70%"},
        "groups": [{
            "kp_path": "OS/进程", "kp_name": "进程管理",
            "question_count": 5, "difficulty": 3,
            "source": "课程资料", "reason": "薄弱",
        }],
        "total_count": 5,
        "time_limit_minutes": 60,
        "acceptance": {"pass_condition": "平均正确率 ≥ 70%", "fallback_action": "追加一轮"},
        "created_at": now,
        "updated_at": now,
        "published_at": None,
    }
    base.update(overrides)
    return Task(**base)


def make_completion(content: str, usage=None):
    m = MagicMock()
    m.choices = [MagicMock()]
    m.choices[0].message.content = content
    if usage is None:
        u = MagicMock()
        u.prompt_tokens = 10
        u.completion_tokens = 5
        u.total_tokens = 15
        usage = u
    m.usage = usage
    return m


@pytest.fixture
def task_app():
    app = FastAPI()
    app.include_router(tasks_router)
    return app


def make_client(app: FastAPI, user: User, session: FakeSession) -> TestClient:
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_session] = lambda: session
    return TestClient(app)


# ═══════════════════════════════════════════════════════════
# 1. _sanitize_llm_output（结构白名单 + 数量封顶）
# ═══════════════════════════════════════════════════════════

VALID_KPS = [
    {"kp_path": "OS/进程", "kp_name": "进程管理"},
    {"kp_path": "OS/内存", "kp_name": "内存管理"},
]


class TestSanitizeLLM:
    def test_valid_structure_ok(self):
        from coursepilot.agent.skills.generate_task import _sanitize_llm_output
        raw = {
            "goal": {"metric": "practice_correct_rate", "description": "提升到 70%"},
            "groups": [
                {"kp_path": "OS/进程", "kp_name": "进程管理", "question_count": 5,
                 "difficulty": 2, "source": "讲义", "reason": "薄弱"},
            ],
            "time_limit_minutes": 90,
            "acceptance": {"pass_condition": "≥70%", "fallback_action": "追加一轮"},
        }
        out = _sanitize_llm_output(raw, VALID_KPS)
        assert out is not None
        assert out["total_count"] == 5
        assert out["goal"]["metric"] == "practice_correct_rate"
        assert out["groups"][0]["kp_path"] == "OS/进程"
        assert out["time_limit_minutes"] == 90

    def test_fabricated_kp_dropped(self):
        from coursepilot.agent.skills.generate_task import _sanitize_llm_output
        raw = {
            "goal": {"metric": "x", "description": "目标"},
            "groups": [
                {"kp_path": "虚构知识点/不存在", "question_count": 5},
                {"kp_path": "OS/进程", "question_count": 3},
            ],
            "acceptance": {"pass_condition": "≥60%"},
        }
        out = _sanitize_llm_output(raw, VALID_KPS)
        # 课程外知识点组被丢，剩课程内一组
        assert out is not None
        assert len(out["groups"]) == 1
        assert out["groups"][0]["kp_path"] == "OS/进程"

    def test_all_groups_invalid_none(self):
        from coursepilot.agent.skills.generate_task import _sanitize_llm_output
        raw = {
            "goal": {"metric": "x", "description": "目标"},
            "groups": [{"kp_path": "编造的", "question_count": 5}],
            "acceptance": {"pass_condition": "≥60%"},
        }
        assert _sanitize_llm_output(raw, VALID_KPS) is None

    def test_empty_groups_none(self):
        from coursepilot.agent.skills.generate_task import _sanitize_llm_output
        raw = {"goal": {"description": "x"}, "groups": [], "acceptance": {"pass_condition": "x"}}
        assert _sanitize_llm_output(raw, VALID_KPS) is None

    def test_quantity_capped(self):
        from coursepilot.agent.skills.generate_task import _sanitize_llm_output
        raw = {
            "goal": {"description": "目标"},
            "groups": [
                {"kp_path": "OS/进程", "question_count": 999, "difficulty": 9},
            ],
            "acceptance": {"pass_condition": "≥60%"},
        }
        out = _sanitize_llm_output(raw, VALID_KPS)
        assert out["groups"][0]["question_count"] == 20   # MAX_PER_GROUP
        assert out["groups"][0]["difficulty"] == 5        # 难度钳制到 1~5

    def test_missing_acceptance_none(self):
        from coursepilot.agent.skills.generate_task import _sanitize_llm_output
        raw = {
            "goal": {"description": "目标"},
            "groups": [{"kp_path": "OS/进程", "question_count": 5}],
        }
        assert _sanitize_llm_output(raw, VALID_KPS) is None


# ═══════════════════════════════════════════════════════════
# 2. _make_fallback（确定性构造）
# ═══════════════════════════════════════════════════════════

class TestMakeFallback:
    def test_weak_kps_priority(self):
        from coursepilot.agent.skills.generate_task import _make_fallback
        diagnosis = {"weak_kps": ["OS/进程", "OS/内存"], "mastery_level": {}}
        out = _make_fallback(diagnosis, VALID_KPS, [])
        assert len(out["groups"]) == 2
        assert all(g["reason"] == "薄弱知识点专项突破" for g in out["groups"])
        assert out["total_count"] == 10
        assert out["goal"]["metric"] == "practice_correct_rate"

    def test_no_weak_uses_course_kps(self):
        from coursepilot.agent.skills.generate_task import _make_fallback
        kps = [MagicMock(kp_path=f"KP/{i}", title=f"点{i}") for i in range(3)]
        out = _make_fallback({"weak_kps": []}, [{"kp_path": k.kp_path, "kp_name": k.title} for k in kps], kps)
        assert len(out["groups"]) == 3
        assert out["groups"][0]["reason"] == "课程核心知识点入门巩固"

    def test_no_kp_at_all(self):
        from coursepilot.agent.skills.generate_task import _make_fallback
        out = _make_fallback({"weak_kps": []}, [], [])
        assert out["groups"] == []
        assert out["total_count"] == 0


# ═══════════════════════════════════════════════════════════
# 3. generate_task 主流程
# ═══════════════════════════════════════════════════════════

DIAG_PROFILE = {
    "mastery_level": {"OS/进程": 0.4},
    "weak_kps": ["OS/进程"],
    "common_mistakes": [],
    "avg_correct_rate": 0.45,
    "class_rank": "低于 30% 的同学",
}


def _patch_db_helpers(weak_kps=None, kp_count=3, course_name="数据结构"):
    """返回 patch 上下文，mock 掉 generate_task 的三个 DB helper。"""
    course_kps = [MagicMock(kp_path=f"OS/K{i}", title=f"知识点{i}") for i in range(kp_count)]
    diag = dict(DIAG_PROFILE)
    if weak_kps is not None:
        diag["weak_kps"] = weak_kps
    return patch.multiple(
        "coursepilot.agent.skills.generate_task",
        _load_course=AsyncMock(return_value=MagicMock(name=course_name)),
        _load_course_kps=AsyncMock(return_value=course_kps),
        _collect_diagnosis=AsyncMock(return_value=(diag, True)),
    )


@pytest.mark.asyncio
async def test_generate_task_no_llm_key_falls_back():
    """无 LLM key → 确定性 fallback（token 0，weak 薄弱题组）。"""
    from coursepilot.agent.skills.generate_task import generate_task
    with _patch_db_helpers(weak_kps=["OS/K1"]):
        with patch("coursepilot.agent.skills.generate_task.settings.llm_api_key", ""):
            data, tokens = await generate_task(session=None, course_id=str(uuid.uuid4()), student_id=str(uuid.uuid4()))
    assert data["has_profile"] is True
    assert data["diagnosis"]["weak_kps"] == ["OS/K1"]
    assert data["total_count"] > 0
    assert all(g["reason"] == "薄弱知识点专项突破" for g in data["groups"])
    assert tokens == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@pytest.mark.asyncio
async def test_generate_task_llm_valid_output():
    """LLM 输出合法 → 结构保留 + token 记录。"""
    from coursepilot.agent.skills.generate_task import generate_task
    content = (
        '{"goal": {"metric": "practice_correct_rate", "description": "OS/K1 提升到 70%"},'
        ' "groups": [{"kp_path": "OS/K1", "kp_name": "知识点1", "question_count": 6,'
        ' "difficulty": 2, "source": "讲义", "reason": "正确率 45%"}],'
        ' "time_limit_minutes": 45,'
        ' "acceptance": {"pass_condition": "平均正确率 ≥ 70%", "fallback_action": "追加一轮"}}'
    )
    with _patch_db_helpers(weak_kps=["OS/K1"]):
        with patch("coursepilot.agent.skills.generate_task.settings.llm_api_key", "sk-test"):
            with patch("coursepilot.agent.skills.generate_task.AsyncOpenAI") as mock_openai:
                client = AsyncMock()
                mock_openai.return_value = client
                client.chat.completions.create = AsyncMock(
                    return_value=make_completion(content)
                )
                data, tokens = await generate_task(
                    session=None, course_id=str(uuid.uuid4()), student_id=str(uuid.uuid4()),
                )
    assert data["groups"][0]["question_count"] == 6
    assert data["goal"]["description"].startswith("OS/K1")
    assert data["time_limit_minutes"] == 45
    assert tokens["total_tokens"] == 15


@pytest.mark.asyncio
async def test_generate_task_llm_invalid_content_falls_back():
    """LLM 输出非 JSON → 丢弃并回退确定性构造（不抛异常）。"""
    from coursepilot.agent.skills.generate_task import generate_task
    with _patch_db_helpers(weak_kps=["OS/K1"]):
        with patch("coursepilot.agent.skills.generate_task.settings.llm_api_key", "sk-test"):
            with patch("coursepilot.agent.skills.generate_task.AsyncOpenAI") as mock_openai:
                client = AsyncMock()
                mock_openai.return_value = client
                client.chat.completions.create = AsyncMock(
                    return_value=make_completion("这不是 JSON{{{")
                )
                data, tokens = await generate_task(
                    session=None, course_id=str(uuid.uuid4()), student_id=str(uuid.uuid4()),
                )
    assert all(g["reason"] == "薄弱知识点专项突破" for g in data["groups"])
    assert tokens == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# ═══════════════════════════════════════════════════════════
# 4. /tasks API（权限 + 状态机）
# ═══════════════════════════════════════════════════════════

DRAFT_DATA = {
    "diagnosis": {"weak_kps": ["OS/K1"]},
    "goal": {"metric": "practice_correct_rate", "description": "提升到 70%"},
    "groups": [{"kp_path": "OS/K1", "kp_name": "K1", "question_count": 5,
                "difficulty": 2, "source": None, "reason": None}],
    "total_count": 5,
    "time_limit_minutes": 60,
    "acceptance": {"pass_condition": "≥70%", "fallback_action": None},
    "has_profile": True,
}


class TestDraftAPI:
    def test_draft_teacher_ok(self, task_app):
        teacher = make_user("teacher")
        student = make_user("student")
        session = FakeSession()
        with (
            patch("coursepilot.api.tasks._is_course_teacher", AsyncMock(return_value=True)),
            patch("coursepilot.api.tasks._is_course_student", AsyncMock(return_value=True)),
            patch("coursepilot.agent.skills.generate_task.generate_task",
                  AsyncMock(return_value=(DRAFT_DATA, {}))),
        ):
            client = make_client(task_app, teacher, session)
            resp = client.post("/tasks/draft", json={
                "course_id": str(uuid.uuid4()), "student_id": str(student.id),
            })
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["status"] == "draft"
        assert body["student_id"] == str(student.id)
        assert body["total_count"] == 5
        assert len(session.added) == 1
        assert isinstance(session.added[0], Task)
        assert session.added[0].created_by == teacher.id
        assert session.commits == 1

    def test_draft_not_course_teacher_403(self, task_app):
        teacher = make_user("teacher")
        session = FakeSession()
        with patch("coursepilot.api.tasks._is_course_teacher", AsyncMock(return_value=False)):
            client = make_client(task_app, teacher, session)
            resp = client.post("/tasks/draft", json={
                "course_id": str(uuid.uuid4()), "student_id": str(uuid.uuid4()),
            })
        assert resp.status_code == 403

    def test_draft_student_not_in_course_400(self, task_app):
        teacher = make_user("teacher")
        session = FakeSession()
        with (
            patch("coursepilot.api.tasks._is_course_teacher", AsyncMock(return_value=True)),
            patch("coursepilot.api.tasks._is_course_student", AsyncMock(return_value=False)),
        ):
            client = make_client(task_app, teacher, session)
            resp = client.post("/tasks/draft", json={
                "course_id": str(uuid.uuid4()), "student_id": str(uuid.uuid4()),
            })
        assert resp.status_code == 400

    def test_draft_student_role_forbidden(self, task_app):
        student = make_user("student")
        session = FakeSession()
        client = make_client(task_app, student, session)
        resp = client.post("/tasks/draft", json={
            "course_id": str(uuid.uuid4()), "student_id": str(uuid.uuid4()),
        })
        assert resp.status_code == 403  # task:assign 需要 teacher


class TestPublishAPI:
    def test_publish_owner_ok(self, task_app):
        teacher = make_user("teacher")
        task = make_task(created_by=teacher.id)
        session = FakeSession(FakeResult(scalar=task))
        client = make_client(task_app, teacher, session)
        resp = client.post(f"/tasks/{task.id}/publish")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "published"
        assert body["published_at"] is not None

    def test_publish_non_owner_403(self, task_app):
        teacher_a = make_user("teacher")
        teacher_b = make_user("teacher")
        task = make_task(created_by=teacher_a.id)
        session = FakeSession(FakeResult(scalar=task))
        client = make_client(task_app, teacher_b, session)
        resp = client.post(f"/tasks/{task.id}/publish")
        assert resp.status_code == 403

    def test_publish_empty_groups_400(self, task_app):
        teacher = make_user("teacher")
        task = make_task(created_by=teacher.id, groups=[], total_count=0)
        session = FakeSession(FakeResult(scalar=task))
        client = make_client(task_app, teacher, session)
        resp = client.post(f"/tasks/{task.id}/publish")
        assert resp.status_code == 400

    def test_publish_already_published_400(self, task_app):
        teacher = make_user("teacher")
        task = make_task(created_by=teacher.id, status="published",
                         published_at=datetime.now(UTC))
        session = FakeSession(FakeResult(scalar=task))
        client = make_client(task_app, teacher, session)
        resp = client.post(f"/tasks/{task.id}/publish")
        assert resp.status_code == 400

    def test_publish_by_student_403(self, task_app):
        student = make_user("student")
        task = make_task()
        session = FakeSession(FakeResult(scalar=task))
        client = make_client(task_app, student, session)
        resp = client.post(f"/tasks/{task.id}/publish")
        assert resp.status_code == 403


class TestDetailVisibility:
    def test_teacher_own_draft_ok(self, task_app):
        teacher = make_user("teacher")
        task = make_task(created_by=teacher.id, status="draft")
        session = FakeSession(FakeResult(scalar=task))
        client = make_client(task_app, teacher, session)
        resp = client.get(f"/tasks/{task.id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "draft"

    def test_teacher_other_draft_404(self, task_app):
        teacher_a = make_user("teacher")
        teacher_b = make_user("teacher")
        task = make_task(created_by=teacher_a.id, status="draft")
        session = FakeSession(FakeResult(scalar=task))
        client = make_client(task_app, teacher_b, session)
        resp = client.get(f"/tasks/{task.id}")
        assert resp.status_code == 404

    def test_student_own_published_ok(self, task_app):
        student = make_user("student")
        task = make_task(student_id=student.id, status="published",
                         published_at=datetime.now(UTC))
        session = FakeSession(FakeResult(scalar=task))
        client = make_client(task_app, student, session)
        resp = client.get(f"/tasks/{task.id}")
        assert resp.status_code == 200

    def test_student_own_draft_invisible(self, task_app):
        student = make_user("student")
        task = make_task(student_id=student.id, status="draft")
        session = FakeSession(FakeResult(scalar=task))
        client = make_client(task_app, student, session)
        resp = client.get(f"/tasks/{task.id}")
        assert resp.status_code == 404  # 草稿对学生不可见

    def test_student_others_published_404(self, task_app):
        student_a = make_user("student")
        student_b = make_user("student")
        task = make_task(student_id=student_a.id, status="published",
                         published_at=datetime.now(UTC))
        session = FakeSession(FakeResult(scalar=task))
        client = make_client(task_app, student_b, session)
        resp = client.get(f"/tasks/{task.id}")
        assert resp.status_code == 404

    def test_teacher_published_by_others_published_404_scope(self, task_app):
        """教师只能读自己创建的任务，即使发布也不能读同课他师任务。"""
        teacher_a = make_user("teacher")
        teacher_b = make_user("teacher")
        task = make_task(created_by=teacher_a.id, status="published",
                         published_at=datetime.now(UTC))
        session = FakeSession(FakeResult(scalar=task))
        client = make_client(task_app, teacher_b, session)
        resp = client.get(f"/tasks/{task.id}")
        assert resp.status_code == 404


class TestUpdateDraft:
    def test_update_goal_and_groups(self, task_app):
        teacher = make_user("teacher")
        task = make_task(created_by=teacher.id)
        session = FakeSession(FakeResult(scalar=task))
        client = make_client(task_app, teacher, session)
        resp = client.put(f"/tasks/{task.id}", json={
            "goal": {"metric": "practice_correct_rate", "description": "新目标到 80%"},
            "groups": [
                {"kp_path": "OS/K2", "kp_name": "K2", "question_count": 3,
                 "difficulty": 4, "source": "真题", "reason": "重新排"},
                {"kp_path": "OS/K1", "kp_name": "K1", "question_count": 7,
                 "difficulty": 2, "source": None, "reason": None},
            ],
            "time_limit_minutes": 120,
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["goal"]["description"] == "新目标到 80%"
        assert body["total_count"] == 10          # 3 + 7 重算
        assert body["time_limit_minutes"] == 120
        assert len(body["groups"]) == 2
        assert task.status == "draft"             # 编辑不改变状态

    def test_update_published_forbidden(self, task_app):
        teacher = make_user("teacher")
        task = make_task(created_by=teacher.id, status="published",
                         published_at=datetime.now(UTC))
        session = FakeSession(FakeResult(scalar=task))
        client = make_client(task_app, teacher, session)
        resp = client.put(f"/tasks/{task.id}", json={"goal": {"description": "x"}})
        assert resp.status_code == 400

    def test_update_invalid_goal_422(self, task_app):
        teacher = make_user("teacher")
        task = make_task(created_by=teacher.id)
        session = FakeSession(FakeResult(scalar=task))
        client = make_client(task_app, teacher, session)
        resp = client.put(f"/tasks/{task.id}", json={"goal": {"foo": "bar"}})
        assert resp.status_code == 422

    def test_update_clear_time_limit(self, task_app):
        teacher = make_user("teacher")
        task = make_task(created_by=teacher.id, time_limit_minutes=60)
        session = FakeSession(FakeResult(scalar=task))
        client = make_client(task_app, teacher, session)
        resp = client.put(f"/tasks/{task.id}", json={"time_limit_minutes": None})
        assert resp.status_code == 200
        assert resp.json()["time_limit_minutes"] is None


class TestListTasks:
    def test_teacher_lists_own(self, task_app):
        teacher = make_user("teacher")
        mine = make_task(created_by=teacher.id)
        session = FakeSession(FakeResult(rows=[mine]))
        client = make_client(task_app, teacher, session)
        resp = client.get("/tasks")
        assert resp.status_code == 200
        assert [t["id"] for t in resp.json()] == [str(mine.id)]

    def test_student_sees_published_only_smoke(self, task_app):
        """学生列表：python 层返回 fake 结果；published 过滤在 SQL 层（代码审查保证）。"""
        student = make_user("student")
        pub = make_task(student_id=student.id, status="published",
                        published_at=datetime.now(UTC))
        session = FakeSession(FakeResult(rows=[pub]))
        client = make_client(task_app, student, session)
        resp = client.get("/tasks")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_student_never_calls_teacher_list(self, task_app):
        """学生 token 不能以教师身份拉取草稿列表（SQL 按 student_id 过滤）。"""
        student = make_user("student")
        session = FakeSession(FakeResult(rows=[]))
        client = make_client(task_app, student, session)
        resp = client.get("/tasks?status_filter=draft")
        assert resp.status_code == 200

    def test_invalid_status_filter_400(self, task_app):
        teacher = make_user("teacher")
        session = FakeSession()
        client = make_client(task_app, teacher, session)
        resp = client.get("/tasks?status_filter=deleted")
        assert resp.status_code == 400
