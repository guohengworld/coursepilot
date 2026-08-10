"""MCP 工具业务逻辑单元测试（按新 API 重写，P1-T4）。

覆盖 tools/{tutor,practice,knowledge}.py 的业务成功路径（mock DB / LLM），
与 tests/test_mcp/ 的协议级测试（test_gateway / test_stdio / test_schemas /
test_validation / test_errors）及租户断言测试（test_tenancy）互补：

- query_knowledge：RAG 问答成功 / 课程不存在
- get_kp_tree：空课程 / 返回知识点路径列表
- search_knowledge_units：检索并保持重排顺序
- diagnose / generate_practice / grade_answers：业务结果 JSON 内容断言
  （租户断言行为见 test_tenancy.py，此处聚焦业务逻辑）

注意：diagnose / generate_practice / grade_answers 带策略装饰器，
测试需先注入 Principal（principal_var）方可进入业务逻辑。
"""
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult

from coursepilot.mcp.principal import (
    Principal,
    principal_var,
    set_principal,
)
from coursepilot.mcp.shared.schemas import (
    DiagnoseParams,
    GeneratePracticeParams,
    GetKPTreeParams,
    GradeAnswersParams,
    QueryKnowledgeParams,
    SearchKnowledgeUnitsParams,
)

ZERO_TOKENS = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

_UUID = str(uuid.uuid4())
_UUID2 = str(uuid.uuid4())


@pytest.fixture
def mock_session():
    """mock DB session + chain: scalars → all → []"""
    session = AsyncMock()
    session.close = AsyncMock(return_value=None)

    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    result.scalars.return_value = result
    result.all.return_value = []
    session.execute = AsyncMock(return_value=result)

    return session


@pytest.fixture(autouse=True)
def _clean_principal():
    """清理 ContextVar，避免跨用例泄漏。"""
    token = set_principal(None)
    yield
    principal_var.reset(token)


# ── query_knowledge（tutor.py，无装饰器）──────────────────────────


@pytest.mark.asyncio
async def test_query_knowledge_ok(mock_session):
    """正常 RAG 问答流程：返回答案 + 来源知识点 + token 用量。"""
    from coursepilot.mcp.tools.tutor import query_knowledge

    params = QueryKnowledgeParams(query="什么是进程调度", course_id=uuid.UUID(_UUID))

    with (
        patch("coursepilot.mcp.tools.tutor._get_session", return_value=mock_session),
        patch(
            "coursepilot.mcp.tools.tutor.build_course_context",
            return_value={"name": "OS", "textbook": "教材", "chapters": ["进程管理"]},
        ),
        patch("coursepilot.mcp.tools.tutor.Retriever") as mock_ret_cls,
        patch("coursepilot.mcp.tools.tutor.Generator") as mock_gen_cls,
    ):
        mock_ret = AsyncMock()
        mock_ret.retrieve.return_value = (
            "检索到的上下文",
            {"source_kp_paths": ["OS/进程管理", "OS/内存管理"]},
        )
        mock_ret_cls.return_value = mock_ret

        mock_gen = AsyncMock()
        mock_gen.generate.return_value = ("进程调度是核心功能", ZERO_TOKENS)
        mock_gen_cls.return_value = mock_gen

        result = await query_knowledge(params)

    assert isinstance(result, CallToolResult)
    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["answer"] == "进程调度是核心功能"
    assert data["source_kps"] == ["OS/进程管理", "OS/内存管理"]
    assert data["tokens"]["total_tokens"] == 0


@pytest.mark.asyncio
async def test_query_knowledge_course_not_found(mock_session):
    """course_id 不存在 → isError=true。"""
    from coursepilot.mcp.tools.tutor import query_knowledge

    params = QueryKnowledgeParams(query="test", course_id=uuid.UUID(_UUID))

    with (
        patch("coursepilot.mcp.tools.tutor._get_session", return_value=mock_session),
        patch("coursepilot.mcp.tools.tutor.build_course_context", return_value={}),
    ):
        result = await query_knowledge(params)

    assert result.is_error is True
    assert "不存在" in result.content[0].text


# ── get_kp_tree（knowledge.py，无装饰器）─────────────────────────


@pytest.mark.asyncio
async def test_get_kp_tree_empty(mock_session):
    """课程没有知识点 → 返回空列表。"""
    from coursepilot.mcp.tools.knowledge import get_kp_tree

    params = GetKPTreeParams(course_id=uuid.UUID(_UUID))

    with patch("coursepilot.mcp.tools.knowledge._get_session",
               return_value=mock_session):
        result = await get_kp_tree(params)

    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["course_id"] == _UUID
    assert data["kp_paths"] == []


@pytest.mark.asyncio
async def test_get_kp_tree_with_paths(mock_session):
    """返回知识点路径列表（保持查询顺序）。"""
    from coursepilot.mcp.tools.knowledge import get_kp_tree

    params = GetKPTreeParams(course_id=uuid.UUID(_UUID))
    # 模拟查询返回两行 kp_path
    mock_session.execute.return_value.all.return_value = [
        ("OS/进程管理",), ("OS/内存管理",),
    ]

    with patch("coursepilot.mcp.tools.knowledge._get_session",
               return_value=mock_session):
        result = await get_kp_tree(params)

    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["kp_paths"] == ["OS/进程管理", "OS/内存管理"]


# ── search_knowledge_units（knowledge.py，无装饰器）──────────────


@pytest.mark.asyncio
async def test_search_knowledge_units_ok(mock_session):
    """检索结果保持重排顺序，且仅返回 top_k 条。"""
    from coursepilot.mcp.tools.knowledge import search_knowledge_units

    params = SearchKnowledgeUnitsParams(
        query="二重积分", course_id=uuid.UUID(_UUID), top_k=2,
    )
    u1, u2, u3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    # 检索返回的 top_uuids 顺序与 DB 行顺序故意不同，验证按重排顺序输出
    metadata = {"top_uuids": [str(u3), str(u1)]}

    rows = [
        (MagicMock(id=u1, content="内容1", summary="摘要1", page_ref="p1"), "OS/1"),
        (MagicMock(id=u2, content="内容2", summary="摘要2", page_ref="p2"), "OS/2"),
        (MagicMock(id=u3, content="内容3", summary="摘要3", page_ref="p3"), "OS/3"),
    ]
    mock_session.execute.return_value.all.return_value = rows

    with (
        patch("coursepilot.mcp.tools.knowledge._get_session",
              return_value=mock_session),
        patch("coursepilot.mcp.tools.knowledge.Retriever") as mock_ret_cls,
    ):
        mock_ret = AsyncMock()
        mock_ret.retrieve.return_value = ("ctx", metadata)
        mock_ret_cls.return_value = mock_ret

        result = await search_knowledge_units(params)

    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert [u["uuid"] for u in data["units"]] == [str(u3), str(u1)]
    assert data["units"][0]["content"] == "内容3"


@pytest.mark.asyncio
async def test_search_knowledge_units_no_results(mock_session):
    """检索无结果 → 返回空列表。"""
    from coursepilot.mcp.tools.knowledge import search_knowledge_units

    params = SearchKnowledgeUnitsParams(query="x", course_id=uuid.UUID(_UUID))

    with (
        patch("coursepilot.mcp.tools.knowledge._get_session",
              return_value=mock_session),
        patch("coursepilot.mcp.tools.knowledge.Retriever") as mock_ret_cls,
    ):
        mock_ret = AsyncMock()
        mock_ret.retrieve.return_value = ("ctx", {"top_uuids": []})
        mock_ret_cls.return_value = mock_ret

        result = await search_knowledge_units(params)

    assert result.is_error is False
    assert json.loads(result.content[0].text) == {"units": []}


# ── diagnose（tutor.py，带租户装饰器：先注入 student 本人）─────────


@pytest.mark.asyncio
async def test_diagnose_ok(mock_session):
    """学情诊断：返回薄弱知识点 / 统计 / 正确率。

    注：diagnose 带租户装饰器，student 角色只能查自己——
    此处以 student 身份查自己（params.user_id 与 principal.user_id 一致）。
    """
    from coursepilot.mcp.tools.tutor import diagnose

    student_id = str(uuid.UUID("00000000-0000-0000-0000-000000000001"))
    set_principal(Principal(
        user_id=student_id, role="student", scopes=frozenset({"read"})))
    params = DiagnoseParams(user_id=uuid.UUID(student_id),
                            course_id=uuid.UUID(_UUID))

    mock_result = {
        "weak_kps": ["OS/进程管理"],
        "kp_stats": {"OS/进程管理": {"total": 10, "correct": 3, "rate": 0.3}},
        "summary": "薄弱知识点：进程管理",
        "total_practiced": 10,
        "overall_rate": 0.3,
    }
    with (
        patch("coursepilot.mcp.tools.tutor._get_session", return_value=mock_session),
        patch("coursepilot.mcp.tools.tutor.diagnose_skill",
              return_value=mock_result),
    ):
        result = await diagnose(params)

    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["weak_kps"] == ["OS/进程管理"]
    assert data["overall_rate"] == 0.3
    assert data["total_practiced"] == 10


# ── generate_practice（practice.py，带 scope 装饰器：注入 teacher）──


@pytest.mark.asyncio
async def test_generate_practice_ok(mock_session):
    """生成练习题：返回题目列表（不含答案）。"""
    from coursepilot.mcp.tools.practice import generate_practice

    set_principal(Principal(
        user_id="u-teacher", role="teacher", scopes=frozenset({"read", "write"})))
    params = GeneratePracticeParams(
        course_id=uuid.UUID(_UUID), kp_path="OS/进程管理", count=1, difficulty=3,
    )

    mock_quiz = {
        "questions": [
            {
                "question_text": "什么是进程？",
                "options": {"A": "程序", "B": "进程实例", "C": "文件"},
                "correct_answer": "B",
                "explanation": "进程是程序在执行过程中的实例",
            }
        ]
    }
    with (
        patch("coursepilot.mcp.tools.practice._get_session",
              return_value=mock_session),
        patch("coursepilot.mcp.tools.practice.build_course_context",
              return_value={"name": "OS", "textbook": "教材", "chapters": []}),
        patch("coursepilot.mcp.tools.practice._find_kp_by_path",
              return_value=MagicMock(id="kp-1")),
        patch("coursepilot.mcp.tools.practice.Retriever") as mock_ret_cls,
        patch("coursepilot.mcp.tools.practice.generate_quiz",
              return_value=(mock_quiz, ZERO_TOKENS)),
    ):
        mock_ret = AsyncMock()
        mock_ret.retrieve.return_value = ("上下文", {})
        mock_ret_cls.return_value = mock_ret

        result = await generate_practice(params)

    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["count"] == 1
    assert data["questions"][0]["question_text"] == "什么是进程？"
    # 不含答案（不泄露 correct_answer）
    assert "correct_answer" not in data["questions"][0]
    assert data["questions"][0]["question_id"]


# ── grade_answers（practice.py，带租户装饰器：注入 teacher）────────


@pytest.mark.asyncio
async def test_grade_answers_ok(mock_session):
    """批改：答案正确写入 PracticeRecord 并返回批改结果。"""
    from coursepilot.mcp.tools.practice import grade_answers

    set_principal(Principal(
        user_id="u-teacher", role="teacher", scopes=frozenset({"read", "write"})))
    params = GradeAnswersParams(
        user_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        question_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
        answer="B",
    )

    q = MagicMock(id=str(params.question_id), correct_answer="B",
                  explanation="进程是实例", kp_id=None)
    # 第一次 execute 查 Question → q；第二次查 kp_path → None
    r_q = MagicMock()
    r_q.scalar_one_or_none.return_value = q
    r_none = MagicMock()
    r_none.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(side_effect=[r_q, r_none])

    with patch("coursepilot.mcp.tools.practice._get_session",
               return_value=mock_session):
        result = await grade_answers(params)

    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["correct"] is True
    assert data["student_answer"] == "B"
    assert data["correct_answer"] == "B"
    assert data["explanation"] == "进程是实例"


@pytest.mark.asyncio
async def test_grade_answers_wrong(mock_session):
    """批改：答案错误 → correct=false。"""
    from coursepilot.mcp.tools.practice import grade_answers

    set_principal(Principal(
        user_id="u-teacher", role="teacher", scopes=frozenset({"read", "write"})))
    params = GradeAnswersParams(
        user_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        question_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
        answer="A",
    )

    q = MagicMock(id=str(params.question_id), correct_answer="B",
                  explanation="进程是实例", kp_id=None)
    r_q = MagicMock()
    r_q.scalar_one_or_none.return_value = q
    r_none = MagicMock()
    r_none.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(side_effect=[r_q, r_none])

    with patch("coursepilot.mcp.tools.practice._get_session",
               return_value=mock_session):
        result = await grade_answers(params)

    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["correct"] is False
    assert data["student_answer"] == "A"
