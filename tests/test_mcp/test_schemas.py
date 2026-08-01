"""MCP 共享 schema 测试。"""

import uuid

import pytest
from pydantic import ValidationError

from coursepilot.mcp.shared.schemas import (
    DiagnoseParams,
    GeneratePracticeParams,
    GetKPTreeParams,
    GetReviewPlanParams,
    GradeAnswersParams,
    QueryKnowledgeParams,
    SearchKnowledgeUnitsParams,
)


class TestQueryKnowledgeParams:
    def test_valid_params(self):
        params = QueryKnowledgeParams(
            query="什么是二重积分",
            course_id=uuid.uuid4(),
            kp_path="微积分/重积分/二重积分",
        )
        assert params.query == "什么是二重积分"
        assert params.idempotency_key is None

    def test_query_empty(self):
        with pytest.raises(ValidationError):
            QueryKnowledgeParams(query="", course_id=uuid.uuid4())

    def test_query_too_long(self):
        with pytest.raises(ValidationError):
            QueryKnowledgeParams(query="问" * 2001, course_id=uuid.uuid4())

    def test_kp_path_invalid_char(self):
        with pytest.raises(ValidationError):
            QueryKnowledgeParams(
                query="test",
                course_id=uuid.uuid4(),
                kp_path="微积分; 注入",
            )

    def test_hallucination_param_ignored(self):
        """未声明参数应被丢弃，不报错。"""
        params = QueryKnowledgeParams(
            query="test",
            course_id=uuid.uuid4(),
            unknown_field="should be ignored",
        )
        assert "unknown_field" not in params.model_dump()

    def test_nfc_normalization(self):
        """NFD 形式应规范化为 NFC。"""
        import unicodedata

        nfd = unicodedata.normalize("NFD", "微积分")
        params = QueryKnowledgeParams(query=nfd, course_id=uuid.uuid4())
        assert params.query == "微积分"


class TestGeneratePracticeParams:
    def test_default_count(self):
        course_id = uuid.uuid4()
        params = GeneratePracticeParams(course_id=course_id)
        assert params.count == 3
        assert params.difficulty == 3

    def test_count_out_of_range(self):
        with pytest.raises(ValidationError):
            GeneratePracticeParams(course_id=uuid.uuid4(), count=0)
        with pytest.raises(ValidationError):
            GeneratePracticeParams(course_id=uuid.uuid4(), count=11)

    def test_difficulty_out_of_range(self):
        with pytest.raises(ValidationError):
            GeneratePracticeParams(course_id=uuid.uuid4(), difficulty=6)


class TestGradeAnswersParams:
    def test_valid(self):
        params = GradeAnswersParams(
            question_id=uuid.uuid4(),
            answer="A",
        )
        assert params.answer == "A"


class TestDiagnoseAndReview:
    def test_diagnose_valid(self):
        params = DiagnoseParams(
            user_id=uuid.uuid4(),
            course_id=uuid.uuid4(),
        )
        assert params.user_id == params.user_id

    def test_review_plan_valid(self):
        params = GetReviewPlanParams(
            user_id=uuid.uuid4(),
            course_id=uuid.uuid4(),
        )
        assert params.idempotency_key is None


class TestSearchAndKPTree:
    def test_search_top_k_default(self):
        params = SearchKnowledgeUnitsParams(
            query="二重积分",
            course_id=uuid.uuid4(),
        )
        assert params.top_k == 10

    def test_search_top_k_out_of_range(self):
        with pytest.raises(ValidationError):
            SearchKnowledgeUnitsParams(
                query="test",
                course_id=uuid.uuid4(),
                top_k=100,
            )

    def test_kp_tree_valid(self):
        course_id = uuid.uuid4()
        params = GetKPTreeParams(course_id=course_id)
        assert params.course_id == course_id
