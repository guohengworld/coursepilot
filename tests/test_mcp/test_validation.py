"""MCP 参数校验专项测试。"""

import uuid

import pytest
from pydantic import ValidationError

from coursepilot.mcp.shared.schemas import (
    GeneratePracticeParams,
    QueryKnowledgeParams,
    validate_kp_path,
)


class TestKPPathValidation:
    def test_valid_paths(self):
        assert validate_kp_path("微积分/重积分/二重积分") == "微积分/重积分/二重积分"
        assert validate_kp_path("OS/process/scheduling") == "OS/process/scheduling"
        assert validate_kp_path("chapter-1_section_1") == "chapter-1_section_1"

    def test_empty_path(self):
        assert validate_kp_path("") == ""

    def test_too_long(self):
        with pytest.raises(ValueError):
            validate_kp_path("a" * 513)

    @pytest.mark.parametrize(
        "path",
        [
            "微积分; 注入",
            "OS/process<script>",
            "chapter 1",  # 空格非法
            "chapter\t1",
            "chapter\n1",
        ],
    )
    def test_invalid_chars(self, path):
        with pytest.raises(ValueError):
            validate_kp_path(path)


class TestUUIDValidation:
    def test_valid_uuid_string(self):
        params = QueryKnowledgeParams(
            query="test",
            course_id="550e8400-e29b-41d4-a716-446655440000",
        )
        assert str(params.course_id) == "550e8400-e29b-41d4-a716-446655440000"

    @pytest.mark.parametrize(
        "bad_uuid",
        [
            "not-a-uuid",
            "12345",
            "550e8400-e29b-41d4-a716-44665544000Z",
            "",
        ],
    )
    def test_invalid_uuid(self, bad_uuid):
        with pytest.raises(ValidationError):
            QueryKnowledgeParams(query="test", course_id=bad_uuid)


class TestExtraParamsIgnored:
    def test_extra_params_dropped(self):
        params = GeneratePracticeParams(
            course_id=uuid.uuid4(),
            count=2,
            malicious_param="drop_me",
        )
        dumped = params.model_dump()
        assert "malicious_param" not in dumped
        assert dumped["count"] == 2
