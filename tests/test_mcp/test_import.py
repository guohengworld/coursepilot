"""验证 MCP 模块可正常导入。"""


def test_mcp_package_imports():
    """所有 MCP 子包应可导入。"""
    from coursepilot.mcp import server
    from coursepilot.mcp.cli import main as cli_main
    from coursepilot.mcp.gateway import app as gateway_app
    from coursepilot.mcp.gateway import main as gateway_main
    from coursepilot.mcp.prompts import diagnosis_report, quiz_blueprint, tutor_socratic
    from coursepilot.mcp.resources import course
    from coursepilot.mcp.shared import errors, schemas
    from coursepilot.mcp.tools import knowledge, practice, tutor

    assert server is not None
    assert cli_main is not None
    assert gateway_app is not None
    assert gateway_main is not None
    assert tutor_socratic is not None
    assert quiz_blueprint is not None
    assert diagnosis_report is not None
    assert course is not None
    assert errors is not None
    assert schemas is not None
    assert knowledge is not None
    assert practice is not None
    assert tutor is not None
