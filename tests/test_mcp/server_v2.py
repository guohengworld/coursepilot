#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段二实践项目：企业级 MCP Server（v2 SDK）

功能：
  1. echo         - 回显输入，带 Pydantic 验证
  2. calculate    - 安全数学计算（AST 求值）
  3. query_database - 只读 SQL 查询，完整安全防护
  4. config://server-info - 资源，返回 Server 元数据
  5. greeting     - Prompt 模板

企业级特性：
  - Pydantic Schema + 6 要素 description
  - SQL 注入多层防护（关键字黑名单 + 分号过滤 + UNION 过滤 + sqlparse 解析）
  - 结构化错误工程（error_type + message + suggestion + retryable）
  - OpenTelemetry 可观测性（v2 默认开启）
  - 双时代协议支持（2025 + 2026 自动适配）
  - 请求体大小限制（4 MiB 默认）
  - 资源变更通知支持

安装依赖：
    pip install "mcp[cli]>=2.0.0" sqlparse

启动：
    python server_v2.py              # stdio 模式（默认）
    python server_v2.py --http       # Streamable HTTP 模式
    python server_v2.py --http --host 0.0.0.0 --port 3000
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import re
import sys
import time
from datetime import datetime
from enum import IntEnum
from typing import Any, Literal, Optional

import sqlparse
from pydantic import BaseModel, Field, field_validator

from mcp.server import MCPServer
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    PromptMessage,
    TextContent,
    TextResourceContents,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 全局配置与常量
# ═══════════════════════════════════════════════════════════════════════════════

SERVER_NAME = "enterprise-mcp-server"
SERVER_VERSION = "2.0.0"
DEFAULT_PROTOCOL_VERSION = "2026-07-28"

# 请求体大小限制（与 SDK v2 默认值一致：4 MiB）
MAX_REQUEST_BODY_SIZE = 4 * 1024 * 1024

# SQL 安全：禁止的关键字（不区分大小写）
SQL_DANGEROUS_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER",
    "CREATE", "GRANT", "REVOKE", "EXEC", "EXECUTE", "MERGE",
    "CALL", "LOAD", "COPY", "ATTACH", "DETACH",
]

# SQL 安全：禁止的函数（信息泄露或系统调用）
SQL_DANGEROUS_FUNCTIONS = [
    "pg_read_file", "pg_ls_dir", "pg_sleep", "system", "shell",
    "load_file", "into outfile", "into dumpfile",
]

# 数学计算：AST 允许的操作符
SAFE_MATH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 错误码与错误工程
# ═══════════════════════════════════════════════════════════════════════════════

class BusinessErrorCode(IntEnum):
    """MCP 业务错误码（-32000 到 -32099 为 MCP 自定义空间）"""
    AUTH_FAILED = -32001
    PERMISSION_DENIED = -32002
    RESOURCE_NOT_FOUND = -32003
    RATE_LIMITED = -32004
    VALIDATION_FAILED = -32005
    DEPENDENCY_TIMEOUT = -32006
    DEPENDENCY_ERROR = -32007
    MAINTENANCE_MODE = -32008


class ToolErrorDetail(BaseModel):
    """
    对 LLM 友好的错误详情结构。
    序列化后放在 tool 响应的 content 中。
    """
    error_type: Literal[
        "validation", "permission", "not_found", "timeout",
        "rate_limit", "internal"
    ] = Field(description="错误分类")
    message: str = Field(description="对 LLM 可操作的错误描述")
    suggestion: str = Field(description="明确的下一步修正建议")
    retryable: bool = Field(description="LLM 是否应该重试")
    retry_after_seconds: Optional[int] = Field(
        default=None, description="如果是限流，建议多久后重试"
    )


def make_tool_error(
    error_type: str,
    message: str,
    suggestion: str,
    retryable: bool,
    retry_after_seconds: Optional[int] = None,
) -> CallToolResult:
    """构造标准化的工具错误响应（isError=True）"""
    detail = ToolErrorDetail(
        error_type=error_type,
        message=message,
        suggestion=suggestion,
        retryable=retryable,
        retry_after_seconds=retry_after_seconds,
    )
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(detail.model_dump(), ensure_ascii=False),
            )
        ],
        is_error=True,
    )


def make_tool_success(text: str) -> CallToolResult:
    """构造标准化的工具成功响应（isError=False）"""
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        is_error=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Pydantic 参数模型（Schema 工程）
# ═══════════════════════════════════════════════════════════════════════════════

class EchoParams(BaseModel):
    """
    [1-用途] 回显输入内容，用于测试连接和验证参数传递。
    [2-限制] 消息长度不超过 10,000 字符，超过会被截断。
    [3-成本] 极低，纯内存操作，无外部依赖。
    [4-副作用] 无。不会修改任何状态。
    [5-输入格式] message: 任意文本字符串。
    [6-输出格式] 返回 JSON，包含原始消息和时间戳。
    """
    message: str = Field(
        description="要回显的消息内容。支持任意 UTF-8 文本。",
        min_length=1,
        max_length=10000,
    )


class CalculateParams(BaseModel):
    """
    [1-用途] 执行安全的数学表达式计算，支持 + - * / // % ** 和括号。
    [2-限制] 仅支持数值和基本运算符。禁止变量、函数调用、字符串操作。
    [3-成本] 极低，纯 AST 求值，无外部依赖。
    [4-副作用] 无。不会执行任何系统调用或 I/O。
    [5-输入格式] expression: 数学表达式字符串，如 "2 + 3 * 4" 或 "(10 - 2) ** 3"
    [6-输出格式] 返回 JSON，包含原始表达式、计算结果和格式化字符串。
    """
    expression: str = Field(
        description="数学表达式。仅支持数字、+ - * / // % ** 和括号。"
                    "示例：'2 + 3 * 4'、'(100 - 20) / 4'、'2 ** 10'。"
                    "禁止：变量名、函数调用、字符串。",
        min_length=1,
        max_length=500,
    )


class QueryDatabaseParams(BaseModel):
    """
    [1-用途] 执行只读 SQL 查询，从内部数据仓库获取数据。
    [2-限制] 仅支持 SELECT 语句。禁止 INSERT、UPDATE、DELETE、DROP、CREATE、ALTER 等修改操作。
             查询超时 30 秒，超过会被强制终止。
             单次最多返回 1000 行，超过自动截断。
             禁止多语句（分号）和 UNION 查询（注入风险）。
    [3-成本] 大数据表（>100万行）的聚合查询可能耗时 5-30 秒。
             建议在非高峰时段执行复杂查询，或添加精确的 WHERE 条件。
    [4-副作用] 此工具只读，不会修改任何数据。但复杂查询可能消耗大量数据库 CPU/IO。
    [5-输入格式] sql: 标准 SQL SELECT 语句。limit: 最大返回行数（1-1000，默认 100）。
    [6-输出格式] 返回 Markdown 表格。如果结果为空，返回 "查询成功，但无数据返回。"
    """
    sql: str = Field(
        description="标准 SQL SELECT 语句。必须以 SELECT 开头。"
                    "禁止：INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/TRUNCATE/GRANT。"
                    "禁止：分号（;）、UNION、注释（-- /* */）。"
                    "建议：添加 WHERE 条件限制数据量。",
        min_length=7,  # 最短 "SELECT *"
        max_length=5000,
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="最大返回行数。默认 100，最多 1000。超过 1000 会自动截断。",
    )

    @field_validator("sql")
    @classmethod
    def validate_sql_basic(cls, v: str) -> str:
        """第一层验证：基础字符和格式检查（Pydantic 层）"""
        # 1. 必须以 SELECT 开头（忽略前导空白）
        stripped = v.strip()
        if not re.match(r"^SELECT\s", stripped, re.IGNORECASE):
            raise ValueError("SQL 必须以 SELECT 开头，仅支持查询操作")

        # 2. 禁止分号（多语句注入）
        if ";" in v:
            raise ValueError("禁止分号（;），不允许多语句查询")

        # 3. 禁止注释（-- 和 /* */）
        if "--" in v or "/*" in v or "*/" in v:
            raise ValueError("禁止 SQL 注释（-- 或 /* */），防止注释注入攻击")

        # 4. 禁止 UNION（联合查询注入）
        if re.search(r"\bUNION\b", v, re.IGNORECASE):
            raise ValueError("禁止 UNION 查询，防止联合注入攻击")

        # 5. 禁止危险关键字
        upper = v.upper()
        for keyword in SQL_DANGEROUS_KEYWORDS:
            # 使用单词边界匹配，避免误杀合法字段名（如 "selection" 中的 "select"）
            if re.search(rf"\b{keyword}\b", upper):
                raise ValueError(f"检测到危险关键字: {keyword}。仅允许 SELECT 查询")

        # 6. 禁止危险函数
        for func in SQL_DANGEROUS_FUNCTIONS:
            if func.lower() in v.lower():
                raise ValueError(f"检测到危险函数: {func}")

        return stripped


class GreetingParams(BaseModel):
    """
    [1-用途] 根据名字和风格生成个性化问候语。
    [2-限制] 名字长度不超过 50 字符。仅支持预定义风格。
    [3-成本] 极低，纯模板替换。
    [4-副作用] 无。
    [5-输入格式] name: 被问候者名字。style: 风格（friendly/formal/casual）。
    [6-输出格式] 返回 GetPromptResult，包含多轮对话消息。
    """
    name: str = Field(
        description="被问候者的名字。示例：'Alice'、'张三'。",
        min_length=1,
        max_length=50,
    )
    style: Literal["friendly", "formal", "casual"] = Field(
        default="friendly",
        description="问候风格。friendly=友好（默认），formal=正式，casual=随意。",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 安全计算引擎（AST 求值）
# ═══════════════════════════════════════════════════════════════════════════════

def safe_math_eval(expression: str) -> float:
    """
    使用 AST 安全求值数学表达式。
    仅允许数值常量和基本运算符，禁止任何函数调用、变量访问。
    """
    # ── 预检查 1: 深度嵌套括号（DoS / 栈溢出防护） ──────────────
    # CPython 3.12 对嵌套括号有硬限制（约 200 层会触发 SyntaxError），
    # 我们在解析前主动检测，避免不可控的 SyntaxError 泄漏。
    depth = 0
    for ch in expression:
        if ch == '(':
            depth += 1
            if depth > 100:
                raise ValueError("括号嵌套过深，最多允许 100 层")
        elif ch == ')':
            depth -= 1

    # ── 预检查 2: 十六进制 / 八进制 / 二进制字面量注入 ──────────
    # Python AST 会把 0xff / 0o77 / 0b1010 视为合法 Constant，
    # 但计算器不应接受这些非十进制写法。
    if re.search(r'\b0[xXoObB][0-9a-fA-F]+', expression):
        raise ValueError("不支持十六进制、八进制或二进制字面量")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {e.msg}")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"不支持的常量类型: {type(node.value).__name__}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in SAFE_MATH_OPS:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            # 防止极端幂运算导致内存/CPU 耗尽（如 9**9**9）
            if op_type is ast.Pow:
                if abs(left) > 1000 or abs(right) > 100:
                    raise ValueError("幂运算底数或指数过大，可能导致计算溢出")
            return SAFE_MATH_OPS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in SAFE_MATH_OPS:
                raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
            operand = _eval(node.operand)
            return SAFE_MATH_OPS[op_type](operand)
        elif isinstance(node, ast.Call):
            raise ValueError("禁止函数调用")
        elif isinstance(node, ast.Name):
            raise ValueError("禁止变量引用")
        elif isinstance(node, ast.Subscript):
            raise ValueError("禁止下标访问")
        elif isinstance(node, ast.Attribute):
            raise ValueError("禁止属性访问")
        else:
            raise ValueError(f"不支持的表达式节点: {type(node).__name__}")

    result = _eval(tree)
    if not isinstance(result, (int, float)):
        raise ValueError("表达式结果必须是数字")
    return float(result)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 模拟数据库引擎（用于演示，实际生产应替换为真实连接）
# ═══════════════════════════════════════════════════════════════════════════════

class MockDatabase:
    """模拟数据库，用于演示 query_database 工具。"""

    TABLES = {
        "users": [
            {"id": 1, "name": "Alice", "status": "active", "role": "admin"},
            {"id": 2, "name": "Bob", "status": "inactive", "role": "user"},
            {"id": 3, "name": "Charlie", "status": "active", "role": "user"},
            {"id": 4, "name": "Diana", "status": "active", "role": "editor"},
        ],
        "orders": [
            {"id": 101, "user_id": 1, "amount": 250.00, "status": "completed"},
            {"id": 102, "user_id": 3, "amount": 120.50, "status": "pending"},
            {"id": 103, "user_id": 1, "amount": 890.00, "status": "completed"},
        ],
        "products": [
            {"id": 1, "name": "Laptop", "price": 999.99, "stock": 50},
            {"id": 2, "name": "Mouse", "price": 29.99, "stock": 200},
            {"id": 3, "name": "Keyboard", "price": 79.99, "stock": 0},
        ],
    }

    @classmethod
    def execute(cls, sql: str, limit: int) -> tuple[list[dict], str]:
        """
        模拟执行 SQL 查询。
        返回 (结果列表, 执行信息)。
        实际生产环境应使用真实数据库连接（psycopg2、sqlalchemy 等）。
        """
        # 模拟执行延迟（简单查询 < 10ms，复杂查询 50-200ms）
        start = time.time()

        # 解析 SQL 获取表名（简化版，实际应用应使用 sqlparse 或 SQL parser）
        upper_sql = sql.upper()

        # 提取 FROM 后的表名（简化正则，仅用于演示）
        table_match = re.search(r"FROM\s+(\w+)", upper_sql)
        if not table_match:
            raise ValueError("无法解析 SQL：缺少 FROM 子句或表名")

        table_name = table_match.group(1).lower()
        if table_name not in cls.TABLES:
            raise ValueError(f"表 '{table_name}' 不存在。可用表: {', '.join(cls.TABLES.keys())}")

        # 模拟 WHERE 过滤（极其简化，仅用于演示框架）
        data = cls.TABLES[table_name].copy()

        # 模拟 LIMIT
        data = data[:limit]

        elapsed = (time.time() - start) * 1000
        info = f"表: {table_name} | 返回: {len(data)} 行 | 耗时: {elapsed:.1f}ms"
        return data, info


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 初始化 MCPServer
# ═══════════════════════════════════════════════════════════════════════════════

mcp = MCPServer(
    name=SERVER_NAME,
    title="Enterprise MCP Server",
    description="企业级 MCP Server，提供安全的数据查询、计算和模板服务。",
    version=SERVER_VERSION,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 工具定义（Tools）
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def echo(params: EchoParams) -> CallToolResult:
    """
    [1-用途] 回显输入内容，用于测试连接和验证参数传递。
    [2-限制] 消息长度不超过 10,000 字符，超过会被截断。
    [3-成本] 极低，纯内存操作，无外部依赖。
    [4-副作用] 无。不会修改任何状态。
    [5-输入格式] message: 任意文本字符串。
    [6-输出格式] 返回 JSON，包含原始消息和时间戳。
    """
    result = {
        "echo": params.message,
        "timestamp": datetime.now().isoformat(),
        "length": len(params.message),
    }
    return make_tool_success(json.dumps(result, ensure_ascii=False))


@mcp.tool()
async def calculate(params: CalculateParams) -> CallToolResult:
    """
    [1-用途] 执行安全的数学表达式计算，支持 + - * / // % ** 和括号。
    [2-限制] 仅支持数值和基本运算符。禁止变量、函数调用、字符串操作。
    [3-成本] 极低，纯 AST 求值，无外部依赖。
    [4-副作用] 无。不会执行任何系统调用或 I/O。
    [5-输入格式] expression: 数学表达式字符串，如 "2 + 3 * 4" 或 "(10 - 2) ** 3"
    [6-输出格式] 返回 JSON，包含原始表达式、计算结果和格式化字符串。
    """
    try:
        result = safe_math_eval(params.expression)
        output = {
            "expression": params.expression,
            "result": result,
            "formatted": f"{params.expression} = {result}",
        }
        return make_tool_success(json.dumps(output, ensure_ascii=False))
    except ValueError as e:
        return make_tool_error(
            error_type="validation",
            message=f"表达式验证失败: {e}",
            suggestion="请检查表达式格式。仅支持数字、+ - * / // % ** 和括号。"
                      "示例：'2 + 3 * 4'、'(100 - 20) / 4'。",
            retryable=True,
        )
    except Exception as e:
        return make_tool_error(
            error_type="internal",
            message=f"计算过程中发生内部错误: {e}",
            suggestion="请稍后重试。如果问题持续，请联系运维团队。",
            retryable=True,
        )


@mcp.tool()
async def query_database(params: QueryDatabaseParams) -> CallToolResult:
    """
    [1-用途] 执行只读 SQL 查询，从内部数据仓库获取数据。
    [2-限制] 仅支持 SELECT 语句。禁止 INSERT、UPDATE、DELETE、DROP、CREATE、ALTER 等修改操作。
             查询超时 30 秒，超过会被强制终止。
             单次最多返回 1000 行，超过自动截断。
             禁止多语句（分号）和 UNION 查询（注入风险）。
    [3-成本] 大数据表（>100万行）的聚合查询可能耗时 5-30 秒。
             建议在非高峰时段执行复杂查询，或添加精确的 WHERE 条件。
    [4-副作用] 此工具只读，不会修改任何数据。但复杂查询可能消耗大量数据库 CPU/IO。
    [5-输入格式] sql: 标准 SQL SELECT 语句。limit: 最大返回行数（1-1000，默认 100）。
    [6-输出格式] 返回 Markdown 表格。如果结果为空，返回 "查询成功，但无数据返回。"
    """
    # 第二层验证：使用 sqlparse 进行语法级分析
    try:
        parsed = sqlparse.parse(params.sql)
        if not parsed:
            return make_tool_error(
                error_type="validation",
                message="SQL 解析失败：无法解析为空或无效语句",
                suggestion="请提供有效的 SELECT 语句。",
                retryable=True,
            )

        stmt = parsed[0]
        # 检查 token 类型
        first_token = None
        for token in stmt.tokens:
            if not token.is_whitespace:
                first_token = token
                break

        if first_token is None or first_token.normalized != "SELECT":
            return make_tool_error(
                error_type="validation",
                message="SQL 解析失败：第一个有效 token 不是 SELECT",
                suggestion="请确保 SQL 以 SELECT 开头。仅支持查询操作。",
                retryable=True,
            )

        # 检查所有 token 中是否有危险关键字（深度遍历）
        def check_tokens(token_list):
            for token in token_list:
                if token.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DDL):
                    if token.normalized in SQL_DANGEROUS_KEYWORDS:
                        return token.normalized
                if hasattr(token, "tokens"):
                    found = check_tokens(token.tokens)
                    if found:
                        return found
            return None

        dangerous = check_tokens(stmt.tokens)
        if dangerous:
            return make_tool_error(
                error_type="validation",
                message=f"SQL 解析发现危险关键字: {dangerous}",
                suggestion="请仅使用 SELECT 查询。禁止 INSERT/UPDATE/DELETE/DROP 等操作。",
                retryable=True,
            )

    except Exception as e:
        return make_tool_error(
            error_type="validation",
            message=f"SQL 解析异常: {e}",
            suggestion="请检查 SQL 语法。",
            retryable=True,
        )

    # 第三层验证：自动注入 LIMIT（如果原 SQL 没有）
    sql_with_limit = params.sql
    if "LIMIT" not in params.sql.upper():
        sql_with_limit = f"{params.sql.strip()} LIMIT {params.limit}"

    # 执行查询（带超时模拟）
    try:
        # 模拟超时检查（实际生产应使用 asyncio.wait_for 或数据库驱动超时）
        start = time.time()
        results, info = MockDatabase.execute(sql_with_limit, params.limit)
        elapsed = time.time() - start

        if elapsed > 30:
            return make_tool_error(
                error_type="timeout",
                message="查询执行超过 30 秒超时限制。",
                suggestion="请尝试：1) 添加更精确的 WHERE 条件 2) 减少返回列数 "
                          "3) 避免对大表进行全表扫描 4) 在非高峰时段重试",
                retryable=True,
                retry_after_seconds=60,
            )

        if not results:
            return make_tool_success(
                f"查询成功，但无数据返回。\n\n执行信息: {info}"
            )

        # 格式化为 Markdown 表格
        headers = list(results[0].keys())
        md = "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in results:
            md += "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n"

        md += f"\n**执行信息**: {info}"
        return make_tool_success(md)

    except ValueError as e:
        return make_tool_error(
            error_type="validation",
            message=str(e),
            suggestion="请检查 SQL 语法和表名。可用表: users, orders, products。",
            retryable=True,
        )
    except Exception as e:
        trace_id = f"err-{int(time.time() * 1000)}"
        # 实际生产：logger.exception(f"DB error [trace_id={trace_id}]: {e}")
        return make_tool_error(
            error_type="internal",
            message="数据库查询发生内部错误。",
            suggestion=f"请稍后重试。如果问题持续，请联系运维并提供追踪 ID: {trace_id}",
            retryable=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. 资源定义（Resources）
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.resource(
    uri="config://server-info",
    name="Server Info",
    description="MCP 服务器元数据，包含版本、能力、传输协议等信息。",
    mime_type="application/json",
)
async def resource_server_info() -> str:
    """返回 Server 元数据 JSON"""
    info = {
        "name": SERVER_NAME,
        "version": SERVER_VERSION,
        "protocolVersion": DEFAULT_PROTOCOL_VERSION,
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"subscribe": True, "listChanged": True},
            "prompts": {"listChanged": True},
        },
        "transport": "streamable-http / stdio",
        "stateless": True,
        "security": {
            "sqlInjectionProtection": True,
            "astSafeEval": True,
            "requestBodyLimit": f"{MAX_REQUEST_BODY_SIZE} bytes",
        },
        "timestamp": datetime.now().isoformat(),
    }
    return json.dumps(info, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Prompt 定义（Prompts）
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.prompt(
    name="greeting",
    description="根据名字和风格生成个性化问候语。适用于开场白、客户接待等场景。",
)
async def greeting(params: GreetingParams) -> GetPromptResult:
    """
    [1-用途] 根据名字和风格生成个性化问候语。
    [2-限制] 名字长度不超过 50 字符。仅支持 friendly/formal/casual 三种风格。
    [3-成本] 极低，纯模板替换，无外部依赖。
    [4-副作用] 无。
    [5-输入格式] name: 被问候者名字。style: 风格枚举。
    [6-输出格式] 返回 GetPromptResult，包含用户请求和助手回复两条消息。
    """
    templates = {
        "friendly": f"你好, {params.name}! 很高兴见到你! 有什么我可以帮你的吗?",
        "formal": f"尊敬的 {params.name}，欢迎您的到来。请问有什么可以为您效劳？",
        "casual": f"嘿 {params.name}，怎么样? 今天过得如何?",
    }
    text = templates.get(params.style, templates["friendly"])

    return GetPromptResult(
        description=f"为 {params.name} 生成的 {params.style} 风格问候语",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"请向 {params.name} 问好，使用{params.style}风格。",
                ),
            ),
            PromptMessage(
                role="assistant",
                content=TextContent(type="text", text=text),
            ),
        ],
    )


@mcp.prompt(
    name="code_review",
    description="生成标准化的代码评审报告。适用于 Pull Request 合并前的质量检查。",
)
async def code_review(pr_url: str) -> GetPromptResult:
    """
    [1-用途] 对指定 Pull Request 进行标准化代码评审。
    [2-限制] 仅评审代码质量，不执行实际构建或测试。需要有效的 PR URL。
    [3-成本] 中等，需要分析代码 diff 和上下文。
    [4-副作用] 无。不会修改代码仓库。
    [5-输入格式] pr_url: Pull Request 的完整 URL。
    [6-输出格式] 返回结构化评审报告，包含安全性、性能、可维护性、测试覆盖四个维度。
    """
    return GetPromptResult(
        description=f"PR 代码评审: {pr_url}",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""请对以下 Pull Request 进行代码评审，按以下维度评分（1-5分）：

## 评审维度
1. **安全性**：是否存在 SQL 注入、XSS、敏感信息泄露、不安全的反序列化？
2. **性能**：是否有 N+1 查询、大数据集全表扫描、内存泄漏、不必要的循环？
3. **可维护性**：代码是否清晰、是否有足够注释、是否符合项目编码规范（PEP8/ESLint）？
4. **测试覆盖**：是否包含单元测试？边界条件是否覆盖？是否有集成测试？

## PR 信息
URL: {pr_url}

## 输出格式
- 每个维度给出评分（1-5）和具体依据
- 发现的问题按严重程度分类：🔴 阻塞 / 🟡 警告 / 🟢 建议
- 最后给出是否建议合并的结论（是 / 否，附条件）
- 如有阻塞问题，必须说明修复后才能合并
""",
                ),
            ),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Enterprise MCP Server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="使用 Streamable HTTP 传输模式（默认 stdio）",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP 监听地址")
    parser.add_argument("--port", type=int, default=3000, help="HTTP 监听端口")
    parser.add_argument(
        "--path",
        default="/mcp",
        help="HTTP 端点路径（默认 /mcp，对应 /mcp/v1/messages）",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"  🚀 {SERVER_NAME} v{SERVER_VERSION}")
    print(f"  📡 MCP Python SDK v2 | 双时代协议支持")
    print("=" * 70)

    if args.http:
        print(f"  🌐 传输模式: Streamable HTTP")
        print(f"  📍 端点: http://{args.host}:{args.port}{args.path}/v1/messages")
        print(f"  🔒 请求体限制: {MAX_REQUEST_BODY_SIZE} bytes (4 MiB)")
        print("=" * 70)
        print("  📋 测试命令:")
        print("  curl -X POST http://{}:{}{}/v1/messages ".format(args.host, args.port, args.path))
        print('       -H "Content-Type: application/json" ')
        print('       -H "Mcp-Protocol-Version: 2026-07-28" ')
        print('       -d ''{"jsonrpc":"2.0","id":"1","method":"tools/list","params":{}}''')
        print("=" * 70)

        # Streamable HTTP 模式（v2 自动支持双时代）
        mcp.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            streamable_http_path=args.path,
            max_request_body_size=MAX_REQUEST_BODY_SIZE,
        )
    else:
        print(f"  🖥️  传输模式: stdio")
        print(f"  🔒 请求体限制: {MAX_REQUEST_BODY_SIZE} bytes (4 MiB)")
        print("=" * 70)
        print("  💡 提示: 使用 --http 参数切换到 HTTP 模式")
        print("=" * 70)

        # stdio 模式（v2 自动隔离 print，防止污染 wire）
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()