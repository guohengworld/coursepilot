# Coursepilot MCP 层重构 TODO（详细执行清单）

> 状态：实施中（P1-T1 ✅ 已完成，2026-08-10；P1-T2/P1-T3 待启动）
> 更新：2026-08-10
> 用途：把设计文档的 P1–P4 拆成可勾选的具体动作，每个动作标注「涉及文件 / 步骤 / 验收标准 / 风险点」。
> 依赖顺序：P1-T1（Principal 基座 + 透传测试）必须先完成，P1-T2、P1-T3 依赖它；P2/P3 在 P1 之后；P4 按需。

---

## 2026-08-10 核对结论（新增）

对照设计文档逐一核验当前实现（含 mcp 2.0.0 API 实测、测试全量运行）：

| 项 | 结论 |
|---|---|
| `gateway/main.py` 无状态 Streamable HTTP + 访问日志脱敏 | ✅ 正确，保留（设计"已做对"栏一致） |
| `gateway/auth.py` key 校验（前缀脱敏 / 角色映射） | ✅ 正确，测试全绿 |
| `server.py` 工具描述 / ToolAnnotations / Resource URI | ✅ 正确，保留 |
| `cli/main.py` stdio 桥接 | ✅ 正确，测试全绿 |
| **SSE 挂载 `sse_app`** | ❌ 已弃用传输，待删（代码已加 `[DEPRECATED-MARK]`） |
| **私有访问 `mcp._lowlevel_server._session_manager`** | ❌ 待消除（代码已加 `[PRIVATE-API-MARK]`）；**注意：无状态模式同样需要 session manager（实测）**，正确姿势是直接挂 SDK app 用自带 lifespan |
| **`tests/unit/test_mcp.py`** | ❌ 过时：9 个用例全失败（测旧 API：位置参数签名、已删除函数），已标记模块级 skip，待重写或删除 |
| 设计 §4.2 / §11 备用路径 `ctx.request_context.request.headers` | ❌ 实测不存在，应为 `ctx.headers`（已修正设计文档） |
| 设计 §7 "无状态不需要手动 session manager" | ❌ 实测错误（已修正设计文档） |

**本次已落地**：
1. 核对阶段：删除临时探测脚本；`tests/unit/test_mcp.py` 加 skip 标记；`gateway/main.py` 加两处待清理标记；设计文档 §4.2/§7/§10/§11 与本节 TODO 同步修正。
2. **P1-T1 实施（2026-08-10 完成）**：新增 `principal.py` + `auth/{keys,middleware,policy}.py`；`shared/errors.py` 加 `MCPError` / `UnauthenticatedError` / `ToolForbiddenError`；新增 `tests/test_mcp/test_principal_injection.py` 9 用例全绿。**ContextVar 主路径透传实测成立**（中间件 set → 工具 handler 内 `get_principal()` 可读），备用路径未触发。全量回归 87 passed / 10 skipped。

---

## 总览（勾选进度）

- [ ] **P1 安全收口**
  - [x] P1-T1 搭建 Principal 注入基座 + ContextVar 透传集成测试（9 测试全绿）
  - [ ] P1-T2 为 3 工具 / 2 资源加租户断言
  - [ ] P1-T3 清理 gateway/main.py：去 SSE + 去私有依赖
  - [ ] P1-T4（新增）重写或删除过时测试 `tests/unit/test_mcp.py`（现为模块级 skip）
- [ ] **P2 鉴权硬化**（Key 启动载入 + role→scope 强制 + 头路由审计）
- [ ] **P3 可观测**（OTel 工具调用追踪 + 返回内容不可信标注）
- [ ] **P4 生产 auth**（OAuth 2.1 + PKCE + DPoP 适配，仅换 middleware）

---

## P1 安全收口

### P1-T1 搭建 Principal 注入基座 + ContextVar 透传集成测试

> **状态：✅ 已完成（2026-08-10）**。9 个集成测试全绿；**ContextVar 主路径透传实测成立**（无需备用路径）。

**目标**：把「调用方身份」从网关注入到工具/资源层，并验证 `ContextVar` 在 MCP 工具 handler 内能读到。这是关闭 P0 越权的地基。

**涉及文件（新增）**：
- `src/coursepilot/mcp/principal.py` ✅
- `src/coursepilot/mcp/auth/__init__.py`（新包）✅
- `src/coursepilot/mcp/auth/keys.py` ✅
- `src/coursepilot/mcp/auth/middleware.py` ✅
- `src/coursepilot/mcp/auth/policy.py` ✅
- `tests/test_mcp/test_principal_injection.py`（新增集成测试，与现有 `test_gateway` / `test_stdio` 同目录）✅

**涉及文件（已存在，仅扩展）**：
- `src/coursepilot/mcp/shared/errors.py`（已含 `MCPErrorCode` / `ERROR_MESSAGES` / `make_rpc_error`，加 `UnauthenticatedError` / `ToolForbiddenError` 异常类即可）✅

**步骤**：
- [x] `principal.py`：`@dataclass(frozen=True) class Principal(user_id: str, role: str, scopes: frozenset[str])`
- [x] `principal.py`：`principal_var: ContextVar[Principal | None] = ContextVar("cp_principal", default=None)`
- [x] `principal.py`：`get_principal() -> Principal`，为 `None` 时抛 `UnauthenticatedError`
- [x] `shared/errors.py`：定义 `UnauthenticatedError` / `ToolForbiddenError`（含稳定错误码，供工具层 `CallToolResult.isError` 映射）
- [x] `auth/keys.py`：`KeyStore` 启动时从环境变量/secret 一次性载入 `{key: ApiKeyInfo(user_id, role, scopes)}`；提供 `lookup(key) -> ApiKeyInfo | None`
- [x] `auth/middleware.py`：`AuthenticationMiddleware`（Starlette `BaseHTTPMiddleware` 或原生 ASGI）；解析 `Authorization: Bearer cp_xxx`；查 `KeyStore`；通过后 `principal_var.set(Principal(...))`；失败返回 401
- [x] `auth/middleware.py`：访问日志脱敏（仅记 key 前缀 + tool 名，不记参数/响应，符合 R4.9）
- [x] `auth/policy.py`：`require_self_or_privileged(*privileged_roles)` 装饰器（断言 `params.user_id == principal.user_id` 或 role 在特权集）
- [x] `auth/policy.py`：`require_scope(*scopes)` 装饰器（断言 `principal.scopes` 覆盖所需 scope）

**集成测试（最关键，先写先跑）**：
- [x] `test_principal_injection.py`：构造请求流经中间件 → `principal_var.set` → 在模拟工具 handler 内调用 `get_principal()`，断言能读到身份
- [x] 断言未 `set` 时工具抛 `UnauthenticatedError`
- [x] **验证 `ContextVar` 跨 MCP 工具 handler 所在 task 边界的透传**：直接对 `MCPServer` 注册一个测试工具，经 `streamable_http_app` 真实发请求，断言工具内 `get_principal()` 非空
- [x] 若上一条失败：切换到备用路径——~~网关注入 `X-Principal-User-Id` / `X-Principal-Role` / `X-Principal-Scopes` 头，工具从 `ctx.headers` 读取~~ **（未触发：主路径透传已成立；`ctx.headers` 属性存在性仍留有防御性测试）**

**验收标准**：
- ✅ 集成测试全绿；`ContextVar` 透传成立（主路径，实测 `principal=u-001:student:['read']` 在工具 handler 内可读）
- ✅ 未走网关直连调用工具时，统一抛 `UnauthenticatedError`（工具返回 isError），不会以匿名身份执行

**风险点**：`ContextVar` 跨 SDK 内部 task 边界透传未经验证，必须先测；失败则走头注入（设计文档 §4.2）。**结论：实测透传成立，风险关闭。** 附加发现：SDK 捕获工具异常后错误文本为 `Error executing tool <name>: <消息>`（不含类型名），测试按 isError + 消息断言。

---

### P1-T2 为 3 工具 / 2 资源加租户断言

**目标**：关闭 P0 跨租户越权——学生只能读自己，teacher/super 可读任意。

**涉及文件（修改）**：
- `src/coursepilot/mcp/tools/tutor.py`（`diagnose`、`get_review_plan`）
- `src/coursepilot/mcp/tools/practice.py`（`generate_practice`、`grade_answers`）
- `src/coursepilot/mcp/resources/course.py`（`read_report`、`read_mastery`）

**步骤**：
- [ ] `diagnose`：入口加 `@require_self_or_privileged("teacher", "super")`（学生只能查自己）
- [ ] `get_review_plan`（在 `tutor.py`）：加租户断言 + `@require_scope("read")`
- [ ] `generate_practice`（在 `practice.py`）：加 `@require_scope("write")`（仅 teacher/super 可生成练习）
- [ ] `grade_answers`（在 `practice.py`）：加租户断言（`params.user_id` 须与 principal 一致，或 role ∈ privileged）
- [ ] `read_report` / `read_mastery`：入口做 `user_id` 作用域校验（资源层无装饰器，手写断言调用 `get_principal()`）
- [ ] 工具描述文案（`[1-用途]…[6-输出格式]`）保持不变，不破坏现有强项

**验收标准**：
- 持 `student` key、传他人 `user_id` → 返回 `ToolForbiddenError`（或 `isError`），不泄露他人数据
- 持 `teacher`/`super` key、传任意 `user_id` → 正常执行
- 单元测试覆盖上述两分支（可并入 `test_principal_injection.py` 或新建 `test_tenancy.py`）

**风险点**：`generate_practice`/`get_review_plan` 当前 `role` 仅记日志未强制，改动后需确认无现有调用依赖 student 可调用。

---

### P1-T3 清理 gateway/main.py：去 SSE + 去私有依赖

**目标**：去掉已弃用 SSE 传输与双下划线私有依赖，让网关只剩无状态 Streamable HTTP。

**涉及文件（重构）**：
- `src/coursepilot/mcp/gateway/main.py` → 拆为 `gateway/app.py` + `gateway/observability.py`（新增）
- 删除或归档 `gateway/main.py`

**步骤**：
- [ ] `gateway/app.py`：`create_app()` 仅挂载 `mcp.streamable_http_app(stateless_http=True, json_response=True)`
- [ ] 移除 `mcp.sse_app(sse_path="/sse", message_path="/messages/")` 及其路由合并（当前代码已加 `[DEPRECATED-MARK]` 注释标记）
- [ ] 删除 `_lifespan` 中对 `mcp._lowlevel_server._session_manager` 的双下划线私有访问（当前代码已加 `[PRIVATE-API-MARK]` 注释标记）
- [ ] **lifespan 处理（2026-08-10 实测修正）**：无状态模式**同样需要** `session_manager.run()` 初始化 task group（`_handle_request` 首行检查 `_task_group`，不启会抛 `RuntimeError: Task group is not initialized`）。SDK 返回的 Starlette app 自带 `lifespan=session_manager.run`——正确做法是**直接挂载 SDK 返回的 app**（或经 `custom_starlette_routes` 注入 /health 等自定义路由），从而无需手动 lifespan 与私有访问；不要沿用"合并 routes 到 FastAPI + 手写 lifespan"的旧结构
- [ ] 审计逻辑（`_extract_tool`）改为优先读 `MCP-Protocol-Version` / `Mcp-Method` / `Mcp-Name` 头，body 解析仅兜底
- [ ] 拆分 `gateway/observability.py`：承接现有访问日志（延迟/状态/tool 名，脱敏）
- [ ] 更新 `cli/main.py` 与各调用方指向新的 `create_app()` 入口；同步更新 `tests/test_mcp/test_gateway.py`（移除 /sse 相关 2 个用例）与 `tests/test_mcp/test_import.py`（`gateway.main` → `gateway.app`）

**验收标准**：
- 仅存在无状态 HTTP 端点，无 `/sse`、`/messages/` 端点
- `grep -rn "_lowlevel_server\|_session_manager\|sse_app" src/coursepilot/mcp/` 无残留
- 现有 stdio 桥接（`cli/main.py`）与网关集成测试仍通过

**风险点**：~~删除私有依赖后需确认无状态入口不依赖手动 session manager 启动~~ **（已实测：无状态也依赖 task group，故必须用 SDK 自带 lifespan 直接挂载，而不是"删除后不挂"）**；`/sse` 用例删除前先改 `test_gateway.py`，避免测试红。

---

### P1-T4 重写或删除过时测试 `tests/unit/test_mcp.py`（新增）

**背景（2026-08-10 实测）**：该文件 9 个用例全部失败——测的是 MCP 重构前的旧 API：位置参数签名（`query_knowledge("query", "course-id")`）、已删除函数（`student_report` / `query_mastery` / `diagnose_weakness`）与旧 `server.py` 单文件结构。协议级覆盖已由 `tests/test_mcp/` 提供，本文件当前是模块级 `pytest.mark.skip`。

**涉及文件**：`tests/unit/test_mcp.py`

**步骤（二选一）**：
- [ ] 重写：按新 API（`params: XxxParams -> CallToolResult`）为 `tools/{tutor,practice,knowledge}.py` 补单测；P1-T2 的租户断言正/反用例可并入（student 传他人 user_id → `ToolForbiddenError`；teacher/super → 放行）
- [ ] 或直接删除（保留 git 历史）

**验收标准**：
- `tests/unit/` 下不再有引用旧 API 的测试；全量测试无失败
- 删除 `pytestmark = pytest.mark.skip` 后不引入新的失败

**风险点**：删除前确认 `tests/test_mcp/` 已覆盖原单测价值（已验证：test_schemas 15 例、test_validation 7 例、test_errors 5 例、test_gateway / test_stdio 协议级覆盖）。

---

## P2 鉴权硬化

**目标**：让「角色 / scope / key 轮换」真实生效，而非只在日志里看看。

**涉及文件**：`auth/keys.py`、`auth/middleware.py`、`auth/policy.py`（在 P1 基础上增强）

**步骤**：
- [ ] `keys.py`：启动时一次性载入 key 表（不再每次请求 `os.getenv` 解析 JSON）；提供进程内字典 + 可选 `/reload` 端点触发热重载
- [ ] `policy.py` + `middleware.py`：把 `role → scopes` 映射集中定义，`generate_practice`/`get_review_plan` 强制 `require_scope("write"/"read")`（P1-T2 已埋点，此处接通）
- [ ] `middleware.py` 审计：优先读 2.0 头路由（`Mcp-Method`/`Mcp-Name`），body 解析兜底（与 P1-T3 共用）
- [ ] key 轮换：支持新增/吊销 key 后经 `/reload` 或重启生效，不丢在线服务

**验收标准**：
- 持 `student` key 调用 `generate_practice` → 被 `require_scope` 拒绝
- key 轮换（`/reload`）后新 key 立即生效、旧 key 立即失效
- 不再每次请求解析环境变量（性能项，可用简单基准验证）

**风险点**：`/reload` 端点本身需鉴权，避免成为越权入口（建议仅内网/运维端口）。

---

## P3 可观测

**目标**：每个工具调用有分布式 trace，返回内容标注不可信。

**涉及文件**：`gateway/observability.py`（新增 tracing）、`resources/course.py`、`tools/knowledge.py`、`tools/tutor.py`

**步骤**：
- [ ] `observability.py`：每 `tools/call` 生成一个 span，打 `gen_ai.tool.name` / `mcp.server.url`，继承 W3C Traceparent（2.0 规范强制）
- [ ] 与现有 `observability/metrics.py`（业务统计）分层并存，互不污染
- [ ] `read_report` / `read_mastery` / `grade_answers` 返回内容（学员自写文本）标注「不可信」，供消费方做注入/PII 过滤（防御 GitHub MCP 被恶意 Issue 劫持类事故）
- [ ] 可选：`tools/list` 等返回 `ttlMs`/`cacheScope`，稳定上游 prompt 缓存

**验收标准**：
- trace 中能看到每个工具调用的 span 与耗时（p95/p99 可算）
- 返回内容在接口/文档层明确标记为不可信，且有基础内容扫描（可选）

**风险点**：OTel SDK 引入新依赖，需确认与现有 `pyproject.toml` 依赖无冲突。

---

## P4 生产 auth（按需）

**目标**：达到生产级鉴权标准，且不影响业务代码。

**涉及文件**：仅 `auth/middleware.py`（替换校验逻辑）

**步骤**：
- [ ] 仅替换 `auth/middleware.py` 的校验函数为 OAuth 2.1 + PKCE + Resource Indicators (RFC 8707) + DPoP (RFC 9449)
- [ ] 接入 Auth0 / Okta / WorkOS（**不自写授权服务器**）
- [ ] 工具 / 资源 / 策略层零改动（依赖 P1 的 `Principal` 抽象已解耦）

**验收标准**：
- OAuth token 可正常鉴权并注入 `Principal`
- `git diff` 显示仅 `auth/middleware.py` 变更，工具/资源代码未动

**风险点**：OAuth 元数据（issuer 校验 RFC 9207、CIMD）需按 2026-07-28 鉴权硬化要求配置，避免 mix-up 攻击。

---

## 完成定义（Definition of Done）按阶段

| 阶段 | 完成标志 |
|---|---|
| P1 | P0 越权关闭（集成测试证明 student 跨租户被拒）；仅无状态 HTTP 端点；无私有依赖残留 |
| P2 | role/scope 强制生效；key 可轮换；无每次请求读 env |
| P3 | 工具调用有 trace；返回内容不可信标注到位 |
| P4 | OAuth 2.1+PKCE+DPoP 接通，业务代码零改动 |

---

## 备注

- 所有 API 基于已装 `mcp` 2.0.0 实测（`MCPServer`、`streamable_http_app(stateless_http=True)`、`Context.headers` / `ctx.transport.headers` 等均已核验存在；**`ctx.request_context` 不存在，勿用**）。
- 本清单与任务系统里的任务同源；任务系统用于跟踪勾选，本文档用于查看详细步骤。
- 范围外（明确暂缓）：OAuth 适配（P4 按需）、MCP Apps/Tasks 扩展、多实例 key 进程外存储。
