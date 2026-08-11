# AGENTS.md

本文档是给 AI 编码 Agent 的项目指南（"Agent 版 README"）。 humans 看 [README.md](README.md)，本文件只放 Agent 工作需要知道的上下文与命令。

## Project overview

CoursePilot 是一个面向高校课程的 **AI 教学助手**：把课程资料转成可检索知识库，并用 **LangGraph 编排的 Agentic RAG** 回答学生问题、生成练习、诊断学情、制定复习计划。后端为 FastAPI，另有 Vue 3 前端与 Streamlit 调试 UI。

Agentic RAG 的自主检索链路：查询改写 → 混合检索（向量 + BM25 + RRF 融合）→ 重排 → 充分性自检 → 多轮补搜 → 网络搜索兜底。

## Tech stack

- **Web**：FastAPI（`uvicorn` 启动），`/api/v1` 下挂载 `auth` / `courses` / `agent` / `practice` / `admin` 五个 router。Python ≥ 3.12，[`uv`](https://github.com/astral-sh/uv) 管理依赖，`src/` 布局，包名 `coursepilot`。
- **数据库**：PostgreSQL（SQLAlchemy 2 async + `asyncpg`），Alembic 做迁移。开发环境直连本机 PostgreSQL（`docker-compose.yml` 仅起 PG，是可选的容器化方案，非必须）。
- **向量与检索**：Milvus **Lite** 嵌入式（`data/milvus/milvus.db`，无需独立向量服务）+ BM25 + RRF 融合；嵌入用 **BGE-M3**、重排用 **bge-reranker-v2-m3**（均 `FlagEmbedding`，CPU 推理）。
- **LLM**：DeepSeek，经 OpenAI 兼容客户端 `AsyncOpenAI` 调用，`llm_timeout` 默认 60s。`LLM_MODEL` 默认 `deepseek-v4-flash`（请核实你账户下的真实可用模型名）。
- **Agent**：LangGraph 工作流 + `langgraph-checkpoint-postgres` 做会话持久化。
- **MCP**：`coursepilot.mcp.gateway.app`（Streamable HTTP，默认 `0.0.0.0:8080`）+ stdio→HTTP 桥接器 `coursepilot.mcp.cli`（安装后提供 `coursepilot-mcp` 控制台命令）。鉴权为 `Authorization: Bearer <key>`，Gateway 启动时从 `COURSEPILOT_MCP_API_KEY` / `COURSEPILOT_MCP_API_KEYS` 一次性加载进 `KeyStore` 单例。
- **部署形态**：单实例，无 Redis、无消息队列、无多活/单元化。

## Setup

```bash
# 1. 安装依赖到 .venv
uv sync

# 2. 配置环境变量（模板见 .env.example，需至少填 LLM_API_KEY / DATABASE_URL）
cp .env.example .env
#    编辑 .env：LLM_API_KEY、DATABASE_URL 等

# 3. 准备 PostgreSQL（二选一）
#    a) 本机已装 PG：直接保证 DATABASE_URL 指向可达实例
#    b) 用容器：make db-up        # 等价于 docker compose up -d

# 4. 建表 + 首个超级用户
make migrate      # alembic upgrade head
make bootstrap    # python -m coursepilot.auth.bootstrap
```

> 项目 Makefile 已 `export PYTHONPATH := src`。直接用 `python -m ...` 跑模块时，若不在 Makefile 内执行，需自行 `export PYTHONPATH=src`（Windows 用 `set PYTHONPATH=src`）。

## Common commands

```bash
make run-api            # 启动 FastAPI：uvicorn coursepilot.main:app --reload --port 8000
make run-ui             # 启动 Streamlit 调试 UI
make lint               # ruff check + ruff format --check（src/ tests/ scripts/ eval/）
make format             # ruff check --fix + ruff format
make typecheck          # mypy src/
make test               # pytest tests/unit/ -v
make test-integration   # pytest tests/integration/ -v -s

# MCP Gateway（需先设 key，且 key 与客户端配置一致）
set COURSEPILOT_MCP_API_KEY=cp_test1234
.venv/Scripts/python -m uvicorn coursepilot.mcp.gateway.app:create_app --factory --port 8080
```

## Code style

- 统一用 **ruff**（`line-length = 100`，`target-version = py312`）；`ruff format` 负责格式化。
- 类型检查用 **mypy**（`strict = false`）。
- 后端以 **async** 为主（FastAPI / SQLAlchemy async / `AsyncOpenAI`）。
- 函数与公共 API 写类型注解；新增依赖须在 `pyproject.toml` 声明理由。

## Testing

- 测试框架 **pytest**，`asyncio_mode = auto`。
- 目录：`tests/unit/`（单元）、`tests/integration/`（集成）、`tests/` 下另有 `rag` / `mcp`；`e2e` 标记用于全 mock 外部依赖的端到端测试。
- 改代码后跑 `make test`；涉及类型再跑 `make typecheck`；提交前确保 `make lint` 通过。

## Security considerations

- **CORS（待修）**：`src/coursepilot/main.py` 当前为 `CORSMiddleware(allow_origins=["*"], allow_credentials=True)`——既是 `*` + 凭据的不安全组合，浏览器也会拒绝该组合（功能缺陷）。部署前须把 `allow_origins` 限定为具体前端域名；新增/修改 CORS 配置前先确认。
- **密钥**：所有密钥（LLM key、MCP key、JWT secret）经 `.env`（`pydantic-settings`）加载，禁止硬编码；`.env` 在 `.gitignore`。
- **MCP 鉴权**：Gateway 的 `KeyStore` 在进程启动时从环境变量一次性加载，运行中改 `.env` 不会热生效，必须重启 Gateway；`/mcp` 端点要求 `Bearer` 头。

## Machine-specific config（换机器必读）

- `config.py` 中 `embedding_model_path` / `reranker_model_path` 默认值是**作者本机绝对路径**（`F:/all-projs/models/...`），干净机器上不存在会加载失败。换机器前用以下环境变量覆盖为 HuggingFace 模型 id（首次运行自动下载）或你自己的本地路径：
  - `EMBEDDING_MODEL_PATH=BAAI/bge-m3`
  - `RERANKER_MODEL_PATH=BAAI/bge-reranker-v2-m3`
  - `FlagEmbedding` 的 `BGEM3FlagModel` / `FlagReranker` 同时支持本地路径与 HF id。
- 配置（`.env` / 环境变量）改动**不会**被已运行进程热感知，相关进程需重启。
- `deepseek-v4-flash` 为默认 LLM 模型名，请核实你 DeepSeek 账户/套餐下是否真实可用；不可用则通过 `LLM_MODEL` / `LLM_BASE_URL` 覆盖。
