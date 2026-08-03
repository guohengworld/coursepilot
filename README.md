# CoursePilot

> 面向学校课程的 AI 教学助手 —— 用 LangGraph 编排教学工作流，Agentic RAG 提供精准知识检索，MCP 协议将教学能力开放给任意 IDE 与客户端。适用于任意学科。

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![MCP](https://img.shields.io/badge/MCP-2.0-orange)
![Status](https://img.shields.io/badge/status-开发中-yellow)

---

## 项目简介

CoursePilot 是一套**能自主执行教学任务的 AI 教学系统**。它用 LangGraph 有向状态图编排完整的教学工作流：理解学生意图 → 路由到对应 workflow → 节点逐步执行 → 关键步骤独立验证 → 持续改进学生画像。

设计哲学：**纯 Workflow 驱动**——用确定性有向状态图编排教学流程，关键节点内嵌独立验证、确定性护栏、每步追踪与人类接管，而非依赖通用 Planner 动态规划。教学意图有限可枚举、执行路径确定，结果更可预测。适用于任意学科课程。

---

## 核心特性

### 1. LangGraph 教学工作流

用一张有向状态图编排 4 种教学意图（问答 / 练习 / 诊断 / 复习计划），意图分类节点 + 条件边路由到对应子图。关键节点内嵌**生成-验证分离**（条件边 + 重试循环）、**确定性护栏**、**PostgresSaver 断点恢复**、**interrupt() 人类接管**。

### 2. Agentic RAG 引擎

六阶段检索编排：查询改写 → BGE-M3 编码 → 混合检索（Milvus 向量 + BM25 + RRF 融合）→ bge-reranker-v2-m3 重排序 → 知识点金字塔扩展。并具备上下文充足性质检、复杂查询分解、多轮补搜、网络搜索兜底等 Agentic 能力。

### 3. MCP Server

基于 Model Context Protocol 2.0，将课程查询、练习生成、学情诊断、复习计划等能力封装为 MCP 工具/Resources/Prompts，可接入任意 MCP 客户端（已测试 Trae 与 WorkBuddy）。提供 stdio 桥接器与 HTTP/SSE Gateway 两种部署模式。

### 4. RAGAS 评估体系

黄金数据集 + RAGAS 8 项指标 + 配置网格搜索，量化评估 RAG 各阶段表现，支持 baseline 对比与最优配置寻优。

### 5. 治理与可观测性

操作审计日志、RBAC 权限、确定性护栏、LangGraph native tracing、用户画像长期记忆。

---

## 项目结构

```
coursepilot/
├── src/coursepilot/
│   ├── agent/              # LangGraph Agent（graph/nodes/skills/memory）
│   ├── api/                # FastAPI 路由
│   ├── auth/               # JWT 认证
│   ├── evaluation/         # RAGAS 评估
│   ├── governance/         # 治理（audit/guardrails/rbac）
│   ├── ingestion/          # 文档解析管道（PDF/DOCX/Markdown）
│   ├── knowledge/          # 知识点树（分割/构建/大纲解析）
│   ├── mcp/                # MCP Server（gateway/cli/tools/prompts/resources）
│   ├── models/             # SQLAlchemy 模型（14 张表）
│   ├── observability/      # 可观测性
│   ├── rag/                # Agentic RAG 引擎（检索/重排序/生成/改写）
│   └── storage/            # 文件存储（FileStore，UUID 命名）
├── frontend/               # Vue 3 前端
├── scripts/                # 运维脚本（导入/重建/评估/诊断）
├── eval/                   # 评估脚本与数据集
├── tests/                  # 单元/集成/RAG/MCP 测试
├── docs/                   # 设计文档（agent/rag/pipeline/archive）
├── examples/               # MCP 配置示例
├── alembic/                # 数据库迁移
├── docker-compose.yml      # PostgreSQL
├── Makefile                # 开发命令（可选辅助，可能需根据环境调整）
└── pyproject.toml          # 项目配置（uv）
```

---

## 快速开始

### 环境要求

- **Python** ≥ 3.12
- **uv**（包管理器，[安装](https://docs.astral.sh/uv/)）
- **Docker**（用于 PostgreSQL，也可用本地 PG）
- **GPU**（可选，加速 MinerU OCR 与 BGE-M3 编码；CPU 可运行但较慢）

### 安装与运行

```bash
# 1. 克隆仓库
git clone https://github.com/guohengworld/coursepilot.git
cd coursepilot

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，至少填入：
#   DATABASE_URL      PostgreSQL 连接串
#   LLM_API_KEY       DeepSeek API Key
#   JWT_SECRET_KEY    随机字符串（≥32 字符）

# 3. 安装依赖（uv 会创建 .venv 并 editable 安装 coursepilot）
uv sync

# 4. 启动 PostgreSQL 并初始化数据库
docker compose up -d
uv run alembic upgrade head
uv run python -m coursepilot.auth.bootstrap   # 创建首个超级用户

# 5. 导入课程数据（可选，需要教材 PDF）
uv run python -m scripts.seed_knowledge "tests/fixtures/pdfs/你的教材.pdf" --course-name "课程名" --ingest

# 6. 启动服务
uv run uvicorn coursepilot.main:app --reload --host 0.0.0.0 --port 8000   # FastAPI 后端
```

### 前端（Vue 3）

```bash
cd frontend
npm install
npm run dev           # Vite 开发服务器
```

---

## 配置说明

所有配置通过 `.env` 加载（从 [.env.example](.env.example) 复制）。核心项：

| 变量 | 说明 | 示例 |
| ------ | ------ | ------ |
| `DATABASE_URL` | PostgreSQL 异步连接串 | `postgresql+asyncpg://postgres:pwd@localhost:5432/coursepilot` |
| `LLM_API_KEY` | LLM 服务 API Key | `sk-...` |
| `LLM_BASE_URL` | LLM 服务地址 | `https://api.deepseek.com` |
| `LLM_MODEL` | 模型名 | `deepseek-v4-flash` |
| `JWT_SECRET_KEY` | JWT 签名密钥 | 随机字符串 ≥32 字符 |
| `COURSEPILOT_MCP_TRANSPORT` | MCP 启动模式 | `stdio` / `http` |
| `COURSEPILOT_MCP_GATEWAY` | MCP Gateway 地址 | `http://127.0.0.1:8080/mcp` |
| `COURSEPILOT_MCP_API_KEY` | MCP API Key | `cp_your_key` |

完整配置项见 [src/coursepilot/config.py](src/coursepilot/config.py)。

---

## MCP 集成

CoursePilot 的教学能力通过 MCP 协议开放，可接入任意 MCP 客户端（已测试 Trae 与 WorkBuddy）。

### stdio 桥接模式（推荐用于 IDE）

安装后提供 `coursepilot-mcp` 命令。客户端配置示例见 [examples/mcp-config-example.json](examples/mcp-config-example.json)：

```json
{
  "mcpServers": {
    "coursepilot": {
      "command": "python",
      "args": ["-m", "coursepilot.mcp.cli"],
      "env": {
        "COURSEPILOT_MCP_GATEWAY": "http://127.0.0.1:8080/mcp",
        "COURSEPILOT_MCP_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

### HTTP/SSE Gateway 模式（推荐用于多客户端共享）

```bash
# 启动 Gateway（默认 :8080）
COURSEPILOT_MCP_TRANSPORT=http uv run uvicorn coursepilot.main:app --host 0.0.0.0 --port 8000
```

提供的 MCP 能力：4 个工具（课程查询 / 练习生成 / 学情诊断 / 复习计划）、Resources、Prompts。

---

## 评估体系

基于 RAGAS 的量化评估，支持 baseline 对比与配置网格搜索：

```bash
# 基线评估
uv run python -m eval.eval_ragas baseline

# 配置网格搜索
uv run python -m eval.eval_ragas grid --stage 1

# 对比不同配置
uv run python -m eval.eval_ragas compare
```

评估数据集与配置搜索脚本见 [eval/](eval/)，评估体系设计见 [docs/rag/RAG评估体系构建.md](docs/rag/RAG评估体系构建.md)。

---

## 测试

```bash
uv run pytest tests/unit/ -v           # 单元测试（无需外部依赖）
uv run pytest tests/integration/ -v -s # 集成测试（需 PostgreSQL + Milvus + MinerU）
```

测试覆盖：解析器、知识点分割、向量存储、BM25、RAG 组件、Agent 工作流、上下文记忆、MCP 工具、练习 API 等。

---

## 文档

| 文档 | 内容 |
| ------ | ------ |
| [docs/agent/CoursePilot-Agent-Design.md](docs/agent/CoursePilot-Agent-Design.md) | LangGraph Agent 架构设计 v4.0 |
| [docs/agent/Context-Memory-Design.md](docs/agent/Context-Memory-Design.md) | 上下文与记忆系统设计 |
| [docs/rag/RAG_Engine_Design.md](docs/rag/RAG_Engine_Design.md) | RAG 引擎设计 |
| [docs/rag/RAG评估体系构建.md](docs/rag/RAG评估体系构建.md) | RAGAS 评估体系构建 |
| [docs/Agentic_RAG_实现方案.md](docs/Agentic_RAG_实现方案.md) | Agentic RAG 实现方案 |
| [docs/RAG_Optimization_Roadmap.md](docs/RAG_Optimization_Roadmap.md) | RAG 优化路线图 |
| [docs/pipeline/ingestion-pipeline.md](docs/pipeline/ingestion-pipeline.md) | 文档导入管道设计 |
| [CLAUDE.md](CLAUDE.md) | AI 的项目级长期记忆 |

---

## 常用命令速查

| 命令 | 作用 |
| ------ | ------ |
| `uv sync` | 安装依赖 |
| `uv run uvicorn coursepilot.main:app --reload` | 启动 FastAPI 后端 (:8000) |
| `uv run python -m scripts.seed_knowledge PDF --course-name 课程 --ingest` | 从 PDF 种子化课程 |
| `uv run python -m scripts.batch_ingest` | 批量导入教材 |
| `uv run python -m scripts.rebuild_all` | 清空重建所有课程数据 |
| `uv run alembic upgrade head` | 运行数据库迁移 |
| `uv run pytest tests/unit/ -v` | 单元测试 |
| `uv run ruff check src/ tests/ scripts/ eval/` | 代码检查 |

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

## 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) — 有向状态图编排
- [MinerU](https://github.com/opendatalab/MinerU) — PDF 智能解析
- [BGE](https://github.com/FlagOpen/FlagEmbedding) — 向量化与重排序模型
- [RAGAS](https://github.com/explodinggradients/ragas) — RAG 评估框架
- [MCP](https://modelcontextprotocol.io) — Model Context Protocol
