# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在处理本代码仓库时提供指导。

开发规范见 `AGENTS.md`（代码真实性、调试纪律、安全规则、执行纪律），处理本仓库时须一并遵循。

## 常用命令

```bash
# 运行所有单元测试
PYTHONPATH=src .venv/Scripts/python -m pytest tests/unit/ -v

# 运行单个测试
PYTHONPATH=src .venv/Scripts/python -m pytest tests/unit/test_week2.py::TestKPSplitter::test_exact_heading_match -v

# 运行较慢的真实管道集成测试（MinerU，需要数据库）
PYTHONPATH=src .venv/Scripts/python -m pytest tests/integration/ -v -s

# 代码规范检查
.venv/Scripts/ruff check src/ tests/ scripts/ eval/

# 类型检查
.venv/Scripts/mypy src/

# 启动 API 服务器
.venv/Scripts/uvicorn coursepilot.main:app --reload

# 初始化引导（创建首个超级用户）
PYTHONPATH=src .venv/Scripts/python -m coursepilot.auth.bootstrap

# 从 PDF 种子化课程数据（KP 树 + 知识单元，单次解析）
PYTHONPATH=src .venv/Scripts/python -m scripts.seed_knowledge "tests/fixtures/pdfs/书名.pdf" --course-name "课程名" --ingest

# 批量导入全部 8 本教材（5 门课程）
PYTHONPATH=src .venv/Scripts/python -m scripts.batch_ingest

# 格式化检查
.venv/Scripts/ruff format --check src/ tests/

# 启动数据库（PostgreSQL）
docker compose up -d
docker compose down

# RAGAS 评估
PYTHONPATH=src EMBEDDING_MODEL_PATH="F:/all-projs/models/bge-m3" .venv/Scripts/python -m eval.eval_ragas baseline
PYTHONPATH=src EMBEDDING_MODEL_PATH="F:/all-projs/models/bge-m3" .venv/Scripts/python -m eval.eval_ragas grid --stage 1
PYTHONPATH=src EMBEDDING_MODEL_PATH="F:/all-projs/models/bge-m3" .venv/Scripts/python -m eval.eval_ragas compare

# Alembic 数据库迁移
alembic revision --autogenerate -m "描述信息"
alembic upgrade head
```

## 系统架构

CoursePilot 是一个面向学校课程的 AI 教学助手。后端技术栈：FastAPI + SQLAlchemy 异步模式 + PostgreSQL。文档解析：MinerU（PDF OCR）+ python-docx + 自定义 Markdown 解析器。RAG 引擎：BGE-M3 编码 + Milvus Lite 向量存储 + bge-reranker-v2-m3 重排序 + DeepSeek LLM 生成。上层另有：Agent（LangGraph 编排 + 上下文记忆）、MCP 服务器（stdio 桥接 + 工具）、治理层（审计日志 / RBAC / 生成护栏）、评估子系统（RAGAS）。前端为 `frontend/`（Vue 3 + Vite + Element Plus）。

## 关键文件清单

| 文件路径 | 职责说明 |
| :--- | :--- |
| src/coursepilot/config.py | 所有配置项，通过 pydantic-settings 从 .env 加载 |
| src/coursepilot/db.py | 异步引擎，FastAPI 依赖注入用的 get_session()，脚本用的 get_session_etx() |
| src/coursepilot/main.py | FastAPI 应用入口，注册 auth/courses/agent/practice/admin 等路由 |
| src/coursepilot/api/auth.py | 注册、登录（JWT）、获取当前用户信息 |
| src/coursepilot/api/courses.py | 课程增删改查、上传并触发导入、文档列表、KP 树查询 |
| src/coursepilot/api/deps.py | get_current_user（JWT → User）、require_superuser 权限校验 |
| src/coursepilot/api/admin.py | 管理路由（用户/课程管理、审计查询） |
| src/coursepilot/api/agent.py | Agent 会话、问答路由 |
| src/coursepilot/api/practice.py | 练习、诊断、复习路由 |
| src/coursepilot/ingestion/pipeline.py | run_ingestion() — B1→B6 流程编排 |
| src/coursepilot/ingestion/pdf_parser.py | parse_pdf() — 封装 MinerU CLI，返回 {markdown, content_list} |
| src/coursepilot/ingestion/docx_parser.py | parse_docx() — 同步解析（python-docx），返回相同格式 |
| src/coursepilot/ingestion/markdown_parser.py | parse_markdown() — 同步解析，相同的 content_list 格式 |
| src/coursepilot/ingestion/parser_utils.py | extract_knowledge_units() — _split_by_headings() +_split_text() |
| src/coursepilot/knowledge/kp_splitter.py | KPSplitter — 将文本块匹配到 KP（精确匹配→清洗后匹配→关键词匹配→根节点回退） |
| src/coursepilot/knowledge/syllabus_parser.py | SyllabusParser — Markdown/中文编号大纲 → 树节点 |
| src/coursepilot/knowledge/kp_tree.py | KPTree — 递归 CTE 查询，使用 parent_id 回填的批量插入 |
| src/coursepilot/storage/file_store.py | FileStore — 本地文件系统存储，文件保存在 data/uploads/ 下，使用 UUID 命名 |
| src/coursepilot/rag/vector_store.py | VectorStore — Milvus Lite 向量存储 CRUD + 混合检索 + RRF |
| src/coursepilot/rag/encoder.py | Encoder — BGE-M3 编码器（dense 1024-dim + sparse） |
| src/coursepilot/rag/retriever.py | Retriever — 六阶段检索编排（改写→编码→混合检索→重排序→KP 扩展） |
| src/coursepilot/rag/reranker.py | Reranker — bge-reranker-v2-m3 cross-encoder 重排序 |
| src/coursepilot/rag/generator.py | Generator — DeepSeek LLM 生成（含流式 SSE） |
| src/coursepilot/rag/query_rewriter.py | QueryRewriter — DeepSeek 查询改写 |
| src/coursepilot/rag/summary_bridge.py | SummaryBridge — 知识单元摘要并发生成（thinking=disabled） |
| src/coursepilot/rag/citation.py | Citation — 引用来源格式化 |
| src/coursepilot/evaluation/rag_eval.py | RAGEvaluator — RAGAS 四大指标评估器 |
| src/coursepilot/models/ | 14 张表的 SQLAlchemy ORM 模型定义 |
| src/coursepilot/auth/ | 密码哈希（passlib）、JWT 编解码、bootstrap 初始化 |
| src/coursepilot/agent/ | LangGraph Agent（节点、技能、记忆 compactor/context_manager/retriever） |
| src/coursepilot/mcp/ | MCP 服务器（CLI 入口、stdio 网关、工具/资源定义） |
| src/coursepilot/governance/ | 审计日志、RBAC、生成护栏（guardrails） |
| frontend/ | Vue 3 + Vite + Element Plus 前端 |
| scripts/seed_knowledge.py | CLI 工具：解析 PDF → 提取标题 → 构建 KP 树；--ingest 参数可同时执行知识单元导入 |
| scripts/batch_ingest.py | 批量处理 tests/fixtures/pdfs/ 下的全部 8 个 PDF，按课程分组 |
| scripts/rebuild_all.py | 清空重建所有课程的知识点和知识单元 |
| eval/eval_ragas.py | RAGAS 评估 CLI（baseline / grid / compare） |

## 数据目录

- `data/milvus/` — Milvus Lite 向量数据库文件
- `data/parsed/` — MinerU 解析输出（临时调试用）
- `data/uploads/` — 上传的文件存储

## API 设计文档

- `docs/pipeline/` — 导入管道设计文档
- `docs/rag/` — RAG 引擎设计文档
- `docs/archive/` — 历史评估报告、设计初稿等归档文档

## 数据库模型（14 张表）

- `User` → `Course` → `KnowledgePoint`（邻接表模型，`parent_id` 自引用，深度 ≤ 4）→ `KnowledgeUnit`（带 `kp_id` 的文本块）。
- `User` → `Document`（上传的文件，状态流转：pending → processing → ready/failed）。
- `User` → `Question`、`PracticeRecord`、`DiagnosisReport`、`ReviewPlan`、`QARecord`、`EvalMetric`。
- `UserProfile` — 用户画像（薄弱知识点、学习进度等）。
- `AgentSession` — Agent 会话管理（LangGraph 的 thread_id + checkpoint）。
- `AuditLog` — 操作审计日志。

## API 路由

- `POST/GET /api/v1/auth/register|login`，`GET /auth/me`
- `GET/POST /api/v1/courses`，`GET/DELETE /courses/{id}`
- `POST /courses/upload`（multipart 表单：file + course_id，触发内联导入）
- `GET /courses/{id}/documents`，`DELETE /courses/{id}/document/{doc_id}`
- `GET /courses/{id}/knowledge-points`（返回带 parent_id 的扁平列表，供前端渲染树形结构）

## 配置说明

配置文件为项目根目录下的 `.env`。必填项：`DATABASE_URL=postgresql+asyncpg://...`、`JWT_SECRET_KEY=...`。所有默认值详见 `config.py`。

易错配置（默认值可能误导）：

- `LLM_MODEL` 默认 `deepseek-v4-flash`。
- 本地模型路径硬编码在 `config.py`：`F:/all-projs/models/bge-m3`、`F:/all-projs/models/bge-reranker-v2-m3`。
- MCP API key 双别名：`COURSEPILOT_MCP_API_KEY`（优先）/ 旧别名 `MCP_API_KEY`。
- `main.py` 在任何 torch import 前设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，勿移动该行。

## 测试结构

```text
tests/
├── unit/                   # 单元测试（无需数据库/外部服务）
│   ├── test_week2.py       # 解析器、大纲提取、KP 分割器、文件存储
│   ├── test_rag.py         # RAG 组件单元测试
│   ├── test_vector_store.py # 向量存储测试
│   ├── test_summary_bridge.py # SummaryBridge 测试
│   ├── test_import.py      # 导入验证
│   ├── test_bm25.py        # BM25 检索测试
│   ├── test_agent_phase1.py # Agent 阶段 1 测试
│   ├── test_agent_phase2.py # Agent 阶段 2 测试
│   ├── test_context_memory.py # 上下文记忆测试
│   ├── test_practice_api.py # 练习 API 测试
│   ├── test_mcp.py         # MCP 工具测试
│   └── test_phase3.py      # 阶段 3 综合测试
├── integration/            # 集成测试（需要 PostgreSQL + MinerU + Milvus）
│   ├── test_real_pipeline.py # 完整导入管道
│   ├── test_ingestion.py    # PDF/DOCX 解析器
│   ├── test_gpu_availability.py # GPU 环境检测
│   ├── test_agent_workflow.py # Agent 端到端工作流
│   └── test_agent_db.py    # Agent 数据库集成
├── test_mcp/                # MCP 专项测试（CLI、网关、stdio、schema 校验）
├── rag/                    # RAG 专项测试
│   └── test_ragas.py       # RAGAS 评估
├── milvus/                 # Milvus 专项测试
│   └── test_milvus_data.py # Milvus 数据验证
└── fixtures/               # 测试数据
    ├── pdfs/               # 8 本教材 PDF
    └── exported_units.json
```

## 关键设计模式

- 所有数据库操作均为异步（`AsyncSession`）。脚本中使用 `get_session_etx()`，FastAPI 路由中使用 `Depends(get_session)`。
- `KnowledgePoint.kp_path`（如 `"微积分/定积分/牛顿-莱布尼茨公式"`）是规范的层级标识符；`parent_id` 字段经过反规范化处理以支持 CTE 查询。
- `content_list` 是通用的中间数据格式：`[{type, text, text_level, page_idx}, ...]`。三种解析器（PDF/DOCX/MD）均生成该格式。
- `text_level ≤ 4` 表示标题（PDF 中基于字体大小，DOCX 中基于 Heading 样式，MD 中基于 `#` 层级），`99` 表示正文文本。
- `KPSplitter` 使用标题上下文栈机制：遇到标题行时更新 `current_heading`；正文行自动继承当前标题对应的 KP。
