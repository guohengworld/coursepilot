# CoursePilot 系统设计 v1.0

> 数据库 Schema · API 接口定义 · 组件交互
> 版本: 1.0 | 日期: 2026-06-11 | 状态: 待评审

---

## 1. 数据库 Schema 设计

### 1.1 总览

- **数据库**: PostgreSQL 17
- **向量存储**: Milvus Lite（独立于 PostgreSQL，用于向量索引）
- **ORM**: SQLAlchemy 2.0 (async)
- **Migration**: Alembic

### 1.2 ER 图（文字版）

```
User ───1:N──→ PracticeRecord ──N:1──→ Question
  │                                      │
  │                                      │
  └───1:N──→ DiagnosisReport ────N:1────┘
  │
  └───1:N──→ ReviewPlan
  │
  └───1:N──→ Document ────N:1────→ Course
                                     │
                                     └───1:N──→ KnowledgePoint
                                                    │
                                                    └───1:N──→ KnowledgeUnit

QARecord ────N:1──── User
QARecord ────N:1──── Course
```

### 1.3 完整表结构

#### users

```sql
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username        VARCHAR(64)  NOT NULL UNIQUE,
    password_hash   VARCHAR(256) NOT NULL,
    role            VARCHAR(16)  NOT NULL DEFAULT 'student'
                    CHECK (role IN ('super', 'student')),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_users_username ON users (username);
CREATE INDEX idx_users_role     ON users (role);
```

#### courses

```sql
CREATE TABLE courses (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(128) NOT NULL,
    description     TEXT,
    created_by      UUID         NOT NULL REFERENCES users(id),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_courses_created_by ON courses (created_by);
```

#### knowledge_points（知识点树，adjacency list + 递归 CTE）

```sql
CREATE TABLE knowledge_points (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    course_id       UUID         NOT NULL REFERENCES courses(id)
                                   ON DELETE CASCADE,
    parent_id       UUID         REFERENCES knowledge_points(id)
                                   ON DELETE CASCADE,
    kp_path         VARCHAR(512) NOT NULL,
    -- e.g. "OS/process/scheduling/rr"
    -- e.g. "DS/sort/quick_sort"
    title           VARCHAR(256) NOT NULL,
    summary         TEXT,                  -- 知识点概述（由教学大纲提取）
    difficulty      SMALLINT     DEFAULT 1
                    CHECK (difficulty BETWEEN 1 AND 5),
    source          VARCHAR(32)  NOT NULL DEFAULT 'course'
                    CHECK (source IN ('course', 'external')),
    sort_order      INTEGER      DEFAULT 0,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_kp_course_id  ON knowledge_points (course_id);
CREATE INDEX idx_kp_parent_id  ON knowledge_points (parent_id);
CREATE INDEX idx_kp_path       ON knowledge_points (kp_path);
CREATE UNIQUE INDEX idx_kp_course_path
    ON knowledge_points (course_id, kp_path);

-- 递归 CTE 查询示例：查询某节点的完整路径
-- WITH RECURSIVE kp_tree AS (
--     SELECT id, parent_id, kp_path, 1 AS depth
--     FROM knowledge_points WHERE id = :leaf_id
--     UNION ALL
--     SELECT kp.id, kp.parent_id, kp.kp_path, t.depth + 1
--     FROM knowledge_points kp
--     JOIN kp_tree t ON kp.id = t.parent_id
-- )
-- SELECT * FROM kp_tree ORDER BY depth DESC;
```

#### documents

```sql
CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    course_id       UUID         NOT NULL REFERENCES courses(id)
                                   ON DELETE CASCADE,
    filename        VARCHAR(256) NOT NULL,
    file_type       VARCHAR(16)  NOT NULL
                    CHECK (file_type IN ('pdf', 'docx', 'md')),
    file_size       INTEGER,              -- bytes
    file_path       VARCHAR(512) NOT NULL, -- 本地存储路径
    page_count      INTEGER,
    uploader_id     UUID         NOT NULL REFERENCES users(id),
    status          VARCHAR(16)  NOT NULL DEFAULT 'pending'
                    CHECK (status IN (
                        'pending', 'processing', 'ready', 'failed'
                    )),
    error_message   TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_docs_course_id   ON documents (course_id);
CREATE INDEX idx_docs_uploader    ON documents (uploader_id);
CREATE INDEX idx_docs_status      ON documents (status);
```

#### knowledge_units（知识单元，切分后的检索基本单位）

```sql
CREATE TABLE knowledge_units (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kp_id           UUID         NOT NULL REFERENCES knowledge_points(id)
                                   ON DELETE CASCADE,
    document_id     UUID         REFERENCES documents(id)
                                   ON DELETE SET NULL,
    content         TEXT         NOT NULL,     -- 原始文本
    summary         TEXT,                      -- 摘要（LLM 生成）
    seq_order       INTEGER      DEFAULT 0,    -- 在同一知识点内的顺序
    page_ref        VARCHAR(64),               -- 来源页码/e.g."p45-47"
    metadata        JSONB        DEFAULT '{}', -- 扩展元数据
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_ku_kp_id         ON knowledge_units (kp_id);
CREATE INDEX idx_ku_document_id   ON knowledge_units (document_id);
-- 用于 Milvus metadata filter 中的批量查询
CREATE INDEX idx_ku_metadata      ON knowledge_units USING gin (metadata);
```

#### questions

```sql
CREATE TABLE questions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kp_id           UUID         NOT NULL REFERENCES knowledge_points(id),
    question_text   TEXT         NOT NULL,
    question_type   VARCHAR(16)  NOT NULL DEFAULT 'choice_4'
                    CHECK (question_type IN ('choice_4', 'true_false')),
    options         JSONB        NOT NULL,
    -- choice_4: {"A": "text", "B": "text", "C": "text", "D": "text"}
    -- true_false: {"A": "True", "B": "False"}
    correct_answer  VARCHAR(4)   NOT NULL,
    explanation     TEXT         NOT NULL,     -- 答案解析
    difficulty      SMALLINT     DEFAULT 1
                    CHECK (difficulty BETWEEN 1 AND 5),
    source          VARCHAR(32)  NOT NULL DEFAULT 'system'
                    CHECK (source IN ('system', 'manual', 'eval_dataset')),
    verified        BOOLEAN      DEFAULT TRUE,
    -- 是否通过三重自验证
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_q_kp_id          ON questions (kp_id);
CREATE INDEX idx_q_source         ON questions (source);
CREATE INDEX idx_q_verified       ON questions (verified);
```

#### qa_records（问答历史）

```sql
CREATE TABLE qa_records (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID         NOT NULL REFERENCES users(id)
                                   ON DELETE CASCADE,
    course_id       UUID         REFERENCES courses(id),
    query           TEXT         NOT NULL,
    kp_path         VARCHAR(512),
    retrieved_units UUID[]       DEFAULT '{}', -- 引用知识单元 ID 列表
    answer          TEXT         NOT NULL,
    citations       JSONB        DEFAULT '[]',
    -- [{"source": "OSTEP p42", "kp_path": "OS/process/scheduling"}]
    feedback        SMALLINT,                -- 用户反馈：1(有用)/0(无用)/NULL
    latency_ms      INTEGER,                 -- 端到端延迟
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_qa_user_id    ON qa_records (user_id);
CREATE INDEX idx_qa_course_id  ON qa_records (course_id);
CREATE INDEX idx_qa_created_at ON qa_records (created_at DESC);
```

#### practice_records（做题记录）

```sql
CREATE TABLE practice_records (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID         NOT NULL REFERENCES users(id)
                                   ON DELETE CASCADE,
    question_id     UUID         NOT NULL REFERENCES questions(id),
    user_answer     VARCHAR(4),
    correct_flag    BOOLEAN,                 -- NULL = 未作答/跳过
    answered_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_pr_user_id      ON practice_records (user_id);
CREATE INDEX idx_pr_question_id  ON practice_records (question_id);
CREATE INDEX idx_pr_correct_flag ON practice_records (correct_flag)
    WHERE correct_flag IS NOT NULL;
-- 分析用：用户在某课程下的做题概览
CREATE INDEX idx_pr_user_course
    ON practice_records (user_id, question_id);
```

#### diagnosis_reports（错题诊断报告）

```sql
CREATE TABLE diagnosis_reports (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID         NOT NULL REFERENCES users(id)
                                   ON DELETE CASCADE,
    practice_record_id UUID     NOT NULL REFERENCES practice_records(id)
                                   ON DELETE CASCADE,
    error_reason    TEXT         NOT NULL,    -- 错误原因分析
    error_category  VARCHAR(32)  NOT NULL
                    CHECK (error_category IN (
                        'concept_misunderstanding',
                        'calculation_error',
                        'confused_knowledge_point',
                        'careless',
                        'unknown'
                    )),
    remedy_kp_ids   UUID[]       DEFAULT '{}', -- 建议复习的知识点 ID
    report_content  TEXT         NOT NULL,    -- 完整诊断报告（用户可见）
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX_idx_diag_user_id    ON diagnosis_reports (user_id);
CREATE INDEX_idx_diag_practice   ON diagnosis_reports (practice_record_id);
```

#### review_plans（复习计划）

```sql
CREATE TABLE review_plans (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID         NOT NULL REFERENCES users(id)
                                   ON DELETE CASCADE,
    course_id       UUID         NOT NULL REFERENCES courses(id),
    items           JSONB        NOT NULL DEFAULT '[]',
    -- [{"kp_id": "...", "kp_path": "OS/...", "priority": 1,
    --   "reason": "错题率 60%", "status": "pending"}]
    -- priority: 1(最薄弱) ~ 5(已掌握)
    -- status: 'pending' | 'reviewed'
    reviewed_count  SMALLINT     DEFAULT 0,
    total_count     SMALLINT     DEFAULT 0,
    generated_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX_idx_rp_user_id      ON review_plans (user_id);
CREATE INDEX_idx_rp_course_id    ON review_plans (course_id);
```

#### eval_metrics（评估指标记录，P2 功能）

```sql
CREATE TABLE eval_metrics (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_type       VARCHAR(32)  NOT NULL,
    metric_name     VARCHAR(64)  NOT NULL,
    metric_value    FLOAT        NOT NULL,
    sample_size     INTEGER,
    metadata        JSONB        DEFAULT '{}',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX_idx_em_type ON eval_metrics (eval_type, created_at DESC);
```

### 1.4 Milvus Collection 设计

向量数据不存 PostgreSQL，存在 Milvus Lite（嵌入式）。

```python
from pymilvus import CollectionSchema, FieldSchema, DataType

collection_name = "coursepilot_knowledge"

schema = CollectionSchema([
    FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema("uuid", DataType.VARCHAR, max_length=36),  # 对应 knowledge_units.id
    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=1024),  # bge-m3 输出维度
    FieldSchema("kp_id", DataType.VARCHAR, max_length=36),
    FieldSchema("course_id", DataType.VARCHAR, max_length=36),
    FieldSchema("kp_path", DataType.VARCHAR, max_length=512),
    FieldSchema("content", DataType.VARCHAR, max_length=8192),
])

index_params = {
    "metric_type": "IP",       # Inner Product（与 cosine 等价，性能更好）
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}

# metadata filter 用在检索阶段：
# expr = 'kp_path startswith "OS/process/scheduling/rr"'
```

---

## 2. API 接口定义

### 2.1 基础约定

- **Base URL**: `http://localhost:8000/api/v1`
- **认证方式**: JWT Bearer Token（`Authorization: Bearer <token>`）
- **请求体**: JSON（`application/json`）
- **响应格式**:

```json
{
    "success": true,
    "data": { ... },
    "error": null
}

{
    "success": false,
    "data": null,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "用户名已存在"
    }
}
```

### 2.2 认证接口

#### POST /auth/register

注册新用户（默认 role = student）。

```
Request:
{
    "username": "string (3-64)",
    "password": "string (8-128)"
}

Response 201:
{
    "user_id": "uuid",
    "username": "string",
    "role": "student"
}

Error 409: 用户名已存在
Error 422: 参数校验失败
```

#### POST /auth/login

```
Request:
{
    "username": "string",
    "password": "string"
}

Response 200:
{
    "token": "jwt_string",
    "expires_in": 86400,
    "user": {
        "id": "uuid",
        "username": "string",
        "role": "super|student"
    }
}

Error 401: 用户名或密码错误
```

#### GET /auth/me

获取当前用户信息（需要认证）。

```
Response 200:
{
    "id": "uuid",
    "username": "string",
    "role": "super|student",
    "created_at": "datetime"
}
```

### 2.3 课程资料接口（SuperUser only）

#### POST /courses/upload

上传课程资料，触发 ingestion pipeline。

```
Request: multipart/form-data
{
    "file": "binary (pdf/docx/md)",
    "course_id": "uuid",
    "auto_index": true (default)
}

Response 202:
{
    "document_id": "uuid",
    "status": "processing",
    "estimated_time_seconds": 120
}

Error 400: 不支持的格式
Error 404: course_id 不存在
```

#### GET /courses

获取课程列表。

```
Response 200:
{
    "courses": [
        {
            "id": "uuid",
            "name": "string",
            "description": "string",
            "document_count": 5,
            "kp_count": 120,
            "created_at": "datetime"
        }
    ]
}
```

#### POST /courses

创建新课程。

```
Request:
{
    "name": "string (2-128)",
    "description": "string (optional)"
}

Response 201:
{
    "id": "uuid",
    "name": "string",
    "description": "string"
}
```

#### DELETE /courses/{course_id}

删除课程（级联删除所有关联数据）。

```
Response 200:
{
    "deleted": true
}
```

#### GET /courses/{course_id}/documents

获取某课程下的资料列表。

```
Response 200:
{
    "documents": [
        {
            "id": "uuid",
            "filename": "string",
            "file_type": "pdf|docx|md",
            "status": "ready|processing|failed",
            "page_count": 120,
            "uploaded_at": "datetime"
        }
    ]
}
```

#### DELETE /documents/{document_id}

删除某份资料。

```
Response 200:
{
    "deleted": true
}
```

### 2.4 知识问答接口

#### POST /qa/ask

提问。

```
Request:
{
    "query": "string (10-2000)",
    "course_id": "uuid",
    "kp_path": "string (optional, 指定知识点范围)"
}

Response 200:
{
    "qa_record_id": "uuid",
    "answer": "带引用来源的回答文本",
    "citations": [
        {
            "source": "OSTEP 第3章 p42",
            "kp_path": "OS/process/scheduling",
            "relevance_score": 0.92
        }
    ],
    "latency_ms": 3842,
    "kp_path": "OS/process/scheduling/rr"
}

Error 422: query 太短或太长
```

#### GET /qa/history

获取问答历史。

```
Query params:
  - course_id (optional)
  - limit (default 20)
  - offset (default 0)

Response 200:
{
    "records": [
        {
            "id": "uuid",
            "query": "string",
            "answer_preview": "前100字...",
            "kp_path": "OS/process/...",
            "created_at": "datetime"
        }
    ],
    "total": 156
}
```

#### POST /qa/feedback

提交问答反馈。

```
Request:
{
    "qa_record_id": "uuid",
    "feedback": 1 | 0
}

Response 200:
{
    "updated": true
}
```

### 2.5 题目练习接口

#### POST /practice/generate

生成练习题。

```
Request:
{
    "kp_id": "uuid (指定知识点)",
    "count": 1-5 (default 1),
    "question_type": "choice_4" | "true_false" (default "choice_4")
}

Response 200:
{
    "questions": [
        {
            "id": "uuid",
            "question_text": "string",
            "question_type": "choice_4",
            "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
            "difficulty": 3
        }
    ]
}

# 注意：返回给用户时不含 correct_answer 和 explanation
# 用户在提交作答后才能看到答案和解析

Error 422: 当前知识点题⽬生成失败，请重试或换一个知识点
```

#### POST /practice/submit

提交作答。

```
Request:
{
    "question_id": "uuid",
    "answer": "A|B|C|D"
}

Response 200:
{
    "correct": true | false,
    "correct_answer": "A",
    "explanation": "解析文本",
    "practice_record_id": "uuid"
}
```

#### GET /practice/records

获取做题记录。

```
Query params:
  - course_id (optional)
  - correct_only (optional boolean)
  - limit (default 50)
  - offset (default 0)

Response 200:
{
    "records": [
        {
            "id": "uuid",
            "question_id": "uuid",
            "question_text": "string",
            "kp_path": "OS/...",
            "user_answer": "A",
            "correct_answer": "B",
            "correct": false,
            "answered_at": "datetime"
        }
    ],
    "stats": {
        "total": 50,
        "correct": 32,
        "correct_rate": 0.64
    }
}
```

### 2.6 错题诊断接口

#### POST /practice/diagnose

对某条做题记录生成诊断。

```
Request:
{
    "practice_record_id": "uuid"
}

Response 200:
{
    "diagnosis_id": "uuid",
    "error_reason": "学生混淆了FIFO和LRU的置换策略...",
    "error_category": "confused_knowledge_point",
    "remedial_materials": [
        {
            "kp_path": "OS/vm/page_replacement/fifo",
            "title": "FIFO页面置换",
            "summary": "...",
            "documents": [
                {"filename": "OS_Chapter9.pdf", "page_ref": "p45-47"}
            ]
        },
        {
            "kp_path": "OS/vm/page_replacement/lru",
            "title": "LRU页面置换",
            "summary": "..."
        }
    ],
    "report_content": "【错误原因】...【正确理解】...【建议复习】...",
    "created_at": "datetime"
}
```

### 2.7 复习计划接口

#### POST /review/plan

生成复习计划。

```
Request:
{
    "course_id": "uuid"
}

Response 200:
{
    "plan_id": "uuid",
    "items": [
        {
            "kp_id": "uuid",
            "kp_path": "OS/vm/page_replacement",
            "title": "页面置换算法",
            "priority": 1,
            "reason": "错题率 80%（5 题错 4 题）",
            "status": "pending"
        }
    ],
    "stats": {
        "total": 20,
        "reviewed": 0,
        "weak_points": 3
    }
}
```

#### GET /review/plan/{plan_id}

获取复习计划详情。

#### PUT /review/plan/{plan_id}/items/{kp_id}

标记某知识点为"已复习"。

```
Request:
{
    "status": "reviewed"
}

Response 200:
{
    "updated": true,
    "reviewed_count": 5,
    "total_count": 20
}
```

### 2.8 管理接口（SuperUser only）

#### GET /admin/users

获取用户列表。

```
Response 200:
{
    "users": [
        {
            "id": "uuid",
            "username": "string",
            "role": "super|student",
            "qa_count": 42,
            "practice_count": 15,
            "created_at": "datetime"
        }
    ]
}
```

#### PUT /admin/users/{user_id}/role

升级/降级用户角色。

```
Request:
{
    "role": "super" | "student"
}

Response 200:
{
    "updated": true
}
```

#### GET /admin/eval/metrics

获取评估指标。

```
Response 200:
{
    "rag": {
        "context_precision": 0.88,
        "context_recall": 0.92,
        "answer_relevancy": 0.87,
        "faithfulness": 0.94
    },
    "question_gen": {
        "available_rate": 0.87,
        "accuracy_rate": 0.96,
        "eval_sample_size": 50
    }
}
```

---

## 3. 组件交互图

### 3.1 分层架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Presentation Layer (Streamlit)                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ 问答页面  │ │ 练习页面  │ │ 诊断页面  │ │ 复习页面  │ │ 管理后台     │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘  │
└───────┼────────────┼────────────┼────────────┼───────────────┼──────────┘
        │            │            │            │               │
        ▼            ▼            ▼            ▼               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  API Layer (FastAPI)                                                    │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌──────────┐  │
│  │ /qa/*     │ │ /practice │ │ /auth/*   │ │ /courses/*│ │ /admin/* │  │
│  │           │ │ /*         │ │           │ │           │ │          │  │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └────┬─────┘  │
└────────┼─────────────┼─────────────┼─────────────┼────────────┼─────────┘
         │             │             │             │            │
         ▼             ▼             ▼             ▼            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Agent Layer (LangGraph)                                                │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      StateGraph                                   │   │
│  │  ┌────────┐  ┌────────┐  ┌──────────┐  ┌────────┐  ┌─────────┐  │   │
│  │  │ Router  │→│ Parse  │→│ Retrieve  │→│Generate│→│ Verify  │  │   │
│  │  │(LLM)   │  │(LLM+KP)│  │(RAG工具)  │  │(LLM)   │  │(LLM*3) │  │   │
│  │  └────────┘  └────────┘  └──────────┘  └────────┘  └─────────┘  │   │
│  │                                          ↑__________________↓     │   │
│  │                                          │ retry (max 3)        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│         │           │            │                                      │
│         │           │            │                                      │
│         ▼           ▼            ▼                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                                │
│  │ QA Subgraph││Diagnose  ││ReviewPlan│                                │
│  └──────────┘ └──────────┘ └──────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
         │           │            │
         ▼           ▼            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  RAG Engine Layer                                                       │
│                                                                         │
│  ┌─────────────┐  ┌────────────────────┐  ┌──────────────┐  ┌────────┐  │
│  │ bge-m3      │  │ bge-m3             │  │ bge-reranker │  │  RRF   │  │
│  │ (Dense)     │  │ (Learned Sparse)   │  │ (Rerank)     │  │ (Fusion)│  │
│  └──────┬──────┘  └──────┬─────────────┘  └──────┬───────┘  └────┬───┘  │
│         │                │                       │               │      │
│         ▼                ▼                       ▼               ▼      │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Milvus Lite (hybrid_search: dense ANN + sparse + 内置 RRF)      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  PostgreSQL (metadata + KP 树 + 知识单元 + 查询日志)               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
         │                │                  │
         ▼                ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Data Layer                                                             │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  PostgreSQL  │  │  Milvus Lite │  │  本地文件系统  │                  │
│  │  (关系数据)   │  │  (向量索引)   │  │  (原始文件)   │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Ingestion Pipeline 数据流（已实施）

```
用户上传 PDF/DOCX/MD
       │
       ▼
┌──────────────┐
│  文件保存      │ → 存入本地文件系统
│  (FastAPI)   │ → documents 表 INSERT (status=pending)
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  run_ingestion(session, document_id)                         │
│                                                              │
│  B0: _ensure_kp_tree()          ← 自动构建/合并知识点树      │
│      从 content_list 提取标题 (text_level ≤ 4)                │
│      → headings_to_syllabus() → 按 kp_path 去重 → INSERT     │
│                                                              │
│  B1: 格式解析                   ← 已有（或复用 preparsed）    │
│      MinerU (PDF) / python-docx / 自定义 Markdown            │
│      → content_list                                          │
│                                                              │
│  B2: extract_knowledge_units()  ← 阶段 A 改造                │
│      _filter_garbage → _split_by_headings (heading 追踪)     │
│      → _split_text_v2 (数学块感知 + 段落边界优先)             │
│                                                              │
│  B3: KPSplitter.assign()        ← 已有                       │
│      标题匹配 → 清洗后匹配 → 关键词匹配 → 根节点兜底          │
│                                                              │
│  B4: SummaryBridge.run()        ← 新增                       │
│      DeepSeek 为每 unit 生成 ≤80 字中文摘要                   │
│                                                              │
│  B5: _encode_units()            ← 新增                       │
│      BGE-M3 一次 forward → dense (1024) + learned sparse     │
│      → Milvus hybrid_search collection insert                │
│                                                              │
│  B6: KnowledgeUnit INSERT       ← PG 入库                    │
│      → Document.status = "ready"                             │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 核心依赖关系

```
Component              Depends On                    Communication
────────               ──────────                    ─────────────
Streamlit UI           FastAPI API                   HTTP (localhost)
FastAPI API Layer      Agent Layer (LangGraph)       Python in-process
Agent (LangGraph)      RAG Engine + LLM API          Python in-process
RAG Engine             Milvus Lite + BM25 + LLM API  Python in-process
PostgreSQL             SQLAlchemy 2.0                asyncpg driver
Milvus Lite            Local file (.db)              pymilvus
LLM API (DeepSeek)     Internet / API key            HTTP (httpx)
```

### 3.4 错误处理策略

| 层 | 错误类型 | 处理方式 |
|----|----------|----------|
| API | 参数校验失败 | 422 + 明确错误字段 |
| API | 认证失败 | 401 + "请重新登录" |
| API | 无权限 | 403 + "仅超级用户可执行此操作" |
| Agent | LLM API 超时 | 重试 2 次，间隔 1s，失败则返回"服务暂时不可用" |
| Agent | RAG 检索为空 | 降级返回"未找到相关资料，请换个问法" |
| Agent | 自验证 3 次失败 | 返回"当前无法生成该知识点题目，请换一个知识点" |
| Ingestion | 文件解析失败 | status=failed + error_message，API 返回可读提示 |
| Ingestion | Milvus 写入失败 | 回滚：删除已写入数据，标记文档为 failed |

---

## 4. 项目目录结构（细化版）

```
coursepilot/
├── pyproject.toml
├── docker-compose.yml
├── alembic.ini
├── .env.example
├── src/coursepilot/
│   ├── __init__.py
│   ├── config.py                     # Pydantic Settings
│   ├── main.py                       # FastAPI app entry
│   ├── app.py                        # Streamlit entry
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py                   # 依赖注入（db session, current user）
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── courses.py
│   │   │   ├── qa.py
│   │   │   ├── practice.py
│   │   │   ├── review.py
│   │   │   └── admin.py
│   │   └── schemas.py                # Pydantic request/response models
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── session.py                # AsyncSession factory
│   │   ├── models/                   # SQLAlchemy models
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   ├── course.py
│   │   │   ├── knowledge.py
│   │   │   ├── document.py
│   │   │   ├── question.py
│   │   │   ├── practice.py
│   │   │   ├── qa_record.py
│   │   │   └── eval.py
│   │   └── migrations/               # Alembic
│   │
    ├── ingestion/
    │   ├── __init__.py
    │   ├── pdf_parser.py
    │   ├── docx_parser.py
    │   ├── markdown_parser.py
    │   ├── parser_utils.py            # extract_knowledge_units + 垃圾过滤 + 数学块切分
    │   └── pipeline.py                # run_ingestion: B0-B6 完整管线

    ├── knowledge/
    │   ├── __init__.py
    │   ├── kp_tree.py                 # 知识点树操作（CRUD + CTE 查询）
    │   ├── kp_splitter.py             # 文本块 → KP 匹配分配
    │   └── syllabus_parser.py         # 大纲解析 + extract_headings + headings_to_syllabus

    ├── rag/
    │   ├── __init__.py
    │   ├── config.py                  # RAGConfig 降级开关与阈值
    │   ├── encoder.py                 # BGE-M3 dense + learned sparse 统一编码
    │   ├── vector_store.py            # Milvus Lite CRUD + hybrid_search
    │   ├── query_rewriter.py          # DeepSeek 查询改写（阶段0）
    │   ├── reranker.py                # bge-reranker-v2-m3 + 层级惩罚（阶段3）
    │   ├── retriever.py               # 五阶段检索编排 + KP 扩展（阶段1-4）
    │   ├── generator.py               # DeepSeek LLM 生成 + prompt 组装（阶段5）
    │   ├── summary_bridge.py          # 导入时为 unit 生成摘要
    │   ├── citation.py                # <ref> 标签解析与验证
    │   └── logger.py                  # 结构化查询日志

    ├── agent/                         # 预留：LangGraph StateGraph
    │   └── ...

    ├── evaluation/                    # 预留：RAGAS 评估 + 题目自验证
    │   └── ...

    ├── ui/                           # Streamlit pages
    │   ├── __init__.py
    │   ├── login.py
    │   ├── qa_page.py
    │   ├── practice_page.py
    │   ├── diagnose_page.py
    │   ├── review_page.py
    │   ├── admin_courses.py
    │   ├── admin_users.py
    │   └── admin_eval.py
    │
    └── llm/
        ├── __init__.py
        ├── client.py                 # DeepSeek API client 封装
        └── prompts.py                # 所有 prompt 模板
├── tests/
    │   ├── conftest.py
    │   ├── test_api/
    │   ├── test_rag/
    │   ├── test_agent/
    │   └── test_ingestion/
    │
    ├── scripts/
    │   ├── seed_course.py
    │   ├── seed_knowledge.py
    │   ├── seed_questions.py
    │   ├── run_eval.py
    │   └── init_db.py
    │
    └── data/
    ├── sample_course/                # 示例数据（脱敏）
    └── sample_questions/
```
