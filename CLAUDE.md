# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Run all unit tests
PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_week2.py -v

# Run a single test
PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_week2.py::TestKPSplitter::test_exact_heading_match -v

# Run the slow real-pipeline integration test (MinerU, requires DB)
PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_real_pipeline.py -v -s

# Lint
.venv/Scripts/ruff check src/ tests/ scripts/

# Type-check
.venv/Scripts/mypy src/

# Start the API server
.venv/Scripts/uvicorn coursepilot.main:app --reload

# Bootstrap (create first superuser)
PYTHONPATH=src .venv/Scripts/python -m coursepilot.auth.bootstrap

# Seed a course from a PDF (KP tree + knowledge units, one parse)
PYTHONPATH=src .venv/Scripts/python -m scripts.seed_knowledge "tests/fixtures/pdfs/书名.pdf" --course-name "课程名" --ingest

# Batch ingest all 8 textbooks (5 courses)
PYTHONPATH=src .venv/Scripts/python -m scripts.batch_ingest

# Alembic
alembic revision --autogenerate -m "描述"
alembic upgrade head
```

## Architecture

CoursePilot is an AI teaching assistant for CS courses. **Backend**: FastAPI + SQLAlchemy async + PostgreSQL. **Parsing**: MinerU (PDF OCR) + python-docx + custom Markdown. **Planned**: Milvus vector store, LangGraph agent, Streamlit UI.

### Two-Phase Ingestion Pipeline (Critical)

The core data flow has two phases that run at **different times**:

**Phase A — Build Knowledge Point tree (one-time per course):**
`scripts/seed_knowledge.py` or `scripts/batch_ingest.py` →
parse file once → extract headings (text_level ≤ 4) → `headings_to_syllabus()` builds `kp_path` hierarchy → INSERT `knowledge_points` (adjacency list with `parent_id`)

**Phase B — Convert document to Knowledge Units (per upload):**
`POST /courses/upload` → `pipeline.run_ingestion()` →
parse file → `parser_utils.extract_knowledge_units()` splits by headings → `KPSplitter.assign()` matches each chunk to a KP → INSERT `knowledge_units`

**Critical rule: Never parse the same file twice.** MinerU takes ~100 min for a 300-page PDF. `run_ingestion()` accepts `preparsed_content_list` to skip re-parsing. `seed_knowledge.py --ingest` does both phases in one parse. The batch script processes multi-volume courses (e.g., 上下册) by merging all headings into a shared KP tree while creating separate Documents per volume.

### Key Files

| File | Role |
|------|------|
| `src/coursepilot/config.py` | All settings via pydantic-settings, loaded from `.env` |
| `src/coursepilot/db.py` | Async engine, `get_session()` for FastAPI DI, `get_session_etx()` for scripts |
| `src/coursepilot/main.py` | FastAPI app, registers auth + courses routers |
| `src/coursepilot/api/auth.py` | Register, login (JWT), me |
| `src/coursepilot/api/courses.py` | Course CRUD, upload + trigger ingestion, document list, KP tree query |
| `src/coursepilot/api/deps.py` | `get_current_user` (JWT → User), `require_superuser` |
| `src/coursepilot/ingestion/pipeline.py` | `run_ingestion()` — the B1→B6 orchestration |
| `src/coursepilot/ingestion/pdf_parser.py` | `parse_pdf()` — wraps MinerU CLI, returns `{markdown, content_list}` |
| `src/coursepilot/ingestion/docx_parser.py` | `parse_docx()` — sync (python-docx), returns same format |
| `src/coursepilot/ingestion/markdown_parser.py` | `parse_markdown()` — sync, same content_list format |
| `src/coursepilot/ingestion/parser_utils.py` | `extract_knowledge_units()` — `_split_by_headings()` + `_split_text()` |
| `src/coursepilot/knowledge/kp_splitter.py` | `KPSplitter` — matches text blocks to KPs (exact→cleaned→keyword→root fallback) |
| `src/coursepilot/knowledge/syllabus_parser.py` | `SyllabusParser` — markdown/Chinese-numbered outline → tree nodes |
| `src/coursepilot/knowledge/kp_tree.py` | `KPTree` — recursive CTE queries, batch insert with parent_id backfill |
| `src/coursepilot/storage/file_store.py` | `FileStore` — local filesystem, UUID filenames under `data/uploads/<course_id>/` |
| `scripts/seed_knowledge.py` | CLI: parse PDF → headings → KP tree; `--ingest` also does knowledge units |
| `scripts/batch_ingest.py` | Batch process all 8 PDFs in `tests/fixtures/pdfs/`, grouped by course |

### Database Models (11 tables)

`User` → `Course` → `KnowledgePoint` (adjacency list, `parent_id` self-ref, depth ≤ 4) → `KnowledgeUnit` (chunks with `kp_id`).
`User` → `Document` (uploaded file, status: pending→processing→ready/failed).
`User` → `Question`, `PracticeRecord`, `DiagnosisReport`, `ReviewPlan`, `QARecord`, `EvalMetric`.

### API Routes

- `POST/GET /api/v1/auth/register|login`, `GET /auth/me`
- `GET/POST /api/v1/courses`, `GET/DELETE /courses/{id}`
- `POST /courses/upload` (multipart: file + course_id, triggers ingestion inline)
- `GET /courses/{id}/documents`, `DELETE /courses/{id}/document/{doc_id}`
- `GET /courses/{id}/knowledge-points` (flat list with parent_id for frontend tree rendering)

### Configuration

`.env` at project root. Required: `DATABASE_URL=postgresql+asyncpg://...`, `JWT_SECRET_KEY=...`, `MINERU_MODEL_SOURCE=local`. See `config.py` for all defaults.

### Test Structure

- `tests/test_week2.py` — 56 unit tests covering parsers, syllabus extraction, KP splitter, file store, edge cases (no DB needed, 2 skipped require DB)
- `tests/test_real_pipeline.py` — single slow integration test, 8 steps, requires PostgreSQL + MinerU

### Key Patterns

- All DB operations are async (`AsyncSession`). Use `get_session_etx()` in scripts, `Depends(get_session)` in FastAPI routes.
- `KnowledgePoint.kp_path` like `"微积分/定积分/牛顿-莱布尼茨公式"` is the canonical hierarchical identifier; `parent_id` is denormalized for CTE queries.
- `content_list` is the universal intermediate format: `[{type, text, text_level, page_idx}, ...]`. All three parsers (PDF/DOCX/MD) produce it.
- `text_level ≤ 4` = heading (font size in PDF, Heading style in DOCX, `#` depth in MD), `99` = body text.
- The `KPSplitter` uses a heading-context stack: when a heading line is encountered, it updates `current_heading`; body lines inherit the current heading's KP.
