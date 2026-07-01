.PHONY: help lock sync setup db-up db-down migrate bootstrap seed batch-ingest rebuild run-api run-ui lint format typecheck test test-unit test-integration clean

# ── 默认目标 ──
help:
	@echo "CoursePilot 开发命令"
	@echo ""
	@echo "  环境管理"
	@echo "    make lock           解析依赖，生成 uv.lock"
	@echo "    make sync           安装依赖到 .venv"
	@echo "    make setup          首次初始化（lock → sync → db → migrate → bootstrap）"
	@echo ""
	@echo "  数据库"
	@echo "    make db-up          启动 PostgreSQL"
	@echo "    make db-down        停止 PostgreSQL"
	@echo "    make migrate        运行 Alembic 迁移"
	@echo "    make bootstrap      创建首个超级用户"
	@echo ""
	@echo "  数据导入"
	@echo "    make seed PDF=path  从 PDF 种子化课程数据"
	@echo "    make batch-ingest   批量导入全部教材"
	@echo "    make rebuild        清空重建所有课程"
	@echo ""
	@echo "  运行"
	@echo "    make run-api        启动 FastAPI 服务"
	@echo "    make run-ui         启动 Streamlit 前端"
	@echo ""
	@echo "  质量检查"
	@echo "    make lint           代码规范检查"
	@echo "    make format         自动格式化"
	@echo "    make typecheck      类型检查"
	@echo "    make test           运行单元测试"
	@echo "    make test-integration  运行集成测试"
	@echo ""
	@echo "  清理"
	@echo "    make clean          删除 .venv + uv.lock（重新初始化环境）"

# ── 环境变量 ──
export PYTHONPATH := src
PY := .venv/Scripts/python
UV := uv

# ── 环境管理 ──

lock:
	$(UV) lock

sync:
	$(UV) sync

setup: lock sync db-up migrate bootstrap
	@echo ""
	@echo "初始化完成。运行 make run-api 启动服务。"

# ── 数据库 ──

db-up:
	docker compose up -d

db-down:
	docker compose down

migrate:
	alembic upgrade head

bootstrap:
	$(PY) -m coursepilot.auth.bootstrap

# ── 数据导入 ──

seed:
	$(PY) -m scripts.seed_knowledge "$(PDF)" --course-name "$(COURSE)" --ingest

batch-ingest:
	$(PY) -m scripts.batch_ingest

rebuild:
	$(PY) -m scripts.rebuild_all

# ── 运行 ──

run-api:
	$(PY) -m uvicorn coursepilot.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	$(PY) -m streamlit run src/coursepilot/ui/app.py

# ── 质量检查 ──

lint:
	ruff check src/ tests/ scripts/ eval/
	ruff format --check src/ tests/ scripts/ eval/

format:
	ruff check --fix src/ tests/ scripts/ eval/
	ruff format src/ tests/ scripts/ eval/

typecheck:
	mypy src/

test:
	$(PY) -m pytest tests/unit/ -v

test-integration:
	$(PY) -m pytest tests/integration/ -v -s

# ── 清理 ──

clean:
	@echo "删除 .venv 和 uv.lock..."
	rm -rf .venv uv.lock
	@echo "已清理。运行 make setup 重新初始化。"
