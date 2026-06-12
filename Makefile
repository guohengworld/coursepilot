.PHONY: install db-up db-down migrate run-api run-ui lint test

install:
	uv sync

db-up:
	docker compose up -d

db-down:
	docker compose down

migrate:
	alembic upgrade head

run-api:
	uvicorn coursepilot.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	streamlit run src/coursepilot/ui/app.py

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

test:
	pytest -v
