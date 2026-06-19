---
name: verify
description: Run lint, type-check, and unit tests to verify code changes are correct. Use after making edits to confirm nothing is broken.
---

Run these checks sequentially. Stop early if any step fails and report which step failed.

```bash
.venv/Scripts/ruff check src/ tests/ scripts/
```
```bash
.venv/Scripts/mypy src/
```
```bash
PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_week2.py -v
```

If all three pass, report: "All checks passed — ruff, mypy, and 56 tests."
