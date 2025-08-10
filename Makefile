.PHONY: install browser-install dev test test-ui eval compare report clean

install:
	python3 -m pip install -r requirements-dev.txt

browser-install:
	python3 -m playwright install chromium

dev:
	python3 -m uvicorn backend.app.api:app --reload --host $${APP_HOST:-127.0.0.1} --port $${APP_PORT:-8000}

test:
	python3 -m pytest

test-ui:
	python3 scripts/browser_smoke.py

eval:
	python3 -m backend.app.evaluation evals/sample_questions.jsonl

compare:
	python3 -m backend.app.comparison --dataset evals/sample_questions.jsonl --docs sample_docs

report:
	python3 scripts/build_report.py

clean:
	rm -rf .pytest_cache .rag_data
