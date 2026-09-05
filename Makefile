.PHONY: install dev test lint run eval docker

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

test:
	pytest

lint:
	ruff check . && ruff format --check .

run:
	streamlit run app.py

eval:
	python -m eval.run_eval --dataset eval/dataset.jsonl

docker:
	docker compose up --build
