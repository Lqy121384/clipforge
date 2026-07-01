.PHONY: install dev test lint check docker

install:
	python -m pip install -e ".[dev]"

dev:
	python -m app

test:
	pytest --cov=app --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .

check: lint test

docker:
	docker compose up --build

