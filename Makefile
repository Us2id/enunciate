.PHONY: install lint format typecheck test integration evaluate serve docker

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy

test:
	pytest

integration:
	pytest -m integration

evaluate:
	python scripts/evaluate.py

serve:
	enunciate serve --log-level info

docker:
	docker build -t enunciate .
