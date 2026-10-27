.PHONY: setup test lint data

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

data:
	uv run python -m data.download

