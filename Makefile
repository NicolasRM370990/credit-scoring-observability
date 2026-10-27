.PHONY: setup test lint data prepare

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

data:
	uv run python -m creditwatch.data.download

prepare:
	uv run python -m creditwatch.data.prepare