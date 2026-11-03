.PHONY: setup test lint data prepare train evaluate score

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

train:
	uv run python -m creditwatch.model.train

evaluate:
	uv run python -m creditwatch.model.evaluate

score:
	uv run python -m creditwatch.model.predict $(INPUT) --output $(OUTPUT)
