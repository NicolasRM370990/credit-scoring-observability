.PHONY: setup test lint data prepare train evaluate evaluate-labeled score bias demo-smoke demo demo-video

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

evaluate-labeled:
	uv run python -m creditwatch.model.evaluate --labeled-dir data/production/concept_drift

score:
	uv run python -m creditwatch.model.predict $(INPUT) --output $(OUTPUT)

bias:
	uv run python -m creditwatch.model.fairness

demo-smoke:
	uv run python -m creditwatch.pipeline.run --smoke

demo:
	uv run python -m creditwatch.pipeline.run --scenario no_drift
	uv run python -m creditwatch.pipeline.run --scenario covariate_moderate
	uv run python -m creditwatch.pipeline.run --scenario covariate_severe
	uv run python -m creditwatch.pipeline.run --scenario invalid

demo-video:
	uv run python -m creditwatch.pipeline.run --demo-video
