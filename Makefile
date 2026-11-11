.PHONY: setup test lint data prepare train evaluate evaluate-labeled score bias demo-smoke demo demo-video \
        mlflow-ui mlflow-training dashboard alerts-summary evidence

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

# --- Observabilidade (P4) ---
mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000

mlflow-training:
	uv run python -m creditwatch.observability.mlflow_tracker --training

dashboard:
	uv run python -m creditwatch.observability.dashboard

alerts-summary:
	uv run python -m creditwatch.observability.alerts
	uv run python -m creditwatch.observability.metrics

# copia evidências pequenas (dashboard + alertas + histórico) para docs/evidence/
evidence: dashboard
	uv run python -m creditwatch.observability.dashboard --cdn --output docs/evidence/dashboard.html
	uv run python -c "import shutil,pathlib; d=pathlib.Path('docs/evidence'); [shutil.copy2(p, d/p.name) for p in map(pathlib.Path, ['artifacts/alerts/alerts.jsonl','artifacts/metrics/run_history.jsonl']) if p.exists()]"
