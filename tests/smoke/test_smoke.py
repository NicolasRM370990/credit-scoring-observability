"""Smoke tests: o projeto instala, importa, lê a configuração e as CLIs sobem."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from creditwatch.interfaces import LOG_REQUIRED_FIELDS, STAGES

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).resolve().parents[2]

MODULES = [
    "creditwatch.interfaces",
    "creditwatch.contracts",
    "creditwatch.contracts.validate",
    "creditwatch.data.download",
    "creditwatch.data.prepare",
    "creditwatch.model.train",
    "creditwatch.model.predict",
    "creditwatch.model.evaluate",
    "creditwatch.model.fairness",
    "creditwatch.simulation",
    "creditwatch.simulation.generate",
    "creditwatch.simulation.scenarios",
    "creditwatch.pipeline.run",
]

CLIS = [
    "creditwatch.pipeline.run",
    "creditwatch.model.predict",
    "creditwatch.model.evaluate",
    "creditwatch.model.train",
    "creditwatch.model.fairness",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    assert importlib.import_module(module) is not None


@pytest.mark.parametrize(
    "name",
    ["alerts", "contract", "data", "drift_scenarios", "drift_thresholds", "model"],
)
def test_config_is_valid_yaml_mapping(name):
    config = yaml.safe_load((ROOT / "config" / f"{name}.yaml").read_text(encoding="utf-8"))
    assert isinstance(config, dict) and config


def test_model_metadata_is_present_and_parseable():
    metadata = json.loads((ROOT / "models" / "baseline_v1.json").read_text(encoding="utf-8"))
    assert metadata["model_version"] == "baseline_v1"


@pytest.mark.parametrize("module", CLIS)
def test_cli_help_exits_cleanly(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_pipeline_smoke_cli_emits_json_log_for_every_stage(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "creditwatch.pipeline.run", "--smoke"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.strip().splitlines()[-1])

    log_path = tmp_path / "artifacts" / "logs" / "pipeline.jsonl"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    assert [event["stage"] for event in events] == list(STAGES)
    assert all(set(LOG_REQUIRED_FIELDS) <= set(event) for event in events)
    assert {event["run_id"] for event in events} == {summary["run_id"]}


def test_pipeline_cli_without_arguments_fails_with_usage(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "creditwatch.pipeline.run"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 2
    assert "informe --smoke" in result.stderr
