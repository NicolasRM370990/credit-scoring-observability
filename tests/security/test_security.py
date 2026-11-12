"""Testes de segurança: padrões inseguros no código, segredos, privacidade e integridade."""
from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest
from conftest import make_credit_frame

from creditwatch.contracts import validate_batch
from creditwatch.model.predict import score_batch
from creditwatch.pipeline import run as pipeline

pytestmark = pytest.mark.security

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = sorted((ROOT / "src").rglob("*.py"))

SECRET_PATTERNS = {
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "kaggle_key": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
    "generic_secret": re.compile(
        r"(?i)(?:password|passwd|secret|api_key|token)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
    ),
}


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def _calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            owner = func.value.id if isinstance(func, ast.Attribute) and isinstance(
                func.value, ast.Name) else ""
            yield node, owner, name


# ------------------------------------------------------------------ SAST (AST)
@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_dangerous_builtins(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    banned = {"eval", "exec", "compile", "__import__"}
    found = [name for _, owner, name in _calls(tree) if not owner and name in banned]
    assert not found, f"{path.name} usa {found}"


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_yaml_is_always_loaded_safely(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    unsafe = [
        node.lineno for node, owner, name in _calls(tree)
        if owner == "yaml" and name in {"load", "unsafe_load", "full_load", "load_all"}
    ]
    assert not unsafe, f"{path.name}: yaml inseguro nas linhas {unsafe}"


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_shell_injection_vectors(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node, owner, name in _calls(tree):
        assert not (owner == "os" and name in {"system", "popen"}), f"{path.name}:{node.lineno}"
        if owner == "subprocess":
            shell = [kw for kw in node.keywords if kw.arg == "shell"]
            assert not any(
                isinstance(kw.value, ast.Constant) and kw.value.value for kw in shell
            ), f"{path.name}:{node.lineno} usa shell=True"
            assert node.args and isinstance(node.args[0], ast.List), (
                f"{path.name}:{node.lineno} deve passar argumentos como lista"
            )


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_raw_pickle_deserialization(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = [node.lineno for node, owner, name in _calls(tree)
             if owner in {"pickle", "marshal", "dill"} and name in {"load", "loads"}]
    assert not found, f"{path.name}: desserialização sem verificação nas linhas {found}"


# ------------------------------------------------------------------ segredos
def test_no_secrets_in_tracked_files():
    findings = []
    for path in _tracked_files():
        if path.suffix in {".lock", ".parquet", ".joblib", ".png", ".jpg"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        findings += [f"{path.relative_to(ROOT)}: {name}"
                     for name, pattern in SECRET_PATTERNS.items() if pattern.search(text)]
    assert not findings, findings


def test_sensitive_files_are_not_tracked():
    tracked = {path.relative_to(ROOT).as_posix() for path in _tracked_files()}
    sensitive = re.compile(r"(^|/)(\.env(\..*)?|kaggle\.json|.*\.pem|.*\.key|.*\.joblib)$")
    assert not [name for name in tracked if sensitive.search(name)]
    assert not [name for name in tracked
                if name.startswith("data/") and not name.lower().endswith(("readme.md", ".gitkeep"))]


def test_gitignore_protects_secrets_and_data():
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for required in (".env", "kaggle.json", "*.pem", "*.key", "data/**", "artifacts/*"):
        assert required in rules, required


# ------------------------------------------------------------------ privacidade (LGPD)
def test_validation_result_never_exposes_row_values():
    batch = make_credit_frame(1_200).drop(columns="target")
    batch["monthly_income"] = batch["monthly_income"].astype(float)
    batch.loc[0, "customer_id"] = "CPF-123.456.789-00"
    batch.loc[1, "customer_id"] = "CPF-123.456.789-00"
    batch.loc[2, "monthly_income"] = 987654.321
    batch.loc[3, "age"] = 7

    result = validate_batch(batch, batch_id="privacy")
    payload = result.model_dump_json()

    assert result.status == "BLOCKED"
    assert "CPF-123.456.789-00" not in payload
    assert "987654.321" not in payload
    for check in result.checks:
        assert all(set(example) == {"row_index"} for example in check.examples)


def test_validation_examples_are_capped():
    batch = make_credit_frame(1_200).drop(columns="target")
    batch["age"] = 5

    result = validate_batch(batch, batch_id="cap")

    age_check = next(check for check in result.checks if check.rule_id == "R01_AGE_RANGE")
    assert age_check.n_failures == len(batch)
    assert len(age_check.examples) <= 10


def test_pipeline_log_events_do_not_carry_row_level_data(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "LOG_PATH", tmp_path / "pipeline.jsonl")
    pipeline.run_smoke()
    for line in (tmp_path / "pipeline.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        assert event["extra"] == {}
        assert not {"customer_id", "monthly_income", "age"} & set(event)


# ------------------------------------------------------------------ integridade do modelo
def test_tampered_model_is_rejected(trained_model, tmp_path):
    model_path, metadata_path = trained_model
    tampered = tmp_path / "baseline_v1.joblib"
    tampered.write_bytes(model_path.read_bytes() + b"\x00")

    with pytest.raises(ValueError, match="SHA-256"):
        score_batch(make_credit_frame(10), tampered, metadata_path)


def test_scoring_never_uses_target_as_feature(trained_model):
    model_path, _ = trained_model
    batch = make_credit_frame(300)
    flipped = batch.assign(target=1 - batch["target"])

    assert score_batch(batch, model_path)["score"].equals(score_batch(flipped, model_path)["score"])
