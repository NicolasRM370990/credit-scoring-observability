"""Obtenção e conferência do dataset bruto (E1-01).

Uso: python -m creditwatch.data.download [--print-hash]
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path("config/data.yaml")
KAGGLE_COMPETITION = "GiveMeSomeCredit"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def try_kaggle_download(raw_dir: Path) -> bool:
    """Tenta baixar via Kaggle CLI. Retorna False se não for possível."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION, "-p", str(raw_dir)],
            check=True,
        )
        zip_path = raw_dir / f"{KAGGLE_COMPETITION}.zip"
        if zip_path.exists():
            import zipfile

            with zipfile.ZipFile(zip_path) as z:
                z.extractall(raw_dir)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def load_raw(cfg: dict) -> pd.DataFrame:
    """Lê o CSV bruto sem a coluna de índice e renomeia para snake_case."""
    raw = Path(cfg["dataset"]["raw_file"])
    df = pd.read_csv(raw, index_col=0)
    return df.rename(columns=cfg["rename"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-hash", action="store_true", help="só imprime o SHA-256 do arquivo bruto")
    args = parser.parse_args()

    cfg = load_config()
    raw = Path(cfg["dataset"]["raw_file"])

    if not raw.exists() and not args.print_hash:
        print(f"{raw} não encontrado. Tentando Kaggle CLI...")
        try_kaggle_download(raw.parent)
    if not raw.exists():
        print(
            f"ERRO: {raw} não existe. Baixe manualmente (ver data/README.md) e salve nesse caminho.",
            file=sys.stderr,
        )
        return 1

    digest = sha256_of(raw)
    if args.print_hash:
        print(digest)
        return 0

    expected = cfg["dataset"]["sha256"]
    if expected in ("", "PREENCHER"):
        print("ERRO: SHA-256 não configurado em config/data.yaml.", file=sys.stderr)
        return 1
    elif digest != expected:
        print(f"ERRO: SHA-256 diferente.\n  esperado: {expected}\n  obtido:   {digest}", file=sys.stderr)
        return 1
    else:
        print("SHA-256 confere.")

    df = load_raw(cfg)
    required = set(cfg["rename"].values())
    missing = required - set(df.columns)
    if missing:
        print(f"ERRO: colunas esperadas ausentes no CSV: {sorted(missing)}", file=sys.stderr)
        return 1
    target = cfg["target"]
    if not df[target].dropna().isin([0, 1]).all():
        print(f"ERRO: target {target!r} precisa conter somente 0 e 1.", file=sys.stderr)
        return 1
    if len(df) < 5000:
        print(f"ERRO: dataset tem {len(df)} linhas; mínimo esperado é 5.000.", file=sys.stderr)
        return 1
    out_dir = Path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "full.parquet"
    df.to_parquet(out, index=False)
    print(f"{len(df)} linhas gravadas em {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
