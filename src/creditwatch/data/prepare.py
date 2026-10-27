"""Preparação dos dados (E1-02): limpeza e divisão em referência e pool de produção.

Uso: python -m creditwatch.data.prepare

Entrada : data/raw/cs-training.csv (via config/data.yaml)
Saída   : data/processed/{reference_train,reference_val,reference_test,production_pool}.parquet
          data/processed/split_manifest.json
Os limites de limpeza vêm de config/contract.yaml, para o dado de referência
passar no contrato por construção.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from creditwatch.data.download import CONFIG_PATH, load_config, load_raw, sha256_of

CONTRACT_PATH = Path("config/contract.yaml")


def load_contract(path: Path = CONTRACT_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def clean(df: pd.DataFrame, cfg: dict, contract: dict) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aplica as regras de limpeza e devolve (dados limpos, linhas removidas por passo)."""
    rules = contract["block_rules"]
    age_rule, income_rule = rules["R01_AGE_RANGE"], rules["R02_INCOME"]
    age_col, income_col = age_rule["column"], income_rule["column"]
    steps: dict[str, int] = {}

    n = len(df)
    df = df.drop_duplicates()
    steps["duplicates"] = n - len(df)

    n = len(df)
    df = df[df[age_col].between(age_rule["min"], age_rule["max"])]
    steps["age_out_of_range"] = n - len(df)

    n = len(df)
    df = df[df[income_col].notna()]
    steps["income_null"] = n - len(df)

    n = len(df)
    df = df[(df[income_col] >= income_rule["min"]) & (df[income_col] <= income_rule["max"])]
    steps["income_out_of_range"] = n - len(df)

    df = df.copy()
    special = cfg["special_codes"]
    cells = 0
    for col in special["columns"]:
        is_code = df[col].isin(special["values"])
        cells += int(is_code.sum())
        df[col] = df[col].mask(is_code)
    steps["special_code_cells_set_to_nan"] = cells

    return df.reset_index(drop=True), steps


def add_customer_id(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    shuffled = df.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
    shuffled.insert(0, cfg["id_col"], [f"C{i:07d}" for i in range(len(shuffled))])
    return shuffled


def split_data(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Divide em duas etapas, sempre estratificando pelo target."""
    split, target, seed = cfg["split"], cfg["target"], cfg["seed"]

    reference, pool = train_test_split(
        df, test_size=split["production_pool"], stratify=df[target], random_state=seed
    )
    val_test = split["reference_val"] + split["reference_test"]
    train, rest = train_test_split(
        reference,
        test_size=val_test / (1 - split["production_pool"]),
        stratify=reference[target],
        random_state=seed,
    )
    val, test = train_test_split(
        rest,
        test_size=split["reference_test"] / val_test,
        stratify=rest[target],
        random_state=seed,
    )
    parts = {
        "reference_train": train,
        "reference_val": val,
        "reference_test": test,
        "production_pool": pool,
    }
    return {name: part.reset_index(drop=True) for name, part in parts.items()}


def prepare(
    cfg_path: Path = CONFIG_PATH,
    contract_path: Path = CONTRACT_PATH,
    out_dir: Path | None = None,
) -> dict:
    cfg = load_config(cfg_path)
    contract = load_contract(contract_path)
    out_dir = Path(out_dir or cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = Path(cfg["dataset"]["raw_file"])
    df = load_raw(cfg)
    columns = [*cfg["features"]["numerical"], *cfg["features"]["categorical"], cfg["target"]]
    df = df[columns]
    rows_raw = len(df)

    df, removed = clean(df, cfg, contract)
    df = add_customer_id(df, cfg)
    ordered = [cfg["id_col"], *columns]
    parts = split_data(df[ordered], cfg)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": cfg["seed"],
        "raw_file": str(raw_path),
        "raw_sha256": sha256_of(raw_path),
        "contract_version": contract["version"],
        "rows_raw": rows_raw,
        "removed": removed,
        "rows_clean": len(df),
        "splits": {},
    }
    for name, part in parts.items():
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
        manifest["splits"][name] = {
            "rows": len(part),
            "target_rate": round(float(part[cfg["target"]].mean()), 4),
        }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> int:
    manifest = prepare()
    print(f"Linhas brutas: {manifest['rows_raw']}")
    for step, n in manifest["removed"].items():
        print(f"  {step}: {n}")
    print(f"Linhas limpas: {manifest['rows_clean']}")
    for name, info in manifest["splits"].items():
        print(f"  {name}: {info['rows']} linhas, taxa do target {info['target_rate']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
