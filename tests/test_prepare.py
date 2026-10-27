from pathlib import Path

import pandas as pd
import pytest

from creditwatch.data.download import load_config
from creditwatch.data.prepare import (
    add_customer_id,
    clean,
    load_contract,
    prepare,
    split_data,
)

CFG = load_config()
CONTRACT = load_contract()
RAW = Path(CFG["dataset"]["raw_file"])


def _row(**overrides) -> dict:
    base = {
        "revolving_utilization": 0.3, "age": 40, "past_due_30_59": 0, "debt_ratio": 0.3,
        "monthly_income": 5000.0, "open_credit_lines": 5, "times_90_days_late": 0,
        "real_estate_loans": 1, "past_due_60_89": 0, "dependents": 1, "target": 0,
    }
    return {**base, **overrides}


def test_clean_removes_each_kind_of_invalid_row():
    cap = CONTRACT["block_rules"]["R02_INCOME"]["max"]
    rows = [
        _row(),                                   # ok
        _row(),                                   # duplicata do anterior
        _row(age=17, monthly_income=1.0),         # menor de idade
        _row(age=111, monthly_income=2.0),        # idade acima do máximo
        _row(monthly_income=float("nan")),        # renda nula
        _row(monthly_income=cap + 1),             # renda acima do teto
        _row(age=30, monthly_income=4000.0, past_due_30_59=98),  # código especial: fica, vira NaN
        _row(age=50, monthly_income=7000.0),      # ok
    ]
    out, steps = clean(pd.DataFrame(rows), CFG, CONTRACT)

    assert len(out) == 3
    assert steps == {
        "duplicates": 1,
        "age_out_of_range": 2,
        "income_null": 1,
        "income_out_of_range": 1,
        "special_code_cells_set_to_nan": 1,
    }
    assert out["past_due_30_59"].isna().sum() == 1


def test_add_customer_id_is_unique_and_first_column():
    df = pd.DataFrame([_row(age=20 + i) for i in range(50)])
    out = add_customer_id(df, CFG)
    assert out.columns[0] == CFG["id_col"]
    assert out[CFG["id_col"]].is_unique
    assert len(out) == len(df)


def test_split_data_sizes_overlap_and_target_rate():
    n = 5000
    df = pd.DataFrame([_row(age=20 + i % 60, target=int(i % 15 == 0)) for i in range(n)])
    df = add_customer_id(df, CFG)
    parts = split_data(df, CFG)

    assert sum(len(p) for p in parts.values()) == n
    ids = [set(p[CFG["id_col"]]) for p in parts.values()]
    assert sum(len(s) for s in ids) == len(set().union(*ids))   # nenhum id em dois splits

    for name, frac in CFG["split"].items():
        assert abs(len(parts[name]) / n - frac) < 0.01
    overall = df["target"].mean()
    for part in parts.values():
        assert abs(part["target"].mean() - overall) < 0.01


@pytest.mark.skipif(not RAW.exists(), reason="CSV bruto não está em data/raw")
def test_prepare_end_to_end_respects_the_contract(tmp_path):
    manifest = prepare(out_dir=tmp_path)
    parts = {name: pd.read_parquet(tmp_path / f"{name}.parquet") for name in manifest["splits"]}
    all_rows = pd.concat(parts.values())
    age_rule = CONTRACT["block_rules"]["R01_AGE_RANGE"]
    income_rule = CONTRACT["block_rules"]["R02_INCOME"]

    assert len(all_rows) == manifest["rows_clean"]
    assert all_rows["customer_id"].is_unique
    assert all_rows["age"].between(age_rule["min"], age_rule["max"]).all()
    assert all_rows["monthly_income"].notna().all()
    assert all_rows["monthly_income"].between(income_rule["min"], income_rule["max"]).all()
    assert not all_rows[CFG["special_codes"]["columns"]].isin(CFG["special_codes"]["values"]).any().any()
    assert not all_rows.drop(columns="customer_id").duplicated().any()
    assert all_rows["target"].isin([0, 1]).all()
    assert (tmp_path / "split_manifest.json").exists()
