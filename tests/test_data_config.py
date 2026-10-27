from pathlib import Path

import yaml

CFG = yaml.safe_load(Path("config/data.yaml").read_text(encoding="utf-8"))


def test_rename_matches_features_and_target():
    renamed = set(CFG["rename"].values())
    expected = set(CFG["features"]["numerical"]) | set(CFG["features"]["categorical"]) | {CFG["target"]}
    assert renamed == expected


def test_split_fractions_sum_to_one():
    assert abs(sum(CFG["split"].values()) - 1.0) < 1e-9


def test_special_code_columns_are_features():
    assert set(CFG["special_codes"]["columns"]) <= set(CFG["features"]["numerical"])