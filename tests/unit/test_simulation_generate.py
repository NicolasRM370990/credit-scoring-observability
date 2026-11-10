import json
from datetime import date, timedelta
from itertools import pairwise

import pandas as pd
import pytest

from creditwatch.simulation import generate_scenario


def test_generate_no_drift(tmp_path, production_pool):
    pool = production_pool

    output_dir = tmp_path / "production"

    generated = generate_scenario(
        pool=pool,
        scenario="no_drift",
        config_path="config/drift_scenarios.yaml",
        out_dir=output_dir,
    )

    assert len(generated) == 4

    for path in generated:
        assert path.exists()

        batch = pd.read_parquet(path)

        assert len(batch) == 3000
        assert set(batch.columns) == set(pool.columns) | {"event_date"}
        assert pd.notna(batch["event_date"]).all()

    manifest = output_dir / "no_drift" / "MANIFEST.json"

    assert manifest.exists()

    manifest_path = output_dir / "no_drift" / "MANIFEST.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(manifest["batches"]) == 4

    for index, batch_info in enumerate(manifest["batches"]):
        expected_date = date(2026, 1, 1) + timedelta(days=index)

        assert batch_info["event_date"] == expected_date.isoformat()
        assert batch_info["intensity"] == 0.0

def test_generate_no_drift_is_reproducible(tmp_path, production_pool):
    pool = production_pool

    first = generate_scenario(
        pool=pool,
        scenario="no_drift",
        config_path="config/drift_scenarios.yaml",
        out_dir=tmp_path / "first",
    )

    second = generate_scenario(
        pool=pool,
        scenario="no_drift",
        config_path="config/drift_scenarios.yaml",
        out_dir=tmp_path / "second",
    )

    for first_path, second_path in zip(first, second, strict=True):
        first_df = pd.read_parquet(first_path)
        second_df = pd.read_parquet(second_path)

        pd.testing.assert_frame_equal(first_df, second_df)



def test_generate_covariate_moderate(tmp_path, production_pool):
    pool = production_pool

    generated = generate_scenario(
        pool=pool,
        scenario="covariate_moderate",
        config_path="config/drift_scenarios.yaml",
        out_dir=tmp_path,
    )

    assert len(generated) == 4

    for path in generated:
        batch = pd.read_parquet(path)

        assert len(batch) == 3000
        assert (batch["monthly_income"] <= 25000).all()
        assert batch["revolving_utilization"].between(0, 3).all()
        assert batch["age"].between(18, 110).all()


def test_generate_covariate_severe(tmp_path, production_pool):
    pool = production_pool

    generated = generate_scenario(
        pool=pool,
        scenario="covariate_severe",
        config_path="config/drift_scenarios.yaml",
        out_dir=tmp_path,
    )

    assert len(generated) == 6

    for path in generated:
        batch = pd.read_parquet(path)

        assert len(batch) == 3000
        assert (batch["monthly_income"] <= 25000).all()
        assert batch["revolving_utilization"].between(0, 3).all()
        assert (batch["debt_ratio"] >= 0).all()
        assert batch["age"].between(18, 110).all()



def test_generate_covariate_moderate_ramp(tmp_path, production_pool):
    output_dir = tmp_path / "production"

    generate_scenario(
        pool=production_pool,
        scenario="covariate_moderate",
        config_path="config/drift_scenarios.yaml",
        out_dir=output_dir,
    )

    manifest_path = (
        output_dir / "covariate_moderate" / "MANIFEST.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batches = manifest["batches"]

    intensities = [batch["intensity"] for batch in batches]

    assert intensities == pytest.approx([0.25, 0.50, 0.75, 1.0])
    assert all(
        current < following
        for current, following in pairwise(intensities)
    )

    expected_factors = [1.0625, 1.125, 1.1875, 1.25]
    expected_shifts = [0.0375, 0.075, 0.1125, 0.15]

    for batch_info, expected_factor, expected_shift in zip(
        batches, expected_factors, expected_shifts
    ):
        transformations = batch_info["transformations"]

        income = next(
            item for item in transformations
            if item["feature"] == "monthly_income"
        )
        utilization = next(
            item for item in transformations
            if item["feature"] == "revolving_utilization"
        )

        assert income["factor"] == pytest.approx(expected_factor)
        assert utilization["value"] == pytest.approx(expected_shift)



def test_generate_covariate_severe_ramp(tmp_path, production_pool):
    output_dir = tmp_path / "production"

    generate_scenario(
        pool=production_pool,
        scenario="covariate_severe",
        config_path="config/drift_scenarios.yaml",
        out_dir=output_dir,
    )

    manifest_path = (
        output_dir / "covariate_severe" / "MANIFEST.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batches = manifest["batches"]

    intensities = [batch["intensity"] for batch in batches]

    assert intensities == pytest.approx(
        [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0]
    )

    assert all(
        current < following
        for current, following in pairwise(intensities)
    )

    for batch_info in batches:
        intensity = batch_info["intensity"]
        transformations = batch_info["transformations"]

        income = next(
            item for item in transformations
            if item["feature"] == "monthly_income"
        )
        utilization = next(
            item for item in transformations
            if item["feature"] == "revolving_utilization"
        )
        debt_ratio = next(
            item for item in transformations
            if item["feature"] == "debt_ratio"
        )

        assert income["factor"] == pytest.approx(1 + 0.40 * intensity)
        assert utilization["value"] == pytest.approx(0.30 * intensity)
        assert debt_ratio["factor"] == pytest.approx(1 + 0.60 * intensity)
