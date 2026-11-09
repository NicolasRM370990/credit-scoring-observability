
import pandas as pd

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
        assert set(batch.columns) == set(pool.columns)

    manifest = output_dir / "no_drift" / "MANIFEST.json"

    assert manifest.exists()


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
