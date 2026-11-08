from pathlib import Path

import pandas as pd
import pytest
import yaml

from creditwatch.simulation import (
    apply_resample_mix,
    apply_scale,
    apply_scenario_transformations,
    apply_shift_add,
    apply_transformations,
    load_scenario_config,
)


def test_apply_scale_multiplies_values():
    df = pd.DataFrame(
        {
            "monthly_income": [1000.0, 2000.0, 3000.0],
            "age": [25, 35, 45],
        }
    )

    result = apply_scale(
        df=df,
        feature="monthly_income",
        factor=1.25,
    )

    assert result["monthly_income"].tolist() == [
        1250.0,
        2500.0,
        3750.0,
    ]


def test_apply_scale_preserves_other_columns():
    df = pd.DataFrame(
        {
            "monthly_income": [1000.0, 2000.0],
            "age": [25, 35],
        }
    )

    result = apply_scale(df, "monthly_income", 1.25)

    pd.testing.assert_series_equal(result["age"], df["age"])


def test_apply_scale_does_not_modify_original_dataframe():
    df = pd.DataFrame(
        {
            "monthly_income": [1000.0, 2000.0],
        }
    )

    original = df.copy(deep=True)

    apply_scale(df, "monthly_income", 1.25)

    pd.testing.assert_frame_equal(df, original)


def test_apply_scale_preserves_null_values():
    df = pd.DataFrame(
        {
            "monthly_income": [1000.0, None, 3000.0],
        }
    )

    result = apply_scale(df, "monthly_income", 1.25)

    assert result["monthly_income"].iloc[0] == 1250.0
    assert pd.isna(result["monthly_income"].iloc[1])
    assert result["monthly_income"].iloc[2] == 3750.0


def test_apply_scale_raises_error_for_missing_feature():
    df = pd.DataFrame({"age": [25, 35]})

    with pytest.raises(ValueError, match="não existe"):
        apply_scale(df, "monthly_income", 1.25)


def test_apply_scale_raises_error_for_non_numeric_feature():
    df = pd.DataFrame({"name": ["Ana", "João"]})

    with pytest.raises(ValueError, match="numéricos"):
        apply_scale(df, "name", 1.25)


@pytest.mark.parametrize("factor", [float("nan"), float("inf"), float("-inf")])
def test_apply_scale_rejects_non_finite_factor(factor):
    df = pd.DataFrame({"monthly_income": [1000.0]})

    with pytest.raises(ValueError, match="finito"):
        apply_scale(df, "monthly_income", factor)


def test_apply_scale_rejects_string_factor():
    df = pd.DataFrame({"monthly_income": [1000.0]})

    with pytest.raises(TypeError, match="número real"):
        apply_scale(df, "monthly_income", "1.25")


def test_apply_shift_add_adds_value():
    df = pd.DataFrame({"revolving_utilization": [0.1, 1.2, 2.0]})

    result = apply_shift_add(df, "revolving_utilization", 0.15)

    assert result["revolving_utilization"].tolist() == pytest.approx([0.25, 1.35, 2.15])


def test_apply_shift_add_applies_clip():
    df = pd.DataFrame({"revolving_utilization": [-0.2, 1.2, 2.95]})

    result = apply_shift_add(
        df,
        "revolving_utilization",
        0.15,
        clip=[0, 3],
    )

    assert result["revolving_utilization"].tolist() == pytest.approx([0.0, 1.35, 3.0])


def test_apply_shift_add_preserves_null_values():
    df = pd.DataFrame({"revolving_utilization": [0.1, None, 2.0]})

    result = apply_shift_add(df, "revolving_utilization", 0.15)

    assert result["revolving_utilization"].iloc[0] == pytest.approx(0.25)
    assert pd.isna(result["revolving_utilization"].iloc[1])
    assert result["revolving_utilization"].iloc[2] == pytest.approx(2.15)


def test_apply_shift_add_preserves_other_columns():
    df = pd.DataFrame(
        {
            "revolving_utilization": [0.1, 1.2],
            "age": [25, 35],
        }
    )

    result = apply_shift_add(df, "revolving_utilization", 0.15)

    pd.testing.assert_series_equal(result["age"], df["age"])


def test_apply_shift_add_does_not_modify_original_dataframe():
    df = pd.DataFrame({"revolving_utilization": [0.1, 1.2]})
    original = df.copy(deep=True)

    apply_shift_add(df, "revolving_utilization", 0.15)

    pd.testing.assert_frame_equal(df, original)


def test_apply_shift_add_raises_error_for_missing_feature():
    df = pd.DataFrame({"age": [25, 35]})

    with pytest.raises(ValueError, match="não existe"):
        apply_shift_add(df, "revolving_utilization", 0.15)


def test_apply_shift_add_raises_error_for_invalid_clip():
    df = pd.DataFrame({"revolving_utilization": [0.1, 1.2]})

    with pytest.raises(ValueError, match="dois limites"):
        apply_shift_add(df, "revolving_utilization", 0.15, clip=[0, 3, 4])


def test_apply_shift_add_raises_error_for_inverted_clip():
    df = pd.DataFrame({"revolving_utilization": [0.1, 1.2]})

    with pytest.raises(ValueError, match="não pode superar"):
        apply_shift_add(df, "revolving_utilization", 0.15, clip=[3, 0])


def test_apply_shift_add_rejects_non_finite_value():
    df = pd.DataFrame({"revolving_utilization": [0.1, 1.2]})

    with pytest.raises(ValueError, match="finito"):
        apply_shift_add(df, "revolving_utilization", float("inf"))


def test_apply_resample_mix_returns_requested_sample_size():
    df = pd.DataFrame({"customer_id": range(100)})

    result = apply_resample_mix(df, sample_size=50, seed=42)

    assert len(result) == 50


def test_apply_resample_mix_is_reproducible():
    df = pd.DataFrame({"customer_id": range(100)})

    result_1 = apply_resample_mix(df, 50, seed=42)
    result_2 = apply_resample_mix(df, 50, seed=42)

    pd.testing.assert_frame_equal(result_1, result_2)


def test_apply_resample_mix_allows_replacement():
    df = pd.DataFrame({"customer_id": range(10)})

    result = apply_resample_mix(
        df,
        sample_size=100,
        seed=42,
        replace=True,
    )

    assert len(result) == 100
    assert result["customer_id"].nunique() <= 10


def test_apply_resample_mix_without_replacement():
    df = pd.DataFrame({"customer_id": range(100)})

    result = apply_resample_mix(
        df,
        sample_size=50,
        seed=42,
        replace=False,
    )

    assert len(result) == 50
    assert result["customer_id"].nunique() == 50


def test_apply_resample_mix_preserves_columns():
    df = pd.DataFrame(
        {
            "customer_id": range(100),
            "monthly_income": range(1000, 1100),
            "age": range(20, 120),
        }
    )

    result = apply_resample_mix(df, 50, seed=42)

    assert list(result.columns) == list(df.columns)


def test_apply_resample_mix_does_not_modify_original_dataframe():
    df = pd.DataFrame({"customer_id": range(100)})
    original = df.copy(deep=True)

    apply_resample_mix(df, 50, seed=42)

    pd.testing.assert_frame_equal(df, original)


def test_apply_resample_mix_rejects_invalid_sample_size():
    df = pd.DataFrame({"customer_id": range(10)})

    with pytest.raises(ValueError, match="maior que zero"):
        apply_resample_mix(df, 0, seed=42)


def test_apply_resample_mix_rejects_sample_size_without_replacement():
    df = pd.DataFrame({"customer_id": range(10)})

    with pytest.raises(ValueError, match="não pode ser maior"):
        apply_resample_mix(df, 20, seed=42, replace=False)


def test_apply_resample_mix_rejects_empty_dataframe():
    df = pd.DataFrame({"customer_id": []})

    with pytest.raises(ValueError, match="DataFrame vazio"):
        apply_resample_mix(df, 10, seed=42)


def test_apply_resample_mix_rejects_invalid_seed():
    df = pd.DataFrame({"customer_id": range(10)})

    with pytest.raises(ValueError, match="seed deve estar"):
        apply_resample_mix(df, 5, seed=-1)


def test_apply_transformations_executes_scale():
    df = pd.DataFrame({"monthly_income": [1000.0, 2000.0]})

    result = apply_transformations(
        df,
        [{"feature": "monthly_income", "op": "scale", "factor": 1.25}],
        seed=42,
    )

    assert result["monthly_income"].tolist() == [1250.0, 2500.0]


def test_apply_transformations_executes_shift_add_with_clip():
    df = pd.DataFrame({"revolving_utilization": [0.1, 2.95]})

    result = apply_transformations(
        df,
        [
            {
                "feature": "revolving_utilization",
                "op": "shift_add",
                "value": 0.15,
                "clip": [0, 3],
            }
        ],
        seed=42,
    )

    assert result["revolving_utilization"].tolist() == pytest.approx([0.25, 3.0])


def test_apply_transformations_executes_multiple_operations_in_order():
    df = pd.DataFrame({"monthly_income": [1000.0, 2000.0]})

    result = apply_transformations(
        df,
        [
            {"feature": "monthly_income", "op": "scale", "factor": 1.25},
            {"feature": "monthly_income", "op": "shift_add", "value": 100},
        ],
        seed=42,
    )

    assert result["monthly_income"].tolist() == [1350.0, 2600.0]


def test_apply_transformations_executes_resample_mix():
    df = pd.DataFrame({"customer_id": range(100)})

    result = apply_transformations(
        df,
        [{"op": "resample_mix", "sample_size": 50}],
        seed=42,
    )

    assert len(result) == 50


def test_apply_transformations_rejects_unknown_operation():
    df = pd.DataFrame({"monthly_income": [1000.0]})

    with pytest.raises(ValueError, match="não suportada"):
        apply_transformations(
            df,
            [{"feature": "monthly_income", "op": "unknown"}],
            seed=42,
        )


def test_apply_transformations_rejects_invalid_parameters():
    df = pd.DataFrame({"monthly_income": [1000.0]})

    with pytest.raises(ValueError, match="Parâmetros inválidos"):
        apply_transformations(
            df,
            [{"feature": "monthly_income", "op": "scale"}],
            seed=42,
        )


def test_apply_transformations_does_not_modify_original_dataframe():
    df = pd.DataFrame({"monthly_income": [1000.0, 2000.0]})
    original = df.copy(deep=True)

    apply_transformations(
        df,
        [{"feature": "monthly_income", "op": "scale", "factor": 1.25}],
        seed=42,
    )

    pd.testing.assert_frame_equal(df, original)


def test_apply_transformations_rejects_invalid_seed():
    df = pd.DataFrame({"monthly_income": [1000.0]})

    with pytest.raises(ValueError, match="seed deve estar"):
        apply_transformations(df, [], seed=-1)


def test_load_scenario_config_reads_yaml():
    config_path = Path("config/drift_scenarios.yaml")

    config = load_scenario_config(config_path)

    assert config["version"] == 1
    assert config["seed"] == 42
    assert "covariate_moderate" in config["scenarios"]


def test_apply_scenario_transformations_moderate():
    df = pd.DataFrame(
        {
            "monthly_income": [1000.0, 2000.0],
            "revolving_utilization": [0.1, 2.95],
            "age": [25, 35],
        }
    )

    config_path = Path("config/drift_scenarios.yaml")

    result = apply_scenario_transformations(
        df=df,
        scenario="covariate_moderate",
        config_path=config_path,
    )

    assert result["monthly_income"].tolist() == [1250.0, 2500.0]

    assert result["revolving_utilization"].tolist() == pytest.approx([0.25, 3.0])

    pd.testing.assert_series_equal(result["age"], df["age"])


def test_apply_scenario_transformations_no_drift():
    df = pd.DataFrame(
        {
            "monthly_income": [1000.0, 2000.0],
            "revolving_utilization": [0.1, 2.95],
        }
    )

    config_path = Path("config/drift_scenarios.yaml")

    result = apply_scenario_transformations(
        df=df,
        scenario="no_drift",
        config_path=config_path,
    )

    pd.testing.assert_frame_equal(result, df)

    assert result is not df


def test_apply_scenario_transformations_rejects_unknown_scenario():
    df = pd.DataFrame({"age": [25, 35]})

    config_path = Path("config/drift_scenarios.yaml")

    with pytest.raises(ValueError, match="Cenário não encontrado"):
        apply_scenario_transformations(
            df=df,
            scenario="invalid_scenario",
            config_path=config_path,
        )


def test_load_scenario_config_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        load_scenario_config(Path("config/arquivo_inexistente.yaml"))


def test_load_scenario_config_rejects_invalid_yaml(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("scenarios: [invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML inválido"):
        load_scenario_config(config_path)


def test_load_scenario_config_rejects_missing_required_keys(tmp_path):
    config_path = tmp_path / "incomplete.yaml"
    config_path.write_text(
        yaml.safe_dump({"version": 1, "seed": 42}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="campos obrigatórios"):
        load_scenario_config(config_path)
