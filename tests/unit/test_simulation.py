import pandas as pd
import pytest

from creditwatch.simulation import apply_scale, apply_shift_add


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
