import pandas as pd
import pytest

from creditwatch.simulation import apply_scale


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
