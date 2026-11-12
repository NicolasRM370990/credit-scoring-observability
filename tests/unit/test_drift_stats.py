
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from creditwatch.drift.stats import ks_test, psi


def test_psi_same_distribution_is_approximately_zero():
    reference = pd.Series(np.arange(1000))
    current = reference.copy()

    result = psi(reference, current)

    assert result == pytest.approx(0.0, abs=1e-10)


def test_psi_increases_with_distribution_shift():
    reference = pd.Series(np.random.default_rng(42).normal(0, 1, 5000))
    current = pd.Series(np.random.default_rng(42).normal(2, 1, 5000))

    result_no_drift = psi(reference, reference.copy())
    result_drift = psi(reference, current)

    assert result_drift > result_no_drift
    assert result_drift > 0.1


def test_ks_test_matches_scipy():
    reference = pd.Series([1, 2, 3, 4, 5])
    current = pd.Series([2, 3, 4, 5, 6])

    statistic, pvalue = ks_test(reference, current)
    expected = ks_2samp(
        reference,
        current,
        method="asymp",
    )

    assert statistic == pytest.approx(expected.statistic)
    assert pvalue == pytest.approx(expected.pvalue)


@pytest.mark.parametrize(
    "function",
    [psi, ks_test],
)
def test_empty_samples_raise_value_error(function):
    reference = pd.Series([], dtype=float)
    current = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="vazias"):
        function(reference, current)


def test_psi_rejects_invalid_bins():
    with pytest.raises(ValueError, match="bins"):
        psi(pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), bins=0)


def test_psi_rejects_invalid_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        psi(pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), epsilon=0)