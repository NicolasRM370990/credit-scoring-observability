"""Testes da configuração e validação dos thresholds de drift."""

import pytest

from creditwatch.drift.detector import _validate_thresholds
from creditwatch.interfaces import DriftThresholds
from creditwatch.pipeline.run import _thresholds


def test_load_thresholds_from_yaml():
    """Carrega os thresholds definidos na configuração do projeto."""
    thresholds = _thresholds()

    assert isinstance(thresholds, DriftThresholds)
    assert thresholds.psi_warn == pytest.approx(0.015639)
    assert thresholds.psi_critical == pytest.approx(0.017247)
    assert thresholds.ks_stat_min == pytest.approx(0.042700)
    assert thresholds.drift_share_alert == pytest.approx(0.30)


def test_valid_thresholds_are_accepted():
    thresholds = DriftThresholds(
        psi_warn=0.015639,
        psi_critical=0.017247,
        ks_stat_min=0.042700,
        drift_share_alert=0.30,
    )

    _validate_thresholds(thresholds)


@pytest.mark.parametrize(
    "values",
    [
        {
            "psi_warn": 0.25,
            "psi_critical": 0.10,
            "ks_stat_min": 0.10,
            "drift_share_alert": 0.30,
        },
        {
            "psi_warn": -0.01,
            "psi_critical": 0.10,
            "ks_stat_min": 0.10,
            "drift_share_alert": 0.30,
        },
        {
            "psi_warn": 0.10,
            "psi_critical": 0.25,
            "ks_stat_min": 1.10,
            "drift_share_alert": 0.30,
        },
        {
            "psi_warn": 0.10,
            "psi_critical": 0.25,
            "ks_stat_min": 0.10,
            "drift_share_alert": 1.10,
        },
    ],
)
def test_invalid_thresholds_are_rejected(values):
    thresholds = DriftThresholds(**values)

    with pytest.raises(ValueError):
        _validate_thresholds(thresholds)
