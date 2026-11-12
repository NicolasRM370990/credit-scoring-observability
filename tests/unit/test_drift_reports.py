from pathlib import Path

import pandas as pd
import pytest

from creditwatch.drift.reports import generate_batch_report, generate_psi_chart
from creditwatch.interfaces import (
    DriftResult,
    DriftThresholds,
    FeatureDrift,
)


@pytest.fixture
def dataframes():
    reference = pd.DataFrame({
        "feature_a": [10, 11, 12, 13, 14] * 20,
        "feature_b": [20, 21, 22, 23, 24] * 20,
        "score": [0.1, 0.2, 0.3, 0.4, 0.5] * 20,
    })

    current = pd.DataFrame({
        "feature_a": [30, 31, 32, 33, 34] * 20,
        "feature_b": [50, 51, 52, 53, 54] * 20,
        "score": [0.2, 0.3, 0.4, 0.5, 0.6] * 20,
    })

    return reference, current


@pytest.fixture
def drift_result():
    return DriftResult(
        run_id="test-run",
        batch_id="batch_01",
        scenario="test",
        reference_rows=100,
        current_rows=100,
        dataset_drift=True,
        drift_share=1.0,
        features=[
            FeatureDrift(
                name="feature_a",
                psi=0.8,
                ks_stat=0.9,
                ks_pvalue=0.001,
                evidently_drifted=True,
                drifted=True,
                severity="CRITICAL",
            ),
            FeatureDrift(
                name="feature_b",
                psi=0.7,
                ks_stat=0.8,
                ks_pvalue=0.001,
                evidently_drifted=True,
                drifted=True,
                severity="CRITICAL",
            ),
        ],
        thresholds=DriftThresholds(
            psi_warn=0.1,
            psi_critical=0.25,
            ks_stat_min=0.1,
            drift_share_alert=0.3,
        ),
    )


def test_generate_batch_report(
    dataframes,
    drift_result,
    tmp_path: Path,
):
    reference, current = dataframes

    updated_result = generate_batch_report(
        reference=reference,
        current=current,
        result=drift_result,
        output_dir=tmp_path,
    )

    html_path = tmp_path / "batch_01.html"
    json_path = tmp_path / "batch_01.json"

    assert html_path.exists()
    assert html_path.stat().st_size > 0

    assert json_path.exists()
    assert json_path.stat().st_size > 0

    assert updated_result.report_html == str(html_path)
    assert drift_result.report_html is None


def test_generate_batch_report_rejects_empty_data(
    drift_result,
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="não podem estar vazios"):
        generate_batch_report(
            reference=pd.DataFrame(),
            current=pd.DataFrame(),
            result=drift_result,
            output_dir=tmp_path,
        )


def test_generate_psi_chart(
    drift_result,
    tmp_path: Path,
):
    second_result = drift_result.model_copy(
        update={
            "batch_id": "batch_02",
            "features": [
                feature.model_copy(update={"psi": feature.psi + 0.2})
                for feature in drift_result.features
            ],
        }
    )

    output_path = tmp_path / "psi_evolution.png"

    generated_path = generate_psi_chart(
        results=[drift_result, second_result],
        output_path=output_path,
    )

    assert generated_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_psi_chart_rejects_empty_results(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="não pode estar vazia"):
        generate_psi_chart(
            results=[],
            output_path=tmp_path / "psi.png",
        )


def test_generate_psi_chart_rejects_duplicate_batches(
    drift_result,
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="batch_ids duplicados"):
        generate_psi_chart(
            results=[drift_result, drift_result],
            output_path=tmp_path / "psi.png",
        )