from creditwatch.interfaces import (
    LOG_REQUIRED_FIELDS,
    RunContext,
    ValidationCheck,
    ValidationResult,
)


def test_run_context_ids_are_unique():
    assert RunContext.new().run_id != RunContext.new().run_id


def test_validation_result_json_roundtrip():
    result = ValidationResult(
        run_id="r1",
        batch_id="b1",
        status="BLOCKED",
        n_rows=10,
        n_rows_failed=1,
        checks=[
            ValidationCheck(
                rule_id="R01_AGE_RANGE", severity="BLOCK", column="age",
                passed=False, n_failures=1, failure_rate=0.1,
                examples=[{"row_index": 3}],
            )
        ],
        duration_s=0.1,
    )
    again = ValidationResult.model_validate_json(result.model_dump_json())
    assert again.status == "BLOCKED"
    assert again.checks[0].examples == [{"row_index": 3}]


def test_log_fields_include_run_and_stage():
    assert {"run_id", "stage", "status", "duration_ms"} <= set(LOG_REQUIRED_FIELDS)