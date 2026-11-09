from creditwatch.contracts import validate_batch


def test_blocks_invalid_age(production_pool):
    df = production_pool.copy()

    df.loc[df.index[0], "age"] = 17

    result = validate_batch(
        df,
        batch_id="test_invalid_age",
    )

    assert result.status == "BLOCKED"

    age_check = next(check for check in result.checks if check.rule_id == "R01_AGE_RANGE")

    assert not age_check.passed
    assert age_check.n_failures >= 1


def test_blocks_negative_income(production_pool):
    df = production_pool.copy()
    df.loc[df.index[0], "monthly_income"] = -100

    result = validate_batch(df, batch_id="test_negative_income")

    assert result.status == "BLOCKED"
    assert any(check.rule_id == "R02_INCOME" and not check.passed for check in result.checks)


def test_blocks_duplicate_customer_id(production_pool):
    df = production_pool.copy()
    df.loc[df.index[1], "customer_id"] = df.loc[df.index[0], "customer_id"]

    result = validate_batch(df, batch_id="test_duplicate_id")

    assert result.status == "BLOCKED"
    assert any(check.rule_id == "R03_NO_DUPLICATES" and not check.passed for check in result.checks)


def test_warns_when_volume_is_below_minimum(production_pool):
    df = production_pool.head(500).copy()

    result = validate_batch(df, batch_id="test_low_volume")

    assert result.status == "PASSED_WITH_WARNINGS"
    assert any(check.rule_id == "W02_MIN_VOLUME" and not check.passed for check in result.checks)


def test_blocks_missing_required_column(production_pool):
    df = production_pool.copy()

    df = df.drop(columns=["age"])

    result = validate_batch(
        df,
        batch_id="test_missing_age",
    )

    assert result.status == "BLOCKED"
    assert any(check.rule_id == "R04_SCHEMA" and not check.passed for check in result.checks)


def test_blocks_null_in_non_nullable_column(production_pool):
    df = production_pool.copy()
    df.loc[df.index[0], "monthly_income"] = None

    result = validate_batch(df, batch_id="test_null_income")

    assert result.status == "BLOCKED"
    assert any(check.rule_id == "R04_SCHEMA" and not check.passed for check in result.checks)


def test_blocks_invalid_target(production_pool):
    df = production_pool.copy()
    df.loc[df.index[0], "target"] = 2

    result = validate_batch(df, batch_id="test_invalid_target")

    assert result.status == "BLOCKED"
    assert any(check.rule_id == "R04_SCHEMA" and not check.passed for check in result.checks)
