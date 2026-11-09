"""Fixtures sintéticos compartilhados; testes unitários não dependem de dados locais."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def production_pool() -> pd.DataFrame:
    """Pool válido grande o bastante para os cenários de drift configurados."""
    n_rows = 18_000
    row = np.arange(n_rows)
    return pd.DataFrame(
        {
            "customer_id": [f"C{index:07d}" for index in row],
            "revolving_utilization": (row % 280) / 100,
            "age": 18 + row % 80,
            "past_due_30_59": row % 5,
            "debt_ratio": (row % 700) / 100,
            "monthly_income": 1_000 + (row % 240) * 100,
            "open_credit_lines": row % 25,
            "times_90_days_late": row % 4,
            "real_estate_loans": row % 5,
            "past_due_60_89": row % 4,
            "dependents": row % 6,
            "target": (row % 11 == 0).astype("int8"),
        }
    )
