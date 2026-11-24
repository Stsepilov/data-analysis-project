import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def daily():
    dates = pd.date_range("2020-01-01", periods=100, freq="D", name="date")
    return pd.DataFrame(
        {
            "level_cm": np.arange(100, dtype=float) - 50,
            "temperature_mean_c": np.arange(100, dtype=float),
            "precipitation_mm": np.ones(100),
            "humidity_mean_pct": np.full(100, 70.0),
        },
        index=dates,
    )
