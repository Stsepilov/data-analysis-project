"""Синтетические данные для офлайн-демо с фиксируемым seed."""

import numpy as np
import pandas as pd


def make_demo(seed: int = 42, days: int = 730) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=days, freq="D", name="date")
    day = np.arange(days)
    rain = rng.gamma(shape=0.7, scale=3.0, size=days)
    temperature = 5 + 15 * np.sin(2 * np.pi * (day - 100) / 365) + rng.normal(0, 2, days)
    level = np.zeros(days)
    level[0] = -80
    for i in range(1, days):
        seasonal = -100 + 250 * np.exp(-((((i % 365) - 110) / 25) ** 2))
        level[i] = 0.95 * level[i - 1] + 0.05 * seasonal + 0.8 * rain[i - 1]
        level[i] += rng.normal(0, 3)
    return pd.DataFrame(
        {
            "level_cm": np.round(level, 2),
            "temperature_mean_c": np.round(temperature, 2),
            "precipitation_mm": np.round(rain, 2),
            "humidity_mean_pct": np.round(rng.uniform(40, 95, days), 2),
        },
        index=dates,
    )
