"""Оценка наивного прогноза на последних датах ряда."""

import numpy as np
import pandas as pd

from river_forecast.features import build_target


def evaluate_baseline(daily: pd.DataFrame, test_fraction: float = 0.2) -> tuple[dict, pd.DataFrame]:
    if not 0 < test_fraction < 1:
        raise ValueError("Доля теста test_fraction должна быть строго между 0 и 1.")
    if len(daily) < 30:
        raise ValueError("Для оценки бейзлайна нужно хотя бы 30 календарных дней.")
    # Сначала выбираем границу по календарю. Пропуски не должны её сдвигать.
    boundary = daily.index[int(len(daily) * (1 - test_fraction))]
    target = build_target(daily)
    result = pd.DataFrame({"actual_cm": target, "prediction_cm": daily["level_cm"]})
    # Делим по дате прогноза. На каждый следующий день берём новый факт.
    result["target_date"] = result.index.shift(1, freq="D")
    candidates = result.loc[
        (result["target_date"] >= boundary) & (result["target_date"] <= daily.index.max())
    ]
    result = candidates.dropna(subset=["actual_cm", "prediction_cm"]).copy()
    if len(result) < 5:
        raise ValueError("В тестовом периоде меньше пяти пар «прогноз — факт».")
    error = result["actual_cm"] - result["prediction_cm"]
    metrics = {
        "model": "persistence",
        "horizon_days": 1,
        "mae_cm": float(error.abs().mean()),
        "rmse_cm": float(np.sqrt(np.mean(error**2))),
        "n_predictions": len(result),
        "n_skipped_pairs": len(candidates) - len(result),
        "evaluation_target_start": boundary.date().isoformat(),
        "evaluation_target_end": result["target_date"].max().date().isoformat(),
        "protocol": (
            f"one-step rolling evaluation on the last {test_fraction:.0%} of calendar dates"
        ),
    }
    result.index.name = "origin_date"
    return metrics, result
