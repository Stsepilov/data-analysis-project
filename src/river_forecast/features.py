"""Единый расчёт признаков для обучения и прогноза."""

import numpy as np
import pandas as pd

from river_forecast.data import WEATHER_COLUMNS, quality_report


def build_features(daily: pd.DataFrame, weather_delay_days: int = 5) -> pd.DataFrame:
    """На дату t берём уровень за t и погоду до t минус задержка.

    Сдвиги идут по календарю. Последнюю строку сохраняем для прогноза.
    """
    quality_report(daily)
    if weather_delay_days < 0:
        raise ValueError("Задержка погоды weather_delay_days должна быть от 0 дней.")
    level = daily["level_cm"]
    features = pd.DataFrame(index=daily.index)
    features["level_current_cm"] = level
    for lag in (1, 2, 7):
        features[f"level_lag_{lag}_cm"] = level.shift(lag)
    for window in (3, 7):
        features[f"level_mean_{window}_cm"] = level.rolling(window, min_periods=window).mean()
    features["level_change_1_cm"] = level.diff()
    delayed = daily.loc[:, list(WEATHER_COLUMNS)].shift(weather_delay_days)
    for column in WEATHER_COLUMNS:
        features[f"{column}_available"] = delayed[column]
    features["precipitation_sum_7_mm"] = delayed["precipitation_mm"].rolling(7).sum()
    # Цикл гладко проходит через Новый год, учитываем и 29 февраля.
    year_length = np.where(daily.index.is_leap_year, 366, 365)
    phase = 2 * np.pi * (daily.index.dayofyear - 1) / year_length
    features["day_of_year_sin"] = np.sin(phase)
    features["day_of_year_cos"] = np.cos(phase)
    return features


def build_target(daily: pd.DataFrame, horizon_days: int = 1) -> pd.Series:
    quality_report(daily)
    if horizon_days < 1:
        raise ValueError("Горизонт horizon_days должен быть хотя бы 1 день.")
    return daily["level_cm"].shift(-horizon_days).rename("target_level_cm")
