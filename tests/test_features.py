import numpy as np
import pandas as pd
import pytest

from river_forecast.data import merge_daily, quality_report
from river_forecast.features import build_features, build_target


def test_truncating_future_does_not_change_any_past_feature(daily):
    cutoff = daily.index[49]
    complete = build_features(daily)
    truncated = build_features(daily.loc[:cutoff])
    pd.testing.assert_frame_equal(complete.loc[:cutoff], truncated)
    modified = daily.copy()
    modified.loc[modified.index > cutoff, "level_cm"] = 999_999
    modified.loc[modified.index > cutoff, "temperature_mean_c"] = -999
    pd.testing.assert_frame_equal(complete.loc[:cutoff], build_features(modified).loc[:cutoff])


def test_weather_delay_and_window_use_only_available_days(daily):
    features = build_features(daily, weather_delay_days=5)
    origin = daily.index[20]
    assert features.loc[origin, "temperature_mean_c_available"] == 15
    assert features.loc[origin, "precipitation_sum_7_mm"] == 7
    assert features.loc[origin, "level_mean_3_cm"] == daily["level_cm"].iloc[18:21].mean()
    assert build_target(daily).loc[origin] == daily["level_cm"].iloc[21]
    assert np.isnan(build_target(daily).iloc[-1])
    assert features["level_current_cm"].iloc[-1] == daily["level_cm"].iloc[-1]


def test_missing_dates_are_not_treated_as_adjacent_days(daily):
    hydro = daily[["level_cm"]].drop(daily.index[10]).reset_index()
    weather = daily.drop(columns="level_cm").reset_index()
    merged = merge_daily(hydro, weather, "2020-01-01", "2020-04-09")
    features = build_features(merged)
    assert np.isnan(features["level_lag_1_cm"].iloc[11])
    assert np.isnan(build_target(merged).iloc[9])
    assert merged["level_cm"].iloc[9] == -41  # Уровень ниже нуля поста допустим.
    assert quality_report(merged)["missing"]["level_cm"] == 1


def test_irregular_calendar_is_rejected_before_feature_generation(daily):
    with pytest.raises(ValueError, match="В календаре пропущены дни"):
        build_features(daily.drop(daily.index[10]))
