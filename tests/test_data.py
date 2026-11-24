import numpy as np
import pytest

from river_forecast.data import daily_index, quality_report


def test_duplicate_measurements_require_an_explicit_resolution(daily):
    frame = daily[["level_cm"]].reset_index()
    frame.loc[1, "date"] = frame.loc[0, "date"]
    with pytest.raises(ValueError, match="дубли дат"):
        daily_index(frame, ("date", "level_cm"))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("precipitation_mm", -1, "осадков ниже нуля"),
        ("humidity_mean_pct", 101, "Влажность"),
        ("level_cm", np.inf, "встретилось inf"),
    ],
)
def test_invalid_values_fail_instead_of_becoming_training_data(daily, column, value, message):
    daily.loc[daily.index[0], column] = value
    with pytest.raises(ValueError, match=message):
        quality_report(daily)
