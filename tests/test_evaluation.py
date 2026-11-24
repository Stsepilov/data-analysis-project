import pytest

from river_forecast.evaluation import evaluate_baseline


def test_baseline_uses_only_previous_day_and_known_targets(daily):
    metrics, predictions = evaluate_baseline(daily)
    assert metrics["mae_cm"] == pytest.approx(1)
    assert metrics["rmse_cm"] == pytest.approx(1)
    assert metrics["n_predictions"] == 20
    assert metrics["n_skipped_pairs"] == 0
    assert predictions.index.min() == daily.index[79]
    assert predictions["target_date"].max() == daily.index.max()


def test_missing_target_and_origin_are_excluded_without_filling(daily):
    daily.loc[daily.index[85], "level_cm"] = float("nan")
    metrics, predictions = evaluate_baseline(daily)
    assert metrics["n_predictions"] == 18
    assert metrics["n_skipped_pairs"] == 2
    assert daily.index[84] not in predictions.index
    assert daily.index[85] not in predictions.index
