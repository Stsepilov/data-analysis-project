from datetime import date

from river_forecast.demo import make_demo
from river_forecast.pipeline import evaluate_dataset


def test_reserved_holdout_cannot_influence_development_baseline(tmp_path):
    daily = make_demo()
    metadata = {"kind": "synthetic_test"}
    kwargs = {
        "processed_dir": tmp_path / "processed",
        "artifact_dir": tmp_path / "artifacts",
        "metadata": metadata,
        "holdout_start": date(2023, 1, 1),
    }
    before = evaluate_dataset(daily, **kwargs)
    predictions = (tmp_path / "artifacts/predictions.csv").read_bytes()
    daily.loc["2023-01-01":, "level_cm"] = 99_999
    after = evaluate_dataset(daily, **kwargs)
    assert before == after
    assert (tmp_path / "artifacts/predictions.csv").read_bytes() == predictions
    assert before["evaluation_target_end"] == "2022-12-31"
