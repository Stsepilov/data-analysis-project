"""Загрузка, сборка датасета и журнал запусков."""

import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from river_forecast import __version__
from river_forecast.artifacts import frame_sha256, write_json
from river_forecast.config import Config
from river_forecast.data import merge_daily, quality_report
from river_forecast.errors import describe_error
from river_forecast.evaluation import evaluate_baseline
from river_forecast.features import build_features
from river_forecast.http import make_session
from river_forecast.sources import fetch_hydrology, fetch_weather


def config_metadata(config: Config) -> dict:
    return {
        **asdict(config),
        "start": config.start.isoformat(),
        "end": config.end.isoformat(),
        "holdout_start": config.holdout_start.isoformat(),
    }


@contextmanager
def journal(root: Path, command: str, metadata: dict):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "runs.jsonl"
    record = {
        "run_id": str(uuid4()),
        "command": command,
        "started_at_utc": datetime.now(UTC).isoformat(),
        **metadata,
    }

    def append(event: dict):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    append({**record, "status": "started"})
    try:
        yield record
    except Exception as exc:
        append(
            {
                **record,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": describe_error(exc),
                "finished_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        raise
    else:
        append({**record, "status": "success", "finished_at_utc": datetime.now(UTC).isoformat()})


def fetch_dataset(config: Config, processed_dir: Path, raw_dir: Path) -> pd.DataFrame:
    with make_session() as session:
        hydro = fetch_hydrology(session, config, raw_dir)
        weather = fetch_weather(session, config, raw_dir)
    daily = merge_daily(hydro, weather, config.start.isoformat(), config.end.isoformat())
    processed_dir.mkdir(parents=True, exist_ok=True)
    hydro.to_csv(processed_dir / "hydrology.csv", index=False)
    weather.to_csv(processed_dir / "weather.csv", index=False)
    daily.to_csv(processed_dir / "daily.csv")
    write_json(
        processed_dir / "dataset.json",
        {
            "kind": "historical_observations_and_reanalysis",
            "config": config_metadata(config),
            "daily_sha256": frame_sha256(processed_dir / "daily.csv"),
            "hydrology_source": "USGS Water Data API v1 / daily",
            "weather_source": "Open-Meteo / ERA5",
            "station_metadata": hydro.attrs["station_metadata"],
            "level_statistic": "00065 / 00003: daily mean gage height",
            "discharge_statistic": "00060 / 00003: daily mean discharge",
            "level_reference": "USGS gage reference; not absolute elevation",
            "source_units": {"level": "ft", "discharge": "ft^3/s"},
            "canonical_units": {"level": "cm", "discharge": "m3/s"},
            "time_series_ids": {
                column: hydro[f"{column}_time_series_id"].dropna().unique().tolist()
                for column in ("level_cm", "discharge_m3s")
            },
        },
    )
    return daily


def evaluate_dataset(
    daily: pd.DataFrame,
    processed_dir: Path,
    artifact_dir: Path,
    metadata: dict,
    delay: int = 5,
    holdout_start: date | None = None,
) -> dict:
    report = quality_report(daily)
    features = build_features(daily, weather_delay_days=delay)
    development = daily if holdout_start is None else daily.loc[daily.index < str(holdout_start)]
    metrics, predictions = evaluate_baseline(development)
    metrics["data_kind"] = metadata["kind"]
    metrics["holdout_start_excluded"] = str(holdout_start) if holdout_start else None
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(processed_dir / "features.csv")
    predictions.to_csv(artifact_dir / "predictions.csv")
    write_json(artifact_dir / "quality.json", report)
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(
        artifact_dir / "run.json",
        {
            "project_version": __version__,
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "dataset": metadata,
            "dataset_sha256": hashlib.sha256(daily.to_csv().encode()).hexdigest(),
            "feature_columns": list(features.columns),
            "weather_delay_days": delay,
            "complete_feature_rows": int(features.notna().all(axis=1).sum()),
            "metrics": metrics,
        },
    )
    return metrics
