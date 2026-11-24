import json
from datetime import date

import pandas as pd
import pytest
import requests

from river_forecast.config import Config, Station
from river_forecast.sources import (
    HYDRO_BASE,
    fetch_hydrology,
    fetch_weather,
    parse_hydrology_csv,
    parse_weather,
)


def hydro_frame(parameter="00065", unit="ft"):
    # Две тестовые строки в формате выгрузки USGS v1.
    return pd.DataFrame(
        {
            "monitoring_location_id": ["USGS-02320500"] * 2,
            "parameter_code": [parameter] * 2,
            "statistic_id": ["00003"] * 2,
            "time": ["2023-01-02", "2023-01-01"],
            "value": [2.0, 1.0],
            "unit_of_measure": [unit] * 2,
            "time_series_id": ["example-series"] * 2,
            "approval_status": ["Provisional", "Approved"],
            "qualifier": [None, None],
            "last_modified": ["2023-01-03T12:00:00Z"] * 2,
        }
    )


def parse(frame, parameter="00065"):
    return parse_hydrology_csv(frame.to_csv(index=False).encode(), "02320500", parameter)


def test_usgs_level_is_converted_from_feet_to_cm_and_provenance_is_preserved():
    frame = parse(hydro_frame())
    assert frame["date"].tolist() == ["2023-01-01", "2023-01-02"]
    assert frame["level_cm"].tolist() == pytest.approx([30.48, 60.96])
    assert frame["level_cm_approval_status"].tolist() == ["Approved", "Provisional"]


def test_usgs_discharge_is_converted_from_cubic_feet_per_second():
    frame = parse(hydro_frame("00060", "ft^3/s"), parameter="00060")
    assert frame["discharge_m3s"].tolist() == pytest.approx([0.028316846592, 0.056633693184])


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("monitoring_location_id", "USGS-01646500"),
        ("parameter_code", "00060"),
        ("statistic_id", "00001"),
        ("unit_of_measure", "m"),
        ("approval_status", "unknown"),
    ],
)
def test_wrong_station_parameter_statistic_units_or_status_are_rejected(column, value):
    frame = hydro_frame()
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        parse(frame)


def test_multiple_usgs_series_are_not_silently_averaged():
    frame = hydro_frame()
    frame.loc[0, "time_series_id"] = "another-sensor"
    with pytest.raises(ValueError, match="time_series_id"):
        parse(frame)


def test_duplicate_dates_are_rejected_and_missing_values_stay_missing():
    frame = hydro_frame()
    frame.loc[0, "value"] = float("nan")
    parsed = parse(frame)
    assert pd.isna(parsed["level_cm"].iloc[1])
    frame.loc[0, "time"] = frame.loc[1, "time"]
    with pytest.raises(ValueError, match="дубли дат"):
        parse(frame)


def test_weather_units_and_parallel_array_lengths_are_checked():
    payload = {
        "daily": {
            "time": ["2023-01-01"],
            "temperature_2m_mean": [-10],
            "precipitation_sum": [0],
            "relative_humidity_2m_mean": [80],
        },
        "daily_units": {
            "temperature_2m_mean": "°C",
            "precipitation_sum": "mm",
            "relative_humidity_2m_mean": "%",
        },
    }
    assert parse_weather(payload)["temperature_mean_c"].iloc[0] == -10
    payload["daily_units"]["precipitation_sum"] = "inch"
    with pytest.raises(ValueError, match="единицы"):
        parse_weather(payload)
    payload["daily_units"]["precipitation_sum"] = "mm"
    payload["daily"]["precipitation_sum"] = []
    with pytest.raises(ValueError, match="Массивы Open-Meteo"):
        parse_weather(payload)


@pytest.fixture
def source_client():
    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            response = requests.Response()
            response.status_code = 200
            response.encoding = "utf-8"
            if url.endswith("monitoring-locations/items"):
                payload = {
                    "features": [
                        {
                            "properties": {"id": "USGS-02320500"},
                            "geometry": {"type": "Point", "coordinates": [-82.9276, 29.9558]},
                        }
                    ]
                }
                response._content = json.dumps(payload).encode()
            elif url.endswith("daily/items"):
                parameter = kwargs["params"]["parameter_code"]
                unit = "ft" if parameter == "00065" else "ft^3/s"
                response._content = hydro_frame(parameter, unit).to_csv(index=False).encode()
            else:
                payload = {
                    "daily": {
                        "time": ["2023-01-01", "2023-01-02"],
                        "temperature_2m_mean": [10, 11],
                        "precipitation_sum": [0, 1],
                        "relative_humidity_2m_mean": [70, 80],
                    },
                    "daily_units": {
                        "temperature_2m_mean": "°C",
                        "precipitation_sum": "mm",
                        "relative_humidity_2m_mean": "%",
                    },
                }
                response._content = json.dumps(payload).encode()
            return response

    config = Config(
        station=Station(
            "02320500", "Suwannee at Branford", "Suwannee", 29.9558, -82.9276, "Etc/GMT+5"
        ),
        start=date(2023, 1, 1),
        end=date(2023, 1, 2),
        holdout_start=date(2023, 1, 2),
    )
    return FakeSession(), config


def test_client_combines_both_parameters_and_sends_api_key_only_to_usgs(
    source_client, tmp_path, monkeypatch
):
    session, config = source_client
    monkeypatch.setenv("USGS_API_KEY", "private-test-key")
    hydro = fetch_hydrology(session, config, tmp_path)
    fetch_weather(session, config, tmp_path)
    assert hydro["monitoring_location_id"].eq("USGS-02320500").all()
    assert hydro["level_cm"].tolist() == pytest.approx([30.48, 60.96])
    assert hydro["discharge_m3s"].tolist() == pytest.approx([0.028316846592, 0.056633693184])
    assert len(session.calls) == 4
    for url, kwargs in session.calls:
        if url.startswith(HYDRO_BASE):
            assert kwargs["headers"]["X-Api-Key"] == "private-test-key"
        else:
            assert "X-Api-Key" not in kwargs.get("headers", {})
    for path in tmp_path.rglob("*.json"):
        assert "private-test-key" not in path.read_text(encoding="utf-8")


def test_saturated_usgs_response_is_rejected_instead_of_silently_truncated(
    source_client, tmp_path, monkeypatch
):
    session, config = source_client
    monkeypatch.setattr("river_forecast.sources.USGS_PAGE_LIMIT", 2)
    with pytest.raises(ValueError, match="лимит строк"):
        fetch_hydrology(session, config, tmp_path)
