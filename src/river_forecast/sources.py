"""Загрузка суточных данных из USGS и Open-Meteo."""

import os
from datetime import timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from river_forecast.artifacts import snapshot
from river_forecast.config import Config
from river_forecast.data import WEATHER_COLUMNS
from river_forecast.http import TIMEOUT

HYDRO_BASE = "https://api.waterdata.usgs.gov/ogcapi/v1/collections"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARIABLES = ("temperature_2m_mean", "precipitation_sum", "relative_humidity_2m_mean")
USGS_STATISTIC = "00003"  # Код суточного среднего.
USGS_PAGE_LIMIT = 10_000
USGS_PARAMETERS = {
    "00065": ("level_cm", "ft", 30.48),
    "00060": ("discharge_m3s", "ft^3/s", 0.028316846592),
}


def parse_hydrology_csv(content: bytes, station_code: str, parameter_code: str) -> pd.DataFrame:
    """Разбираем CSV USGS и сохраняем ID ряда и флаги качества."""
    if parameter_code not in USGS_PARAMETERS:
        raise ValueError(f"Параметр USGS {parameter_code} не поддерживается.")
    column, expected_unit, multiplier = USGS_PARAMETERS[parameter_code]
    frame = pd.read_csv(BytesIO(content), dtype="string", encoding="utf-8-sig")
    required = {
        "monitoring_location_id",
        "parameter_code",
        "statistic_id",
        "time",
        "value",
        "unit_of_measure",
        "time_series_id",
        "approval_status",
        "qualifier",
        "last_modified",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"В CSV USGS не хватает полей: {', '.join(sorted(missing))}")
    expected = {
        "monitoring_location_id": f"USGS-{station_code}",
        "parameter_code": parameter_code,
        "statistic_id": USGS_STATISTIC,
        "unit_of_measure": expected_unit,
    }
    for field, value in expected.items():
        if not frame[field].eq(value).fillna(False).all():
            raise ValueError(f"USGS вернул другое значение {field}. Ожидалось {value}.")
    if not frame["time"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False).all():
        raise ValueError("USGS вернул дату не в формате ГГГГ-ММ-ДД.")
    try:
        pd.to_datetime(frame["time"], format="%Y-%m-%d", errors="raise")
    except ValueError as exc:
        raise ValueError("USGS вернул некорректную календарную дату.") from exc
    if frame["time_series_id"].isna().any() or frame["time_series_id"].nunique() > 1:
        raise ValueError("В ответе USGS должен быть ровно один time_series_id на параметр.")
    if frame["time"].duplicated().any():
        raise ValueError("В ряду USGS есть дубли дат.")
    if not frame["approval_status"].isin(["Approved", "Provisional"]).all():
        raise ValueError("USGS вернул неизвестный статус проверки approval_status.")
    try:
        values = pd.to_numeric(frame["value"], errors="raise").astype(float)
    except ValueError as exc:
        raise ValueError("В поле value ответа USGS есть нечисловое значение.") from exc
    if np.isinf(values.to_numpy()).any():
        raise ValueError("В измерениях USGS встретилось inf.")
    if parameter_code == "00060" and (values < 0).any():
        raise ValueError("USGS вернул отрицательный расход воды.")
    result = pd.DataFrame(
        {
            "date": frame["time"],
            column: values * multiplier,
            f"{column}_time_series_id": frame["time_series_id"],
            f"{column}_approval_status": frame["approval_status"],
            f"{column}_qualifier": frame["qualifier"],
            f"{column}_last_modified": frame["last_modified"],
        }
    )
    return result.sort_values("date").reset_index(drop=True)


def parse_weather(payload: dict) -> pd.DataFrame:
    daily = payload.get("daily", {})
    units = payload.get("daily_units", {})
    expected_units = dict(zip(DAILY_VARIABLES, ("°C", "mm", "%"), strict=True))
    if any(units.get(key) != unit for key, unit in expected_units.items()):
        raise ValueError("Open-Meteo вернул неожиданные единицы в daily_units.")
    keys = ("time", *DAILY_VARIABLES)
    if any(key not in daily for key in keys):
        raise ValueError("В ответе Open-Meteo не хватает полей в daily.")
    if not daily["time"] or len({len(daily[key]) for key in keys}) != 1:
        raise ValueError("Массивы Open-Meteo пусты или не совпадают по длине.")
    return pd.DataFrame(
        {
            "date": daily["time"],
            **dict(zip(WEATHER_COLUMNS, (daily[k] for k in DAILY_VARIABLES), strict=True)),
        }
    )


def fetch_hydrology(session: requests.Session, config: Config, raw_dir: Path) -> pd.DataFrame:
    # Ключ передаём только в запросах к USGS.
    key = os.environ.get("USGS_API_KEY")
    headers = {"X-Api-Key": key} if key else {}
    catalogue_url = f"{HYDRO_BASE}/monitoring-locations/items"
    params = {"f": "json", "monitoring_location_number": config.station.code, "limit": 2}
    response = session.get(catalogue_url, params=params, headers=headers, timeout=TIMEOUT)
    response.raise_for_status()
    snapshot(
        raw_dir / "usgs/stations",
        response.content,
        "geojson",
        {"url": catalogue_url, "query": params},
    )
    stations = response.json()["features"]
    location_id = f"USGS-{config.station.code}"
    if len(stations) != 1 or stations[0]["properties"]["id"] != location_id:
        raise ValueError(f"Каталог USGS не вернул однозначную запись для {location_id}.")
    station = stations[0]
    longitude, latitude = station["geometry"]["coordinates"]
    if (
        abs(latitude - config.station.latitude) > 0.01
        or abs(longitude - config.station.longitude) > 0.01
    ):
        raise ValueError(f"Координаты {location_id} в конфиге расходятся с каталогом USGS.")
    tables = []
    for parameter_code, (column, _, _) in USGS_PARAMETERS.items():
        chunks = []
        start = config.start
        while start <= config.end:
            # Берём до 9 000 дней, чтобы один ряд уместился в лимит API.
            end = min(start + timedelta(days=8_999), config.end)
            params = {
                "f": "csv",
                "monitoring_location_id": location_id,
                "parameter_code": parameter_code,
                "statistic_id": USGS_STATISTIC,
                "time": f"{start.isoformat()}/{end.isoformat()}",
                "limit": USGS_PAGE_LIMIT,
                "sortby": "time",
            }
            url = f"{HYDRO_BASE}/daily/items"
            response = session.get(url, params=params, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            snapshot(raw_dir / "usgs/daily", response.content, "csv", {"url": url, "query": params})
            chunk = parse_hydrology_csv(response.content, config.station.code, parameter_code)
            if len(chunk) >= USGS_PAGE_LIMIT:
                raise ValueError("Ответ USGS упёрся в лимит строк. Уменьшите диапазон дат.")
            if not chunk["date"].between(start.isoformat(), end.isoformat()).all():
                raise ValueError(f"USGS вернул даты вне диапазона {start} — {end}.")
            chunks.append(chunk)
            start = end + timedelta(days=1)
        table = pd.concat(chunks, ignore_index=True)
        if table.empty or not table[column].notna().any():
            raise ValueError(f"USGS не вернул измерений по параметру {parameter_code}.")
        if table["date"].duplicated().any() or table[f"{column}_time_series_id"].nunique() != 1:
            raise ValueError(f"У параметра USGS {parameter_code} есть дубли дат или сменился ряд.")
        tables.append(table)
    result = tables[0].merge(tables[1], on="date", how="outer", validate="one_to_one")
    result = result.sort_values("date").reset_index(drop=True)
    result["monitoring_location_id"] = location_id
    result["hydrology_source"] = "USGS Water Data API v1"
    result.attrs["station_metadata"] = station["properties"]
    return result


def fetch_weather(session: requests.Session, config: Config, raw_dir: Path) -> pd.DataFrame:
    params = {
        "latitude": config.station.latitude,
        "longitude": config.station.longitude,
        "start_date": config.start.isoformat(),
        "end_date": config.end.isoformat(),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": config.station.timezone,
        "models": "era5",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }
    response = session.get(WEATHER_URL, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    snapshot(raw_dir / "weather", response.content, "json", {"url": WEATHER_URL, "query": params})
    return parse_weather(response.json())
