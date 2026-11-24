"""Проверка TOML. Даты считаем по календарю гидропоста."""

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True)
class Station:
    code: str
    name: str
    river: str
    latitude: float
    longitude: float
    timezone: str


@dataclass(frozen=True)
class Config:
    station: Station
    start: date
    end: date
    holdout_start: date
    horizon_days: int = 1
    weather_delay_days: int = 5
    seed: int = 42

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError("В конфиге start не может быть позже end.")
        if not self.start < self.holdout_start <= self.end:
            raise ValueError("Холд-аут holdout_start должен быть после start и не позже end.")
        if (
            not isinstance(self.station.code, str)
            or not self.station.code.isascii()
            or not self.station.code.isdigit()
            or not 8 <= len(self.station.code) <= 15
        ):
            raise ValueError("Код поста station.code укажите строкой из 8–15 цифр.")
        if not (-90 <= self.station.latitude <= 90 and -180 <= self.station.longitude <= 180):
            raise ValueError("Координаты поста вне диапазона: широта ±90°, долгота ±180°.")
        try:
            ZoneInfo(self.station.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Неизвестный часовой пояс: {self.station.timezone}.") from exc
        if self.horizon_days != 1:
            raise ValueError("Пока поддерживается прогноз на сутки. Укажите horizon_days = 1.")
        if self.weather_delay_days < 5:
            raise ValueError("Архив ERA5 запаздывает. Укажите weather_delay_days не меньше 5.")


def load_config(path: Path) -> Config:
    with path.open("rb") as handle:
        values = tomllib.load(handle)
    dataset = values["dataset"]
    dates = {}
    for key in ("start", "end", "holdout_start"):
        try:
            dates[key] = date.fromisoformat(str(dataset[key]))
        except ValueError as exc:
            raise ValueError(f"Укажите {key} в формате ГГГГ-ММ-ДД.") from exc
    return Config(
        station=Station(**values["station"]),
        horizon_days=dataset.get("horizon_days", 1),
        weather_delay_days=dataset.get("weather_delay_days", 5),
        seed=dataset.get("seed", 42),
        **dates,
    )
