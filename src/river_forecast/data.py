"""Суточная таблица и проверки качества данных."""

import numpy as np
import pandas as pd

WEATHER_COLUMNS = ("temperature_mean_c", "precipitation_mm", "humidity_mean_pct")


def daily_index(frame: pd.DataFrame, required: tuple[str, ...]) -> pd.DataFrame:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"В таблице не хватает столбцов: {', '.join(sorted(missing))}")
    result = frame.loc[:, list(required)].copy()
    try:
        dates = pd.to_datetime(result.pop("date"), errors="raise")
    except ValueError as exc:
        raise ValueError("Не удалось прочитать date. Проверьте формат и значения дат.") from exc
    if dates.isna().any() or dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("В date нужны местные даты без пропусков, времени и часового пояса.")
    result.index = pd.DatetimeIndex(dates, name="date")
    if result.index.duplicated().any():
        raise ValueError("Есть дубли дат. Оставьте одно наблюдение на сутки.")
    if result.empty:
        raise ValueError("Источник вернул пустую таблицу.")
    for column in result.columns:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        except ValueError as exc:
            raise ValueError(f"В столбце {column} есть нечисловое значение.") from exc
        if np.isinf(result[column].to_numpy()).any():
            raise ValueError(f"В столбце {column} встретилось inf.")
    return result.sort_index()


def merge_daily(
    hydrology: pd.DataFrame,
    weather: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    hydro_columns = ("date", "level_cm")
    if "discharge_m3s" in hydrology:
        hydro_columns += ("discharge_m3s",)
    hydro = daily_index(hydrology, hydro_columns)
    meteo = daily_index(weather, ("date", *WEATHER_COLUMNS))
    calendar = pd.date_range(start, end, freq="D", name="date")
    result = hydro.join(meteo, how="outer").reindex(calendar)
    quality_report(result)
    if not result["level_cm"].notna().any():
        raise ValueError(f"С {start} по {end} замеров уровня воды нет.")
    return result


def quality_report(frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise ValueError("Суточный датасет пуст.")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise ValueError("Нужен DatetimeIndex без пропусков.")
    if frame.index.duplicated().any() or not frame.index.is_monotonic_increasing:
        raise ValueError("Отсортируйте даты по возрастанию и уберите дубли.")
    if not frame.index.equals(pd.date_range(frame.index.min(), frame.index.max(), name="date")):
        raise ValueError("В календаре пропущены дни. Восстановите сетку перед расчётом лагов.")
    for column in ("level_cm", *WEATHER_COLUMNS):
        if column not in frame:
            raise ValueError(f"В таблице нет столбца {column}.")
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise ValueError(f"В столбце {column} встретилось inf.")
    if (frame["precipitation_mm"] < 0).any():
        raise ValueError("Сумма осадков ниже нуля. Проверьте данные погоды.")
    if "discharge_m3s" in frame:
        if np.isinf(frame["discharge_m3s"].to_numpy(dtype=float)).any():
            raise ValueError("В расходе воды встретилось inf.")
        if (frame["discharge_m3s"] < 0).any():
            raise ValueError("В таблице есть отрицательный расход воды.")
    humidity = frame["humidity_mean_pct"].dropna()
    if not humidity.between(0, 100).all():
        raise ValueError("Влажность вышла за диапазон 0–100%.")
    # Уровень может быть ниже нуля поста. Скачки считаем отдельно.
    return {
        "rows": len(frame),
        "start": frame.index.min().date().isoformat(),
        "end": frame.index.max().date().isoformat(),
        "missing": {column: int(count) for column, count in frame.isna().sum().items()},
        "level_jumps_over_100_cm": int((frame["level_cm"].diff().abs() > 100).sum()),
        "level_min_cm": _optional_float(frame["level_cm"].min()),
        "level_max_cm": _optional_float(frame["level_cm"].max()),
    }


def _optional_float(value: float) -> float | None:
    return None if pd.isna(value) else float(value)
