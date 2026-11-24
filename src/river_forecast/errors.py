"""Короткие сообщения для терминала и журнала запусков."""

import json
import tomllib
from urllib.parse import urlsplit

import pandas as pd
import requests


def describe_error(exc: Exception) -> str:
    if isinstance(exc, requests.RequestException):
        request = exc.request
        host = urlsplit(request.url).hostname if request is not None and request.url else None
        source = {
            "api.waterdata.usgs.gov": "USGS",
            "archive-api.open-meteo.com": "Open-Meteo",
        }.get(host, host or "источник данных")
        if isinstance(exc, requests.exceptions.JSONDecodeError):
            return f"{source} вернул некорректный JSON."
        if isinstance(exc, requests.Timeout):
            return f"Таймаут запроса к {source}. Попробуйте ещё раз."
        if exc.response is not None:
            status = exc.response.status_code
            if status == 429:
                return f"{source} ограничил число запросов (HTTP 429). Повторите загрузку позже."
            return f"{source} вернул HTTP {status}."
        if isinstance(exc, requests.exceptions.RetryError):
            return f"Не удалось получить ответ от {source} после повторных запросов."
        return f"Сбой запроса к {source} ({type(exc).__name__})."
    if isinstance(exc, FileNotFoundError) and exc.filename is not None:
        return f"Файл не найден: {exc.filename}"
    if isinstance(exc, PermissionError) and exc.filename is not None:
        return f"Нет доступа к файлу: {exc.filename}"
    if isinstance(exc, tomllib.TOMLDecodeError):
        return "Не удалось прочитать TOML. Проверьте синтаксис конфига."
    if isinstance(exc, json.JSONDecodeError):
        return f"Некорректный JSON в строке {exc.lineno}, столбце {exc.colno}."
    if isinstance(exc, pd.errors.EmptyDataError):
        return "CSV пуст. Проверьте исходный файл или ответ API."
    if isinstance(exc, (pd.errors.ParserError, UnicodeError)):
        return "Не удалось прочитать данные. Проверьте формат и кодировку."
    if isinstance(exc, KeyError):
        return f"Не хватает поля {exc}."
    if isinstance(exc, ValueError):
        return str(exc)
    return f"Сбой выполнения ({type(exc).__name__}): {exc}"
