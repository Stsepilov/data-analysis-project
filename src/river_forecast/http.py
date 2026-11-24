"""Таймауты и повторы для запросов к источникам."""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TIMEOUT = (5, 30)


def make_session() -> requests.Session:
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers["User-Agent"] = "river-forecast/0.1 (educational research)"
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session
