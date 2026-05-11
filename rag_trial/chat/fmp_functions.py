import os
from typing import Any, Dict, Optional

import requests


BASE_URL = "https://financialmodelingprep.com/stable"


def _fmp_get(
    path: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    timeout: int = 30,
) -> Any:
    request_params: Dict[str, Any] = dict(params or {})
    request_params["apikey"] = os.environ["FMP_API_KEY"]
    response = requests.get(f"{BASE_URL}/{path}", params=request_params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_grades_historical(symbol: str, *, timeout: int = 30) -> Any:
    return _fmp_get("grades-historical", {"symbol": symbol}, timeout=timeout)


def get_grades(symbol: str, *, timeout: int = 30) -> Any:
    return _fmp_get("grades", {"symbol": symbol}, timeout=timeout)


def get_price_target_consensus(symbol: str, *, timeout: int = 30) -> Any:
    return _fmp_get("price-target-consensus", {"symbol": symbol}, timeout=timeout)


def get_price_target_summary(symbol: str, *, timeout: int = 30) -> Any:
    return _fmp_get("price-target-summary", {"symbol": symbol}, timeout=timeout)


def get_ratings_historical(symbol: str, *, timeout: int = 30) -> Any:
    return _fmp_get("ratings-historical", {"symbol": symbol}, timeout=timeout)


def get_ratings_snapshot(symbol: str, *, timeout: int = 30) -> Any:
    return _fmp_get("ratings-snapshot", {"symbol": symbol}, timeout=timeout)


def get_analyst_estimates(
    symbol: str,
    *,
    period: str = "annual",
    page: int = 0,
    limit: int = 10,
    timeout: int = 30,
) -> Any:
    return _fmp_get(
        "analyst-estimates",
        {"symbol": symbol, "period": period, "page": page, "limit": limit},
        timeout=timeout,
    )


def get_fmp_articles(*, page: int = 0, limit: int = 20, timeout: int = 30) -> Any:
    return _fmp_get("fmp-articles", {"page": page, "limit": limit}, timeout=timeout)


def get_news_general_latest(*, page: int = 0, limit: int = 20, timeout: int = 30) -> Any:
    return _fmp_get("news/general-latest", {"page": page, "limit": limit}, timeout=timeout)


def get_news_stock_latest(*, page: int = 0, limit: int = 20, timeout: int = 30) -> Any:
    return _fmp_get("news/stock-latest", {"page": page, "limit": limit}, timeout=timeout)


def get_news_forex_latest(*, page: int = 0, limit: int = 20, timeout: int = 30) -> Any:
    return _fmp_get("news/forex-latest", {"page": page, "limit": limit}, timeout=timeout)


def get_news_stock(symbols: str, *, timeout: int = 30) -> Any:
    return _fmp_get("news/stock", {"symbols": symbols}, timeout=timeout)


def get_news_forex(symbols: str, *, timeout: int = 30) -> Any:
    return _fmp_get("news/forex", {"symbols": symbols}, timeout=timeout)
