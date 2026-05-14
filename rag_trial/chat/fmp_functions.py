import os
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests


BASE_URL = "https://financialmodelingprep.com/stable"

_US_EASTERN = ZoneInfo("America/New_York")
_REGULAR_OPEN = time(9, 30)
_REGULAR_CLOSE = time(16, 0)


def _previous_weekday(before: date) -> date:
    d = before - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _us_equity_session_window_et() -> tuple[datetime, datetime, date]:
    """Regular-session window in US/Eastern: prior weekday before Mon–Fri 9:30, else that day 9:30–now or 9:30–16:00."""
    now_et = datetime.now(_US_EASTERN)
    today_d = now_et.date()
    clock = now_et.time()

    if now_et.weekday() >= 5:
        session_date = _previous_weekday(today_d)
        start = datetime.combine(session_date, _REGULAR_OPEN, tzinfo=_US_EASTERN)
        end = datetime.combine(session_date, _REGULAR_CLOSE, tzinfo=_US_EASTERN)
        return start, end, session_date

    if clock < _REGULAR_OPEN:
        session_date = _previous_weekday(today_d)
        start = datetime.combine(session_date, _REGULAR_OPEN, tzinfo=_US_EASTERN)
        end = datetime.combine(session_date, _REGULAR_CLOSE, tzinfo=_US_EASTERN)
        return start, end, session_date

    if clock <= _REGULAR_CLOSE:
        session_date = today_d
        start = datetime.combine(session_date, _REGULAR_OPEN, tzinfo=_US_EASTERN)
        end = min(now_et, datetime.combine(session_date, _REGULAR_CLOSE, tzinfo=_US_EASTERN))
        return start, end, session_date

    session_date = today_d
    start = datetime.combine(session_date, _REGULAR_OPEN, tzinfo=_US_EASTERN)
    end = datetime.combine(session_date, _REGULAR_CLOSE, tzinfo=_US_EASTERN)
    return start, end, session_date


def _parse_fmp_chart_datetime(s: str) -> Optional[datetime]:
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _filter_intraday_by_et_window(data: Any, start_et: datetime, end_et: datetime) -> Any:
    if not isinstance(data, list):
        return data
    out: List[Any] = []
    for row in data:
        if not isinstance(row, dict):
            out.append(row)
            continue
        raw = row.get("date")
        if raw is None:
            out.append(row)
            continue
        bar_dt = _parse_fmp_chart_datetime(str(raw))
        if bar_dt is None:
            out.append(row)
            continue
        if bar_dt.tzinfo is None:
            bar_dt = bar_dt.replace(tzinfo=_US_EASTERN)
        else:
            bar_dt = bar_dt.astimezone(_US_EASTERN)
        if start_et <= bar_dt <= end_et:
            out.append(row)
    return out


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


def get_grades_consensus(symbol: str, *, timeout: int = 30) -> Any:
    return _fmp_get("grades-consensus", {"symbol": symbol}, timeout=timeout)


def get_price_target_consensus(symbol: str, *, timeout: int = 30) -> Any:
    return _fmp_get("price-target-consensus", {"symbol": symbol}, timeout=timeout)


def get_price_target_summary(symbol: str, *, timeout: int = 30) -> Any:
    return _fmp_get("price-target-summary", {"symbol": symbol}, timeout=timeout)


def get_ratings_historical(symbol: str, *, timeout: int = 30) -> Any:
    return _fmp_get("ratings-historical", {"symbol": symbol}, timeout=timeout)


def get_ratings_snapshot(symbol: str, *, timeout: int = 30) -> Any:
    return _fmp_get("ratings-snapshot", {"symbol": symbol}, timeout=timeout)


def get_shares_float(symbol: str, *, timeout: int = 30) -> Any:
    return _fmp_get("shares-float", {"symbol": symbol}, timeout=timeout)


def get_historical_chart(
    symbol: str,
    *,
    interval: str = "1min",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    timeout: int = 30,
) -> Any:
    params: Dict[str, Any] = {"symbol": symbol}
    use_session = date_from is None and date_to is None
    start_et: Optional[datetime] = None
    end_et: Optional[datetime] = None
    if use_session:
        start_et, end_et, session_date = _us_equity_session_window_et()
        day = session_date.isoformat()
        params["from"] = day
        params["to"] = day
    else:
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to
    raw = _fmp_get(f"historical-chart/{interval}", params, timeout=timeout)
    if use_session and start_et is not None and end_et is not None:
        return _filter_intraday_by_et_window(raw, start_et, end_et)
    return raw


def get_analyst_estimates(
    symbol: str,
    *,
    period: str = "annual",
    page: int = 0,
    limit: int = 10,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    timeout: int = 30,
) -> Any:
    params: Dict[str, Any] = {"symbol": symbol, "period": period, "page": page, "limit": limit}
    if date_from:
        params["from"] = date_from
    if date_to:
        params["to"] = date_to
    return _fmp_get(
        "analyst-estimates",
        params,
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
