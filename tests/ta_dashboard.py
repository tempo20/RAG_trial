# Run with:
# streamlit run ta_dashboard.py

from __future__ import annotations

from collections import Counter
from dataclasses import MISSING, fields as dataclass_fields
from datetime import date, datetime, timedelta, timezone
import importlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (_PROJECT_ROOT, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rag_trial.db import ta_cache
from ta_pipe import PipelineConfig, SignalResult

if not all(
    hasattr(ta_cache, name)
    for name in (
        "load_fresh_fundamental_analysis",
        "load_articles_for_ticker",
        "load_articles_since",
        "load_stock_pick_dates",
        "load_stock_pick_snapshot",
        "load_bullish_candidate_snapshot",
        "load_latest_signal_snapshot",
        "load_company_profiles",
    )
):
    ta_cache = importlib.reload(ta_cache)


FUNDAMENTAL_TOP_N = 15
STOCK_PICK_HISTORY_START_DATE = "2017-12-31"
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
_BULLISH_SNAPSHOT_CACHE_HIT_ATTR = "_bullish_candidate_snapshot_cache_hit"
_SIGNAL_SERIES_FIELDS = {
    "spread",
    "close",
    "sma_50",
    "sma_200",
    "vwma_50",
    "regular_bullish_trend",
    "strong_bullish_confirmation",
    "regular_bearish_trend",
    "strong_bearish_confirmation",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi",
    "donchian_20_high",
    "donchian_55_high",
    "ret_z_20",
    "atr_14",
    "atr_move",
    "relative_volume",
    "ema_20",
    "extension_atr",
    "stoch_rsi",
    "bb_position",
    "distance_from_sma50",
    "ema_8",
    "ema_21",
    "ema_50",
}


def require_turso_dashboard(db_path: str | Path | None = None) -> None:
    if db_path is not None:
        return
    ta_cache.require_turso_configured()
    os.environ["TA_CACHE_READ_ONLY"] = "1"
    os.environ["TA_CACHE_FORCE_TURSO"] = "1"


st.set_page_config(
    page_title="TA Candidate Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none;
    }

    .st-key-candidate_table_sticky {
        position: fixed;
        top: 3.25rem;
        left: 0;
        right: 0;
        width: auto;
        max-width: none;
        z-index: 1000;
        background: var(--background-color);
        padding: 0.6rem 0 0.6rem;
        border-bottom: 1px solid var(--border-color);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.18);
    }

    .st-key-candidate_table_sticky [data-testid="stHorizontalBlock"],
    .st-key-candidate_table_sticky [data-testid="column"],
    .st-key-candidate_table_sticky .stMarkdown {
        background: var(--background-color);
        color: var(--text-color);
        margin-bottom: 0;
    }

    .st-key-candidate_table_sticky .stMarkdown p,
    .st-key-candidate_table_sticky .stMarkdown strong {
        color: var(--text-color);
    }

    .st-key-candidate_table_sticky [data-testid="stButton"] button {
        background: var(--secondary-background-color) !important;
        border: 2px solid var(--border-color) !important;
        color: var(--text-color) !important;
        font-size: 1.35rem;
        font-weight: 700;
        min-height: 2.75rem;
        padding: 0;
    }

    @media (max-width: 900px) {
        .st-key-candidate_table_sticky {
            left: 0;
            right: 0;
            width: auto;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _format_signal_date(value) -> object:
    if value is None:
        return None
    if hasattr(value, "date"):
        return value.date()
    return value


def _latest_valid_value(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    valid = series.replace([float("inf"), -float("inf")], pd.NA).dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def result_summary_df(results: list[SignalResult], top_n: int) -> pd.DataFrame:
    rows = []
    for rank, r in enumerate(results[:top_n], start=1):
        latest_spread = _latest_valid_value(r.spread)
        rows.append({
            "rank": rank,
            "ticker": r.ticker,
            "regime_label": r.regime_label,
            "final_bullish_score": round(r.final_bullish_score, 2) if r.final_bullish_score is not None else None,
            "bullish_impulse_score": round(r.bullish_impulse_score, 2) if r.bullish_impulse_score is not None else None,
            "pre_golden_score": round(r.pre_golden_score, 2) if r.pre_golden_score is not None else None,
            "relative_strength_score": round(r.relative_strength_score, 2) if r.relative_strength_score is not None else None,
            "overbought_status": r.overbought_status,
            "overbought_score": round(r.overbought_score, 2) if r.overbought_score is not None else None,
            "spread_%": round(latest_spread * 100, 2) if latest_spread is not None else None,
            "est_days_to_cross": round(r.days_to_cross_estimate, 1) if r.days_to_cross_estimate is not None else None,
            "latest_signal": r.latest_signal,
            "signal_date": _format_signal_date(r.latest_signal_date),
            "total_crosses": len(r.cross_dates),
            "golden_crosses": r.cross_colors.count("green"),
            "death_crosses": r.cross_colors.count("red"),
        })
    return pd.DataFrame(rows)


def selected_result_details_df(r: SignalResult) -> pd.DataFrame:
    latest_spread = _latest_valid_value(r.spread)
    latest_extension_atr = _latest_valid_value(r.extension_atr)
    latest_distance_from_sma50 = _latest_valid_value(r.distance_from_sma50)
    latest_stoch_rsi = _latest_valid_value(r.stoch_rsi)
    latest_bb_position = _latest_valid_value(r.bb_position)
    latest_relative_volume = _latest_valid_value(r.relative_volume)
    values = [
        ("Ticker", r.ticker),
        ("Regime", r.regime_label or "None"),
        (
            "Final bullish score",
            f"{r.final_bullish_score:.2f}" if r.final_bullish_score is not None else "None",
        ),
        (
            "Bullish impulse score",
            f"{r.bullish_impulse_score:.2f}" if r.bullish_impulse_score is not None else "None",
        ),
        (
            "Pre-golden score",
            f"{r.pre_golden_score:.2f}" if r.pre_golden_score is not None else "None",
        ),
        (
            "Relative strength score",
            f"{r.relative_strength_score:.2f}" if r.relative_strength_score is not None else "None",
        ),
        (
            "Liquidity/volume score",
            f"{r.liquidity_volume_score:.2f}" if r.liquidity_volume_score is not None else "None",
        ),
        ("Overbought status", r.overbought_status or "None"),
        (
            "Overbought score",
            f"{r.overbought_score:.2f}" if r.overbought_score is not None else "None",
        ),
        (
            "Extension vs EMA20 (ATR)",
            f"{latest_extension_atr:.2f}" if latest_extension_atr is not None else "None",
        ),
        (
            "Distance from SMA50",
            f"{latest_distance_from_sma50 * 100:.2f}%"
            if latest_distance_from_sma50 is not None
            else "None",
        ),
        (
            "StochRSI",
            f"{latest_stoch_rsi:.2f}" if latest_stoch_rsi is not None else "None",
        ),
        (
            "Bollinger position",
            f"{latest_bb_position:.2f}" if latest_bb_position is not None else "None",
        ),
        (
            "Relative volume",
            f"{latest_relative_volume:.2f}" if latest_relative_volume is not None else "None",
        ),
        (
            "Latest spread %",
            f"{latest_spread * 100:.2f}" if latest_spread is not None else "None",
        ),
        (
            "Estimated days to cross",
            f"{r.days_to_cross_estimate:.1f}"
            if r.days_to_cross_estimate is not None
            else "None",
        ),
        ("Latest signal", r.latest_signal or "None"),
        ("Signal date", str(_format_signal_date(r.latest_signal_date))),
        ("Total crosses", str(len(r.cross_dates))),
        ("Golden crosses", str(r.cross_colors.count("green"))),
        ("Death crosses", str(r.cross_colors.count("red"))),
    ]
    return pd.DataFrame(values, columns=["metric", "value"])


def selected_result_reasons_df(r: SignalResult) -> pd.DataFrame:
    rows = (
        [{"category": "bullish_impulse", "reason": reason} for reason in r.bullish_impulse_reasons]
        + [{"category": "pre_golden", "reason": reason} for reason in r.pre_golden_reasons]
        + [{"category": "relative_strength", "reason": reason} for reason in r.relative_strength_reasons]
        + [{"category": "overbought", "reason": reason} for reason in r.overbought_reasons]
    )
    if not rows:
        return pd.DataFrame([{"reason": "None"}])
    return pd.DataFrame(rows)


def ticker_news_df(
    ticker: str,
    *,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    rows = ta_cache.load_articles_for_ticker(ticker, db_path=db_path)
    if not rows:
        return pd.DataFrame(columns=["published", "source", "title", "url"])

    return pd.DataFrame(
        {
            "published": row.get("publishedDate") or "",
            "source": row.get("site") or "",
            "title": row.get("title") or "",
            "url": row.get("url") or "",
        }
        for row in rows
    )


def ticker_counts_df(
    cfg: PipelineConfig,
    *,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    cutoff_date = (
        datetime.now(timezone.utc) - timedelta(days=cfg.news_days)
    ).date().isoformat()
    week_articles = ta_cache.load_articles_since(cutoff_date, db_path=db_path)
    ticker_counter = Counter(
        str(article.get("symbol") or "").strip().upper()
        for article in week_articles
        if article.get("symbol")
    )

    rows = [
        {
            "ticker": ticker,
            "article_count": count,
        }
        for ticker, count in ticker_counter.most_common()
    ]
    return pd.DataFrame(rows, columns=["ticker", "article_count"])


def _stock_pick_snapshot_date(today: date | None = None) -> str:
    return (today or datetime.now(timezone.utc).date()).isoformat()


def _stock_pick_price_start_date(pick_date: str) -> str:
    try:
        parsed = date.fromisoformat(str(pick_date))
    except ValueError:
        return str(pick_date)
    return (parsed - timedelta(days=10)).isoformat()


def save_stock_pick_snapshot(
    ranked: list[SignalResult],
    cfg: PipelineConfig,
    *,
    today: date | None = None,
    db_path: str | Path | None = None,
) -> bool:
    tickers: list[str] = []
    seen: set[str] = set()
    for result in ranked[: cfg.top_n]:
        ticker = str(result.ticker or "").strip().upper()
        if not TICKER_PATTERN.fullmatch(ticker) or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)

    if not tickers:
        return False

    ta_cache.upsert_stock_pick_snapshot(
        _stock_pick_snapshot_date(today),
        tickers,
        db_path=db_path,
    )
    return True


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return _json_safe_value(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _serialize_series(series: pd.Series | None) -> list[list[Any]]:
    if series is None or series.empty:
        return []
    rows: list[list[Any]] = []
    for index, value in series.items():
        timestamp = index.to_timestamp() if hasattr(index, "to_timestamp") else pd.Timestamp(index)
        rows.append([timestamp.isoformat(), _json_safe_value(value)])
    return rows


def _deserialize_series(rows: Any) -> pd.Series | None:
    if not isinstance(rows, list) or not rows:
        return None

    index: list[pd.Timestamp] = []
    values: list[Any] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 2:
            continue
        timestamp = pd.to_datetime(row[0], errors="coerce")
        if pd.isna(timestamp):
            continue
        index.append(timestamp)
        values.append(row[1])

    if not index:
        return None
    return pd.Series(values, index=pd.DatetimeIndex(index))


def _parse_cached_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.to_pydatetime()


def _field_default(field) -> Any:
    if field.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field.default_factory()  # type: ignore[misc]
    if field.default is not MISSING:
        return field.default
    return None


def _serialize_signal_result(result: SignalResult) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in dataclass_fields(SignalResult):
        value = getattr(result, field.name)
        if field.name in _SIGNAL_SERIES_FIELDS:
            payload[field.name] = _serialize_series(value)
        else:
            payload[field.name] = _json_safe_value(value)
    return payload


def _deserialize_signal_result(payload: dict[str, Any]) -> SignalResult:
    kwargs: dict[str, Any] = {}
    for field in dataclass_fields(SignalResult):
        value = payload.get(field.name, _field_default(field))
        if field.name in _SIGNAL_SERIES_FIELDS:
            value = _deserialize_series(value)
        elif field.name == "latest_signal_date":
            value = _parse_cached_datetime(value)
        elif field.name == "cross_dates":
            raw_values = value if isinstance(value, list) else []
            value = [
                parsed
                for parsed in (_parse_cached_datetime(item) for item in raw_values)
                if parsed is not None
            ]
        kwargs[field.name] = value

    kwargs["ticker"] = str(kwargs.get("ticker") or "").upper()
    for list_field in (
        "cross_dates",
        "cross_prices",
        "cross_colors",
        "forward_returns",
        "pre_golden_reasons",
        "bullish_impulse_reasons",
        "relative_strength_reasons",
        "overbought_reasons",
    ):
        if kwargs.get(list_field) is None:
            kwargs[list_field] = []
    return SignalResult(**kwargs)


def _pipeline_config_to_snapshot(cfg: PipelineConfig) -> dict[str, Any]:
    return {
        field.name: _json_safe_value(getattr(cfg, field.name))
        for field in dataclass_fields(PipelineConfig)
    }


def _pipeline_config_from_snapshot(payload: dict[str, Any]) -> PipelineConfig:
    valid_fields = {field.name for field in dataclass_fields(PipelineConfig)}
    kwargs = {
        key: value
        for key, value in payload.items()
        if key in valid_fields
    }
    if isinstance(kwargs.get("screener_noise"), list):
        kwargs["screener_noise"] = set(kwargs["screener_noise"])
    return PipelineConfig(**kwargs)


def save_bullish_candidate_snapshot(
    ranked: list[SignalResult],
    cfg: PipelineConfig,
    *,
    today: date | None = None,
    db_path: str | Path | None = None,
) -> bool:
    if not ranked:
        return False

    ta_cache.upsert_bullish_candidate_snapshot(
        _stock_pick_snapshot_date(today),
        [_serialize_signal_result(result) for result in ranked],
        _pipeline_config_to_snapshot(cfg),
        db_path=db_path,
    )
    return True


def load_bullish_candidate_snapshot_results(
    pick_date: str,
    *,
    db_path: str | Path | None = None,
) -> tuple[list[SignalResult], PipelineConfig] | None:
    snapshot = ta_cache.load_bullish_candidate_snapshot(pick_date, db_path=db_path)
    if snapshot is None:
        return None

    ranked_payload = snapshot.get("ranked")
    cfg_payload = snapshot.get("cfg")
    if not isinstance(ranked_payload, list) or not isinstance(cfg_payload, dict):
        return None

    ranked = [
        _deserialize_signal_result(row)
        for row in ranked_payload
        if isinstance(row, dict)
    ]
    if not ranked:
        return None

    cfg = _pipeline_config_from_snapshot(cfg_payload)
    setattr(cfg, _BULLISH_SNAPSHOT_CACHE_HIT_ATTR, True)
    return ranked, cfg


def _bullish_snapshot_cache_hit(cfg: PipelineConfig) -> bool:
    return bool(getattr(cfg, _BULLISH_SNAPSHOT_CACHE_HIT_ATTR, False))


def load_cached_fundamental_analyses(
    ranked: list[SignalResult],
    *,
    top_n: int = FUNDAMENTAL_TOP_N,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    for result in ranked[:top_n]:
        cached = ta_cache.load_fresh_fundamental_analysis(result.ticker, db_path=db_path)
        if cached is not None:
            analyses.append(_row_from_cached_analysis(cached))
    return analyses


def _price_from_bar(row: dict[str, Any]) -> float | None:
    value = row.get("adj_close")
    if value is None:
        value = row.get("close")
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _return_from_bar_rows(
    pick_date: str,
    ticker: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    clean = sorted(
        (
            {
                "bar_date": str(row.get("bar_date") or ""),
                "price": _price_from_bar(row),
            }
            for row in rows
        ),
        key=lambda row: row["bar_date"],
    )
    clean = [
        row
        for row in clean
        if row["bar_date"] and row["price"] is not None
    ]
    base_rows = [row for row in clean if row["bar_date"] >= pick_date]
    prior_rows = [row for row in clean if row["bar_date"] <= pick_date]

    base = base_rows[0] if base_rows else (prior_rows[-1] if prior_rows else None)
    latest = clean[-1] if clean else None
    change_pct = None
    if base is not None and latest is not None and base["price"] not in (None, 0):
        change_pct = round((latest["price"] / base["price"] - 1.0) * 100, 2)

    return {
        "ticker": ticker,
        "base_date": None if base is None else base["bar_date"],
        "base_close": None if base is None else round(float(base["price"]), 4),
        "latest_date": None if latest is None else latest["bar_date"],
        "latest_close": None if latest is None else round(float(latest["price"]), 4),
        "change_%": change_pct,
    }


def _history_for_stock_pick_ticker(
    downloaded: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame | None:
    if downloaded is None or downloaded.empty:
        return None

    if isinstance(downloaded.columns, pd.MultiIndex):
        level0 = downloaded.columns.get_level_values(0)
        level1 = downloaded.columns.get_level_values(1)
        if ticker in set(level0):
            return downloaded[ticker].dropna(how="all")
        if ticker in set(level1):
            return downloaded.xs(ticker, axis=1, level=1).dropna(how="all")
        return None

    return downloaded.dropna(how="all")


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _history_frame_to_stock_pick_bar_rows(
    ticker: str,
    hist: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    if hist is None or hist.empty:
        return []

    rows: list[dict[str, Any]] = []
    for index, row in hist.iterrows():
        timestamp = index.to_timestamp() if hasattr(index, "to_timestamp") else pd.Timestamp(index)
        rows.append({
            "ticker": ticker,
            "bar_date": timestamp.date().isoformat(),
            "open": _float_or_none(row.get("Open")),
            "high": _float_or_none(row.get("High")),
            "low": _float_or_none(row.get("Low")),
            "close": _float_or_none(row.get("Close")),
            "adj_close": _float_or_none(row.get("Adj Close")),
            "volume": _float_or_none(row.get("Volume")),
        })
    return rows


def _fetch_missing_stock_pick_bars(
    tickers: list[str],
    *,
    start_date: str,
    end_date: str,
    db_path: str | Path | None = None,
) -> str | None:
    return None


def stock_pick_returns_df(
    pick_date: str,
    tickers: list[str],
    *,
    today: date | None = None,
    db_path: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    date_key = str(pick_date or "").strip()
    if not date_key or not tickers:
        return pd.DataFrame(
            columns=[
                "rank",
                "pick_date",
                "ticker",
                "base_date",
                "base_close",
                "latest_date",
                "latest_close",
                "change_%",
            ]
        ), []

    end_date = (today or datetime.now(timezone.utc).date()).isoformat()
    price_start_date = _stock_pick_price_start_date(date_key)
    normalized = []
    seen: set[str] = set()
    for ticker in tickers:
        ticker_key = str(ticker or "").strip().upper()
        if not TICKER_PATTERN.fullmatch(ticker_key) or ticker_key in seen:
            continue
        seen.add(ticker_key)
        normalized.append(ticker_key)

    rows_by_ticker = {
        ticker: ta_cache.load_daily_bars(
            ticker,
            price_start_date,
            end_date,
            provider="financetoolkit",
            db_path=db_path,
        )
        for ticker in normalized
    }
    returns_by_ticker = {
        ticker: _return_from_bar_rows(date_key, ticker, rows)
        for ticker, rows in rows_by_ticker.items()
    }
    missing = [
        ticker
        for ticker, row in returns_by_ticker.items()
        if row["base_close"] is None or row["latest_close"] is None
    ]

    warnings: list[str] = []

    out_rows = []
    for rank, ticker in enumerate(normalized, start=1):
        row = returns_by_ticker.get(ticker) or _return_from_bar_rows(date_key, ticker, [])
        out_rows.append({
            "rank": rank,
            "pick_date": date_key,
            **row,
        })

    missing_after_fetch = [
        row["ticker"]
        for row in out_rows
        if row["base_close"] is None or row["latest_close"] is None
    ]
    if missing_after_fetch:
        warnings.append(
            "Missing price data for: " + ", ".join(missing_after_fetch)
        )

    return pd.DataFrame(out_rows), warnings


def normalize_search_ticker(value: str) -> str | None:
    ticker = str(value or "").strip().upper()
    return ticker if TICKER_PATTERN.fullmatch(ticker) else None


def _empty_profile_payload(ticker: str, error: str | None = None) -> dict[str, Any]:
    return {
        "ticker": ticker.upper(),
        "description": None,
        "sector": None,
        "industry": None,
        "error": error,
    }


def _profile_error_text(exc: Exception, api_key: str | None = None) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if api_key:
        text = text.replace(api_key, "<redacted>")
    return text


def _clean_profile_value(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "n/a"}:
        return None
    return text


def _profile_value(profile: pd.DataFrame, field: str, ticker: str) -> str | None:
    ticker_key = ticker.upper()
    field_keys = [field, field.lower(), field.upper()]

    for field_key in field_keys:
        if field_key in profile.index:
            row = profile.loc[field_key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if isinstance(row, pd.Series):
                if ticker_key in row.index:
                    return _clean_profile_value(row.loc[ticker_key])
                if len(row) == 1:
                    return _clean_profile_value(row.iloc[0])
            return _clean_profile_value(row)

        if field_key in profile.columns and ticker_key in profile.index:
            return _clean_profile_value(profile.loc[ticker_key, field_key])

    return None


def _profile_payload_from_frame(
    ticker: str,
    profile: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "ticker": ticker.upper(),
        "description": _profile_value(profile, "Description", ticker),
        "sector": _profile_value(profile, "Sector", ticker),
        "industry": _profile_value(profile, "Industry", ticker),
        "error": None,
    }


def _normalized_profile_tickers(tickers: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for ticker in tickers:
        ticker_key = normalize_search_ticker(ticker)
        if ticker_key is not None and ticker_key not in seen:
            seen.add(ticker_key)
            normalized.append(ticker_key)
    return tuple(normalized)


@st.cache_data(show_spinner=True, ttl=86400)
def load_company_profiles(
    tickers: tuple[str, ...],
    db_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    require_turso_dashboard(db_path)
    normalized = _normalized_profile_tickers(tickers)
    if not normalized:
        return {}
    cached = ta_cache.load_company_profiles(normalized, db_path=db_path)
    return {
        ticker: cached.get(ticker, _empty_profile_payload(ticker))
        for ticker in normalized
    }


def add_company_profile_columns(
    summary: pd.DataFrame,
    profiles: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    if summary.empty:
        return summary

    with_profiles = summary.copy()
    with_profiles.insert(
        2,
        "sector",
        [
            profiles.get(str(ticker).upper(), {}).get("sector")
            for ticker in with_profiles["ticker"]
        ],
    )
    with_profiles.insert(
        3,
        "industry",
        [
            profiles.get(str(ticker).upper(), {}).get("industry")
            for ticker in with_profiles["ticker"]
        ],
    )
    return with_profiles


def render_company_profile(
    ticker: str,
    profile: dict[str, Any] | None,
) -> None:
    st.subheader(f"{ticker} company profile")
    if profile is None:
        st.info("No company profile is available for the selected ticker.")
        return

    if profile.get("error"):
        st.info(f"Company profile unavailable: {profile['error']}")
        return

    cols = st.columns(2)
    cols[0].metric("Sector", profile.get("sector") or "N/A")
    cols[1].metric("Industry", profile.get("industry") or "N/A")
    st.markdown(profile.get("description") or "No description available.")


def _fundamental_query(ticker: str) -> str:
    return f"Is {ticker.upper()} fundamentally sound based on its financial statements?"


def _extract_fundamental_fields(answer: str) -> tuple[str | None, int | None]:
    assessment_match = re.search(
        r"(?im)^\s*Fundamental Assessment:\s*(Sound|Mixed|Unsound)\s*$",
        answer or "",
    )
    score_match = re.search(
        r"(?im)^\s*Fundamental Score \(0-10\):\s*(10|[0-9])\s*$",
        answer or "",
    )
    assessment = assessment_match.group(1) if assessment_match else None
    score = int(score_match.group(1)) if score_match else None
    return assessment, score


def _row_from_cached_analysis(row: dict[str, Any]) -> dict[str, Any]:
    trace_json = row.get("retrieval_trace_json") or "{}"
    logs_json = row.get("logs_json") or "[]"
    try:
        retrieval_trace = json.loads(trace_json)
    except json.JSONDecodeError:
        retrieval_trace = {}
    try:
        logs = json.loads(logs_json)
    except json.JSONDecodeError:
        logs = []

    return {
        "ticker": row["ticker"],
        "company_name": row.get("company_name"),
        "query": row.get("query"),
        "answer": row.get("answer") or "",
        "decision": row.get("decision"),
        "route_type": row.get("route_type"),
        "fundamental_assessment": row.get("fundamental_assessment"),
        "fundamental_score": row.get("fundamental_score"),
        "finance_context_present": bool(row.get("finance_context_present")),
        "news_context_present": bool(row.get("news_context_present")),
        "news_query": row.get("news_query"),
        "news_item_count": int(row.get("news_item_count") or 0),
        "retrieval_trace": retrieval_trace,
        "logs": logs,
        "generated_at_utc": row.get("generated_at_utc"),
        "updated_at_utc": row.get("updated_at_utc"),
        "cache_status": "cache",
        "stored": True,
    }


def load_or_generate_fundamental_analyses(
    ranked: list[SignalResult],
    *,
    top_n: int = FUNDAMENTAL_TOP_N,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    return load_cached_fundamental_analyses(ranked, top_n=top_n, db_path=db_path)


def _add_series_trace(
    fig: go.Figure,
    series: pd.Series | None,
    *,
    name: str,
    row: int,
    col: int,
    color: str,
    dash: str | None = None,
) -> None:
    if series is None or series.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series,
            mode="lines",
            name=name,
            line={"color": color, **({"dash": dash} if dash else {})},
        ),
        row=row,
        col=col,
    )


def make_signal_chart(r: SignalResult, cfg: PipelineConfig) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.50, 0.18, 0.17, 0.15],
        subplot_titles=("Price", "SMA spread %", "MACD", "RSI"),
    )

    _add_series_trace(fig, r.close, name="Close", row=1, col=1, color="#4da3ff")
    _add_series_trace(
        fig,
        r.sma_50,
        name=f"SMA {cfg.short_sma_period}",
        row=1,
        col=1,
        color="#f59e0b",
        dash="dash",
    )
    _add_series_trace(
        fig,
        r.sma_200,
        name=f"SMA {cfg.long_sma_period}",
        row=1,
        col=1,
        color="#facc15",
    )
    _add_series_trace(
        fig,
        r.vwma_50,
        name=f"VWMA {cfg.vwma_period}",
        row=1,
        col=1,
        color="#fb7185",
        dash="dot",
    )
    _add_series_trace(
        fig,
        r.ema_8,
        name=f"EMA {cfg.impulse_ema_fast}",
        row=1,
        col=1,
        color="#93c5fd",
    )
    _add_series_trace(
        fig,
        r.ema_21,
        name=f"EMA {cfg.impulse_ema_mid}",
        row=1,
        col=1,
        color="#a7f3d0",
        dash="dash",
    )
    _add_series_trace(
        fig,
        r.ema_50,
        name=f"EMA {cfg.impulse_ema_slow}",
        row=1,
        col=1,
        color="#c4b5fd",
        dash="dot",
    )

    golden = []
    death = []
    for date, price, color in zip(r.cross_dates, r.cross_prices, r.cross_colors):
        if color == "green":
            golden.append((date, price))
        elif color == "red":
            death.append((date, price))

    if golden:
        dates, prices = zip(*golden)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                mode="markers",
                name=f"Golden crosses ({len(golden)})",
                marker={
                    "color": "#22c55e",
                    "size": 11,
                    "symbol": "triangle-up",
                    "line": {"color": "white", "width": 1},
                },
            ),
            row=1,
            col=1,
        )

    if death:
        dates, prices = zip(*death)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                mode="markers",
                name=f"Death crosses ({len(death)})",
                marker={
                    "color": "#ef4444",
                    "size": 11,
                    "symbol": "triangle-down",
                    "line": {"color": "white", "width": 1},
                },
            ),
            row=1,
            col=1,
        )

    if r.regime_label == "bullish_impulse" and r.close is not None and not r.close.dropna().empty:
        latest_close = r.close.dropna().iloc[-1]
        latest_date = r.close.dropna().index[-1]
        fig.add_trace(
            go.Scatter(
                x=[latest_date],
                y=[latest_close],
                mode="markers",
                name="Latest bullish impulse",
                marker={
                    "color": "#38bdf8",
                    "size": 15,
                    "symbol": "star",
                    "line": {"color": "white", "width": 1},
                },
            ),
            row=1,
            col=1,
        )

    if r.spread is not None and not r.spread.empty:
        spread_pct = r.spread * 100
        fig.add_trace(
            go.Scatter(
                x=spread_pct.index,
                y=spread_pct,
                mode="lines",
                name="Spread %",
                line={"color": "#22d3ee"},
            ),
            row=2,
            col=1,
        )

    fig.add_hline(y=0, line_dash="dash", row=2, col=1)
    fig.add_hline(y=cfg.min_spread * 100, line_dash="dot", row=2, col=1)
    fig.add_hline(y=-cfg.min_spread * 100, line_dash="dot", row=2, col=1)

    if r.macd_hist is not None and not r.macd_hist.empty:
        hist = r.macd_hist.dropna()
        if not hist.empty:
            fig.add_trace(
                go.Bar(
                    x=hist.index,
                    y=hist,
                    name="MACD hist",
                    marker_color=[
                        "#22c55e" if value >= 0 else "#ef4444"
                        for value in hist
                    ],
                    opacity=0.55,
                ),
                row=3,
                col=1,
            )
    _add_series_trace(fig, r.macd, name="MACD", row=3, col=1, color="#60a5fa")
    _add_series_trace(fig, r.macd_signal, name="Signal", row=3, col=1, color="#f97316")
    fig.add_hline(y=0, line_dash="dash", row=3, col=1)

    _add_series_trace(fig, r.rsi, name="RSI", row=4, col=1, color="#a78bfa")
    fig.add_hline(y=45, line_dash="dot", row=4, col=1)
    fig.add_hline(y=cfg.overbought_rsi_level, line_dash="dot", row=4, col=1)
    fig.add_hline(y=cfg.severe_overbought_rsi_level, line_dash="dash", row=4, col=1)

    final_score_text = f"{r.final_bullish_score:.2f}" if r.final_bullish_score is not None else "N/A"
    overbought_score_text = f"{r.overbought_score:.2f}" if r.overbought_score is not None else "N/A"
    fig.update_layout(
        title=(
            f"{r.ticker} | Regime={r.regime_label or 'N/A'} | "
            f"Final Score={final_score_text} | "
            f"Overbought={r.overbought_status or 'N/A'} ({overbought_score_text})"
        ),
        height=980,
        hovermode="x unified",
        template="plotly_dark",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
        },
        margin={"l": 40, "r": 170, "t": 80, "b": 40},
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Spread %", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    return fig


def dashboard_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        top_n=FUNDAMENTAL_TOP_N,
        plot=False,
        save_plots=False,
        start_date="2022-01-01",
        news_days=7,
        forward_days=20,
    )


@st.cache_data(show_spinner=True, ttl=3600)
def load_results(db_path: str | Path | None = None) -> tuple[list[SignalResult], PipelineConfig]:
    require_turso_dashboard(db_path)
    cfg = dashboard_pipeline_config()
    today = _stock_pick_snapshot_date()
    cached = load_bullish_candidate_snapshot_results(today, db_path=db_path)
    if cached is not None:
        return cached

    setattr(cfg, _BULLISH_SNAPSHOT_CACHE_HIT_ATTR, False)
    return [], cfg


@st.cache_data(show_spinner=True, ttl=3600)
def load_search_result(
    ticker: str,
    db_path: str | Path | None = None,
) -> tuple[SignalResult, PipelineConfig, dict[str, Any]] | None:
    require_turso_dashboard(db_path)
    snapshot = ta_cache.load_latest_signal_snapshot(ticker, db_path=db_path)
    if snapshot is None:
        return None
    result_payload = snapshot.get("result")
    cfg_payload = snapshot.get("cfg")
    if not isinstance(result_payload, dict) or not isinstance(cfg_payload, dict):
        return None
    return (
        _deserialize_signal_result(result_payload),
        _pipeline_config_from_snapshot(cfg_payload),
        snapshot,
    )


def analysis_for_ticker(
    analyses: list[dict[str, Any]],
    ticker: str,
) -> dict[str, Any] | None:
    ticker_key = ticker.upper()
    return next(
        (
            analysis
            for analysis in analyses
            if analysis["ticker"] == ticker_key
        ),
        None,
    )


def render_ticker_view(
    result: SignalResult,
    cfg: PipelineConfig,
    analysis: dict[str, Any] | None,
    profile: dict[str, Any] | None,
) -> None:
    if result.error:
        st.warning(f"{result.ticker}: {result.error}")
    elif result.close is None or result.close.dropna().empty:
        st.warning(f"{result.ticker}: no price data")
    else:
        fig = make_signal_chart(result, cfg)
        st.plotly_chart(fig, width="stretch")

    st.subheader(f"{result.ticker} details")
    st.dataframe(
        selected_result_details_df(result),
        width="stretch",
        hide_index=True,
    )
    st.dataframe(
        selected_result_reasons_df(result),
        width="stretch",
        hide_index=True,
    )

    render_company_profile(result.ticker, profile)

    st.subheader(f"{result.ticker} fundamental analysis")
    if analysis is None:
        st.info("No fundamental analysis is available for the selected ticker.")
    else:
        source = "cache" if analysis["cache_status"] == "cache" else "new"
        stored_note = "" if analysis["stored"] else " (not cached)"
        st.caption(f"Source: {source}{stored_note}")
        cols = st.columns(5)
        cols[0].metric("Decision", analysis.get("decision") or "N/A")
        cols[1].metric("Assessment", analysis.get("fundamental_assessment") or "N/A")
        score = analysis.get("fundamental_score")
        cols[2].metric("Score", "N/A" if score is None else str(score))
        cols[3].metric("Financial [F]", "yes" if analysis.get("finance_context_present") else "no")
        cols[4].metric("News Items", str(analysis.get("news_item_count") or 0))
        st.markdown(analysis.get("answer") or "No analysis available.")

    st.subheader(f"{result.ticker} cached news")
    news_df = ticker_news_df(result.ticker)
    if news_df.empty:
        st.info("No cached news articles found for the selected ticker.")
    else:
        st.caption(f"{len(news_df)} cached articles from Turso")
        st.dataframe(
            news_df,
            width="stretch",
            hide_index=True,
            column_config={
                "url": st.column_config.LinkColumn("URL"),
            },
        )


def render_bullish_candidates_tab(
    ranked: list[SignalResult],
    cfg: PipelineConfig,
) -> None:
    if not ranked:
        st.warning("No ranked candidates found in today's Turso snapshot. Run the updater first.")
        return

    summary = result_summary_df(ranked, cfg.top_n)
    st.caption("Loaded from today's Turso bullish-candidate snapshot.")
    company_profiles = load_company_profiles(tuple(summary["ticker"].astype(str).tolist()))
    summary = add_company_profile_columns(summary, company_profiles)
    fundamental_analyses = load_cached_fundamental_analyses(ranked, top_n=FUNDAMENTAL_TOP_N)
    with st.container(key="candidate_table_sticky"):
        toggle_label = "^" if st.session_state.candidate_table_expanded else "v"
        toggle_col, title_col = st.columns(
            [0.06, 0.94],
            vertical_alignment="center",
        )
        if toggle_col.button(
            toggle_label,
            key="toggle_candidate_table",
            width="stretch",
            help=(
                "Collapse candidate table"
                if st.session_state.candidate_table_expanded
                else "Expand candidate table"
            ),
        ):
            st.session_state.candidate_table_expanded = not st.session_state.candidate_table_expanded
            st.rerun()
        title_col.markdown("**Top bullish candidates**")

        if st.session_state.candidate_table_expanded:
            table_state = st.dataframe(
                summary,
                width="stretch",
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                height=280,
            )
            selected_rows = table_state.selection.rows
            if selected_rows and selected_rows[0] < len(summary):
                st.session_state.selected_candidate_index = selected_rows[0]

    spacer_height = 410 if st.session_state.candidate_table_expanded else 140
    st.markdown(
        f'<div style="height: {spacer_height}px;"></div>',
        unsafe_allow_html=True,
    )
    selected_index = st.session_state.selected_candidate_index
    if selected_index >= len(summary):
        selected_index = 0
        st.session_state.selected_candidate_index = selected_index
    selected_ticker = summary.iloc[selected_index]["ticker"]
    selected_result = next(r for r in ranked if r.ticker == selected_ticker)
    selected_analysis = analysis_for_ticker(fundamental_analyses, selected_result.ticker)
    selected_profile = company_profiles.get(selected_result.ticker.upper())

    render_ticker_view(selected_result, cfg, selected_analysis, selected_profile)


def render_ticker_search_tab(cfg: PipelineConfig) -> None:
    if "searched_ticker" not in st.session_state:
        st.session_state.searched_ticker = ""

    search_col, button_col = st.columns([0.82, 0.18], vertical_alignment="bottom")
    raw_ticker = search_col.text_input(
        "Ticker",
        value=st.session_state.searched_ticker,
        key="ticker_search_input",
    )
    if button_col.button("Search", width="stretch"):
        ticker = normalize_search_ticker(raw_ticker)
        if ticker is None:
            st.session_state.searched_ticker = ""
            st.warning("Enter a valid ticker.")
        else:
            st.session_state.searched_ticker = ticker

    searched_ticker = normalize_search_ticker(st.session_state.searched_ticker)
    if searched_ticker is None:
        return

    cached = load_search_result(searched_ticker)
    if cached is None:
        st.info(
            f"{searched_ticker} is not cached yet. Run `python -B tests\\ta_cache_updater.py` "
            "to refresh Turso before searching this ticker."
        )
        return

    result, search_cfg, snapshot = cached
    st.caption(f"Loaded cached signal snapshot from {snapshot.get('snapshot_date')}.")
    analyses = load_cached_fundamental_analyses([result], top_n=1)
    company_profiles = load_company_profiles((searched_ticker,))
    render_ticker_view(
        result,
        search_cfg or cfg,
        analysis_for_ticker(analyses, searched_ticker),
        company_profiles.get(searched_ticker),
    )


def render_ticker_counts_tab(cfg: PipelineConfig) -> None:
    counts = ticker_counts_df(cfg)
    if counts.empty:
        st.info("No cached ticker articles found for the current news window.")
        return

    st.caption(f"Cached article-symbol counts from the last {cfg.news_days} days")
    st.dataframe(
        counts,
        width="stretch",
        hide_index=True,
    )


def render_stock_pick_returns_tab() -> None:
    pick_dates = ta_cache.load_stock_pick_dates()
    if not pick_dates:
        st.info("No cached stock pick dates found yet.")
        return

    today = datetime.now(timezone.utc).date().isoformat()
    default_index = next(
        (index for index, pick_date in enumerate(pick_dates) if pick_date < today),
        0,
    )
    selected_date = st.selectbox(
        "Pick date",
        pick_dates,
        index=default_index,
    )
    snapshot = ta_cache.load_stock_pick_snapshot(selected_date)
    if not snapshot or not snapshot.get("tickers"):
        st.info("No stock picks found for the selected date.")
        return

    returns, warnings = stock_pick_returns_df(
        selected_date,
        list(snapshot["tickers"]),
    )
    st.caption(f"Stock picks cached for {selected_date}")
    if warnings:
        for warning in warnings:
            st.warning(warning)
    st.dataframe(
        returns,
        width="stretch",
        hide_index=True,
    )


def main() -> None:
    if "candidate_table_expanded" not in st.session_state:
        st.session_state.candidate_table_expanded = True
    if "selected_candidate_index" not in st.session_state:
        st.session_state.selected_candidate_index = 0

    try:
        ranked, cfg = load_results()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    candidates_tab, search_tab, ticker_counts_tab, stock_pick_returns_tab = st.tabs(
        [
            "Bullish candidates",
            "Ticker search",
            "Ticker News Mentions",
            "Stock Pick Returns",
        ]
    )
    with candidates_tab:
        render_bullish_candidates_tab(ranked, cfg)
    with search_tab:
        render_ticker_search_tab(cfg)
    with ticker_counts_tab:
        render_ticker_counts_tab(cfg)
    with stock_pick_returns_tab:
        render_stock_pick_returns_tab()


if __name__ == "__main__":
    main()
