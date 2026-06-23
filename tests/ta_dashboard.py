# Run with:
# streamlit run ta_dashboard.py

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, timezone
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (_PROJECT_ROOT, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from financetoolkit import Toolkit
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rag_trial.chat import chatter
from rag_trial.db import ta_cache
from ta_pipe import PipelineConfig, compute_signals, run_pipeline, SignalResult

if not all(
    hasattr(ta_cache, name)
    for name in (
        "load_fresh_fundamental_analysis",
        "load_articles_for_ticker",
        "load_articles_since",
        "load_stock_pick_dates",
        "load_stock_pick_snapshot",
        "upsert_stock_pick_snapshot",
    )
):
    ta_cache = importlib.reload(ta_cache)


FUNDAMENTAL_TOP_N = 15
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
TICKER_NEWS_DAY_OPTIONS = [7, 6, 5, 4, 3, 2, 1]


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
    news_days: int | None = None,
    articles: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if articles is None:
        articles = load_ticker_news_window_articles(
            cfg,
            db_path=db_path,
            news_days=news_days,
        )

    ticker_counter = Counter(
        str(article.get("symbol") or "").strip().upper()
        for article in articles
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


def load_ticker_news_window_articles(
    cfg: PipelineConfig,
    *,
    db_path: str | Path | None = None,
    news_days: int | None = None,
) -> list[dict[str, Any]]:
    selected_news_days = news_days if news_days is not None else cfg.news_days
    cutoff_date = (
        datetime.now(timezone.utc) - timedelta(days=selected_news_days)
    ).date().isoformat()
    return ta_cache.load_articles_since(cutoff_date, db_path=db_path)


def ticker_articles_df(
    cfg: PipelineConfig,
    *,
    db_path: str | Path | None = None,
    news_days: int | None = None,
    articles: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if articles is None:
        articles = load_ticker_news_window_articles(
            cfg,
            db_path=db_path,
            news_days=news_days,
        )
    rows = [
        {
            "published": article.get("publishedDate") or "",
            "ticker": str(article.get("symbol") or "").strip().upper(),
            "source": article.get("site") or "",
            "title": article.get("title") or "",
            "url": article.get("url") or "",
        }
        for article in articles
        if article.get("symbol")
    ]
    return pd.DataFrame(
        rows,
        columns=["published", "ticker", "source", "title", "url"],
    )


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


def _fetch_stock_pick_bars(
    tickers: list[str],
    *,
    start_date: str,
    end_date: str,
    toolkit_factory: Callable[..., Any] = Toolkit,
    db_path: str | Path | None = None,
) -> str | None:
    if not tickers:
        return None

    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return "FMP_API_KEY is not set; stock-pick price rows were not refreshed."

    try:
        toolkit = toolkit_factory(
            list(tickers),
            api_key=api_key,
            start_date=start_date,
            end_date=end_date,
        )
        downloaded = toolkit.get_historical_data()
    except Exception as exc:
        return f"FinanceToolkit price fetch failed: {type(exc).__name__}: {exc}"

    for ticker in tickers:
        hist = _history_for_stock_pick_ticker(downloaded, ticker)
        rows = _history_frame_to_stock_pick_bar_rows(ticker, hist)
        if rows:
            ta_cache.upsert_daily_bars(
                rows,
                provider="financetoolkit",
                db_path=db_path,
            )
    return None


def _needs_stock_pick_price_refresh(
    row: dict[str, Any],
    end_date: str,
) -> bool:
    latest_date = row.get("latest_date")
    return (
        row.get("base_close") is None
        or row.get("latest_close") is None
        or not latest_date
        or str(latest_date) < end_date
    )


def stock_pick_returns_df(
    pick_date: str,
    tickers: list[str],
    *,
    today: date | None = None,
    toolkit_factory: Callable[..., Any] = Toolkit,
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
    refresh_tickers = [
        ticker
        for ticker, row in returns_by_ticker.items()
        if _needs_stock_pick_price_refresh(row, end_date)
    ]

    warnings: list[str] = []
    fetch_warning = _fetch_stock_pick_bars(
        refresh_tickers,
        start_date=price_start_date,
        end_date=end_date,
        toolkit_factory=toolkit_factory,
        db_path=db_path,
    )
    if fetch_warning:
        warnings.append(fetch_warning)

    if refresh_tickers and not fetch_warning:
        for ticker in refresh_tickers:
            rows_by_ticker[ticker] = ta_cache.load_daily_bars(
                ticker,
                price_start_date,
                end_date,
                provider="financetoolkit",
                db_path=db_path,
            )
            returns_by_ticker[ticker] = _return_from_bar_rows(
                date_key,
                ticker,
                rows_by_ticker[ticker],
            )

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
def load_company_profiles(tickers: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    normalized = _normalized_profile_tickers(tickers)
    if not normalized:
        return {}

    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        error = "FMP_API_KEY is not set."
        return {ticker: _empty_profile_payload(ticker, error) for ticker in normalized}

    try:
        toolkit = Toolkit(list(normalized), api_key=api_key)
        profile = toolkit.get_profile()
    except Exception as exc:
        error = _profile_error_text(exc, api_key)
        return {ticker: _empty_profile_payload(ticker, error) for ticker in normalized}

    if profile is None or profile.empty:
        return {
            ticker: _empty_profile_payload(ticker, "FinanceToolkit returned no profile data.")
            for ticker in normalized
        }

    return {
        ticker: _profile_payload_from_frame(ticker, profile)
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


def _successful_fundamental_result(result: dict[str, Any]) -> bool:
    trace = result.get("retrieval_trace") or {}
    return (
        str(result.get("decision") or "").lower() == "answer"
        and result.get("route_type") == "single_ticker_financial"
        and bool(trace.get("finance_context_present"))
        and bool(str(result.get("answer") or "").strip())
    )


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


def _analysis_payload_from_route(
    ticker: str,
    company_name: str | None,
    result: dict[str, Any],
    *,
    cache_status: str,
    stored: bool,
) -> dict[str, Any]:
    trace = result.get("retrieval_trace") or {}
    assessment, score = _extract_fundamental_fields(result.get("answer") or "")
    resolved_target = result.get("resolved_target") or {}
    return {
        "ticker": ticker.upper(),
        "company_name": resolved_target.get("display_name") or company_name,
        "query": result.get("query") or _fundamental_query(ticker),
        "answer": result.get("answer") or "",
        "decision": result.get("decision"),
        "route_type": result.get("route_type"),
        "fundamental_assessment": assessment,
        "fundamental_score": score,
        "finance_context_present": bool(trace.get("finance_context_present")),
        "news_context_present": bool(trace.get("news_context_present")),
        "news_query": trace.get("news_query") or "",
        "news_item_count": int(trace.get("news_item_count") or 0),
        "retrieval_trace": trace,
        "logs": result.get("logs") or [],
        "generated_at_utc": None,
        "updated_at_utc": None,
        "cache_status": cache_status,
        "stored": stored,
    }


def _analysis_payload_from_error(
    ticker: str,
    company_name: str | None,
    exc: Exception,
) -> dict[str, Any]:
    error_text = f"{type(exc).__name__}: {exc}"
    return {
        "ticker": ticker.upper(),
        "company_name": company_name,
        "query": _fundamental_query(ticker),
        "answer": f"Fundamental analysis failed: {error_text}",
        "decision": "error",
        "route_type": "single_ticker_financial",
        "fundamental_assessment": None,
        "fundamental_score": None,
        "finance_context_present": False,
        "news_context_present": False,
        "news_query": "",
        "news_item_count": 0,
        "retrieval_trace": {"error": error_text},
        "logs": [error_text],
        "generated_at_utc": None,
        "updated_at_utc": None,
        "cache_status": "error",
        "stored": False,
    }


def _persist_fundamental_result(
    ticker: str,
    company_name: str | None,
    result: dict[str, Any],
    *,
    db_path: str | Path | None = None,
) -> None:
    trace = result.get("retrieval_trace") or {}
    assessment, score = _extract_fundamental_fields(result.get("answer") or "")
    resolved_target = result.get("resolved_target") or {}
    ta_cache.upsert_fundamental_analysis(
        ticker=ticker,
        company_name=resolved_target.get("display_name") or company_name,
        query=result.get("query") or _fundamental_query(ticker),
        answer=result.get("answer") or "",
        decision=result.get("decision"),
        route_type=result.get("route_type"),
        fundamental_assessment=assessment,
        fundamental_score=score,
        finance_context_present=bool(trace.get("finance_context_present")),
        news_context_present=bool(trace.get("news_context_present")),
        news_query=trace.get("news_query") or "",
        news_item_count=int(trace.get("news_item_count") or 0),
        retrieval_trace_json=json.dumps(trace, ensure_ascii=False),
        logs_json=json.dumps(result.get("logs") or [], ensure_ascii=False),
        db_path=db_path,
    )


def run_single_ticker_dashboard_analysis(
    ticker: str,
    company_name: str | None = None,
) -> dict[str, Any]:
    prompt = chatter.SINGLE_TICKER_FINANCIAL_PROMPT_TEMPLATE.format(
        date_min="N/A",
        date_max="N/A",
    )
    return chatter.run_single_ticker_fundamental_route(
        query=_fundamental_query(ticker),
        ticker=ticker,
        company_name=company_name,
        gen_client=chatter.create_generation_client(),
        base_single_ticker_financial_prompt=prompt,
        dump_query_contexts=False,
    )


def load_or_generate_fundamental_analyses(
    ranked: list[SignalResult],
    *,
    top_n: int = FUNDAMENTAL_TOP_N,
    route_runner: Callable[[str, str | None], dict[str, Any]] | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    runner = route_runner or run_single_ticker_dashboard_analysis
    analyses: list[dict[str, Any]] = []

    for result in ranked[:top_n]:
        ticker = result.ticker.upper()
        cached = ta_cache.load_fresh_fundamental_analysis(ticker, db_path=db_path)
        if cached is not None:
            analyses.append(_row_from_cached_analysis(cached))
            continue

        try:
            route_result = runner(ticker, None)
        except Exception as exc:
            analyses.append(_analysis_payload_from_error(ticker, None, exc))
            continue

        should_store = _successful_fundamental_result(route_result)
        if should_store:
            _persist_fundamental_result(ticker, None, route_result, db_path=db_path)
        analyses.append(
            _analysis_payload_from_route(
                ticker,
                None,
                route_result,
                cache_status="generated",
                stored=should_store,
            )
        )

    return analyses


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
def load_results() -> tuple[list[SignalResult], PipelineConfig]:
    cfg = dashboard_pipeline_config()
    ranked = run_pipeline(cfg)
    save_stock_pick_snapshot(ranked, cfg)
    return ranked, cfg


def compute_search_result(ticker: str) -> tuple[SignalResult, PipelineConfig]:
    cfg = dashboard_pipeline_config()
    return compute_signals(ticker, cfg), cfg


@st.cache_data(show_spinner=True, ttl=3600)
def load_search_result(ticker: str) -> tuple[SignalResult, PipelineConfig]:
    return compute_search_result(ticker)


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
        st.caption(f"{len(news_df)} cached articles from ta_cache.db")
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
        st.warning("No ranked candidates found.")
        return

    summary = result_summary_df(ranked, cfg.top_n)
    company_profiles = load_company_profiles(tuple(summary["ticker"].astype(str).tolist()))
    summary = add_company_profile_columns(summary, company_profiles)
    fundamental_analyses = load_or_generate_fundamental_analyses(ranked, top_n=FUNDAMENTAL_TOP_N)
    with st.container(key="candidate_table"):
        st.markdown("**Top bullish candidates**")
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

    selected_index = st.session_state.selected_candidate_index
    if selected_index >= len(summary):
        selected_index = 0
        st.session_state.selected_candidate_index = selected_index
    selected_ticker = summary.iloc[selected_index]["ticker"]
    selected_result = next(r for r in ranked if r.ticker == selected_ticker)
    selected_analysis = analysis_for_ticker(fundamental_analyses, selected_result.ticker)
    selected_profile = company_profiles.get(selected_result.ticker.upper())

    render_ticker_view(selected_result, cfg, selected_analysis, selected_profile)

    _, refresh_col = st.columns([0.82, 0.18])
    if refresh_col.button("Refresh pipeline", width="stretch"):
        load_results.clear()
        st.session_state.selected_candidate_index = 0
        st.rerun()


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

    result, search_cfg = load_search_result(searched_ticker)
    analyses = load_or_generate_fundamental_analyses([result], top_n=1)
    company_profiles = load_company_profiles((searched_ticker,))
    render_ticker_view(
        result,
        search_cfg or cfg,
        analysis_for_ticker(analyses, searched_ticker),
        company_profiles.get(searched_ticker),
    )


def render_ticker_counts_tab(cfg: PipelineConfig) -> None:
    selected_news_days = st.selectbox(
        "News mention window",
        TICKER_NEWS_DAY_OPTIONS,
        index=0,
        format_func=lambda days: f"Last {days} day{'s' if days != 1 else ''}",
    )
    articles = load_ticker_news_window_articles(
        cfg,
        news_days=selected_news_days,
    )
    counts = ticker_counts_df(
        cfg,
        news_days=selected_news_days,
        articles=articles,
    )
    if counts.empty:
        st.info(
            "No cached ticker articles found for the selected news mention window."
        )
        return

    st.caption(
        f"Cached article-symbol counts from the last {selected_news_days} "
        f"day{'s' if selected_news_days != 1 else ''}"
    )
    st.dataframe(
        counts,
        width="stretch",
        hide_index=True,
    )

    article_rows = ticker_articles_df(
        cfg,
        news_days=selected_news_days,
        articles=articles,
    )
    st.caption(
        f"Cached articles from the last {selected_news_days} "
        f"day{'s' if selected_news_days != 1 else ''}"
    )
    st.dataframe(
        article_rows,
        width="stretch",
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("URL"),
        },
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
    if "selected_candidate_index" not in st.session_state:
        st.session_state.selected_candidate_index = 0

    ranked, cfg = load_results()

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
