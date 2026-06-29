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
from ta_pipe import (
    PipelineConfig,
    SignalResult,
    classify_regime,
    compute_signals,
    run_pipeline,
)

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


def _series_value_on_date(
    series: pd.Series | None,
    target_date: date,
) -> float | None:
    """Return the value on target_date, or the nearest prior valid value."""
    if series is None or series.empty:
        return None

    values = pd.to_numeric(series, errors="coerce").replace(
        [float("inf"), -float("inf")],
        pd.NA,
    ).dropna()
    if values.empty:
        return None

    values = values.copy()
    values.index = pd.to_datetime(values.index)
    if isinstance(values.index, pd.DatetimeIndex) and values.index.tz is not None:
        values.index = values.index.tz_localize(None)
    values = values.sort_index()
    prior = values[values.index <= pd.Timestamp(target_date)]
    if prior.empty:
        return None
    return float(prior.iloc[-1])


def _clean_close_series(result: SignalResult) -> pd.Series:
    if result.close is None:
        return pd.Series(dtype="float64")

    close = pd.to_numeric(result.close, errors="coerce").replace(
        [float("inf"), -float("inf")],
        pd.NA,
    ).dropna()
    if close.empty:
        return close.astype("float64")

    close = close.astype("float64").copy()
    close.index = pd.to_datetime(close.index)
    if isinstance(close.index, pd.DatetimeIndex) and close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    return close.sort_index()


def _position_slice_from_entry(
    close: pd.Series,
    entry_date: date,
) -> dict[str, Any]:
    """Resolve the market bar and price slice used for entry-date context."""
    if close is None or close.empty:
        empty = close.iloc[0:0] if close is not None else pd.Series(dtype="float64")
        return {
            "entry_bar_date": None,
            "entry_bar_close": None,
            "from_entry": empty,
            "entry_data_warning": "No close price data is available.",
        }

    first_available = close.index.min().date()
    last_available = close.index.max().date()
    entry_ts = pd.Timestamp(entry_date)

    if entry_date < first_available:
        return {
            "entry_bar_date": None,
            "entry_bar_close": None,
            "from_entry": close.iloc[0:0],
            "entry_data_warning": (
                f"Entry date {entry_date} is before the available price history "
                f"starting on {first_available}. Date-based position metrics are unavailable."
            ),
        }

    from_raw_entry = close[close.index >= entry_ts]
    up_to_entry = close[close.index <= entry_ts]
    if not from_raw_entry.empty:
        entry_bar_date = from_raw_entry.index[0].date()
        entry_bar_close = float(from_raw_entry.iloc[0])
        warning = None
    elif not up_to_entry.empty:
        entry_bar_date = up_to_entry.index[-1].date()
        entry_bar_close = float(up_to_entry.iloc[-1])
        warning = (
            f"Entry date {entry_date} is after the latest loaded price bar "
            f"({last_available}); using the latest available bar for date-based metrics."
        )
    else:
        entry_bar_date = None
        entry_bar_close = None
        warning = "Could not resolve an entry bar from the available price history."

    from_entry = (
        close[close.index >= pd.Timestamp(entry_bar_date)]
        if entry_bar_date is not None
        else close.iloc[0:0]
    )
    return {
        "entry_bar_date": entry_bar_date,
        "entry_bar_close": entry_bar_close,
        "from_entry": from_entry,
        "entry_data_warning": warning,
    }


def compute_position_context(
    result: SignalResult,
    cfg: PipelineConfig,
    entry_price: float,
    entry_date: date,
) -> dict[str, Any]:
    close = _clean_close_series(result)
    slice_info = _position_slice_from_entry(close, entry_date)
    entry_bar_date = slice_info["entry_bar_date"]
    from_entry = slice_info["from_entry"]

    if not from_entry.empty:
        highest_close_since_entry = float(from_entry.max())
        highest_close_date = from_entry.idxmax().date()
    else:
        highest_close_since_entry = None
        highest_close_date = None

    if entry_bar_date is not None:
        regime_on_entry = classify_regime(
            latest_close=_series_value_on_date(result.close, entry_bar_date),
            latest_sma_50=_series_value_on_date(result.sma_50, entry_bar_date),
            latest_sma_200=_series_value_on_date(result.sma_200, entry_bar_date),
            latest_spread=_series_value_on_date(result.spread, entry_bar_date),
            bullish_impulse_score=result.bullish_impulse_score,
            pre_golden_score=result.pre_golden_score,
            cfg=cfg,
        )
        regime_on_entry_is_approx = True
    else:
        regime_on_entry = None
        regime_on_entry_is_approx = False

    trading_days_held = max(len(from_entry) - 1, 0) if not from_entry.empty else 0
    latest_close = _latest_valid_value(result.close)
    return_pct = (
        (latest_close - entry_price) / entry_price * 100
        if latest_close is not None and entry_price > 0
        else None
    )
    days_remaining_in_window = max(cfg.forward_days - trading_days_held, 0)

    return {
        "entry_bar_date": entry_bar_date,
        "entry_bar_close": slice_info["entry_bar_close"],
        "entry_data_warning": slice_info["entry_data_warning"],
        "latest_close": latest_close,
        "return_pct": return_pct,
        "days_held": max((date.today() - entry_date).days, 0),
        "trading_days_held": trading_days_held,
        "days_remaining_in_window": days_remaining_in_window,
        "window_expired": trading_days_held >= cfg.forward_days,
        "highest_close_since_entry": highest_close_since_entry,
        "highest_close_date": highest_close_date,
        "regime_on_entry": regime_on_entry,
        "regime_on_entry_is_approx": regime_on_entry_is_approx,
        "regime_now": result.regime_label,
        "regime_changed": (
            regime_on_entry is not None
            and result.regime_label is not None
            and regime_on_entry != result.regime_label
        ),
    }


def compute_exit_levels(
    result: SignalResult,
    cfg: PipelineConfig,
    entry_price: float,
    entry_date: date,
) -> dict[str, float | None]:
    close = _clean_close_series(result)
    from_entry = _position_slice_from_entry(close, entry_date)["from_entry"]
    latest_atr = _latest_valid_value(result.atr_14)
    latest_close = _latest_valid_value(result.close)

    stop_atr_1x = entry_price - latest_atr if latest_atr is not None else None
    stop_atr_2x = entry_price - 2.0 * latest_atr if latest_atr is not None else None
    highest_since_entry = float(from_entry.max()) if not from_entry.empty else None
    stop_trailing_atr_2x = (
        highest_since_entry - 2.0 * latest_atr
        if highest_since_entry is not None and latest_atr is not None
        else None
    )

    target_atr_1x = entry_price + latest_atr if latest_atr is not None else None
    target_atr_2x = entry_price + 2.0 * latest_atr if latest_atr is not None else None
    target_atr_3x = entry_price + 3.0 * latest_atr if latest_atr is not None else None
    target_extension_1_5x = (
        entry_price + cfg.overbought_extension_atr * latest_atr
        if latest_atr is not None
        else None
    )
    target_extension_2_5x = (
        entry_price + cfg.severe_overbought_extension_atr * latest_atr
        if latest_atr is not None
        else None
    )
    target_extension_3x = (
        entry_price + cfg.exhaustion_extension_atr * latest_atr
        if latest_atr is not None
        else None
    )

    if close.empty:
        target_bb_upper = None
    else:
        rolling_mean = close.rolling(cfg.bollinger_period).mean()
        rolling_std = close.rolling(cfg.bollinger_period).std()
        target_bb_upper = _latest_valid_value(
            rolling_mean + cfg.bollinger_std_mult * rolling_std
        )

    risk = entry_price - stop_atr_2x if stop_atr_2x is not None else None

    def _rr(target: float | None) -> float | None:
        if target is None or risk is None or risk <= 0:
            return None
        return (target - entry_price) / risk

    return {
        "latest_atr": latest_atr,
        "latest_close": latest_close,
        "atr_pct": (
            latest_atr / entry_price * 100
            if latest_atr is not None and entry_price > 0
            else None
        ),
        "highest_since_entry": highest_since_entry,
        "stop_atr_1x": stop_atr_1x,
        "stop_atr_2x": stop_atr_2x,
        "stop_trailing_atr_2x": stop_trailing_atr_2x,
        "stop_sma50": _latest_valid_value(result.sma_50),
        "stop_ema21": _latest_valid_value(result.ema_21),
        "target_atr_1x": target_atr_1x,
        "target_atr_2x": target_atr_2x,
        "target_atr_3x": target_atr_3x,
        "target_extension_1_5x": target_extension_1_5x,
        "target_extension_2_5x": target_extension_2_5x,
        "target_extension_3x": target_extension_3x,
        "target_donchian_20": _latest_valid_value(result.donchian_20_high),
        "target_donchian_55": _latest_valid_value(result.donchian_55_high),
        "target_bb_upper": target_bb_upper,
        "target_sma200": _latest_valid_value(result.sma_200),
        "rr_atr_1x": _rr(target_atr_1x),
        "rr_atr_2x": _rr(target_atr_2x),
        "rr_atr_3x": _rr(target_atr_3x),
        "rr_extension_1_5x": _rr(target_extension_1_5x),
        "rr_extension_2_5x": _rr(target_extension_2_5x),
        "rr_extension_3x": _rr(target_extension_3x),
        "rr_donchian_20": _rr(_latest_valid_value(result.donchian_20_high)),
        "rr_donchian_55": _rr(_latest_valid_value(result.donchian_55_high)),
        "rr_bb_upper": _rr(target_bb_upper),
        "rr_sma200": _rr(_latest_valid_value(result.sma_200)),
    }


def recommended_exits(
    levels: dict[str, float | None],
    context: dict[str, Any],
    entry_price: float,
    cfg: PipelineConfig,
) -> list[dict[str, Any]]:
    del entry_price
    exits: list[dict[str, Any]] = []
    keyed_positions: dict[str, int] = {}

    if context["window_expired"]:
        exits.append({
            "label": "Signal window expired — review exit",
            "type": "stop",
            "price": None,
            "rr": None,
            "rationale": (
                f"Position has been held for {context['trading_days_held']} trading days, "
                f"beyond the {cfg.forward_days}-trading-day signal window. This is a review "
                "flag, not an automatic sell signal."
            ),
            "priority": 1,
        })
    if context["regime_changed"] and context["regime_now"] in {
        "bearish_or_weak",
        "neutral",
    }:
        exits.append({
            "label": "Regime deteriorated — review position",
            "type": "stop",
            "price": None,
            "rr": None,
            "rationale": (
                f"Regime changed from {context['regime_on_entry']} to "
                f"{context['regime_now']} since entry."
            ),
            "priority": 1,
        })

    def add(
        label: str,
        kind: str,
        price_key: str,
        rr_key: str | None,
        rationale: str,
        priority: int,
    ) -> None:
        item = {
            "label": label,
            "type": kind,
            "price": levels.get(price_key),
            "rr": levels.get(rr_key) if rr_key else None,
            "rationale": rationale,
            "priority": priority,
        }
        existing_position = keyed_positions.get(price_key)
        if existing_position is None:
            keyed_positions[price_key] = len(exits)
            exits.append(item)
        elif priority < exits[existing_position]["priority"]:
            exits[existing_position] = item

    targets = {
        "target_atr_1x": (
            "1× ATR target",
            "rr_atr_1x",
            "Conservative volatility-based profit objective.",
        ),
        "target_atr_2x": (
            "2× ATR target",
            "rr_atr_2x",
            "Standard volatility-based profit objective.",
        ),
        "target_extension_1_5x": (
            "1.5× ATR extension",
            "rr_extension_1_5x",
            "Configured overbought-extension threshold for a fast bullish impulse.",
        ),
        "target_extension_2_5x": (
            "2.5× ATR extension",
            "rr_extension_2_5x",
            "Configured severe-overbought extension threshold.",
        ),
        "target_extension_3x": (
            "Exhaustion ceiling",
            "rr_extension_3x",
            "Informational ATR level associated with price exhaustion risk.",
        ),
        "target_donchian_20": (
            "20-day high resistance",
            "rr_donchian_20",
            "Near-term Donchian breakout or resistance level.",
        ),
        "target_donchian_55": (
            "55-day high resistance",
            "rr_donchian_55",
            "Longer-term Donchian breakout or resistance level.",
        ),
        "target_bb_upper": (
            "Bollinger upper band",
            "rr_bb_upper",
            "Current upper Bollinger price band.",
        ),
        "target_sma200": (
            "SMA200 reclaim target",
            "rr_sma200",
            "Key long-term trend and price-reclaim level.",
        ),
    }

    regime_targets = {
        "confirmed_bullish": [
            ("target_atr_2x", 1),
            ("target_donchian_55", 2),
            ("target_extension_2_5x", 2),
        ],
        "bullish_impulse": [
            ("target_extension_1_5x", 1),
            ("target_atr_2x", 2),
            ("target_donchian_20", 2),
        ],
        "bullish_transition": [
            ("target_sma200", 1),
            ("target_atr_1x", 2),
            ("target_bb_upper", 2),
        ],
        "pre_golden_setup": [
            ("target_sma200", 1),
            ("target_atr_1x", 2),
            ("target_bb_upper", 2),
        ],
    }
    selected_targets = regime_targets.get(
        context["regime_now"],
        [("target_atr_1x", 1), ("target_bb_upper", 2)],
    )
    for price_key, priority in selected_targets:
        label, rr_key, rationale = targets[price_key]
        add(label, "target", price_key, rr_key, rationale, priority)

    for price_key in ("target_donchian_55", "target_extension_3x"):
        label, rr_key, rationale = targets[price_key]
        add(label, "target", price_key, rr_key, rationale, 3)

    add(
        "Trailing 2× ATR stop",
        "stop",
        "stop_trailing_atr_2x",
        None,
        "Two ATR below the highest close since entry.",
        1,
    )
    add(
        "2× ATR stop from entry",
        "stop",
        "stop_atr_2x",
        None,
        "Primary volatility-adjusted stop measured from entry price.",
        1,
    )
    add(
        "1× ATR stop, tight",
        "stop",
        "stop_atr_1x",
        None,
        "Tighter volatility-adjusted stop measured from entry price.",
        2,
    )
    add(
        "SMA50 structural stop",
        "stop",
        "stop_sma50",
        None,
        "Current intermediate-trend support level.",
        2,
    )
    add(
        "EMA21 trailing stop",
        "stop",
        "stop_ema21",
        None,
        "Current fast-trend support, emphasized during bullish impulse regimes.",
        2 if context["regime_now"] == "bullish_impulse" else 3,
    )
    return exits


def _percentile_rank(value: float | None, values: list[float]) -> float | None:
    if value is None or not values:
        return None
    less = sum(1 for item in values if item < value)
    equal = sum(1 for item in values if item == value)
    if equal == 0:
        return round(100.0 * less / len(values), 1)
    return round(100.0 * (less + (equal + 1) / 2) / len(values), 1)


def result_summary_df(results: list[SignalResult], top_n: int) -> pd.DataFrame:
    all_rs_scores = [
        r.relative_strength_score
        for r in results
        if r.relative_strength_score is not None
    ]
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
            "rs_pct_rank": _percentile_rank(r.relative_strength_score, all_rs_scores),
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
    latest_adx = _latest_valid_value(r.adx)
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
        (
            "ADX",
            f"{latest_adx:.2f}" if latest_adx is not None else "None",
        ),
        (
            "Momentum score",
            f"{r.momentum_score:.2f}" if r.momentum_score is not None else "None",
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
        + [{"category": "momentum", "reason": reason} for reason in r.momentum_reasons]
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
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.43, 0.16, 0.15, 0.13, 0.13],
        subplot_titles=("Price", "SMA spread %", "MACD", "RSI", "OBV"),
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

    _add_series_trace(fig, r.obv, name="OBV", row=5, col=1, color="#34d399")

    final_score_text = f"{r.final_bullish_score:.2f}" if r.final_bullish_score is not None else "N/A"
    overbought_score_text = f"{r.overbought_score:.2f}" if r.overbought_score is not None else "N/A"
    fig.update_layout(
        title=(
            f"{r.ticker} | Regime={r.regime_label or 'N/A'} | "
            f"Final Score={final_score_text} | "
            f"Overbought={r.overbought_status or 'N/A'} ({overbought_score_text})"
        ),
        height=1100,
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
    fig.update_yaxes(title_text="OBV", row=5, col=1)
    fig.update_xaxes(title_text="Date", row=5, col=1)
    return fig


def _make_exit_plan_chart(
    result: SignalResult,
    cfg: PipelineConfig,
    entry_price: float,
    entry_date: date,
    exits: list[dict[str, Any]],
    context: dict[str, Any],
) -> go.Figure:
    fig = make_signal_chart(result, cfg)
    fig.add_vline(
        x=pd.Timestamp(entry_date),
        line_dash="dash",
        line_color="#ffffff",
    )
    fig.add_annotation(
        x=pd.Timestamp(entry_date),
        y=1,
        xref="x",
        yref="paper",
        text=f"Entry ${entry_price:.2f}",
        showarrow=False,
        xanchor="left",
        yanchor="top",
    )
    fig.add_hline(
        y=entry_price,
        line_dash="dash",
        line_color="#ffffff",
        annotation_text="Entry",
        annotation_position="top right",
        row=1,
        col=1,
    )

    line_styles = {
        ("target", 1): ("solid", "#22c55e"),
        ("target", 2): ("dash", "#fbbf24"),
        ("target", 3): ("dot", "#94a3b8"),
        ("stop", 1): ("solid", "#ef4444"),
        ("stop", 2): ("dash", "#f97316"),
        ("stop", 3): ("dot", "#f97316"),
    }
    for item in exits:
        price = item.get("price")
        if price is None:
            continue
        line_dash, line_color = line_styles[(item["type"], item["priority"])]
        fig.add_hline(
            y=price,
            line_dash=line_dash,
            line_color=line_color,
            annotation_text=item["label"],
            annotation_position="top right",
            row=1,
            col=1,
        )

    highest_close = context.get("highest_close_since_entry")
    if highest_close is not None:
        fig.add_hline(
            y=highest_close,
            line_dash="dot",
            line_color="#38bdf8",
            annotation_text=f"High since entry ${highest_close:.2f}",
            annotation_position="top left",
            row=1,
            col=1,
        )
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


def render_exit_planner_tab(cfg: PipelineConfig) -> None:
    st.subheader("Exit Planner")
    st.caption(
        "Monitor current exit levels for an existing position. Levels use the latest loaded "
        "indicators and are not a historical backtest."
    )

    col_ticker, col_price, col_date, col_button = st.columns(
        [0.50, 0.20, 0.20, 0.10],
        vertical_alignment="bottom",
    )
    with col_ticker:
        raw_ticker = st.text_input("Ticker", key="exit_planner_ticker_input")
    with col_price:
        raw_entry_price = st.number_input(
            "Entry price",
            min_value=0.01,
            step=0.01,
            format="%.2f",
            key="exit_planner_price_input",
        )
    with col_date:
        raw_entry_date = st.date_input(
            "Entry date",
            key="exit_planner_date_input",
            max_value=date.today(),
        )
    with col_button:
        clicked = st.button(
            "Search",
            width="stretch",
            key="exit_planner_search_button",
        )

    if not clicked:
        st.info("Enter a ticker, entry price, and entry date, then click Search.")
        return

    ticker = normalize_search_ticker(raw_ticker)
    if ticker is None:
        st.warning("Enter a valid ticker (e.g. AAPL).")
        return
    if raw_entry_price is None or raw_entry_price <= 0:
        st.warning("Entry price must be greater than zero.")
        return
    if raw_entry_date > date.today():
        st.warning("Entry date cannot be in the future.")
        return

    entry_price = float(raw_entry_price)
    entry_date = raw_entry_date
    result, exit_cfg = load_search_result(ticker)
    exit_cfg = exit_cfg or cfg

    error = getattr(result, "error", None)
    if error or result.close is None or result.close.dropna().empty:
        st.warning(f"{ticker}: {error or 'no price data'}")
        return

    context = compute_position_context(result, exit_cfg, entry_price, entry_date)
    levels = compute_exit_levels(result, exit_cfg, entry_price, entry_date)
    exits = recommended_exits(levels, context, entry_price, exit_cfg)

    if context.get("entry_data_warning"):
        st.warning(context["entry_data_warning"])
    if context["regime_on_entry_is_approx"]:
        st.caption(
            "Regime on entry is approximate: entry-date moving-average values are used, "
            "but impulse/pre-golden scores come from the current loaded signal result."
        )

    def fmt_money(value: float | None) -> str:
        return f"${value:.2f}" if value is not None else "N/A"

    def fmt_pct(value: float | None) -> str:
        return f"{value:+.2f}%" if value is not None else "N/A"

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Latest close", fmt_money(context["latest_close"]))
    m2.metric(
        "Return",
        fmt_pct(context["return_pct"]),
        delta=(
            fmt_pct(context["return_pct"])
            if context["return_pct"] is not None
            else None
        ),
    )
    m3.metric(
        "Days held",
        f"{context['days_held']}d ({context['trading_days_held']} trading)",
    )
    m4.metric(
        "Signal window",
        (
            "Expired"
            if context["window_expired"]
            else f"{context['days_remaining_in_window']} trading days left"
        ),
    )
    m5.metric(
        "ATR % of entry",
        f"{levels['atr_pct']:.2f}%" if levels["atr_pct"] is not None else "N/A",
    )
    m6.metric("Overbought", result.overbought_status or "N/A")

    regime_label = context["regime_now"] or "N/A"
    if context["regime_changed"]:
        regime_label += " — changed"
    left, right = st.columns(2)
    left.info(f"Regime on entry: {context['regime_on_entry'] or 'N/A'}")
    right.info(f"Regime now: {regime_label}")

    if context["highest_close_since_entry"] is not None:
        trailing_stop = levels["stop_trailing_atr_2x"]
        trailing_note = (
            f" | Trailing 2× ATR stop: ${trailing_stop:.2f}"
            if trailing_stop is not None
            else ""
        )
        st.caption(
            f"Highest close since entry: ${context['highest_close_since_entry']:.2f} "
            f"on {context['highest_close_date']}{trailing_note}"
        )

    sorted_exits = sorted(
        exits,
        key=lambda item: (
            item["priority"],
            item["price"] is None,
            -(item["price"] if item["price"] is not None else 0.0),
        ),
    )
    display_df = pd.DataFrame(sorted_exits)
    display_df["price"] = display_df["price"].map(
        lambda value: f"${value:.2f}" if pd.notna(value) else "—"
    )
    display_df["rr"] = display_df["rr"].map(
        lambda value: f"{value:.2f}x" if pd.notna(value) and value > 0 else "—"
    )
    st.caption(
        f"Exit plan for {ticker} | Entry: ${entry_price:.2f} on {entry_date} | "
        f"Current regime: {context['regime_now'] or 'N/A'}"
    )
    st.dataframe(display_df, hide_index=True, width="stretch")

    with st.expander("All computed levels"):
        rows = [
            {
                "level": key,
                "value": f"{value:.4f}" if isinstance(value, (int, float)) else "—",
            }
            for key, value in levels.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    fig = _make_exit_plan_chart(
        result,
        exit_cfg,
        entry_price,
        entry_date,
        exits,
        context,
    )
    st.plotly_chart(fig, width="stretch")


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

    (
        candidates_tab,
        search_tab,
        ticker_counts_tab,
        stock_pick_returns_tab,
        exit_tab,
    ) = st.tabs([
        "Bullish candidates",
        "Ticker search",
        "Ticker News Mentions",
        "Stock Pick Returns",
        "Exit Planner",
    ])
    with candidates_tab:
        render_bullish_candidates_tab(ranked, cfg)
    with search_tab:
        render_ticker_search_tab(cfg)
    with ticker_counts_tab:
        render_ticker_counts_tab(cfg)
    with stock_pick_returns_tab:
        render_stock_pick_returns_tab()
    with exit_tab:
        render_exit_planner_tab(cfg)


if __name__ == "__main__":
    main()
