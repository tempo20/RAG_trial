# Run with:
# streamlit run ta_dashboard.py

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import importlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

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
    )
):
    ta_cache = importlib.reload(ta_cache)


FUNDAMENTAL_TOP_N = 15
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


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


def main() -> None:
    if "candidate_table_expanded" not in st.session_state:
        st.session_state.candidate_table_expanded = True
    if "selected_candidate_index" not in st.session_state:
        st.session_state.selected_candidate_index = 0

    ranked, cfg = load_results()

    candidates_tab, search_tab, ticker_counts_tab = st.tabs(
        ["Bullish candidates", "Ticker search", "Ticker News Mentions"]
    )
    with candidates_tab:
        render_bullish_candidates_tab(ranked, cfg)
    with search_tab:
        render_ticker_search_tab(cfg)
    with ticker_counts_tab:
        render_ticker_counts_tab(cfg)


if __name__ == "__main__":
    main()
