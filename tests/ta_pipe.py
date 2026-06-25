"""
Stock Screening & Signal Pipeline
==================================
Stages:
  1. collect_candidates  — news momentum + Yahoo screener overlap
  2. compute_signals     — SMA/VWMA crossover logic, no plotting
  3. rank_candidates     — rank strongest pre-golden-cross setups
  4. plot_top            — render charts only for the top N tickers

Usage:
    python ta_pipe.py

    # Or import and call individually:
    from ta_pipe import run_pipeline
    results = run_pipeline(top_n=10, plot=True)

Environment variables required:
    FMP_API_KEY   — Financial Modelling Prep API key
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from dataclasses import dataclass, field

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from financetoolkit import Toolkit

from rag_trial.chat.fmp_functions import (
    get_news_stock_latest,
    _is_blocked_stock_news_publisher,
    _parse_fmp_news_datetime,
)
from rag_trial.db import ta_cache

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    news_days: int = 7                 
    news_limit: int = 250             
    news_max_pages: int = 100
    news_sleep_s: float = 0.25
    news_min_mentions: int = 2
    news_recency_decay: bool = True     

    # Signal screeners count as actual candidate evidence. Momentum and noise
    # screeners are weak context only and never satisfy hard screener overlap.
    signal_screeners: list[str] = field(default_factory=lambda: [
        "undervalued_growth_stocks",
        "undervalued_large_caps",
    ])
    momentum_screeners: list[str] = field(default_factory=lambda: [
        "day_gainers",
        "small_cap_gainers",
        "aggressive_small_caps",
    ])
    noise_screeners: list[str] = field(default_factory=lambda: [
        "most_actives",
    ])
    screener_count: int = 100
    screener_min_overlap: int = 2

    # Backward-compatibility fields retained for external callers. The
    # collector uses the explicit categories above as the source of truth.
    screeners: list[str] | None = None
    screener_noise: set[str] | None = None

    candidate_score_threshold: float = 0.25
    candidate_pool_limit: int = 250
    news_recency_weight: float = 1.0
    news_mentions_weight: float = 0.5
    screener_signal_weight: float = 1.0
    screener_momentum_weight: float = 0.15
    screener_noise_weight: float = 0.05
    debug_candidate_reasons: bool = True
    debug_candidate_top_n: int = 30

    cheap_prefilter: bool = True
    cheap_history_period: str = "1y"
    cheap_batch_size: int = 80
    cheap_top_n: int = 80
    cheap_min_price: float = 2.0
    cheap_min_avg_dollar_volume: float = 5_000_000.0
    cheap_trend_weight: float = 0.5
    cheap_momentum_20d_weight: float = 1.0
    cheap_momentum_50d_weight: float = 0.5
    cheap_relative_volume_weight: float = 0.25
    cheap_liquidity_weight: float = 0.05
    prefilter_52w_high_max_distance: float = 0.20

    start_date: str = "2022-01-01"
    vwma_period: int = 50
    short_sma_period: int = 50
    long_sma_period: int = 200
    confirmation_days: int = 5
    min_spread: float = 0.002
    forward_days: int = 20
    signal_sleep_s: float = 0.3
    top_n: int = 10
    plot: bool = True
    show_plots_at_end: bool = True
    save_plots: bool = False
    plot_dir: str = "plots"
    impulse_donchian_short: int = 20
    impulse_donchian_long: int = 55
    impulse_return_z_window: int = 20
    impulse_return_z_threshold: float = 2.0
    impulse_atr_period: int = 14
    impulse_atr_move_threshold: float = 1.5
    impulse_relative_volume_short: int = 5
    impulse_relative_volume_long: int = 20
    impulse_relative_volume_threshold: float = 1.5
    impulse_ema_fast: int = 8
    impulse_ema_mid: int = 21
    impulse_ema_slow: int = 50
    impulse_min_score: float = 7.0
    transition_min_score: float = 5.0
    relative_strength_benchmarks: list[str] = field(
        default_factory=lambda: ["QQQ", "SOXX", "SMH"]
    )
    relative_strength_window: int = 20
    # Explicit momentum-enhancement weights total 0.85 as requested.
    bullish_impulse_weight: float = 0.25
    pre_golden_weight: float = 0.20
    relative_strength_weight: float = 0.15
    liquidity_volume_weight: float = 0.10
    adx_period: int = 14
    adx_trending_threshold: float = 25.0
    adx_strong_threshold: float = 40.0
    obv_slope_window: int = 10
    obv_high_window: int = 20
    momentum_weight: float = 0.15
    overbought_rsi_level: float = 70.0
    severe_overbought_rsi_level: float = 80.0
    stoch_rsi_period: int = 14
    overbought_stoch_rsi_level: float = 0.80
    severe_overbought_stoch_rsi_level: float = 0.90
    atr_period: int = 14
    ema_extension_period: int = 20
    overbought_extension_atr: float = 1.5
    severe_overbought_extension_atr: float = 2.5
    exhaustion_extension_atr: float = 3.0
    overbought_sma50_distance: float = 0.08
    severe_overbought_sma50_distance: float = 0.15
    bollinger_period: int = 20
    bollinger_std_mult: float = 2.0
    overbought_bb_position: float = 1.0
    severe_overbought_bb_position: float = 1.25
    volume_climax_relative_volume: float = 2.0
    volume_climax_extension_atr: float = 2.0
    moderately_extended_threshold: float = 2.0
    overbought_threshold: float = 4.0
    severely_overbought_threshold: float = 7.0


@dataclass
class NewsTickerScores:
    recency_scores: Counter
    mention_counts: Counter


@dataclass
class ScreenerTickerScores:
    combined_scores: Counter
    signal_counts: Counter
    momentum_counts: Counter
    noise_counts: Counter


@dataclass
class CheapMarketSignal:
    ticker: str
    score: float
    latest_price: float
    avg_dollar_volume_20d: float
    momentum_20d: float
    momentum_50d: float
    relative_volume_5d: float
    above_sma_200: bool | None
    distance_from_52w_high: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Candidate collection
# ─────────────────────────────────────────────────────────────────────────────

def _cache_enabled() -> bool:
    return ta_cache.cache_enabled()


def _score_news_articles(
    articles: list[dict],
    cfg: PipelineConfig,
    *,
    cutoff: datetime,
    now: datetime,
) -> NewsTickerScores:
    recency_scores: Counter = Counter()
    mention_counts: Counter = Counter()
    seen: set = set()

    for article in articles:
        if not isinstance(article, dict):
            continue

        published = _parse_fmp_news_datetime(
            article.get("publishedDate")
            or article.get("date")
            or article.get("publishedAt")
        )
        if published and published < cutoff:
            continue

        ticker = str(article.get("symbol") or "").strip().upper()
        if not ticker:
            continue

        key = article.get("article_id") or article.get("url") or (
            article.get("title"),
            article.get("publishedDate"),
        )
        key = (key, ticker)
        if key in seen:
            continue
        seen.add(key)

        if cfg.news_recency_decay and published:
            age_days = max((now - published).total_seconds() / 86400, 0.01)
            weight = 1.0 / age_days
        else:
            weight = 1.0

        recency_scores[ticker] += weight
        mention_counts[ticker] += 1

    return NewsTickerScores(recency_scores=recency_scores, mention_counts=mention_counts)


def _fetch_news_articles_online(
    cfg: PipelineConfig,
    *,
    cutoff: datetime,
    stop_at_or_before: datetime | None = None,
) -> list[dict]:
    articles: list[dict] = []
    seen: set = set()

    for page in range(cfg.news_max_pages):
        try:
            batch = get_news_stock_latest(page=page, limit=cfg.news_limit)
        except Exception as exc:
            log.warning("FMP news page %d failed: %s", page, exc)
            break

        if not batch:
            break

        oldest_in_batch = None

        for article in batch:
            if not isinstance(article, dict):
                continue

            if _is_blocked_stock_news_publisher(article):
                continue

            published = _parse_fmp_news_datetime(
                article.get("publishedDate")
                or article.get("date")
                or article.get("publishedAt")
            )

            if published:
                oldest_in_batch = (
                    published
                    if oldest_in_batch is None
                    else min(oldest_in_batch, published)
                )

            if published and published < cutoff:
                continue

            key = article.get("url") or (
                article.get("title"),
                article.get("publishedDate"),
            )
            if key in seen:
                continue
            seen.add(key)

            if not article.get("symbol"):
                continue

            articles.append(article)

        log.info("News page=%d  batch=%d  kept_articles=%d", page, len(batch), len(articles))

        if oldest_in_batch and oldest_in_batch < cutoff:
            break
        if stop_at_or_before and oldest_in_batch and oldest_in_batch <= stop_at_or_before:
            break

        time.sleep(cfg.news_sleep_s)

    return articles


def collect_news_tickers(cfg: PipelineConfig) -> NewsTickerScores:
    """
    Return separate news recency scores and raw mention counts.
    Delegates fetching and publisher filtering to fmp_functions so there
    is a single source of truth for both the API call and blocked publishers.
    If recency_decay is enabled, articles are weighted by 1 / days_old.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=cfg.news_days)
    cutoff_date = cutoff.date().isoformat()

    fetched_articles: list[dict] = []

    if _cache_enabled():
        try:
            latest_cached = _parse_fmp_news_datetime(ta_cache.latest_article_published_at())
            if latest_cached:
                log.info("Refreshing FMP news cache after latest cached publish time %s", latest_cached)
            fetched_articles = _fetch_news_articles_online(
                cfg,
                cutoff=cutoff,
                stop_at_or_before=latest_cached,
            )
            if fetched_articles:
                ta_cache.upsert_articles(fetched_articles)
            cached_articles = ta_cache.load_articles_since(cutoff_date)
            if cached_articles:
                log.info("News cache loaded: %d cached article-symbol rows", len(cached_articles))
                return _score_news_articles(
                    cached_articles,
                    cfg,
                    cutoff=cutoff,
                    now=now,
                )
        except Exception as exc:
            log.warning("TA news cache refresh failed; falling back to FMP-only news: %s", exc)

    if not fetched_articles:
        fetched_articles = _fetch_news_articles_online(cfg, cutoff=cutoff)

    if _cache_enabled() and fetched_articles:
        try:
            ta_cache.upsert_articles(fetched_articles)
            cached_articles = ta_cache.load_articles_since(cutoff_date)
            if cached_articles:
                return _score_news_articles(
                    cached_articles,
                    cfg,
                    cutoff=cutoff,
                    now=now,
                )
        except Exception as exc:
            log.warning("TA news cache write failed; using fetched news only: %s", exc)

    return _score_news_articles(fetched_articles, cfg, cutoff=cutoff, now=now)


def collect_screener_tickers(cfg: PipelineConfig) -> ScreenerTickerScores:
    """
    Collect categorized Yahoo screener evidence.

    Signal screeners count as actual candidate evidence. Momentum screeners
    are weak context/tie-breakers. Noise screeners are activity or liquidity
    sources and receive only very weak context weight. Hard screener overlap
    is based only on signal screeners.
    """
    signal_counts: Counter = Counter()
    momentum_counts: Counter = Counter()
    noise_counts: Counter = Counter()

    screener_groups = (
        ("Signal", cfg.signal_screeners, signal_counts),
        ("Momentum", cfg.momentum_screeners, momentum_counts),
        ("Noise", cfg.noise_screeners, noise_counts),
    )

    for category, screener_ids, counter in screener_groups:
        for scr_id in screener_ids:
            try:
                data = yf.screen(scr_id, count=cfg.screener_count)
                quotes = data.get("quotes", [])
                tickers = [q["symbol"] for q in quotes if q.get("symbol")]
                log.info("%s screener %-35s %d tickers", category, scr_id, len(tickers))
                counter.update(tickers)

            except Exception as exc:
                log.warning("%s screener %s failed: %s", category, scr_id, exc)

    # Merge: signal overlap is primary evidence; momentum/noise are tie-breakers.
    combined: Counter = Counter()
    for t in set(signal_counts) | set(momentum_counts) | set(noise_counts):
        combined[t] = (
            signal_counts.get(t, 0) * cfg.screener_signal_weight
            + momentum_counts.get(t, 0) * cfg.screener_momentum_weight
            + noise_counts.get(t, 0) * cfg.screener_noise_weight
        )

    return ScreenerTickerScores(
        combined_scores=combined,
        signal_counts=signal_counts,
        momentum_counts=momentum_counts,
        noise_counts=noise_counts,
    )


def _collect_candidates_overlap_legacy(cfg: PipelineConfig) -> list[str]:
    """
    Intersect news momentum with screener overlap to produce a ranked
    candidate list. Tickers must satisfy BOTH filters to be included.
    """
    log.info("── Stage 1: collecting candidates ───────────────────────────")

    news = collect_news_tickers(cfg)
    screeners = collect_screener_tickers(cfg)
    news_scores = news.recency_scores
    news_mentions = news.mention_counts
    screener_combined = screeners.combined_scores
    screener_signal = screeners.signal_counts

    # Hard filter: must appear in cfg.screener_min_overlap signal screeners
    screener_qualified = {
        t for t, n in screener_signal.items()
        if n >= cfg.screener_min_overlap
    }

    # Hard filter: must meet minimum raw news mentions
    # (score threshold is mentions * average_weight ≈ cfg.news_min_mentions for equal weight)
    news_qualified = {
        t for t, n in news_mentions.items()
        if n >= cfg.news_min_mentions
    }

    overlap = screener_qualified & news_qualified

    # Rank by combined score: normalised news + normalised screener
    max_news = max(news_scores.values(), default=1)
    max_scr = max(screener_combined.values(), default=1)

    ranked = sorted(
        overlap,
        key=lambda t: (
            news_scores[t] / max_news + screener_combined[t] / max_scr
        ),
        reverse=True,
    )

    log.info(
        "Candidates: %d screener-qualified, %d news-qualified, %d overlap",
        len(screener_qualified),
        len(news_qualified),
        len(ranked),
    )
    return ranked

def _norm(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return float(value) / float(maximum)


def _ordered_union(*iterables) -> list[str]:
    ordered: list[str] = []
    seen: set = set()
    for values in iterables:
        for ticker in values:
            if ticker in seen:
                continue
            seen.add(ticker)
            ordered.append(ticker)
    return ordered


def log_candidate_reasons(
    *,
    ranked: list[str],
    scored: dict[str, float],
    news: NewsTickerScores,
    screeners: ScreenerTickerScores,
    cfg: PipelineConfig,
) -> None:
    if not cfg.debug_candidate_reasons:
        return

    for ticker in ranked[: cfg.debug_candidate_top_n]:
        total_score = scored.get(ticker, 0.0)
        news_mentions = news.mention_counts.get(ticker, 0)
        signal_hits = screeners.signal_counts.get(ticker, 0)
        log.info(
            "Candidate %-8s score=%.3f news_mentions=%d news_recency=%.3f "
            "signal_hits=%d momentum_hits=%d noise_hits=%d "
            "passes_score=%s passes_news=%s passes_signal_screener=%s",
            ticker,
            total_score,
            news_mentions,
            news.recency_scores.get(ticker, 0.0),
            signal_hits,
            screeners.momentum_counts.get(ticker, 0),
            screeners.noise_counts.get(ticker, 0),
            total_score >= cfg.candidate_score_threshold,
            news_mentions >= cfg.news_min_mentions,
            signal_hits >= cfg.screener_min_overlap,
        )


def _rank_by_candidate_score(
    *,
    news: NewsTickerScores,
    screeners: ScreenerTickerScores,
    cfg: PipelineConfig,
) -> tuple[list[str], dict[str, float]]:
    all_tickers = _ordered_union(
        screeners.combined_scores,
        screeners.signal_counts,
        screeners.momentum_counts,
        screeners.noise_counts,
        news.recency_scores,
        news.mention_counts,
    )

    max_news_recency = max(news.recency_scores.values(), default=0)
    max_news_mentions = max(news.mention_counts.values(), default=0)
    max_screener_signal = max(screeners.signal_counts.values(), default=0)
    max_screener_momentum = max(screeners.momentum_counts.values(), default=0)
    max_screener_noise = max(screeners.noise_counts.values(), default=0)

    scored: dict[str, float] = {}
    for ticker in all_tickers:
        news_recency = _norm(news.recency_scores.get(ticker, 0), max_news_recency)
        news_mentions = _norm(news.mention_counts.get(ticker, 0), max_news_mentions)
        screener_signal = _norm(screeners.signal_counts.get(ticker, 0), max_screener_signal)
        screener_momentum = _norm(
            screeners.momentum_counts.get(ticker, 0),
            max_screener_momentum,
        )
        screener_noise = _norm(screeners.noise_counts.get(ticker, 0), max_screener_noise)

        total_score = (
            news_recency * cfg.news_recency_weight
            + news_mentions * cfg.news_mentions_weight
            + screener_signal * cfg.screener_signal_weight
            + screener_momentum * cfg.screener_momentum_weight
            + screener_noise * cfg.screener_noise_weight
        )

        passes_score = total_score >= cfg.candidate_score_threshold
        passes_news = news.mention_counts.get(ticker, 0) >= cfg.news_min_mentions
        # Hard screener confirmation is intentionally signal-only.
        passes_screener = screeners.signal_counts.get(ticker, 0) >= cfg.screener_min_overlap
        if passes_score or passes_news or passes_screener:
            scored[ticker] = total_score

    order_index = {ticker: index for index, ticker in enumerate(all_tickers)}
    ranked = sorted(scored, key=lambda t: (-scored[t], order_index.get(t, 0)))
    if cfg.candidate_pool_limit > 0:
        ranked = ranked[: cfg.candidate_pool_limit]

    log_candidate_reasons(
        ranked=ranked,
        scored=scored,
        news=news,
        screeners=screeners,
        cfg=cfg,
    )

    return ranked, scored


def _history_start_for_period(period: str, end_day: date) -> str:
    value = str(period or "").strip().lower()
    if value.endswith("y") and value[:-1].isdigit():
        return (pd.Timestamp(end_day) - pd.DateOffset(years=int(value[:-1]))).date().isoformat()
    if value.endswith("mo") and value[:-2].isdigit():
        return (pd.Timestamp(end_day) - pd.DateOffset(months=int(value[:-2]))).date().isoformat()
    if value.endswith("wk") and value[:-2].isdigit():
        return (pd.Timestamp(end_day) - pd.DateOffset(weeks=int(value[:-2]))).date().isoformat()
    if value.endswith("d") and value[:-1].isdigit():
        return (pd.Timestamp(end_day) - pd.DateOffset(days=int(value[:-1]))).date().isoformat()
    return (pd.Timestamp(end_day) - pd.DateOffset(years=1)).date().isoformat()


def _bars_rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    rename = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj_close": "Adj Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename)
    df.index = pd.to_datetime(df["bar_date"])
    columns = [col for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if col in df.columns]
    return df[columns].sort_index().dropna(how="all")


def _history_frame_is_fresh(hist: pd.DataFrame | None, start_date: str, end_date: str) -> bool:
    if hist is None or hist.empty:
        return False
    index = pd.to_datetime(hist.index)
    first = index.min().date()
    latest = index.max().date()
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    return first <= (start + timedelta(days=7)) and latest >= (end - timedelta(days=3))


def _cached_history_frame(
    ticker: str,
    *,
    start_date: str,
    end_date: str,
    provider: str,
) -> pd.DataFrame | None:
    if not _cache_enabled():
        return None

    rows = ta_cache.load_daily_bars(
        ticker,
        start_date,
        end_date,
        provider=provider,
    )
    hist = _bars_rows_to_frame(rows)
    if _history_frame_is_fresh(hist, start_date, end_date):
        return hist
    return None


def _to_float_or_none(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _history_frame_to_bar_rows(
    ticker: str,
    hist: pd.DataFrame | None,
) -> list[dict]:
    if hist is None or hist.empty:
        return []

    rows: list[dict] = []
    for index, row in hist.iterrows():
        timestamp = index.to_timestamp() if hasattr(index, "to_timestamp") else pd.Timestamp(index)
        bar_date = timestamp.date().isoformat()
        rows.append({
            "ticker": ticker,
            "bar_date": bar_date,
            "open": _to_float_or_none(row.get("Open")),
            "high": _to_float_or_none(row.get("High")),
            "low": _to_float_or_none(row.get("Low")),
            "close": _to_float_or_none(row.get("Close")),
            "adj_close": _to_float_or_none(row.get("Adj Close")),
            "volume": _to_float_or_none(row.get("Volume")),
        })
    return rows


def _upsert_history_frame(
    ticker: str,
    hist: pd.DataFrame | None,
    *,
    provider: str,
) -> None:
    if not _cache_enabled():
        return
    rows = _history_frame_to_bar_rows(ticker, hist)
    if rows:
        ta_cache.upsert_daily_bars(rows, provider=provider)


def _combine_ticker_history(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    clean = {ticker: frame for ticker, frame in frames.items() if frame is not None and not frame.empty}
    if not clean:
        return pd.DataFrame()
    return pd.concat(clean, axis=1)


def _download_history_batch(tickers: list[str], cfg: PipelineConfig) -> pd.DataFrame:
    end_date = date.today().isoformat()
    start_date = _history_start_for_period(cfg.cheap_history_period, date.today())

    if not _cache_enabled():
        return yf.download(
            tickers=tickers,
            period=cfg.cheap_history_period,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

    cached_frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for ticker in tickers:
        try:
            cached = _cached_history_frame(
                ticker,
                start_date=start_date,
                end_date=end_date,
                provider="yahoo",
            )
        except Exception as exc:
            log.warning("TA history cache read failed for %s: %s", ticker, exc)
            cached = None
        if cached is None:
            missing.append(ticker)
        else:
            cached_frames[ticker] = cached

    if missing:
        downloaded = yf.download(
            tickers=missing,
            period=cfg.cheap_history_period,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        for ticker in missing:
            hist = _history_for_ticker(downloaded, ticker)
            if hist is not None and not hist.empty:
                try:
                    _upsert_history_frame(ticker, hist, provider="yahoo")
                    cached = _cached_history_frame(
                        ticker,
                        start_date=start_date,
                        end_date=end_date,
                        provider="yahoo",
                    )
                    cached_frames[ticker] = cached if cached is not None else hist
                except Exception as exc:
                    log.warning("TA history cache write failed for %s: %s", ticker, exc)
                    cached_frames[ticker] = hist

    return _combine_ticker_history(cached_frames)


def _history_for_ticker(downloaded: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
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


def _cheap_market_signal(
    ticker: str,
    hist: pd.DataFrame,
    base_score: float,
    cfg: PipelineConfig,
) -> CheapMarketSignal | None:
    if hist is None or hist.empty or "Volume" not in hist.columns:
        return None

    price_column = "Adj Close" if "Adj Close" in hist.columns else "Close"
    if price_column not in hist.columns:
        return None

    close = pd.to_numeric(hist[price_column], errors="coerce").dropna()
    volume = pd.to_numeric(hist["Volume"], errors="coerce").reindex(close.index).dropna()
    close = close.reindex(volume.index).dropna()
    volume = volume.reindex(close.index).dropna()
    if close.empty or volume.empty:
        return None

    latest_price = float(close.iloc[-1])
    avg_dollar_volume = float((close * volume).tail(20).mean())
    if latest_price < cfg.cheap_min_price:
        return None
    if avg_dollar_volume < cfg.cheap_min_avg_dollar_volume:
        return None

    momentum_20d = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 20 else 0.0
    momentum_50d = float(close.iloc[-1] / close.iloc[-51] - 1) if len(close) > 50 else 0.0
    relative_volume = (
        float(volume.tail(5).mean() / volume.tail(20).mean())
        if len(volume) >= 20 and volume.tail(20).mean() > 0
        else 1.0
    )
    above_sma_200 = None
    if len(close) >= 200:
        sma_200 = float(close.rolling(200).mean().iloc[-1])
        above_sma_200 = latest_price > sma_200

    high_52w = (
        float(close.rolling(252).max().iloc[-1])
        if len(close) >= 252
        else float(close.max())
    )
    distance_from_52w_high = None
    if high_52w > 0:
        distance_from_52w_high = (latest_price - high_52w) / high_52w
        if distance_from_52w_high < -cfg.prefilter_52w_high_max_distance:
            return None

    trend_score = cfg.cheap_trend_weight if above_sma_200 else 0.0
    momentum_score = (
        max(momentum_20d, 0.0) * cfg.cheap_momentum_20d_weight
        + max(momentum_50d, 0.0) * cfg.cheap_momentum_50d_weight
    )
    relative_volume_score = (
        min(max(relative_volume - 1.0, 0.0), 2.0)
        * cfg.cheap_relative_volume_weight
    )
    liquidity_score = (
        min(avg_dollar_volume / cfg.cheap_min_avg_dollar_volume, 5.0)
        * cfg.cheap_liquidity_weight
    )
    score = base_score + trend_score + momentum_score + relative_volume_score + liquidity_score

    return CheapMarketSignal(
        ticker=ticker,
        score=score,
        latest_price=latest_price,
        avg_dollar_volume_20d=avg_dollar_volume,
        momentum_20d=momentum_20d,
        momentum_50d=momentum_50d,
        relative_volume_5d=relative_volume,
        above_sma_200=above_sma_200,
        distance_from_52w_high=distance_from_52w_high,
    )


def prefilter_candidates(
    candidates: list[str],
    candidate_scores: dict[str, float],
    cfg: PipelineConfig,
) -> list[str]:
    if not cfg.cheap_prefilter or not candidates:
        return candidates

    signals: list[CheapMarketSignal] = []
    for start in range(0, len(candidates), cfg.cheap_batch_size):
        batch = candidates[start : start + cfg.cheap_batch_size]
        try:
            downloaded = _download_history_batch(batch, cfg)
        except Exception as exc:
            log.warning("Cheap market prefilter batch failed: %s", exc)
            continue

        for ticker in batch:
            hist = _history_for_ticker(downloaded, ticker)
            signal = _cheap_market_signal(
                ticker=ticker,
                hist=hist,
                base_score=candidate_scores.get(ticker, 0.0),
                cfg=cfg,
            )
            if signal is not None:
                signals.append(signal)

    if not signals:
        log.warning("Cheap market prefilter returned no usable rows; using scored candidates unchanged.")
        return candidates

    signals.sort(key=lambda s: s.score, reverse=True)
    top_signals = signals[: cfg.cheap_top_n] if cfg.cheap_top_n > 0 else signals
    log.info(
        "Cheap prefilter: %d scored candidates -> %d liquid/momentum candidates",
        len(candidates),
        len(top_signals),
    )
    return [signal.ticker for signal in top_signals]


def collect_candidates(cfg: PipelineConfig) -> list[str]:
    """
    Build a broad ticker union, score it by news and categorized screener
    evidence, then optionally apply a cheap market-data prefilter before the
    expensive FinanceToolkit signal scan.

    Signal screeners can pass hard screener overlap. Momentum and noise
    screeners only provide weak tie-breaker context.
    """
    log.info("Stage 1: collecting candidates")

    news = collect_news_tickers(cfg)
    screeners = collect_screener_tickers(cfg)
    ranked, candidate_scores = _rank_by_candidate_score(
        news=news,
        screeners=screeners,
        cfg=cfg,
    )

    log.info(
        "Candidates: %d news tickers, %d screener tickers, %d scored candidates",
        len(news.mention_counts),
        len(screeners.combined_scores),
        len(ranked),
    )
    return prefilter_candidates(ranked, candidate_scores, cfg)


@dataclass
class SignalResult:
    ticker: str
    cross_dates: list
    cross_prices: list
    cross_colors: list
    forward_returns: list           # float | None per cross
    latest_signal: str | None       # "green" | "red" | None
    latest_signal_date: datetime | None
    spread: pd.Series | None = None
    close: pd.Series | None = None
    sma_50: pd.Series | None = None
    sma_200: pd.Series | None = None
    vwma_50: pd.Series | None = None
    regular_bullish_trend: pd.Series | None = None
    strong_bullish_confirmation: pd.Series | None = None
    regular_bearish_trend: pd.Series | None = None
    strong_bearish_confirmation: pd.Series | None = None
    pre_golden_score: float | None = None
    pre_golden_reasons: list[str] = field(default_factory=list)
    days_to_cross_estimate: float | None = None
    macd: pd.Series | None = None
    macd_signal: pd.Series | None = None
    macd_hist: pd.Series | None = None
    rsi: pd.Series | None = None
    bullish_impulse_score: float | None = None
    bullish_impulse_reasons: list[str] = field(default_factory=list)
    relative_strength_score: float | None = None
    relative_strength_reasons: list[str] = field(default_factory=list)
    liquidity_volume_score: float | None = None
    momentum_score: float | None = None
    momentum_reasons: list[str] = field(default_factory=list)
    final_bullish_score: float | None = None
    regime_label: str | None = None
    overbought_score: float | None = None
    overbought_status: str | None = None
    overbought_reasons: list[str] = field(default_factory=list)
    donchian_20_high: pd.Series | None = None
    donchian_55_high: pd.Series | None = None
    ret_z_20: pd.Series | None = None
    atr_14: pd.Series | None = None
    atr_move: pd.Series | None = None
    relative_volume: pd.Series | None = None
    ema_20: pd.Series | None = None
    extension_atr: pd.Series | None = None
    stoch_rsi: pd.Series | None = None
    adx: pd.Series | None = None
    obv: pd.Series | None = None
    bb_position: pd.Series | None = None
    distance_from_sma50: pd.Series | None = None
    ema_8: pd.Series | None = None
    ema_21: pd.Series | None = None
    ema_50: pd.Series | None = None
    error: str | None = None


@dataclass
class OverboughtMetrics:
    score: float
    status: str
    reasons: list[str]
    atr: pd.Series
    ema_20: pd.Series
    extension_atr: pd.Series
    stoch_rsi: pd.Series
    bb_position: pd.Series
    distance_from_sma50: pd.Series
    relative_volume: pd.Series


def _latest_valid_value(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def _series_value(series: pd.Series | None, offset: int = 0) -> float | None:
    if series is None or len(series) <= offset:
        return None
    value = series.iloc[-1 - offset]
    if pd.isna(value) or not np.isfinite(value):
        return None
    return float(value)


def _value_days_ago(series: pd.Series | None, days: int) -> float | None:
    return _series_value(series, days)


def compute_pre_golden_metrics(
    close: pd.Series,
    volume: pd.Series,
    sma_50: pd.Series,
    sma_200: pd.Series,
    vwma_50: pd.Series,
) -> tuple[float, list[str], float | None, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Score pre-golden-cross setups while SMA50 remains below SMA200."""
    spread = ((sma_50 - sma_200) / sma_200).replace([np.inf, -np.inf], np.nan)
    spread_slope_5d = spread.diff(5)
    spread_slope_10d = spread.diff(10)
    spread_accel = spread_slope_5d.diff(5)
    sma50_slope_10d = sma_50.pct_change(10).replace([np.inf, -np.inf], np.nan)
    sma200_slope_20d = sma_200.pct_change(20).replace([np.inf, -np.inf], np.nan)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)

    volume_5d = volume.rolling(5).mean()
    volume_20d = volume.rolling(20).mean()
    relative_volume = (volume_5d / volume_20d.replace(0, np.nan)).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    latest_spread = _series_value(spread)
    latest_spread_slope_5d = _series_value(spread_slope_5d)
    if latest_spread is None or latest_spread_slope_5d is None:
        return 0.0, [], None, macd, macd_signal, macd_hist, rsi

    if latest_spread >= 0:
        return 0.0, ["already crossed"], None, macd, macd_signal, macd_hist, rsi

    score = 0.0
    reasons: list[str] = []

    latest_spread_slope_10d = _series_value(spread_slope_10d)
    latest_spread_accel = _series_value(spread_accel)
    latest_sma50_slope_10d = _series_value(sma50_slope_10d)
    latest_sma200_slope_20d = _series_value(sma200_slope_20d)
    latest_close = _series_value(close)
    latest_sma_50 = _series_value(sma_50)
    latest_sma_200 = _series_value(sma_200)
    latest_vwma_50 = _series_value(vwma_50)
    latest_macd = _series_value(macd)
    latest_macd_signal = _series_value(macd_signal)
    latest_macd_hist = _series_value(macd_hist)
    macd_hist_3d_ago = _value_days_ago(macd_hist, 3)
    latest_rsi = _series_value(rsi)
    rsi_5d_ago = _value_days_ago(rsi, 5)
    latest_relative_volume = _series_value(relative_volume)

    if -0.01 <= latest_spread < 0:
        score += 3.0
        reasons.append("SMA50 is within 1% below SMA200")
    elif -0.02 <= latest_spread < -0.01:
        score += 2.0
        reasons.append("SMA50 is within 2% below SMA200")
    elif -0.04 <= latest_spread < -0.02:
        score += 1.0
        reasons.append("SMA50 is within 4% below SMA200")

    if latest_spread_slope_5d > 0:
        score += 2.0
        reasons.append("SMA spread is closing over 5 days")
    if latest_spread_slope_10d is not None and latest_spread_slope_10d > 0:
        score += 1.0
        reasons.append("SMA spread is closing over 10 days")
    if latest_spread_accel is not None and latest_spread_accel > 0:
        score += 1.0
        reasons.append("spread compression is accelerating")
    if latest_sma50_slope_10d is not None and latest_sma50_slope_10d > 0:
        score += 2.0
        reasons.append("SMA50 slope is positive")
    if latest_sma200_slope_20d is not None and latest_sma200_slope_20d < -0.03:
        score -= 1.5
        reasons.append("penalty: SMA200 falling sharply")
    if latest_close is not None and latest_sma_50 is not None and latest_close > latest_sma_50:
        score += 1.5
        reasons.append("price is above SMA50")
    if latest_close is not None and latest_sma_200 is not None and latest_close > latest_sma_200:
        score += 2.0
        reasons.append("price has reclaimed SMA200 before the cross")
    if latest_vwma_50 is not None and latest_sma_50 is not None and latest_vwma_50 > latest_sma_50:
        score += 1.0
        reasons.append("VWMA50 is above SMA50")
    if (
        latest_macd_hist is not None
        and macd_hist_3d_ago is not None
        and latest_macd_hist > macd_hist_3d_ago
    ):
        score += 1.5
        reasons.append("MACD histogram is rising")
    if (
        latest_macd is not None
        and latest_macd_signal is not None
        and latest_macd > latest_macd_signal
    ):
        score += 1.0
        reasons.append("MACD is above signal line")
    if (
        latest_rsi is not None
        and rsi_5d_ago is not None
        and 45 <= latest_rsi <= 70
        and latest_rsi > rsi_5d_ago
    ):
        score += 1.0
        reasons.append("RSI is recovering without being overbought")
    if latest_relative_volume is not None and latest_relative_volume > 1.1:
        score += 1.0
        reasons.append("recent volume is above average")

    daily_spread_velocity = latest_spread_slope_5d / 5
    if daily_spread_velocity > 0:
        days_to_cross = abs(latest_spread / daily_spread_velocity)
    else:
        days_to_cross = None

    return score, reasons, days_to_cross, macd, macd_signal, macd_hist, rsi


def compute_bullish_impulse_metrics(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    sma_50: pd.Series,
    sma_200: pd.Series,
    macd_hist: pd.Series,
    rsi: pd.Series,
    cfg: PipelineConfig,
) -> tuple[
    float,
    list[str],
    float,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Score fast bullish breakouts without using future bars."""
    close = pd.to_numeric(close, errors="coerce").sort_index()
    high = pd.to_numeric(high, errors="coerce").reindex(close.index)
    low = pd.to_numeric(low, errors="coerce").reindex(close.index)
    volume = pd.to_numeric(volume, errors="coerce").reindex(close.index)

    donchian_short_high = close.shift(1).rolling(cfg.impulse_donchian_short).max()
    donchian_long_high = close.shift(1).rolling(cfg.impulse_donchian_long).max()

    ret_1d = close.pct_change()
    ret_mean = ret_1d.rolling(cfg.impulse_return_z_window).mean()
    ret_std = ret_1d.rolling(cfg.impulse_return_z_window).std()
    ret_z = ((ret_1d - ret_mean) / ret_std.replace(0, np.nan)).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    atr = true_range.rolling(cfg.impulse_atr_period).mean()
    atr_move = ((close - prev_close) / atr.replace(0, np.nan)).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    volume_short = volume.rolling(cfg.impulse_relative_volume_short).mean()
    volume_long = volume.rolling(cfg.impulse_relative_volume_long).mean()
    relative_volume = (volume_short / volume_long.replace(0, np.nan)).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    ema_fast = close.ewm(span=cfg.impulse_ema_fast, adjust=False).mean()
    ema_mid = close.ewm(span=cfg.impulse_ema_mid, adjust=False).mean()
    ema_slow = close.ewm(span=cfg.impulse_ema_slow, adjust=False).mean()
    ema_mid_slope_5d = ema_mid.pct_change(5).replace([np.inf, -np.inf], np.nan)

    latest_close = _series_value(close)
    latest_short_high = _series_value(donchian_short_high)
    latest_long_high = _series_value(donchian_long_high)
    latest_ret_z = _series_value(ret_z)
    latest_atr_move = _series_value(atr_move)
    latest_relative_volume = _series_value(relative_volume)
    latest_ema_fast = _series_value(ema_fast)
    latest_ema_mid = _series_value(ema_mid)
    latest_ema_slow = _series_value(ema_slow)
    latest_ema_mid_slope_5d = _series_value(ema_mid_slope_5d)
    latest_sma_50 = _series_value(sma_50)
    latest_sma_200 = _series_value(sma_200)
    latest_macd_hist = _series_value(macd_hist)
    macd_hist_3d_ago = _value_days_ago(macd_hist, 3)
    latest_rsi = _series_value(rsi)

    score = 0.0
    reasons: list[str] = []

    if latest_close is not None and latest_short_high is not None and latest_close > latest_short_high:
        score += 2.0
        reasons.append(f"broke above {cfg.impulse_donchian_short}-day high")
    if latest_close is not None and latest_long_high is not None and latest_close > latest_long_high:
        score += 2.0
        reasons.append(f"broke above {cfg.impulse_donchian_long}-day high")
    if latest_ret_z is not None and latest_ret_z >= cfg.impulse_return_z_threshold:
        score += 2.0
        reasons.append("return z-score above threshold")
    if latest_atr_move is not None and latest_atr_move >= cfg.impulse_atr_move_threshold:
        score += 2.0
        reasons.append("ATR-adjusted price impulse")
    if (
        latest_relative_volume is not None
        and latest_relative_volume >= cfg.impulse_relative_volume_threshold
    ):
        score += 1.5
        reasons.append("relative volume expansion")
    if (
        latest_ema_fast is not None
        and latest_ema_mid is not None
        and latest_ema_fast > latest_ema_mid
    ):
        score += 1.5
        reasons.append(f"EMA{cfg.impulse_ema_fast} above EMA{cfg.impulse_ema_mid}")
    if latest_close is not None and latest_ema_mid is not None and latest_close > latest_ema_mid:
        score += 1.0
        reasons.append(f"price above EMA{cfg.impulse_ema_mid}")
    if latest_ema_mid_slope_5d is not None and latest_ema_mid_slope_5d > 0:
        score += 1.0
        reasons.append(f"EMA{cfg.impulse_ema_mid} rising")
    if (
        latest_ema_mid is not None
        and latest_ema_slow is not None
        and latest_ema_mid > latest_ema_slow
    ):
        score += 1.0
        reasons.append(f"EMA{cfg.impulse_ema_mid} above EMA{cfg.impulse_ema_slow}")
    if latest_close is not None and latest_sma_200 is not None and latest_close > latest_sma_200:
        score += 2.0
        reasons.append("price reclaimed SMA200")
    if latest_close is not None and latest_sma_50 is not None and latest_close > latest_sma_50:
        score += 1.0
        reasons.append("price above SMA50")
    if (
        latest_macd_hist is not None
        and macd_hist_3d_ago is not None
        and latest_macd_hist > macd_hist_3d_ago
    ):
        score += 1.0
        reasons.append("MACD histogram rising")
    if latest_rsi is not None and 50 <= latest_rsi <= 75:
        score += 1.0
        reasons.append("RSI in bullish range")

    liquidity_volume_score = 0.0
    if latest_relative_volume is not None:
        scaled_volume = latest_relative_volume / cfg.impulse_relative_volume_threshold
        liquidity_volume_score = max(0.0, min(10.0, scaled_volume * 5.0))

    return (
        score,
        reasons,
        liquidity_volume_score,
        donchian_short_high,
        donchian_long_high,
        ret_z,
        atr,
        atr_move,
        relative_volume,
        ema_fast,
        ema_mid,
        ema_slow,
    )


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Compute simple rolling ATR from high/low/close series."""
    close = pd.to_numeric(close, errors="coerce").sort_index()
    high = pd.to_numeric(high, errors="coerce").reindex(close.index)
    low = pd.to_numeric(low, errors="coerce").reindex(close.index)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    return true_range.rolling(period).mean().replace([np.inf, -np.inf], np.nan)


def compute_stoch_rsi(rsi: pd.Series, period: int) -> pd.Series:
    """Compute stochastic RSI, leaving flat/insufficient windows as NaN."""
    rsi = pd.to_numeric(rsi, errors="coerce").sort_index()
    rsi_min = rsi.rolling(period).min()
    rsi_max = rsi.rolling(period).max()
    denominator = (rsi_max - rsi_min).replace(0, np.nan)
    return ((rsi - rsi_min) / denominator).replace([np.inf, -np.inf], np.nan)


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute Average Directional Index using Wilder smoothing.

    Returns a Series aligned to close, with NaN where there is insufficient data.
    """
    close = pd.to_numeric(close, errors="coerce").sort_index()
    high = pd.to_numeric(high, errors="coerce").reindex(close.index)
    low = pd.to_numeric(low, errors="coerce").reindex(close.index)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=close.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=close.index,
    )

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    alpha = 1 / period
    atr = true_range.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean().replace(
        [np.inf, -np.inf],
        np.nan,
    )


def compute_obv(
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Compute On-Balance Volume aligned to close."""
    close = pd.to_numeric(close, errors="coerce").sort_index()
    volume = pd.to_numeric(volume, errors="coerce").reindex(close.index).fillna(0.0)
    direction = np.sign(close.diff()).fillna(0.0)
    obv_delta = volume.where(direction > 0, 0.0)
    obv_delta = obv_delta.where(direction >= 0, -volume)
    return obv_delta.cumsum().replace([np.inf, -np.inf], np.nan)


def compute_momentum_metrics(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    cfg: PipelineConfig,
) -> tuple[float, list[str], pd.Series, pd.Series]:
    """Score momentum quality using ADX trend strength and OBV confirmation."""
    close = pd.to_numeric(close, errors="coerce").sort_index()
    high = pd.to_numeric(high, errors="coerce").reindex(close.index)
    low = pd.to_numeric(low, errors="coerce").reindex(close.index)
    volume = pd.to_numeric(volume, errors="coerce").reindex(close.index)

    adx = compute_adx(high=high, low=low, close=close, period=cfg.adx_period)
    obv = compute_obv(close=close, volume=volume)
    score = 0.0
    reasons: list[str] = []

    latest_adx = _latest_valid_value(adx)
    adx_5d_ago = _value_days_ago(adx, 5)
    latest_obv = _latest_valid_value(obv)
    obv_window_ago = _value_days_ago(obv, cfg.obv_slope_window)
    latest_close = _latest_valid_value(close)
    close_window_ago = _value_days_ago(close, cfg.obv_slope_window)

    if latest_adx is not None and latest_adx > cfg.adx_trending_threshold:
        score += 2.0
        reasons.append(f"ADX above {cfg.adx_trending_threshold:g}")
    if latest_adx is not None and latest_adx > cfg.adx_strong_threshold:
        score += 1.0
        reasons.append(f"ADX above {cfg.adx_strong_threshold:g}")
    if latest_adx is not None and adx_5d_ago is not None and latest_adx > adx_5d_ago:
        score += 1.0
        reasons.append("ADX rising over 5 days")

    if latest_obv is not None and obv_window_ago is not None and latest_obv > obv_window_ago:
        score += 1.5
        reasons.append(f"OBV rising over {cfg.obv_slope_window} days")

    obv_high_window = obv.dropna().tail(cfg.obv_high_window)
    if latest_obv is not None and not obv_high_window.empty and latest_obv >= float(obv_high_window.max()):
        score += 1.5
        reasons.append(f"OBV at {cfg.obv_high_window}-day high")

    if (
        latest_close is not None
        and close_window_ago is not None
        and latest_obv is not None
        and obv_window_ago is not None
        and latest_close > close_window_ago
        and latest_obv < obv_window_ago
    ):
        score -= 2.0
        reasons.append("OBV divergence: price rising while OBV falls")

    if not reasons:
        reasons.append("momentum neutral")
    return score, reasons, adx, obv


def classify_overbought(score: float, cfg: PipelineConfig) -> str:
    if score >= cfg.severely_overbought_threshold:
        return "severely_overbought"
    if score >= cfg.overbought_threshold:
        return "overbought"
    if score >= cfg.moderately_extended_threshold:
        return "moderately_extended"
    return "not_overbought"


def compute_overbought_metrics(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    sma_50: pd.Series,
    rsi: pd.Series,
    cfg: PipelineConfig,
) -> OverboughtMetrics:
    """Score entry-risk extension without changing bullish/bearish regime."""
    close = pd.to_numeric(close, errors="coerce").sort_index()
    high = pd.to_numeric(high, errors="coerce").reindex(close.index)
    low = pd.to_numeric(low, errors="coerce").reindex(close.index)
    volume = pd.to_numeric(volume, errors="coerce").reindex(close.index)
    sma_50 = pd.to_numeric(sma_50, errors="coerce").reindex(close.index)
    rsi = pd.to_numeric(rsi, errors="coerce").reindex(close.index)

    atr = compute_atr(high, low, close, cfg.atr_period)
    ema_20 = close.ewm(span=cfg.ema_extension_period, adjust=False).mean()
    extension_atr = ((close - ema_20) / atr.replace(0, np.nan)).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    stoch_rsi = compute_stoch_rsi(rsi, cfg.stoch_rsi_period).reindex(close.index)
    rolling_mean = close.rolling(cfg.bollinger_period).mean()
    rolling_std = close.rolling(cfg.bollinger_period).std()
    upper_band = rolling_mean + cfg.bollinger_std_mult * rolling_std
    lower_band = rolling_mean - cfg.bollinger_std_mult * rolling_std
    band_width = (upper_band - lower_band).replace(0, np.nan)
    bb_position = ((close - lower_band) / band_width).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    distance_from_sma50 = ((close - sma_50) / sma_50.replace(0, np.nan)).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    volume_short = volume.rolling(cfg.impulse_relative_volume_short).mean()
    volume_long = volume.rolling(cfg.impulse_relative_volume_long).mean()
    relative_volume = (volume_short / volume_long.replace(0, np.nan)).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if high.dropna().empty or low.dropna().empty:
        return OverboughtMetrics(
            score=0.0,
            status="not_overbought",
            reasons=["insufficient high/low data for ATR-based extension"],
            atr=atr,
            ema_20=ema_20,
            extension_atr=extension_atr,
            stoch_rsi=stoch_rsi,
            bb_position=bb_position,
            distance_from_sma50=distance_from_sma50,
            relative_volume=relative_volume,
        )

    score = 0.0
    reasons: list[str] = []
    latest_rsi = _latest_valid_value(rsi)
    latest_stoch_rsi = _latest_valid_value(stoch_rsi)
    latest_extension_atr = _latest_valid_value(extension_atr)
    latest_distance_from_sma50 = _latest_valid_value(distance_from_sma50)
    latest_bb_position = _latest_valid_value(bb_position)
    latest_relative_volume = _latest_valid_value(relative_volume)

    if latest_rsi is not None and latest_rsi > cfg.overbought_rsi_level:
        score += 1.0
        reasons.append(f"RSI above {cfg.overbought_rsi_level:g}")
    if latest_rsi is not None and latest_rsi > cfg.severe_overbought_rsi_level:
        score += 2.0
        reasons.append(f"RSI above {cfg.severe_overbought_rsi_level:g}")
    if (
        latest_stoch_rsi is not None
        and latest_stoch_rsi > cfg.overbought_stoch_rsi_level
    ):
        score += 1.0
        reasons.append(f"StochRSI above {cfg.overbought_stoch_rsi_level:.2f}")
    if (
        latest_stoch_rsi is not None
        and latest_stoch_rsi > cfg.severe_overbought_stoch_rsi_level
    ):
        score += 1.0
        reasons.append(f"StochRSI above {cfg.severe_overbought_stoch_rsi_level:.2f}")
    if (
        latest_extension_atr is not None
        and latest_extension_atr > cfg.overbought_extension_atr
    ):
        score += 1.5
        reasons.append(
            f"Price is more than {cfg.overbought_extension_atr:g} ATR above "
            f"EMA{cfg.ema_extension_period}"
        )
    if (
        latest_extension_atr is not None
        and latest_extension_atr > cfg.severe_overbought_extension_atr
    ):
        score += 2.0
        reasons.append(
            f"Price is more than {cfg.severe_overbought_extension_atr:g} ATR above "
            f"EMA{cfg.ema_extension_period}"
        )
    if (
        latest_extension_atr is not None
        and latest_extension_atr > cfg.exhaustion_extension_atr
    ):
        score += 1.0
        reasons.append(
            f"Price is more than {cfg.exhaustion_extension_atr:g} ATR above "
            f"EMA{cfg.ema_extension_period}"
        )
    if (
        latest_distance_from_sma50 is not None
        and latest_distance_from_sma50 > cfg.overbought_sma50_distance
    ):
        score += 1.0
        reasons.append(
            f"Price is more than {cfg.overbought_sma50_distance:.0%} above SMA50"
        )
    if (
        latest_distance_from_sma50 is not None
        and latest_distance_from_sma50 > cfg.severe_overbought_sma50_distance
    ):
        score += 2.0
        reasons.append(
            f"Price is more than {cfg.severe_overbought_sma50_distance:.0%} above SMA50"
        )
    if (
        latest_bb_position is not None
        and latest_bb_position > cfg.overbought_bb_position
    ):
        score += 1.0
        reasons.append("Price is above upper Bollinger Band")
    if (
        latest_bb_position is not None
        and latest_bb_position > cfg.severe_overbought_bb_position
    ):
        score += 1.0
        reasons.append("Price is severely above upper Bollinger Band")
    if (
        latest_relative_volume is not None
        and latest_extension_atr is not None
        and latest_relative_volume > cfg.volume_climax_relative_volume
        and latest_extension_atr > cfg.volume_climax_extension_atr
    ):
        score += 2.0
        reasons.append("Possible volume climax with price extension")

    status = classify_overbought(score, cfg)
    return OverboughtMetrics(
        score=score,
        status=status,
        reasons=reasons,
        atr=atr,
        ema_20=ema_20,
        extension_atr=extension_atr,
        stoch_rsi=stoch_rsi,
        bb_position=bb_position,
        distance_from_sma50=distance_from_sma50,
        relative_volume=relative_volume,
    )


def _normalize_datetime_index(series: pd.Series) -> pd.Series:
    normalized = series.copy()
    normalized.index = pd.to_datetime(normalized.index)
    if getattr(normalized.index, "tz", None) is not None:
        normalized.index = normalized.index.tz_localize(None)
    return normalized.sort_index()


def _fetch_benchmark_closes(
    cfg: PipelineConfig,
    start: str,
    end: str,
) -> dict[str, pd.Series]:
    """Download configured RS benchmarks once per pipeline run."""
    closes: dict[str, pd.Series] = {}
    if not cfg.relative_strength_benchmarks:
        return closes

    for benchmark in cfg.relative_strength_benchmarks:
        try:
            downloaded = yf.download(
                benchmark,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            bench_hist = _history_for_ticker(downloaded, benchmark)
            if bench_hist is None or bench_hist.empty:
                log.warning("Benchmark %s returned no price data", benchmark)
                continue
            price_col = "Adj Close" if "Adj Close" in bench_hist.columns else "Close"
            if price_col not in bench_hist.columns:
                log.warning("Benchmark %s has no close column", benchmark)
                continue
            bench_close = _normalize_datetime_index(
                pd.to_numeric(bench_hist[price_col], errors="coerce").dropna()
            )
            if not bench_close.empty:
                closes[benchmark] = bench_close
        except Exception as exc:
            log.warning("Benchmark download failed for %s: %s", benchmark, exc)

    log.info(
        "Cached %d/%d relative-strength benchmarks",
        len(closes),
        len(cfg.relative_strength_benchmarks),
    )
    return closes


def compute_relative_strength_score(
    ticker: str,
    close: pd.Series,
    cfg: PipelineConfig,
    *,
    benchmark_closes: dict[str, pd.Series] | None = None,
) -> tuple[float, list[str]]:
    """Compare latest ticker relative strength against configured benchmarks."""
    close = _normalize_datetime_index(pd.to_numeric(close, errors="coerce").dropna())
    if close.empty or not cfg.relative_strength_benchmarks:
        return 0.0, ["relative strength unavailable"]

    start = close.index.min().date().isoformat()
    end = (close.index.max().date() + timedelta(days=1)).isoformat()
    scores: list[float] = []
    reasons: list[str] = []

    for benchmark in cfg.relative_strength_benchmarks:
        try:
            bench_close = None
            if benchmark_closes and benchmark in benchmark_closes:
                bench_close = benchmark_closes[benchmark]
            else:
                downloaded = yf.download(
                    benchmark,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                bench_hist = _history_for_ticker(downloaded, benchmark)
                if bench_hist is None or bench_hist.empty:
                    continue
                price_col = "Adj Close" if "Adj Close" in bench_hist.columns else "Close"
                if price_col not in bench_hist.columns:
                    continue
                bench_close = _normalize_datetime_index(
                    pd.to_numeric(bench_hist[price_col], errors="coerce").dropna()
                )
            if bench_close is None or bench_close.empty:
                continue
            aligned_benchmark = bench_close.reindex(close.index).ffill()
            aligned = pd.DataFrame(
                {"ticker": close, "benchmark": aligned_benchmark}
            ).dropna()
            if len(aligned) <= cfg.relative_strength_window:
                continue
            rs = (aligned["ticker"] / aligned["benchmark"].replace(0, np.nan)).replace(
                [np.inf, -np.inf],
                np.nan,
            )
            rs_sma = rs.rolling(cfg.relative_strength_window).mean()
            latest_rs = _series_value(rs)
            latest_rs_sma = _series_value(rs_sma)
            rs_change = _series_value(rs.pct_change(cfg.relative_strength_window))
            if latest_rs is None or latest_rs_sma is None:
                continue

            benchmark_score = 0.0
            if latest_rs > latest_rs_sma:
                benchmark_score += 1.0
                reasons.append(f"RS above {benchmark} average")
            if rs_change is not None and rs_change > 0:
                benchmark_score += 1.0
                reasons.append(f"{cfg.relative_strength_window}d RS rising vs {benchmark}")
            scores.append(benchmark_score)
        except Exception as exc:
            log.debug("Relative strength failed for %s vs %s: %s", ticker, benchmark, exc)

    if not scores:
        return 0.0, ["relative strength unavailable"]

    return float(sum(scores) / len(scores)), reasons or ["relative strength neutral"]


def classify_regime(
    *,
    latest_close: float | None,
    latest_sma_50: float | None,
    latest_sma_200: float | None,
    latest_spread: float | None,
    bullish_impulse_score: float | None,
    pre_golden_score: float | None,
    cfg: PipelineConfig,
) -> str:
    if (
        latest_sma_50 is not None
        and latest_sma_200 is not None
        and latest_close is not None
        and latest_sma_50 > latest_sma_200
        and latest_close > latest_sma_200
    ):
        return "confirmed_bullish"
    if bullish_impulse_score is not None and bullish_impulse_score >= cfg.impulse_min_score:
        return "bullish_impulse"
    if (
        latest_close is not None
        and latest_sma_50 is not None
        and latest_sma_200 is not None
        and latest_close > latest_sma_200
        and latest_sma_50 < latest_sma_200
    ):
        return "bullish_transition"
    if (
        pre_golden_score is not None
        and latest_spread is not None
        and pre_golden_score >= cfg.transition_min_score
        and latest_spread < 0
    ):
        return "pre_golden_setup"
    if (
        latest_sma_50 is not None
        and latest_sma_200 is not None
        and latest_close is not None
        and latest_sma_50 < latest_sma_200
        and latest_close < latest_sma_200
    ):
        return "bearish_or_weak"
    return "neutral"


def _ensure_datetime_index(series: pd.Series) -> pd.Series:
    if hasattr(series.index, "to_timestamp"):
        series.index = series.index.to_timestamp()
    else:
        series.index = pd.to_datetime(series.index)
    return series


def _load_or_fetch_signal_history(
    ticker: str,
    cfg: PipelineConfig,
    *,
    end_date: str,
) -> pd.DataFrame | None:
    provider = "financetoolkit"
    if _cache_enabled():
        try:
            cached = _cached_history_frame(
                ticker,
                start_date=cfg.start_date,
                end_date=end_date,
                provider=provider,
            )
            if cached is not None:
                log.info("Signal history cache hit: %s", ticker)
                return cached
        except Exception as exc:
            log.warning("TA signal history cache read failed for %s: %s", ticker, exc)

    api_key = os.getenv("FMP_API_KEY")
    tk = Toolkit(
        tickers=[ticker],
        api_key=api_key,
        start_date=cfg.start_date,
        end_date=end_date,
    )
    hist_raw = tk.get_historical_data()
    hist = _history_for_ticker(hist_raw, ticker)
    if hist is not None and not hist.empty and _cache_enabled():
        try:
            _upsert_history_frame(ticker, hist, provider=provider)
        except Exception as exc:
            log.warning("TA signal history cache write failed for %s: %s", ticker, exc)
    return hist


def compute_signals(
    ticker: str,
    cfg: PipelineConfig,
    *,
    benchmark_closes: dict[str, pd.Series] | None = None,
) -> SignalResult:
    """Compute all MA signals for a single ticker. Returns SignalResult."""
    empty = SignalResult(
        ticker=ticker,
        cross_dates=[],
        cross_prices=[],
        cross_colors=[],
        forward_returns=[],
        latest_signal=None,
        latest_signal_date=None,
    )

    try:
        end_date = date.today().isoformat()
        hist = _load_or_fetch_signal_history(ticker, cfg, end_date=end_date)
        if hist is None or hist.empty:
            empty.error = "no price data"
            return empty

        close = pd.to_numeric(hist["Close"], errors="coerce").sort_index()
        volume_source = hist["Volume"] if "Volume" in hist.columns else pd.Series(np.nan, index=hist.index)
        high_source = hist["High"] if "High" in hist.columns else pd.Series(np.nan, index=hist.index)
        low_source = hist["Low"] if "Low" in hist.columns else pd.Series(np.nan, index=hist.index)
        volume = pd.to_numeric(volume_source, errors="coerce").sort_index()
        high = pd.to_numeric(high_source, errors="coerce").sort_index()
        low = pd.to_numeric(low_source, errors="coerce").sort_index()

        close = _ensure_datetime_index(close)
        volume = _ensure_datetime_index(volume).reindex(close.index)
        high = _ensure_datetime_index(high).reindex(close.index)
        low = _ensure_datetime_index(low).reindex(close.index)

        if close.dropna().empty:
            empty.error = "no price data"
            return empty

        sma_50 = close.rolling(window=cfg.short_sma_period).mean()
        sma_200 = close.rolling(window=cfg.long_sma_period).mean()
        vwma_50 = (
            (close * volume).rolling(window=cfg.vwma_period).sum()
            / volume.rolling(window=cfg.vwma_period).sum()
        )

        spread = (sma_50 - sma_200) / sma_200
        prev_spread = spread.shift(1)

        (
            pre_golden_score,
            pre_golden_reasons,
            days_to_cross_estimate,
            macd,
            macd_signal,
            macd_hist,
            rsi,
        ) = compute_pre_golden_metrics(
            close=close,
            volume=volume,
            sma_50=sma_50,
            sma_200=sma_200,
            vwma_50=vwma_50,
        )
        (
            bullish_impulse_score,
            bullish_impulse_reasons,
            liquidity_volume_score,
            donchian_20_high,
            donchian_55_high,
            ret_z_20,
            atr_14,
            atr_move,
            relative_volume,
            ema_8,
            ema_21,
            ema_50,
        ) = compute_bullish_impulse_metrics(
            close=close,
            high=high,
            low=low,
            volume=volume,
            sma_50=sma_50,
            sma_200=sma_200,
            macd_hist=macd_hist,
            rsi=rsi,
            cfg=cfg,
        )
        overbought_metrics = compute_overbought_metrics(
            close=close,
            high=high,
            low=low,
            volume=volume,
            sma_50=sma_50,
            rsi=rsi,
            cfg=cfg,
        )
        momentum_score, momentum_reasons, adx, obv = compute_momentum_metrics(
            close=close,
            high=high,
            low=low,
            volume=volume,
            cfg=cfg,
        )
        relative_strength_score, relative_strength_reasons = compute_relative_strength_score(
            ticker=ticker,
            close=close,
            cfg=cfg,
            benchmark_closes=benchmark_closes,
        )
        final_bullish_score = (
            cfg.bullish_impulse_weight * bullish_impulse_score
            + cfg.pre_golden_weight * pre_golden_score
            + cfg.relative_strength_weight * relative_strength_score
            + cfg.liquidity_volume_weight * liquidity_volume_score
            + cfg.momentum_weight * momentum_score
        )
        latest_close = _series_value(close)
        latest_sma_50 = _series_value(sma_50)
        latest_sma_200 = _series_value(sma_200)
        latest_spread = _series_value(spread)
        regime_label = classify_regime(
            latest_close=latest_close,
            latest_sma_50=latest_sma_50,
            latest_sma_200=latest_sma_200,
            latest_spread=latest_spread,
            bullish_impulse_score=bullish_impulse_score,
            pre_golden_score=pre_golden_score,
            cfg=cfg,
        )

        regular_bullish_trend = (close > sma_200) & (sma_50 > sma_200)
        strong_bullish_confirmation = (close > sma_200) & (sma_50 > sma_200) & (vwma_50 > sma_50)
        regular_bearish_trend = (close < sma_200) & (sma_50 < sma_200)
        strong_bearish_confirmation = (close < sma_200) & (sma_50 < sma_200) & (vwma_50 < sma_50)

        cross_dates, cross_prices, cross_colors = [], [], []
        last_confirmed_regime = None

        valid_positions = np.where(sma_200.notna())[0]
        first_valid = valid_positions[0] if len(valid_positions) > 0 else None

        for i in range(1, len(spread)):
            if first_valid is not None and i < first_valid + cfg.confirmation_days:
                continue

            future_window = spread.iloc[i + 1 : i + 1 + cfg.confirmation_days]
            if len(future_window) < cfg.confirmation_days:
                continue
            if pd.isna(prev_spread.iloc[i]) or pd.isna(spread.iloc[i]):
                continue

            is_golden = prev_spread.iloc[i] <= 0 and spread.iloc[i] > 0
            is_death = prev_spread.iloc[i] >= 0 and spread.iloc[i] < 0

            if is_golden and last_confirmed_regime == "bullish":
                last_confirmed_regime = None

            if is_death and last_confirmed_regime == "bearish":
                last_confirmed_regime = None

            confirmed_golden = (
                is_golden
                and (future_window > 0).all()
                and future_window.iloc[-1] > cfg.min_spread
            )
            confirmed_death = (
                is_death
                and (future_window < 0).all()
                and future_window.iloc[-1] < -cfg.min_spread
            )

            if confirmed_golden and last_confirmed_regime != "bullish":
                p0 = (sma_50.iloc[i - 1] + sma_200.iloc[i - 1]) / 2
                p1 = (sma_50.iloc[i] + sma_200.iloc[i]) / 2
                cross_dates.append(spread.index[i])
                cross_prices.append((p0 + p1) / 2)
                cross_colors.append("green")
                last_confirmed_regime = "bullish"

            elif confirmed_death and last_confirmed_regime != "bearish":
                p0 = (sma_50.iloc[i - 1] + sma_200.iloc[i - 1]) / 2
                p1 = (sma_50.iloc[i] + sma_200.iloc[i]) / 2
                cross_dates.append(spread.index[i])
                cross_prices.append((p0 + p1) / 2)
                cross_colors.append("red")
                last_confirmed_regime = "bearish"

        # Forward returns
        fwd_returns = []
        for d in cross_dates:
            idx = close.index.get_loc(d)
            if idx + cfg.forward_days < len(close):
                ret = (close.iloc[idx + cfg.forward_days] - close.iloc[idx]) / close.iloc[idx]
                fwd_returns.append(ret)
            else:
                fwd_returns.append(None)

        latest_signal = cross_colors[-1] if cross_colors else None
        latest_date = cross_dates[-1] if cross_dates else None
        log.info(
            "%s regime=%s final_score=%.2f impulse=%.2f pre_golden=%.2f momentum=%.2f overbought=%s %.2f",
            ticker,
            regime_label,
            final_bullish_score,
            bullish_impulse_score,
            pre_golden_score,
            momentum_score,
            overbought_metrics.status,
            overbought_metrics.score,
        )

        return SignalResult(
            ticker=ticker,
            cross_dates=cross_dates,
            cross_prices=cross_prices,
            cross_colors=cross_colors,
            forward_returns=fwd_returns,
            latest_signal=latest_signal,
            latest_signal_date=latest_date,
            spread=spread,
            close=close,
            sma_50=sma_50,
            sma_200=sma_200,
            vwma_50=vwma_50,
            regular_bullish_trend=regular_bullish_trend,
            strong_bullish_confirmation=strong_bullish_confirmation,
            regular_bearish_trend=regular_bearish_trend,
            strong_bearish_confirmation=strong_bearish_confirmation,
            pre_golden_score=pre_golden_score,
            pre_golden_reasons=pre_golden_reasons,
            days_to_cross_estimate=days_to_cross_estimate,
            macd=macd,
            macd_signal=macd_signal,
            macd_hist=macd_hist,
            rsi=rsi,
            bullish_impulse_score=bullish_impulse_score,
            bullish_impulse_reasons=bullish_impulse_reasons,
            relative_strength_score=relative_strength_score,
            relative_strength_reasons=relative_strength_reasons,
            liquidity_volume_score=liquidity_volume_score,
            momentum_score=momentum_score,
            momentum_reasons=momentum_reasons,
            final_bullish_score=final_bullish_score,
            regime_label=regime_label,
            overbought_score=overbought_metrics.score,
            overbought_status=overbought_metrics.status,
            overbought_reasons=overbought_metrics.reasons,
            donchian_20_high=donchian_20_high,
            donchian_55_high=donchian_55_high,
            ret_z_20=ret_z_20,
            atr_14=overbought_metrics.atr,
            atr_move=atr_move,
            relative_volume=overbought_metrics.relative_volume,
            ema_20=overbought_metrics.ema_20,
            extension_atr=overbought_metrics.extension_atr,
            stoch_rsi=overbought_metrics.stoch_rsi,
            adx=adx,
            obv=obv,
            bb_position=overbought_metrics.bb_position,
            distance_from_sma50=overbought_metrics.distance_from_sma50,
            ema_8=ema_8,
            ema_21=ema_21,
            ema_50=ema_50,
        )

    except Exception as exc:
        log.warning("compute_signals failed for %s: %s", ticker, exc)
        empty.error = str(exc)
        return empty

def rank_candidates(
    results: list[SignalResult],
    require_bullish: bool = False,
) -> list[SignalResult]:
    """
    Rank tickers by fast bullish score while retaining pre-golden setups.

    require_bullish is retained for backward compatibility with older callers.
    """
    log.info("── Stage 3: ranking bullish candidates ─────────────")

    def latest_spread_for(r: SignalResult) -> float | None:
        return _latest_valid_value(r.spread)

    def adjusted_score(r: SignalResult) -> float:
        base = r.final_bullish_score or 0.0
        overbought_score = r.overbought_score or 0.0
        penalty = min(overbought_score * 0.35, 3.0)
        if r.overbought_status == "severely_overbought":
            penalty += 1.5
        return base - penalty

    allowed_regimes = {
        "bullish_impulse",
        "bullish_transition",
        "pre_golden_setup",
        "confirmed_bullish",
    }
    filtered = [
        r for r in results
        if not r.error
        and r.regime_label in allowed_regimes
        and r.final_bullish_score is not None
        and r.final_bullish_score > 0
    ]

    def sort_key(r: SignalResult):
        latest_spread = latest_spread_for(r)
        closest_negative_spread = (
            -abs(latest_spread)
            if latest_spread is not None and latest_spread < 0
            else -9999.0
        )
        days_to_cross = (
            r.days_to_cross_estimate
            if r.days_to_cross_estimate is not None
            else 9999
        )
        overbought_score = r.overbought_score or 0.0
        return (
            adjusted_score(r),
            r.final_bullish_score or 0.0,
            r.bullish_impulse_score or 0.0,
            r.pre_golden_score or 0.0,
            closest_negative_spread,
            -days_to_cross,
            -overbought_score,
        )

    ranked = sorted(filtered, key=sort_key, reverse=True)
    log.info(
        "%d tickers with bullish regimes, returning top candidates",
        len(ranked),
    )
    return ranked


def plot_signal_result(
    result: SignalResult,
    cfg: PipelineConfig,
    *,
    defer_show: bool = False,
) -> None:
    """Render the full chart for a pre-computed SignalResult."""
    r = result
    ticker = r.ticker
    end_date = date.today().isoformat()

    sns.set_theme(style="darkgrid")
    fig, (ax, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [3, 1, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax2.set_facecolor("black")
    ax3.set_facecolor("black")

    # Price lines
    sns.lineplot(x=r.close.index, y=r.close, ax=ax, color="steelblue", linewidth=1.2, label="Close")
    sns.lineplot(x=r.sma_50.index, y=r.sma_50, ax=ax, color="orange", linewidth=1.5, linestyle="--", label=f"SMA({cfg.short_sma_period})")
    sns.lineplot(x=r.sma_200.index, y=r.sma_200, ax=ax, color="#facc15", linewidth=1.8, label=f"SMA({cfg.long_sma_period})")
    sns.lineplot(x=r.vwma_50.index, y=r.vwma_50, ax=ax, color="crimson", linewidth=1.5, linestyle="-.", label=f"VWMA({cfg.vwma_period})")
    if r.ema_8 is not None:
        sns.lineplot(x=r.ema_8.index, y=r.ema_8, ax=ax, color="#93c5fd", linewidth=0.9, alpha=0.75, label=f"EMA({cfg.impulse_ema_fast})")
    if r.ema_21 is not None:
        sns.lineplot(x=r.ema_21.index, y=r.ema_21, ax=ax, color="#a7f3d0", linewidth=0.9, alpha=0.75, linestyle="--", label=f"EMA({cfg.impulse_ema_mid})")
    if r.ema_50 is not None:
        sns.lineplot(x=r.ema_50.index, y=r.ema_50, ax=ax, color="#c4b5fd", linewidth=0.9, alpha=0.75, linestyle=":", label=f"EMA({cfg.impulse_ema_slow})")

    # Regime shading
    ymin, ymax = ax.get_ylim()
    ax.fill_between(r.close.index, ymin, ymax, where=r.regular_bullish_trend.fillna(False), color="mediumturquoise", alpha=0.5, label="Regular bullish")
    ax.fill_between(r.close.index, ymin, ymax, where=r.strong_bullish_confirmation.fillna(False), color="lime", alpha=0.4, label="Strong bullish")
    ax.fill_between(r.close.index, ymin, ymax, where=r.regular_bearish_trend.fillna(False), color="purple", alpha=0.10, label="Regular bearish")
    ax.fill_between(r.close.index, ymin, ymax, where=r.strong_bearish_confirmation.fillna(False), color="red", alpha=0.3, label="Strong bearish")
    ax.set_ylim(ymin, ymax)

    # Crossover markers
    bullish = [(d, p) for d, p, c in zip(r.cross_dates, r.cross_prices, r.cross_colors) if c == "green"]
    bearish = [(d, p) for d, p, c in zip(r.cross_dates, r.cross_prices, r.cross_colors) if c == "red"]

    if bullish:
        bx, by = zip(*bullish)
        ax.scatter(bx, by, zorder=5, s=80, color="green", edgecolors="white", linewidths=0.8,
                   label=f"Confirmed SMA({cfg.short_sma_period}) cross above SMA({cfg.long_sma_period}) [{len(bullish)}]")
    if bearish:
        rx, ry = zip(*bearish)
        ax.scatter(rx, ry, zorder=5, s=80, color="red", edgecolors="white", linewidths=0.8,
                   label=f"Confirmed SMA({cfg.short_sma_period}) cross below SMA({cfg.long_sma_period}) [{len(bearish)}]")
    if r.regime_label == "bullish_impulse" and r.close is not None and not r.close.dropna().empty:
        latest_close = r.close.dropna().iloc[-1]
        latest_date = r.close.dropna().index[-1]
        ax.scatter(
            [latest_date],
            [latest_close],
            zorder=6,
            s=130,
            marker="*",
            color="#38bdf8",
            edgecolors="white",
            linewidths=1.0,
            label="Latest bullish impulse",
        )

    # Forward return labels
    for d, p, fwd in zip(r.cross_dates, r.cross_prices, r.forward_returns):
        if fwd is None:
            ax.annotate("pend.", xy=(d, p), xytext=(0, 15), textcoords="offset points",
                        color="gray", fontsize=7, ha="center")
        else:
            label = f"{'+' if fwd > 0 else ''}{fwd * 100:.1f}%"
            ax.annotate(label, xy=(d, p), xytext=(0, 15), textcoords="offset points",
                        color="lime" if fwd > 0 else "red", fontsize=7, ha="center", fontweight="bold")

    # Spread subplot
    spread_pct = r.spread * 100
    ax2.plot(spread_pct.index, spread_pct, color="cyan", linewidth=1, label="Spread %")
    ax2.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax2.fill_between(spread_pct.index, 0, spread_pct, where=spread_pct > 0, color="lime", alpha=0.25, label="Bullish")
    ax2.fill_between(spread_pct.index, 0, spread_pct, where=spread_pct < 0, color="red", alpha=0.25, label="Bearish")
    ax2.axhline(cfg.min_spread * 100, color="lime", linewidth=0.6, linestyle=":", alpha=0.6)
    ax2.axhline(-cfg.min_spread * 100, color="red", linewidth=0.6, linestyle=":", alpha=0.6)
    ax2.set_ylabel("Spread %", color="white")

    # RSI / overbought subplot
    if r.rsi is not None:
        rsi = r.rsi.replace([np.inf, -np.inf], np.nan)
        ax3.plot(rsi.index, rsi, color="#a78bfa", linewidth=1.1, label="RSI")
    ax3.axhline(cfg.overbought_rsi_level, color="#facc15", linewidth=0.8, linestyle="--", label=f"RSI {cfg.overbought_rsi_level:g}")
    ax3.axhline(cfg.severe_overbought_rsi_level, color="#fb7185", linewidth=0.8, linestyle="--", label=f"RSI {cfg.severe_overbought_rsi_level:g}")
    ax3.axhline(50, color="white", linewidth=0.6, linestyle=":", alpha=0.6)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI", color="white")
    overbought_score_text = (
        f"{r.overbought_score:.2f}" if r.overbought_score is not None else "N/A"
    )
    ax3.set_title(
        f"Overbought: {r.overbought_status or 'N/A'} | score={overbought_score_text}",
        fontsize=10,
        color="white",
    )

    # Shared styling
    for axis in (ax, ax2, ax3):
        axis.tick_params(axis="x", colors="white")
        axis.tick_params(axis="y", colors="white")
        axis.grid(True, color="gray", alpha=0.25)
        for spine in axis.spines.values():
            spine.set_color("white")
        leg = axis.legend(fontsize=8)
        leg.get_frame().set_facecolor("black")
        leg.get_frame().set_edgecolor("white")
        for txt in leg.get_texts():
            txt.set_color("white")

    final_score_text = f"{r.final_bullish_score:.2f}" if r.final_bullish_score is not None else "N/A"
    overbought_title = (
        f"{r.overbought_status or 'N/A'} "
        f"({overbought_score_text})"
    )
    ax.set_title(
        f"{ticker} | Regime={r.regime_label or 'N/A'} | Final Score={final_score_text} | "
        f"Overbought={overbought_title} | {cfg.start_date} to {end_date} | "
        f"Confirmed crosses | {cfg.forward_days}d fwd return",
        fontsize=13,
        color="white",
    )
    ax.set_ylabel("Price", color="white")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()

    if cfg.save_plots:
        os.makedirs(cfg.plot_dir, exist_ok=True)
        path = os.path.join(cfg.plot_dir, f"{ticker}.png")
        plt.savefig(path, dpi=150, facecolor="black")
        log.info("Saved plot → %s", path)

    if cfg.plot:
        if defer_show:
            return
        plt.show()

    plt.close(fig)

def print_summary(ranked: list[SignalResult], cfg: PipelineConfig) -> pd.DataFrame:
    """Print and return a DataFrame summarising the top bullish candidates."""
    rows = []
    for r in ranked[: cfg.top_n]:
        latest_spread = _latest_valid_value(r.spread)
        reasons = (
            list(r.bullish_impulse_reasons[:3])
            + list(r.pre_golden_reasons[:2])
            + list(r.relative_strength_reasons[:2])
        )
        rows.append({
            "ticker":                 r.ticker,
            "regime_label":           r.regime_label,
            "final_bullish_score":    round(r.final_bullish_score, 2) if r.final_bullish_score is not None else None,
            "bullish_impulse_score":  round(r.bullish_impulse_score, 2) if r.bullish_impulse_score is not None else None,
            "pre_golden_score":       round(r.pre_golden_score, 2) if r.pre_golden_score is not None else None,
            "relative_strength_score": round(r.relative_strength_score, 2) if r.relative_strength_score is not None else None,
            "overbought_status":      r.overbought_status,
            "overbought_score":       round(r.overbought_score, 2) if r.overbought_score is not None else None,
            "spread_%":               round(latest_spread * 100, 2) if latest_spread is not None else None,
            "est_days_to_cross":      round(r.days_to_cross_estimate, 1) if r.days_to_cross_estimate is not None else None,
            "latest_signal":          r.latest_signal,
            "signal_date":            r.latest_signal_date.date() if r.latest_signal_date else None,
            "reasons":                " | ".join(reasons),
            "overbought_reasons":     " | ".join(r.overbought_reasons[:4]),
        })

    df = pd.DataFrame(rows)
    print("\n── Top bullish candidates ──────────────────────────")
    print(df.to_string(index=False))
    print(
        "\nInterpretation: a high bullish/pre-golden score with a low overbought "
        "score is preferred. A high bullish/pre-golden score with a high "
        "overbought score can still be bullish, but immediate entry timing risk "
        "is elevated."
    )
    print()
    return df


def run_pipeline(cfg: PipelineConfig | None = None) -> list[SignalResult]:
    """
    Run all four pipeline stages and return the ranked SignalResult list.

    Parameters
    ----------
    cfg : PipelineConfig, optional
        Pass a custom config to override defaults.

    Returns
    -------
    list[SignalResult]
        Ranked list of tickers with pre-golden-cross setups.
    """
    if cfg is None:
        cfg = PipelineConfig()

    candidates = collect_candidates(cfg)
    if not candidates:
        log.warning("No candidates found — check your API keys and screener config.")
        return []

    log.info("%d candidates to process", len(candidates))

    benchmark_end = (date.today() + timedelta(days=1)).isoformat()
    benchmark_closes = _fetch_benchmark_closes(cfg, cfg.start_date, benchmark_end)

    log.info("── Stage 2: computing signals ───────────────────────────────")
    results: list[SignalResult] = []

    for i, ticker in enumerate(candidates, 1):
        log.info("[%d/%d] %s", i, len(candidates), ticker)
        result = compute_signals(ticker, cfg, benchmark_closes=benchmark_closes)
        results.append(result)
        time.sleep(cfg.signal_sleep_s)


    ranked = rank_candidates(results, require_bullish=False)

    if not ranked:
        log.info("No tickers with bullish regimes found.")
        return []

    print_summary(ranked, cfg)

    if cfg.plot or cfg.save_plots:
        log.info("── Stage 4: plotting top %d tickers ─────────────────────────", cfg.top_n)
        for r in ranked[: cfg.top_n]:
            log.info("Plotting %s", r.ticker)
            plot_signal_result(r, cfg, defer_show=cfg.show_plots_at_end)
        if cfg.plot and cfg.show_plots_at_end:
            log.info("Showing %d plot windows", min(len(ranked), cfg.top_n))
            plt.show()
            plt.close("all")

    return ranked

if __name__ == "__main__":
    import os
    print("FMP key loaded:", os.getenv("FMP_API_KEY") is not None)
    cfg = PipelineConfig(
        top_n=10,
        plot=True,
        save_plots=False,
        start_date="2022-01-01",
        news_days=7,
        forward_days=20,
    )
    run_pipeline(cfg)
