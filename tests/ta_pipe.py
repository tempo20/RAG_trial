"""
Stock Screening & Signal Pipeline
==================================
Stages:
  1. collect_candidates  — news momentum + Yahoo screener overlap
  2. compute_signals     — SMA/VWMA crossover logic, no plotting
  3. rank_candidates     — filter to buy signals, sort by recency + strength
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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Candidate collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_news_tickers(cfg: PipelineConfig) -> NewsTickerScores:
    """
    Return separate news recency scores and raw mention counts.
    Delegates fetching and publisher filtering to fmp_functions so there
    is a single source of truth for both the API call and blocked publishers.
    If recency_decay is enabled, articles are weighted by 1 / days_old.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=cfg.news_days)
    recency_scores: Counter = Counter()
    mention_counts: Counter = Counter()
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

            ticker = article.get("symbol")
            if not ticker:
                continue

            if cfg.news_recency_decay and published:
                age_days = max(
                    (datetime.now(timezone.utc) - published).total_seconds() / 86400,
                    0.01,
                )
                weight = 1.0 / age_days
            else:
                weight = 1.0

            recency_scores[ticker] += weight
            mention_counts[ticker] += 1

        log.info("News page=%d  batch=%d  unique tickers=%d", page, len(batch), len(recency_scores))

        if oldest_in_batch and oldest_in_batch < cutoff:
            break

        time.sleep(cfg.news_sleep_s)

    return NewsTickerScores(recency_scores=recency_scores, mention_counts=mention_counts)


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


def _download_history_batch(tickers: list[str], cfg: PipelineConfig) -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        period=cfg.cheap_history_period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )


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
    error: str | None = None


def compute_signals(ticker: str, cfg: PipelineConfig) -> SignalResult:
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
        api_key = os.getenv("FMP_API_KEY")
        end_date = date.today().isoformat()

        tk = Toolkit(
            tickers=[ticker],
            api_key=api_key,
            start_date=cfg.start_date,
            end_date=end_date,
        )
        hist = tk.get_historical_data()

        close = pd.to_numeric(hist["Close"][ticker], errors="coerce").sort_index()
        volume = pd.to_numeric(hist["Volume"][ticker], errors="coerce").sort_index()

        close.index = close.index.to_timestamp()
        volume.index = volume.index.to_timestamp()

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
        )

    except Exception as exc:
        log.warning("compute_signals failed for %s: %s", ticker, exc)
        empty.error = str(exc)
        return empty

def rank_candidates(
    results: list[SignalResult],
    require_bullish: bool = True,
) -> list[SignalResult]:
    """
    Filter to tickers with a recent confirmed golden cross and rank by:
      1. Recency of latest signal (most recent first)
      2. Spread at signal date (larger spread = stronger breakout)
    """
    log.info("── Stage 3: ranking candidates ──────────────────────────────")

    filtered = [
        r for r in results
        if not r.error
        and r.latest_signal is not None
        and (not require_bullish or r.latest_signal == "green")
    ]

    def sort_key(r: SignalResult):
        recency = r.latest_signal_date or datetime.min.replace(tzinfo=timezone.utc)
        # Spread magnitude at the latest signal date as secondary sort
        spread_strength = 0.0
        if r.spread is not None and r.latest_signal_date is not None:
            try:
                idx = r.spread.index.get_loc(r.latest_signal_date)
                spread_strength = abs(r.spread.iloc[idx])
            except Exception:
                pass
        return (recency, spread_strength)

    ranked = sorted(filtered, key=sort_key, reverse=True)
    log.info("%d tickers with active buy signal, returning top candidates", len(ranked))
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
    fig, (ax, ax2) = plt.subplots(
        2, 1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax2.set_facecolor("black")

    # Price lines
    sns.lineplot(x=r.close.index, y=r.close, ax=ax, color="steelblue", linewidth=1.2, label="Close")
    sns.lineplot(x=r.sma_50.index, y=r.sma_50, ax=ax, color="orange", linewidth=1.5, linestyle="--", label=f"SMA({cfg.short_sma_period})")
    sns.lineplot(x=r.sma_200.index, y=r.sma_200, ax=ax, color="white", linewidth=1.8, label=f"SMA({cfg.long_sma_period})")
    sns.lineplot(x=r.vwma_50.index, y=r.vwma_50, ax=ax, color="crimson", linewidth=1.5, linestyle="-.", label=f"VWMA({cfg.vwma_period})")

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

    # Shared styling
    for axis in (ax, ax2):
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

    ax.set_title(
        f"{ticker} | {cfg.start_date} → {end_date} | "
        f"SMA({cfg.short_sma_period}), SMA({cfg.long_sma_period}), VWMA({cfg.vwma_period}) | "
        f"Confirmed crosses | {cfg.forward_days}d fwd return",
        fontsize=13, color="white",
    )
    ax.set_ylabel("Price", color="white")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
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
    """Print and return a DataFrame summarising the top candidates."""
    rows = []
    for r in ranked[: cfg.top_n]:
        last_fwd = r.forward_returns[-1] if r.forward_returns else None
        rows.append({
            "ticker":            r.ticker,
            "latest_signal":     r.latest_signal,
            "signal_date":       r.latest_signal_date.date() if r.latest_signal_date else None,
            f"fwd_{cfg.forward_days}d_%": f"{last_fwd * 100:+.1f}" if last_fwd is not None else "pend.",
            "total_crosses":     len(r.cross_dates),
            "golden_crosses":    r.cross_colors.count("green"),
            "death_crosses":     r.cross_colors.count("red"),
        })

    df = pd.DataFrame(rows)
    print("\n── Top candidates ───────────────────────────────────────────")
    print(df.to_string(index=False))
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
        Ranked list of tickers with active buy signals.
    """
    if cfg is None:
        cfg = PipelineConfig()

    candidates = collect_candidates(cfg)
    if not candidates:
        log.warning("No candidates found — check your API keys and screener config.")
        return []

    log.info("%d candidates to process", len(candidates))


    log.info("── Stage 2: computing signals ───────────────────────────────")
    results: list[SignalResult] = []

    for i, ticker in enumerate(candidates, 1):
        log.info("[%d/%d] %s", i, len(candidates), ticker)
        result = compute_signals(ticker, cfg)
        results.append(result)
        time.sleep(cfg.signal_sleep_s)


    ranked = rank_candidates(results, require_bullish=True)

    if not ranked:
        log.info("No tickers with active buy signals found.")
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
