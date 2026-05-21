"""
Stock Screening & Signal Pipeline
==================================
Stages:
  1. collect_candidates  — news momentum + Yahoo screener overlap
  2. compute_signals     — SMA/VWMA crossover logic, no plotting
  3. rank_candidates     — filter to buy signals, sort by recency + strength
  4. plot_top            — render charts only for the top N tickers

Usage:
    python pipeline.py

    # Or import and call individually:
    from pipeline import run_pipeline
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

# ── Path & env setup ──────────────────────────────────────────────────────────
# ta_pipe.py lives at RAG_TRIAL/tests/ta_pipe.py
# Project root is one level up; added to sys.path so rag_trial is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    _parse_fmp_news_datetime,
    _is_blocked_stock_news_publisher,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # ── Candidate collection ──────────────────────────────────────────────────
    news_days: int = 7                  # lookback window for news articles
    news_limit: int = 250               # articles per FMP page
    news_max_pages: int = 100
    news_sleep_s: float = 0.25          # pause between FMP pages
    news_min_mentions: int = 2          # minimum article count to keep ticker
    news_recency_decay: bool = True     # weight recent articles more heavily

    screeners: list[str] = field(default_factory=lambda: [
        "most_actives",
        "day_gainers",
        "small_cap_gainers",
        "aggressive_small_caps",
        "undervalued_growth_stocks",
        "undervalued_large_caps",
    ])
    screener_count: int = 100           # tickers per screener
    screener_min_overlap: int = 2       # must appear in N+ screeners

    # Screeners excluded from overlap count (used as tiebreaker only)
    screener_noise: set[str] = field(default_factory=lambda: {"most_actives"})

    # blocked_publishers is defined in rag_trial/chat/fmp_functions.py
    # (BLOCKED_STOCK_NEWS_PUBLISHERS) — single source of truth.

    # ── Signal computation ────────────────────────────────────────────────────
    start_date: str = "2022-01-01"
    vwma_period: int = 50
    short_sma_period: int = 50
    long_sma_period: int = 200
    confirmation_days: int = 5
    min_spread: float = 0.002
    forward_days: int = 20
    signal_sleep_s: float = 0.3         # pause between FMP ticker calls

    # ── Output ────────────────────────────────────────────────────────────────
    top_n: int = 10                     # tickers to plot
    plot: bool = True
    save_plots: bool = False            # save PNGs instead of / in addition to showing
    plot_dir: str = "plots"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Candidate collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_news_tickers(cfg: PipelineConfig) -> Counter:
    """
    Return a Counter of {ticker: weighted_score} from recent news.
    Delegates fetching and publisher filtering to fmp_functions so there
    is a single source of truth for both the API call and blocked publishers.
    If recency_decay is enabled, articles are weighted by 1 / days_old.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=cfg.news_days)
    scores: Counter = Counter()
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

            # Delegate publisher filtering to fmp_functions
            if _is_blocked_stock_news_publisher(article):
                continue

            # Delegate date parsing to fmp_functions
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

            scores[ticker] += weight

        log.info("News page=%d  batch=%d  unique tickers=%d", page, len(batch), len(scores))

        if oldest_in_batch and oldest_in_batch < cutoff:
            break

        time.sleep(cfg.news_sleep_s)

    return scores


def collect_screener_tickers(cfg: PipelineConfig) -> Counter:
    """
    Return a Counter of {ticker: screener_count} excluding noisy screeners
    from the overlap count (they're kept as +0.5 tiebreaker weight instead).
    """
    signal_counts: Counter = Counter()
    noise_counts: Counter = Counter()

    for scr_id in cfg.screeners:
        try:
            data = yf.screen(scr_id, count=cfg.screener_count)
            quotes = data.get("quotes", [])
            tickers = [q["symbol"] for q in quotes if q.get("symbol")]
            log.info("Screener %-35s  %d tickers", scr_id, len(tickers))

            if scr_id in cfg.screener_noise:
                noise_counts.update(tickers)
            else:
                signal_counts.update(tickers)

        except Exception as exc:
            log.warning("Screener %s failed: %s", scr_id, exc)

    # Merge: signal overlap is integer, noise adds 0.5 tiebreaker
    combined: Counter = Counter()
    for t, n in signal_counts.items():
        combined[t] = n + noise_counts.get(t, 0) * 0.5

    return combined, signal_counts


def collect_candidates(cfg: PipelineConfig) -> list[str]:
    """
    Intersect news momentum with screener overlap to produce a ranked
    candidate list. Tickers must satisfy BOTH filters to be included.
    """
    log.info("── Stage 1: collecting candidates ───────────────────────────")

    news_scores = collect_news_tickers(cfg)
    screener_combined, screener_signal = collect_screener_tickers(cfg)

    # Hard filter: must appear in cfg.screener_min_overlap signal screeners
    screener_qualified = {
        t for t, n in screener_signal.items()
        if n >= cfg.screener_min_overlap
    }

    # Hard filter: must meet minimum raw news mentions
    # (score threshold is mentions * average_weight ≈ cfg.news_min_mentions for equal weight)
    news_qualified = {
        t for t, s in news_scores.items()
        if s >= cfg.news_min_mentions
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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Signal computation (no plotting)
# ─────────────────────────────────────────────────────────────────────────────

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
        vol_ma = volume.rolling(window=cfg.vwma_period).mean()

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

            future_window = spread.iloc[i : i + cfg.confirmation_days]
            if len(future_window) < cfg.confirmation_days:
                continue
            if pd.isna(prev_spread.iloc[i]) or pd.isna(spread.iloc[i]):
                continue

            is_golden = prev_spread.iloc[i] <= 0 and spread.iloc[i] > 0
            is_death = prev_spread.iloc[i] >= 0 and spread.iloc[i] < 0

            volume_ok = (
                not pd.isna(volume.iloc[i])
                and not pd.isna(vol_ma.iloc[i])
                and volume.iloc[i] > vol_ma.iloc[i]
            )

            confirmed_golden = (
                is_golden
                and (future_window > 0).all()
                and future_window.max() > cfg.min_spread
                and volume_ok
            )
            confirmed_death = (
                is_death
                and (future_window < 0).all()
                and future_window.min() < -cfg.min_spread
                and volume_ok
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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Ranking
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_signal_result(result: SignalResult, cfg: PipelineConfig) -> None:
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
                   label=f"Golden cross + vol [{len(bullish)}]")
    if bearish:
        rx, ry = zip(*bearish)
        ax.scatter(rx, ry, zorder=5, s=80, color="red", edgecolors="white", linewidths=0.8,
                   label=f"Death cross + vol [{len(bearish)}]")

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
        f"Vol-confirmed | {cfg.forward_days}d fwd return",
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
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

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

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    candidates = collect_candidates(cfg)
    if not candidates:
        log.warning("No candidates found — check your API keys and screener config.")
        return []

    log.info("%d candidates to process", len(candidates))

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    log.info("── Stage 2: computing signals ───────────────────────────────")
    results: list[SignalResult] = []

    for i, ticker in enumerate(candidates, 1):
        log.info("[%d/%d] %s", i, len(candidates), ticker)
        result = compute_signals(ticker, cfg)
        results.append(result)
        time.sleep(cfg.signal_sleep_s)

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    ranked = rank_candidates(results, require_bullish=True)

    if not ranked:
        log.info("No tickers with active buy signals found.")
        return []

    # ── Summary table ─────────────────────────────────────────────────────────
    print_summary(ranked, cfg)

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    if cfg.plot or cfg.save_plots:
        log.info("── Stage 4: plotting top %d tickers ─────────────────────────", cfg.top_n)
        for r in ranked[: cfg.top_n]:
            log.info("Plotting %s", r.ticker)
            plot_signal_result(r, cfg)

    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = PipelineConfig(
        top_n=10,
        plot=True,
        save_plots=False,
        start_date="2022-01-01",
        news_days=7,
        forward_days=20,
    )
    run_pipeline(cfg)