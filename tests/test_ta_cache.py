from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pandas as pd

from rag_trial.db import ta_cache

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ta_pipe  # noqa: E402
import ta_dashboard  # noqa: E402


def _cache_path(tmp_path: Path) -> Path:
    return tmp_path / "ta_cache.db"


def _signal(ticker: str) -> ta_pipe.SignalResult:
    return ta_pipe.SignalResult(
        ticker=ticker,
        cross_dates=[],
        cross_prices=[],
        cross_colors=[],
        forward_returns=[],
        latest_signal=None,
        latest_signal_date=None,
    )


def _successful_route_answer(ticker: str, score: int = 8) -> dict:
    return {
        "query": f"Is {ticker} fundamentally sound based on its financial statements?",
        "answer": (
            f"Answer: {ticker} has enough usable financial context for a fundamental assessment [F]\n"
            "Fundamental Assessment: Sound\n"
            f"Fundamental Score (0-10): {score}\n"
            "Score Rationale: The supplied financial context supports a sound profile [F]\n"
            "Key Fundamental Drivers:\n"
            "- Revenue evidence is available in the financial context [F]\n"
            "- Profitability evidence is available in the financial context [F]\n"
            "- Liquidity evidence is available in the financial context [F]\n"
            "Risks / Gaps:\n"
            "- The dashboard route should still treat news as secondary [F]\n"
            "- The cached answer should be refreshed after the age threshold [F]"
        ),
        "decision": "answer",
        "route_type": "single_ticker_financial",
        "resolved_target": {"display_name": f"{ticker} Corp", "ticker": ticker},
        "retrieval_trace": {
            "route_type": "single_ticker_financial",
            "finance_context_present": True,
            "news_context_present": False,
            "news_query": "",
            "news_item_count": 0,
        },
        "logs": ["route"],
    }


def _fail_closed_route_answer(ticker: str) -> dict:
    return {
        "query": f"Is {ticker} fundamentally sound based on its financial statements?",
        "answer": "Answer: Insufficient financial data [F] is available to answer this ticker query.",
        "decision": "abstain",
        "route_type": "single_ticker_financial",
        "resolved_target": {"display_name": f"{ticker} Corp", "ticker": ticker},
        "retrieval_trace": {
            "route_type": "single_ticker_financial",
            "finance_context_present": False,
            "news_context_present": False,
            "news_query": "",
            "news_item_count": 0,
        },
        "logs": ["fail-closed"],
    }


def test_schema_creation_includes_fundamental_analyses(tmp_path: Path) -> None:
    db_path = _cache_path(tmp_path)
    ta_cache.ensure_cache_db(db_path)

    conn = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    finally:
        conn.close()

    assert "ta_fundamental_analyses" in tables


def test_fundamental_analysis_upsert_and_load_by_ticker(tmp_path: Path) -> None:
    db_path = _cache_path(tmp_path)

    ta_cache.upsert_fundamental_analysis(
        ticker="aaa",
        company_name="AAA Corp",
        query="Is AAA fundamentally sound?",
        answer="Answer: AAA is sound [F]",
        decision="answer",
        route_type="single_ticker_financial",
        fundamental_assessment="Sound",
        fundamental_score=8,
        finance_context_present=True,
        news_context_present=False,
        news_query="",
        news_item_count=0,
        retrieval_trace_json='{"finance_context_present": true}',
        logs_json='["ok"]',
        generated_at_utc="2026-05-01T00:00:00+00:00",
        db_path=db_path,
    )

    row = ta_cache.load_fundamental_analysis("AAA", db_path=db_path)

    assert row is not None
    assert row["ticker"] == "AAA"
    assert row["company_name"] == "AAA Corp"
    assert row["fundamental_score"] == 8
    assert row["finance_context_present"] == 1


def test_fundamental_analysis_exactly_90_days_is_fresh(tmp_path: Path) -> None:
    db_path = _cache_path(tmp_path)
    now = datetime(2026, 5, 28, tzinfo=timezone.utc)
    generated_at = now - timedelta(days=90)
    ta_cache.upsert_fundamental_analysis(
        ticker="AAA",
        answer="Answer: cached",
        decision="answer",
        route_type="single_ticker_financial",
        finance_context_present=True,
        generated_at_utc=generated_at.isoformat(),
        db_path=db_path,
    )

    row = ta_cache.load_fresh_fundamental_analysis("AAA", now_utc=now, db_path=db_path)

    assert row is not None


def test_fundamental_analysis_older_than_90_days_is_stale(tmp_path: Path) -> None:
    db_path = _cache_path(tmp_path)
    now = datetime(2026, 5, 28, tzinfo=timezone.utc)
    generated_at = now - timedelta(days=90, seconds=1)
    ta_cache.upsert_fundamental_analysis(
        ticker="AAA",
        answer="Answer: stale",
        decision="answer",
        route_type="single_ticker_financial",
        finance_context_present=True,
        generated_at_utc=generated_at.isoformat(),
        db_path=db_path,
    )

    row = ta_cache.load_fresh_fundamental_analysis("AAA", now_utc=now, db_path=db_path)

    assert row is None


def test_dashboard_fresh_cache_hit_does_not_call_fundamental_route(tmp_path: Path) -> None:
    db_path = _cache_path(tmp_path)
    ta_cache.upsert_fundamental_analysis(
        ticker="AAA",
        answer="Answer: cached [F]\nFundamental Assessment: Sound\nFundamental Score (0-10): 8",
        decision="answer",
        route_type="single_ticker_financial",
        finance_context_present=True,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        db_path=db_path,
    )

    def fail_route(ticker: str, company_name: str | None):
        raise AssertionError("fresh cache should avoid route calls")

    rows = ta_dashboard.load_or_generate_fundamental_analyses(
        [_signal("AAA")],
        route_runner=fail_route,
        db_path=db_path,
    )

    assert rows[0]["cache_status"] == "cache"
    assert rows[0]["answer"].startswith("Answer: cached")


def test_dashboard_missing_cache_calls_route_once_and_persists_success(tmp_path: Path) -> None:
    db_path = _cache_path(tmp_path)
    calls: list[str] = []

    def route(ticker: str, company_name: str | None):
        calls.append(ticker)
        return _successful_route_answer(ticker, score=9)

    rows = ta_dashboard.load_or_generate_fundamental_analyses(
        [_signal("AAA")],
        route_runner=route,
        db_path=db_path,
    )
    cached = ta_cache.load_fundamental_analysis("AAA", db_path=db_path)

    assert calls == ["AAA"]
    assert rows[0]["cache_status"] == "generated"
    assert rows[0]["stored"] is True
    assert cached is not None
    assert cached["fundamental_score"] == 9


def test_dashboard_stale_cache_calls_route_once_and_overwrites(tmp_path: Path) -> None:
    db_path = _cache_path(tmp_path)
    ta_cache.upsert_fundamental_analysis(
        ticker="AAA",
        answer="Answer: old",
        decision="answer",
        route_type="single_ticker_financial",
        finance_context_present=True,
        fundamental_score=5,
        generated_at_utc=(datetime.now(timezone.utc) - timedelta(days=91)).isoformat(),
        db_path=db_path,
    )
    calls: list[str] = []

    def route(ticker: str, company_name: str | None):
        calls.append(ticker)
        return _successful_route_answer(ticker, score=8)

    rows = ta_dashboard.load_or_generate_fundamental_analyses(
        [_signal("AAA")],
        route_runner=route,
        db_path=db_path,
    )
    cached = ta_cache.load_fundamental_analysis("AAA", db_path=db_path)

    assert calls == ["AAA"]
    assert rows[0]["cache_status"] == "generated"
    assert cached is not None
    assert cached["answer"].startswith("Answer: AAA has enough")
    assert cached["fundamental_score"] == 8


def test_dashboard_fail_closed_result_is_shown_but_not_cached(tmp_path: Path) -> None:
    db_path = _cache_path(tmp_path)
    calls: list[str] = []

    def route(ticker: str, company_name: str | None):
        calls.append(ticker)
        return _fail_closed_route_answer(ticker)

    rows = ta_dashboard.load_or_generate_fundamental_analyses(
        [_signal("AAA")],
        route_runner=route,
        db_path=db_path,
    )
    cached = ta_cache.load_fundamental_analysis("AAA", db_path=db_path)

    assert calls == ["AAA"]
    assert rows[0]["cache_status"] == "generated"
    assert rows[0]["stored"] is False
    assert "Insufficient financial data" in rows[0]["answer"]
    assert cached is None


def test_article_upsert_dedupes_by_url(tmp_path: Path) -> None:
    db_path = _cache_path(tmp_path)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    article = {
        "url": "https://example.com/story",
        "title": "Example story",
        "site": "Example",
        "publishedDate": "2026-05-25 10:00:00",
        "symbol": "AAA",
    }

    ta_cache.upsert_articles([article, dict(article)], fetched_at_utc=now, db_path=db_path)
    rows = ta_cache.load_articles_since("2026-05-24", db_path=db_path)

    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAA"
    assert ta_cache.latest_article_published_at(db_path=db_path) == "2026-05-25 10:00:00"


def test_load_articles_for_ticker_returns_all_cached_news_for_symbol(tmp_path: Path) -> None:
    db_path = _cache_path(tmp_path)
    ta_cache.upsert_articles(
        [
            {
                "url": "https://example.com/aaa-old",
                "title": "AAA old article",
                "site": "Example",
                "publishedDate": "2026-05-24 09:00:00",
                "symbol": "AAA",
            },
            {
                "url": "https://example.com/bbb",
                "title": "BBB article",
                "site": "Example",
                "publishedDate": "2026-05-25 09:00:00",
                "symbol": "BBB",
            },
            {
                "url": "https://example.com/aaa-new",
                "title": "AAA newer article",
                "site": "Example",
                "publishedDate": "2026-05-26 09:00:00",
                "symbol": "AAA",
            },
        ],
        fetched_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        db_path=db_path,
    )

    rows = ta_cache.load_articles_for_ticker("aaa", db_path=db_path)

    assert [row["title"] for row in rows] == ["AAA newer article", "AAA old article"]
    assert {row["symbol"] for row in rows} == {"AAA"}


def test_dashboard_ticker_news_df_formats_cached_news(tmp_path: Path) -> None:
    db_path = _cache_path(tmp_path)
    ta_cache.upsert_articles(
        [
            {
                "url": "https://example.com/aaa",
                "title": "AAA article",
                "site": "Example",
                "publishedDate": "2026-05-25 10:00:00",
                "symbol": "AAA",
            }
        ],
        fetched_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        db_path=db_path,
    )

    df = ta_dashboard.ticker_news_df("AAA", db_path=db_path)

    assert df.to_dict("records") == [
        {
            "published": "2026-05-25 10:00:00",
            "source": "Example",
            "title": "AAA article",
            "url": "https://example.com/aaa",
        }
    ]


def test_daily_bar_upsert_dedupes_by_ticker_date_provider(tmp_path: Path) -> None:
    db_path = _cache_path(tmp_path)
    row = {
        "ticker": "AAA",
        "bar_date": "2026-05-25",
        "open": 1.0,
        "high": 2.0,
        "low": 0.5,
        "close": 1.5,
        "adj_close": 1.4,
        "volume": 1000,
    }

    ta_cache.upsert_daily_bars([row, dict(row)], provider="yahoo", db_path=db_path)
    rows = ta_cache.load_daily_bars(
        "AAA",
        "2026-05-24",
        "2026-05-26",
        provider="yahoo",
        db_path=db_path,
    )

    assert len(rows) == 1
    assert rows[0]["close"] == 1.5


def test_collect_news_tickers_refreshes_complete_cache(monkeypatch, tmp_path: Path) -> None:
    db_path = _cache_path(tmp_path)
    monkeypatch.setenv("TA_CACHE_ENABLED", "1")
    monkeypatch.setenv("TA_SQLITE_CACHE_DB", str(db_path))
    today = date.today()
    latest_cached = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc)
    newer_article = latest_cached + timedelta(minutes=30)
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    ta_cache.upsert_articles(
        [
            {
                "url": "https://example.com/old",
                "title": "Old boundary article",
                "publishedDate": (today - timedelta(days=8)).isoformat(),
                "symbol": "OLD",
            },
            {
                "url": "https://example.com/new",
                "title": "Recent article",
                "publishedDate": latest_cached.isoformat(),
                "symbol": "AAA",
            },
        ],
        fetched_at_utc=fetched_at,
        db_path=db_path,
    )

    calls: list[int] = []

    def fake_news(page: int, limit: int):
        calls.append(page)
        if page == 0:
            return [
                {
                    "url": "https://example.com/newer",
                    "title": "Newer article",
                    "publishedDate": newer_article.isoformat(),
                    "symbol": "BBB",
                },
                {
                    "url": "https://example.com/new",
                    "title": "Recent article",
                    "publishedDate": latest_cached.isoformat(),
                    "symbol": "AAA",
                },
            ]
        return []

    monkeypatch.setattr(ta_pipe, "get_news_stock_latest", fake_news)
    cfg = ta_pipe.PipelineConfig(news_days=7, news_max_pages=3, news_sleep_s=0, news_recency_decay=False)

    scores = ta_pipe.collect_news_tickers(cfg)

    assert calls == [0]
    assert scores.mention_counts["AAA"] == 1
    assert scores.mention_counts["BBB"] == 1
    assert "OLD" not in scores.mention_counts


def test_collect_news_tickers_fetches_and_persists_when_cache_missing(monkeypatch, tmp_path: Path) -> None:
    db_path = _cache_path(tmp_path)
    monkeypatch.setenv("TA_CACHE_ENABLED", "1")
    monkeypatch.setenv("TA_SQLITE_CACHE_DB", str(db_path))
    today = date.today().isoformat()
    calls: list[int] = []

    def fake_news(page: int, limit: int):
        calls.append(page)
        if page == 0:
            return [
                {
                    "url": "https://example.com/aaa",
                    "title": "AAA article",
                    "publishedDate": today,
                    "symbol": "AAA",
                },
                {
                    "url": "https://example.com/aaa",
                    "title": "AAA article duplicate",
                    "publishedDate": today,
                    "symbol": "AAA",
                },
            ]
        return []

    monkeypatch.setattr(ta_pipe, "get_news_stock_latest", fake_news)
    cfg = ta_pipe.PipelineConfig(
        news_days=7,
        news_max_pages=2,
        news_sleep_s=0,
        news_recency_decay=False,
    )

    scores = ta_pipe.collect_news_tickers(cfg)
    cached = ta_cache.load_articles_since(today, db_path=db_path)

    assert calls == [0, 1]
    assert scores.mention_counts["AAA"] == 1
    assert len(cached) == 1


def test_download_history_batch_uses_cached_yahoo_shape(monkeypatch, tmp_path: Path) -> None:
    db_path = _cache_path(tmp_path)
    monkeypatch.setenv("TA_CACHE_ENABLED", "1")
    monkeypatch.setenv("TA_SQLITE_CACHE_DB", str(db_path))
    today = date.today().isoformat()
    ta_cache.upsert_daily_bars(
        [
            {
                "ticker": "AAA",
                "bar_date": today,
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "adj_close": 1.4,
                "volume": 1000,
            }
        ],
        provider="yahoo",
        db_path=db_path,
    )

    def fail_download(*args, **kwargs):
        raise AssertionError("online Yahoo download should not be called")

    monkeypatch.setattr(ta_pipe.yf, "download", fail_download)
    cfg = ta_pipe.PipelineConfig(cheap_history_period="1d")

    downloaded = ta_pipe._download_history_batch(["AAA"], cfg)
    hist = ta_pipe._history_for_ticker(downloaded, "AAA")

    assert hist is not None
    assert float(hist["Close"].iloc[-1]) == 1.5


def test_compute_signals_uses_cached_financetoolkit_bars(monkeypatch, tmp_path: Path) -> None:
    db_path = _cache_path(tmp_path)
    monkeypatch.setenv("TA_CACHE_ENABLED", "1")
    monkeypatch.setenv("TA_SQLITE_CACHE_DB", str(db_path))
    start = date.today() - timedelta(days=12)
    rows = []
    for offset in range(13):
        day = start + timedelta(days=offset)
        close = 10.0 + offset
        rows.append({
            "ticker": "AAA",
            "bar_date": day.isoformat(),
            "open": close - 0.5,
            "high": close + 0.5,
            "low": close - 1.0,
            "close": close,
            "adj_close": close,
            "volume": 1000 + offset,
        })
    ta_cache.upsert_daily_bars(rows, provider="financetoolkit", db_path=db_path)

    class FailToolkit:
        def __init__(self, *args, **kwargs):
            raise AssertionError("FinanceToolkit should not be called")

    monkeypatch.setattr(ta_pipe, "Toolkit", FailToolkit)
    cfg = ta_pipe.PipelineConfig(
        start_date=start.isoformat(),
        short_sma_period=2,
        long_sma_period=3,
        vwma_period=2,
        confirmation_days=1,
        forward_days=1,
    )

    result = ta_pipe.compute_signals("AAA", cfg)

    assert result.error is None
    assert isinstance(result.close, pd.Series)
    assert float(result.close.iloc[-1]) == 22.0
