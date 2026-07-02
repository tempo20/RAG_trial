from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rag_trial.db import ta_cache  # noqa: E402
import ta_dashboard as dashboard  # noqa: E402


def test_news_ticker_mentions_schema_contains_only_ticker(tmp_path: Path) -> None:
    db_path = tmp_path / "ta_cache.db"

    ta_cache.ensure_cache_db(db_path)

    conn = ta_cache.connect(db_path)
    try:
        columns = conn.execute(
            "PRAGMA table_info(news_ticker_mentions)"
        ).fetchall()
    finally:
        conn.close()

    assert [column["name"] for column in columns] == ["ticker"]
    assert columns[0]["pk"] == 1


def test_news_ticker_mentions_are_normalized_idempotent_and_cumulative(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "ta_cache.db"

    assert ta_cache.upsert_news_ticker_mentions(
        [" aapl ", "AAPL", "", " msft "],
        db_path=db_path,
    ) == 2
    assert ta_cache.upsert_news_ticker_mentions(
        ["AAPL", " nvda ", "MSFT"],
        db_path=db_path,
    ) == 1

    conn = ta_cache.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT ticker FROM news_ticker_mentions ORDER BY ticker"
        ).fetchall()
    finally:
        conn.close()

    assert [row["ticker"] for row in rows] == ["AAPL", "MSFT", "NVDA"]


def test_ticker_mentions_tab_persists_exact_displayed_tickers(monkeypatch) -> None:
    articles = [
        {"symbol": "msft", "publishedDate": "2026-06-30"},
        {"symbol": " aapl ", "publishedDate": "2026-06-30"},
        {"symbol": "MSFT", "publishedDate": "2026-06-29"},
    ]
    persisted: list[list[str]] = []

    monkeypatch.setattr(dashboard.st, "selectbox", lambda *args, **kwargs: 7)
    monkeypatch.setattr(
        dashboard,
        "load_ticker_news_window_articles",
        lambda *args, **kwargs: articles,
    )
    monkeypatch.setattr(dashboard.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        dashboard.ta_cache,
        "upsert_news_ticker_mentions",
        lambda tickers: persisted.append(list(tickers)),
    )

    dashboard.render_ticker_counts_tab(dashboard.PipelineConfig())

    assert persisted == [["MSFT", "AAPL"]]


def test_empty_ticker_mentions_tab_does_not_write(monkeypatch) -> None:
    messages: list[str] = []

    monkeypatch.setattr(dashboard.st, "selectbox", lambda *args, **kwargs: 7)
    monkeypatch.setattr(
        dashboard,
        "load_ticker_news_window_articles",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        dashboard.st,
        "info",
        lambda message, *args, **kwargs: messages.append(message),
    )

    def unexpected_write(tickers: list[str]) -> int:
        raise AssertionError("empty ticker mention results must not be persisted")

    monkeypatch.setattr(
        dashboard.ta_cache,
        "upsert_news_ticker_mentions",
        unexpected_write,
    )

    dashboard.render_ticker_counts_tab(dashboard.PipelineConfig())

    assert messages == [
        "No cached ticker articles found for the selected news mention window."
    ]
