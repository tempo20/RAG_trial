from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from rag_trial.paths import TA_SQLITE_CACHE_DB_PATH, env_str_path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ta_articles (
    article_id TEXT PRIMARY KEY,
    dedupe_key TEXT NOT NULL UNIQUE,
    url TEXT,
    title TEXT,
    publisher TEXT,
    published_at TEXT,
    provider TEXT NOT NULL,
    fetched_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ta_article_symbols (
    article_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    PRIMARY KEY (article_id, ticker),
    FOREIGN KEY(article_id) REFERENCES ta_articles(article_id)
);

CREATE TABLE IF NOT EXISTS ta_historical_daily_bars (
    ticker TEXT NOT NULL,
    bar_date TEXT NOT NULL,
    provider TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume REAL,
    fetched_at_utc TEXT NOT NULL,
    PRIMARY KEY (ticker, bar_date, provider)
);

CREATE TABLE IF NOT EXISTS ta_stock_pick_snapshots (
    pick_date TEXT PRIMARY KEY,
    tickers_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ta_fundamental_analyses (
    ticker TEXT PRIMARY KEY,
    company_name TEXT,
    query TEXT,
    answer TEXT NOT NULL,
    decision TEXT,
    route_type TEXT,
    fundamental_assessment TEXT,
    fundamental_score INTEGER,
    finance_context_present INTEGER,
    news_context_present INTEGER,
    news_query TEXT,
    news_item_count INTEGER,
    retrieval_trace_json TEXT,
    logs_json TEXT,
    generated_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);
"""


INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_ta_articles_published_at
    ON ta_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_ta_articles_fetched_at
    ON ta_articles(fetched_at_utc);
CREATE INDEX IF NOT EXISTS idx_ta_article_symbols_ticker
    ON ta_article_symbols(ticker);
CREATE INDEX IF NOT EXISTS idx_ta_bars_ticker_date
    ON ta_historical_daily_bars(ticker, bar_date);
"""


def cache_enabled() -> bool:
    value = os.getenv("TA_CACHE_ENABLED", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def cache_db_path() -> Path:
    return Path(env_str_path("TA_SQLITE_CACHE_DB", TA_SQLITE_CACHE_DB_PATH))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_utc_datetime(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper()


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else cache_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_cache_db(db_path: str | Path | None = None) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.executescript(INDEX_SQL)
        conn.commit()
    finally:
        conn.close()


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _published_at(article: dict[str, Any]) -> str:
    return str(
        article.get("publishedDate")
        or article.get("date")
        or article.get("publishedAt")
        or ""
    ).strip()


def article_dedupe_key(article: dict[str, Any]) -> str:
    url = _normalize_text(article.get("url"))
    if url:
        return f"url:{url.rstrip('/')}"

    title = _normalize_text(article.get("title"))
    published_at = _normalize_text(_published_at(article))
    return f"title_date:{title}|{published_at}"


def article_id_for(article: dict[str, Any]) -> str:
    return hashlib.sha256(article_dedupe_key(article).encode("utf-8")).hexdigest()


def _symbols_for(article: dict[str, Any]) -> list[str]:
    raw = (
        article.get("symbol")
        or article.get("ticker")
        or article.get("symbols")
        or article.get("tickers")
        or []
    )
    if isinstance(raw, str):
        values = re.split(r"[,;\s]+", raw)
    elif isinstance(raw, (list, tuple, set)):
        values = [str(item) for item in raw]
    else:
        values = []

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        ticker = str(value or "").strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


def upsert_articles(
    articles: list[dict[str, Any]],
    *,
    provider: str = "fmp_stock_news",
    fetched_at_utc: str | None = None,
    db_path: str | Path | None = None,
) -> int:
    if not articles:
        return 0

    ensure_cache_db(db_path)
    fetched_at = fetched_at_utc or utc_now_iso()
    inserted_or_updated = 0
    conn = connect(db_path)
    try:
        for article in articles:
            if not isinstance(article, dict):
                continue

            article_id = article_id_for(article)
            conn.execute(
                """
                INSERT INTO ta_articles (
                    article_id, dedupe_key, url, title, publisher,
                    published_at, provider, fetched_at_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dedupe_key) DO UPDATE SET
                    url = excluded.url,
                    title = excluded.title,
                    publisher = excluded.publisher,
                    published_at = excluded.published_at,
                    provider = excluded.provider,
                    fetched_at_utc = excluded.fetched_at_utc
                """,
                (
                    article_id,
                    article_dedupe_key(article),
                    str(article.get("url") or "").strip(),
                    str(article.get("title") or "").strip(),
                    str(
                        article.get("site")
                        or article.get("publisher")
                        or article.get("source")
                        or ""
                    ).strip(),
                    _published_at(article),
                    provider,
                    fetched_at,
                ),
            )

            for ticker in _symbols_for(article):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO ta_article_symbols (article_id, ticker)
                    VALUES (?, ?)
                    """,
                    (article_id, ticker),
                )
            inserted_or_updated += 1
        conn.commit()
    finally:
        conn.close()
    return inserted_or_updated


def load_articles_since(
    cutoff_date: str,
    *,
    provider: str = "fmp_stock_news",
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    ensure_cache_db(db_path)
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                a.article_id,
                a.url,
                a.title,
                a.publisher,
                a.published_at,
                s.ticker
            FROM ta_articles a
            JOIN ta_article_symbols s ON s.article_id = a.article_id
            WHERE a.provider = ?
              AND substr(COALESCE(a.published_at, ''), 1, 10) >= ?
            ORDER BY a.published_at DESC, a.article_id, s.ticker
            """,
            (provider, cutoff_date),
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "article_id": row["article_id"],
            "url": row["url"],
            "title": row["title"],
            "site": row["publisher"],
            "publishedDate": row["published_at"],
            "symbol": row["ticker"],
        }
        for row in rows
    ]


def load_articles_for_ticker(
    ticker: str,
    *,
    provider: str = "fmp_stock_news",
    limit: int | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    ticker_key = _normalize_ticker(ticker)
    if not ticker_key:
        return []

    ensure_cache_db(db_path)
    limit_clause = ""
    params: list[Any] = [provider, ticker_key]
    if limit is not None:
        if limit <= 0:
            return []
        limit_clause = "LIMIT ?"
        params.append(int(limit))

    conn = connect(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT
                a.article_id,
                a.url,
                a.title,
                a.publisher,
                a.published_at,
                s.ticker
            FROM ta_articles a
            JOIN ta_article_symbols s ON s.article_id = a.article_id
            WHERE a.provider = ?
              AND s.ticker = ?
            ORDER BY a.published_at DESC, a.article_id
            {limit_clause}
            """,
            tuple(params),
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "article_id": row["article_id"],
            "url": row["url"],
            "title": row["title"],
            "site": row["publisher"],
            "publishedDate": row["published_at"],
            "symbol": row["ticker"],
        }
        for row in rows
    ]


def latest_article_published_at(
    *,
    provider: str = "fmp_stock_news",
    db_path: str | Path | None = None,
) -> str | None:
    ensure_cache_db(db_path)
    conn = connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT MAX(published_at) AS latest_published_at
            FROM ta_articles
            WHERE provider = ?
              AND COALESCE(published_at, '') <> ''
            """,
            (provider,),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return None
    return row["latest_published_at"] or None


def article_cache_has_coverage(
    cutoff_date: str,
    *,
    provider: str = "fmp_stock_news",
    db_path: str | Path | None = None,
    now_utc: datetime | None = None,
) -> bool:
    ensure_cache_db(db_path)
    today = (now_utc or datetime.now(timezone.utc)).date().isoformat()
    conn = connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS row_count,
                MIN(substr(COALESCE(published_at, ''), 1, 10)) AS oldest_date,
                MAX(substr(COALESCE(fetched_at_utc, ''), 1, 10)) AS newest_fetch_date
            FROM ta_articles
            WHERE provider = ?
            """,
            (provider,),
        ).fetchone()
    finally:
        conn.close()

    if not row or int(row["row_count"] or 0) == 0:
        return False
    oldest_date = row["oldest_date"] or ""
    newest_fetch_date = row["newest_fetch_date"] or ""
    return oldest_date <= cutoff_date and newest_fetch_date >= today


def upsert_daily_bars(
    rows: list[dict[str, Any]],
    *,
    provider: str,
    fetched_at_utc: str | None = None,
    db_path: str | Path | None = None,
) -> int:
    if not rows:
        return 0

    ensure_cache_db(db_path)
    fetched_at = fetched_at_utc or utc_now_iso()
    conn = connect(db_path)
    written = 0
    try:
        for row in rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            bar_date = str(row.get("bar_date") or "").strip()
            if not ticker or not bar_date:
                continue
            conn.execute(
                """
                INSERT INTO ta_historical_daily_bars (
                    ticker, bar_date, provider, open, high, low, close,
                    adj_close, volume, fetched_at_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, bar_date, provider) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    adj_close = excluded.adj_close,
                    volume = excluded.volume,
                    fetched_at_utc = excluded.fetched_at_utc
                """,
                (
                    ticker,
                    bar_date,
                    provider,
                    row.get("open"),
                    row.get("high"),
                    row.get("low"),
                    row.get("close"),
                    row.get("adj_close"),
                    row.get("volume"),
                    fetched_at,
                ),
            )
            written += 1
        conn.commit()
    finally:
        conn.close()
    return written


def load_daily_bars(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    provider: str,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    ensure_cache_db(db_path)
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                ticker, bar_date, open, high, low, close, adj_close, volume
            FROM ta_historical_daily_bars
            WHERE ticker = ?
              AND provider = ?
              AND bar_date >= ?
              AND bar_date <= ?
            ORDER BY bar_date ASC
            """,
            (ticker.upper(), provider, start_date, end_date),
        ).fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]


def latest_bar_date(
    ticker: str,
    *,
    provider: str,
    db_path: str | Path | None = None,
) -> str | None:
    ensure_cache_db(db_path)
    conn = connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT MAX(bar_date) AS latest_date
            FROM ta_historical_daily_bars
            WHERE ticker = ? AND provider = ?
            """,
            (ticker.upper(), provider),
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else row["latest_date"]


def _normalize_ticker_list(tickers: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in tickers:
        ticker = _normalize_ticker(value)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


def upsert_stock_pick_snapshot(
    pick_date: str,
    tickers: list[str],
    *,
    db_path: str | Path | None = None,
) -> None:
    date_key = str(pick_date or "").strip()
    if not date_key:
        raise ValueError("pick_date is required")

    normalized = _normalize_ticker_list(tickers)
    ensure_cache_db(db_path)
    conn = connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO ta_stock_pick_snapshots (pick_date, tickers_json)
            VALUES (?, ?)
            ON CONFLICT(pick_date) DO UPDATE SET
                tickers_json = excluded.tickers_json
            """,
            (date_key, json.dumps(normalized)),
        )
        conn.commit()
    finally:
        conn.close()


def load_stock_pick_snapshot(
    pick_date: str,
    *,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    date_key = str(pick_date or "").strip()
    if not date_key:
        return None

    ensure_cache_db(db_path)
    conn = connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT pick_date, tickers_json
            FROM ta_stock_pick_snapshots
            WHERE pick_date = ?
            """,
            (date_key,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    try:
        raw_tickers = json.loads(row["tickers_json"])
    except (TypeError, json.JSONDecodeError):
        raw_tickers = []
    if not isinstance(raw_tickers, list):
        raw_tickers = []

    return {
        "pick_date": row["pick_date"],
        "tickers": _normalize_ticker_list([str(ticker) for ticker in raw_tickers]),
    }


def load_stock_pick_dates(
    *,
    db_path: str | Path | None = None,
) -> list[str]:
    ensure_cache_db(db_path)
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT pick_date
            FROM ta_stock_pick_snapshots
            ORDER BY pick_date DESC
            """
        ).fetchall()
    finally:
        conn.close()
    return [str(row["pick_date"]) for row in rows]


def upsert_fundamental_analysis(
    *,
    ticker: str,
    answer: str,
    company_name: str | None = None,
    query: str | None = None,
    decision: str | None = None,
    route_type: str | None = None,
    fundamental_assessment: str | None = None,
    fundamental_score: int | None = None,
    finance_context_present: bool | None = None,
    news_context_present: bool | None = None,
    news_query: str | None = None,
    news_item_count: int | None = None,
    retrieval_trace_json: str | None = None,
    logs_json: str | None = None,
    generated_at_utc: str | None = None,
    updated_at_utc: str | None = None,
    db_path: str | Path | None = None,
) -> None:
    ticker_key = _normalize_ticker(ticker)
    if not ticker_key:
        raise ValueError("ticker is required")
    if not str(answer or "").strip():
        raise ValueError("answer is required")

    ensure_cache_db(db_path)
    now = utc_now_iso()
    generated_at = generated_at_utc or now
    updated_at = updated_at_utc or now
    conn = connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO ta_fundamental_analyses (
                ticker, company_name, query, answer, decision, route_type,
                fundamental_assessment, fundamental_score,
                finance_context_present, news_context_present,
                news_query, news_item_count, retrieval_trace_json, logs_json,
                generated_at_utc, updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                company_name = excluded.company_name,
                query = excluded.query,
                answer = excluded.answer,
                decision = excluded.decision,
                route_type = excluded.route_type,
                fundamental_assessment = excluded.fundamental_assessment,
                fundamental_score = excluded.fundamental_score,
                finance_context_present = excluded.finance_context_present,
                news_context_present = excluded.news_context_present,
                news_query = excluded.news_query,
                news_item_count = excluded.news_item_count,
                retrieval_trace_json = excluded.retrieval_trace_json,
                logs_json = excluded.logs_json,
                generated_at_utc = excluded.generated_at_utc,
                updated_at_utc = excluded.updated_at_utc
            """,
            (
                ticker_key,
                company_name,
                query,
                answer,
                decision,
                route_type,
                fundamental_assessment,
                fundamental_score,
                None if finance_context_present is None else int(bool(finance_context_present)),
                None if news_context_present is None else int(bool(news_context_present)),
                news_query,
                news_item_count,
                retrieval_trace_json,
                logs_json,
                generated_at,
                updated_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_fundamental_analysis(
    ticker: str,
    *,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    ticker_key = _normalize_ticker(ticker)
    if not ticker_key:
        return None

    ensure_cache_db(db_path)
    conn = connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT
                ticker, company_name, query, answer, decision, route_type,
                fundamental_assessment, fundamental_score,
                finance_context_present, news_context_present,
                news_query, news_item_count, retrieval_trace_json, logs_json,
                generated_at_utc, updated_at_utc
            FROM ta_fundamental_analyses
            WHERE ticker = ?
            """,
            (ticker_key,),
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else dict(row)


def load_fresh_fundamental_analysis(
    ticker: str,
    *,
    max_age_days: int = 90,
    now_utc: datetime | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    row = load_fundamental_analysis(ticker, db_path=db_path)
    if row is None:
        return None
    if str(row.get("decision") or "").lower() != "answer":
        return None
    if row.get("route_type") != "single_ticker_financial":
        return None
    if int(row.get("finance_context_present") or 0) != 1:
        return None
    if not str(row.get("answer") or "").strip():
        return None

    generated_at = _parse_utc_datetime(row.get("generated_at_utc", ""))
    if generated_at is None:
        return None

    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    if now - generated_at <= timedelta(days=max_age_days):
        return row
    return None
