"""
simple_scraper_v2.py - Alpha-primary news scraper with RSS fallback.

Key features:
  1. Alpha Vantage NEWS_SENTIMENT link discovery (primary, direct HTTP)
  2. RSS discovery fallback when Alpha is unavailable or quota-limited
  3. Async article scraping with Trafilatura extraction
  4. Deduplication delegated to ingestion pipeline (single authority)

Usage:
    python simple_scraper_v2.py
    python simple_scraper_v2.py --reset-db
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sqlite3
import time
import re
from html import unescape
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse, urljoin

try:
    import aiohttp
except ImportError:  # pragma: no cover - exercised in dependency-light envs
    aiohttp = None

import feedparser
import requests
from dotenv import load_dotenv
try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - exercised in dependency-light envs
    BeautifulSoup = None
try:
    import trafilatura
except ImportError:  # pragma: no cover - exercised in dependency-light envs
    trafilatura = None
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RSS_SOURCES = [
    {
        "name": "CNBC",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362",
    },
    {
        "name": "BBC",
        "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
    },
    {
        "name": "Nasdaq",
        "url": "https://www.nasdaq.com/feed/rssoutbound?category=Markets",
    },
    {
        "name": "Nasdaq",
        "url": "https://www.nasdaq.com/feed/rssoutbound?category=Investing",
    },
    {
        "name": "CBS MoneyWatch",
        "url": "https://www.cbsnews.com/latest/rss/moneywatch",
    },
    {
        "name": "OilPrice.com",
        "url": "https://oilprice.com/rss/main",
    },
    {
        "name": "Federal Reserve", 
        "url": "https://www.federalreserve.gov/feeds/press_all.xml",
        "content_from_feed": True
    },
    {
        "name": "BLS",
        "url": "https://www.bls.gov/feed/bls_latest.rss",
        "content_from_feed": True
    },
    {
        "name": "US Treasury",
        "url": "https://home.treasury.gov/system/files/press-releases/rss.xml",
        "content_from_feed": True
    },
]

DEFAULT_ALPHA_TOPICS = (
    "financial_markets,economy_macro,economy_monetary,finance,mergers_and_acquisitions"
)
ALPHA_NEWS_ENDPOINT = "https://www.alphavantage.co/query"
ALPHA_SENTIMENT_FUNCTION = "NEWS_SENTIMENT"
ALPHA_SOURCE_FEED = "alpha://news_sentiment"

OUTPUT_JSON = os.getenv("OUTPUT_JSON", "cnbc_articles.json")
SQLITE_DB = os.getenv("SQLITE_DB", "my_database.db")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))
CONCURRENT_REQUESTS = int(os.getenv("CONCURRENT_REQUESTS", "10"))

NEWS_PRIMARY_SOURCE = os.getenv("NEWS_PRIMARY_SOURCE", "alpha").strip().lower()
NEWS_FALLBACK_SOURCE = os.getenv("NEWS_FALLBACK_SOURCE", "rss").strip().lower()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
ALPHA_NEWS_TOPICS = os.getenv("ALPHA_NEWS_TOPICS", DEFAULT_ALPHA_TOPICS).strip()
ALPHA_NEWS_LIMIT = int(os.getenv("ALPHA_NEWS_LIMIT", "120"))
ALPHA_DAILY_BUDGET = int(os.getenv("ALPHA_DAILY_BUDGET", "20"))
ALPHA_TIME_WINDOW_HOURS = int(os.getenv("ALPHA_TIME_WINDOW_HOURS", "72"))
ALPHA_PAGE_LIMIT = int(os.getenv("ALPHA_PAGE_LIMIT", "50"))
ALPHA_SORT = os.getenv("ALPHA_SORT", "LATEST").strip().upper()

HTTP_MAX_ATTEMPTS = int(os.getenv("HTTP_MAX_ATTEMPTS", "4"))
HTTP_BACKOFF_BASE_SECONDS = float(os.getenv("HTTP_BACKOFF_BASE_SECONDS", "0.75"))
HTTP_BACKOFF_MAX_SECONDS = float(os.getenv("HTTP_BACKOFF_MAX_SECONDS", "8"))
HTTP_BACKOFF_JITTER_SECONDS = float(os.getenv("HTTP_BACKOFF_JITTER_SECONDS", "0.35"))

ARTICLE_FETCH_MAX_ATTEMPTS = int(os.getenv("ARTICLE_FETCH_MAX_ATTEMPTS", "3"))

RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

MIN_ARTICLE_WORDS = int(os.getenv("MIN_ARTICLE_WORDS", "80"))
MIN_FEED_WORDS = int(os.getenv("MIN_FEED_WORDS", "40"))
ENABLE_TRADINGECONOMICS_STREAM = os.getenv("ENABLE_TRADINGECONOMICS_STREAM", "0").strip().lower() in {"1", "true", "yes", "on"}
TRADINGECONOMICS_STREAM_URL = os.getenv("TRADINGECONOMICS_STREAM_URL", "https://tradingeconomics.com/stream").strip()
TRADINGECONOMICS_MAX_ITEMS = int(os.getenv("TRADINGECONOMICS_MAX_ITEMS", "60"))
TRADINGECONOMICS_MIN_WORDS = int(os.getenv("TRADINGECONOMICS_MIN_WORDS", "8"))
TRADINGECONOMICS_MAX_WORDS = int(os.getenv("TRADINGECONOMICS_MAX_WORDS", "240"))
TRADINGECONOMICS_MAX_AGE_HOURS = int(os.getenv("TRADINGECONOMICS_MAX_AGE_HOURS", "24"))
TRADINGECONOMICS_STREAM_DEBUG = os.getenv("TRADINGECONOMICS_STREAM_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
TRADINGECONOMICS_DEBUG_HTML_PATH = os.getenv("TRADINGECONOMICS_DEBUG_HTML_PATH", "").strip()
TRADINGECONOMICS_STRICT_FILTER = os.getenv("TRADINGECONOMICS_STRICT_FILTER", "0").strip().lower() in {"1", "true", "yes", "on"}

_TE_STREAM_NAV_TERMS = (
    "home", "markets", "calendar", "forecast", "indicators", "news",
    "stream", "portfolio", "screeners", "currencies", "commodities",
    "stocks", "bonds", "crypto", "watchlist", "screener", "login",
    "sign in", "sign up", "pricing", "advertise", "download app",
)
_TE_STREAM_EXCLUDE_HINTS = (
    "quote", "quotes", "board", "market-board", "calendar", "ticker",
    "watchlist", "navbar", "header", "footer", "menu", "sidenav",
    "breadcrumb", "search", "chart", "price-table",
)
_TE_STREAM_ITEM_HINTS = (
    "stream", "timeline", "news", "event", "update", "story", "brief",
)
_TE_STREAM_MACRO_TERMS = (
    "inflation", "cpi", "pce", "gdp", "unemployment", "payroll",
    "rate", "interest", "central bank", "federal reserve", "ecb",
    "boj", "pboc", "treasury", "bond", "yield", "tariff", "sanction",
    "blockade", "supply disruption", "trade", "oil", "gas", "energy",
    "commodity", "currency", "fx", "fiscal", "monetary", "deficit",
    "export", "import", "economic", "economy",
)
_TE_STREAM_STOCK_BLURB_TERMS = (
    "stock", "shares", "eps", "quarter", "guidance", "buyback",
    "dividend", "earnings",
)
_TE_STREAM_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://tradingeconomics.com/",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _backoff_delay(attempt: int) -> float:
    base = min(HTTP_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), HTTP_BACKOFF_MAX_SECONDS)
    jitter = random.uniform(0.0, HTTP_BACKOFF_JITTER_SECONDS)
    return base + jitter


def _is_retryable_status(status: int) -> bool:
    return status in RETRYABLE_STATUS_CODES


def _alpha_time(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M")


def _parse_alpha_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _normalize_alpha_published(raw: str | None) -> str | None:
    dt = _parse_alpha_datetime(raw)
    if dt is None:
        return None
    return dt.isoformat()


def _split_topics(raw_topics: str) -> list[str]:
    return [topic.strip() for topic in raw_topics.split(",") if topic.strip()]


def normalize_alpha_feed_item(item: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    url = (item.get("url") or "").strip()
    if not url:
        return None

    source_name = (item.get("source") or "Alpha Vantage").strip() or "Alpha Vantage"

    return (
        url,
        {
            "name": source_name,
            "url": ALPHA_SOURCE_FEED,
            "provider": "alpha",
            "published": _normalize_alpha_published(item.get("time_published")),
            "title_hint": item.get("title"),
        },
    )


def _alpha_quota_or_rate_limit(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False

    note_text = " ".join(
        str(payload.get(k, ""))
        for k in ("Note", "Information", "Error Message")
    ).lower()

    limit_markers = (
        "call frequency",
        "rate limit",
        "higher api call",
        "premium",
        "25 requests",
        "retry",
    )
    return any(marker in note_text for marker in limit_markers)


def fetch_json_with_backoff(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    timeout: int = REQUEST_TIMEOUT,
    max_attempts: int = HTTP_MAX_ATTEMPTS,
    request_headers: dict[str, str] | None = None,
) -> tuple[Any | None, str | None, int]:
    """
    Returns (payload, error, requests_made).
    """
    requests_made = 0
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        requests_made += 1
        try:
            resp = session.get(
                url,
                params=params,
                timeout=timeout,
                headers=request_headers or {"User-Agent": "Mozilla/5.0"},
            )

            if _is_retryable_status(resp.status_code):
                raise requests.exceptions.HTTPError(
                    f"retryable_http_status: {resp.status_code}",
                    response=resp,
                )

            resp.raise_for_status()
            return resp.json(), None, requests_made

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = f"network_error: {type(exc).__name__}"
            retryable = True
        except requests.exceptions.HTTPError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = bool(status_code and _is_retryable_status(status_code))
            if status_code:
                last_error = f"http_error: {status_code}"
            else:
                last_error = f"http_error: {type(exc).__name__}"
        except ValueError as exc:
            last_error = f"json_decode_error: {exc}"
            retryable = False
        except requests.exceptions.RequestException as exc:
            last_error = f"request_error: {type(exc).__name__}"
            retryable = False
        except Exception as exc:
            last_error = f"error: {str(exc)[:120]}"
            retryable = False

        if not retryable or attempt >= max_attempts:
            break

        time.sleep(_backoff_delay(attempt))

    return None, last_error, requests_made


def fetch_text_with_backoff(
    session: requests.Session,
    url: str,
    timeout: int = REQUEST_TIMEOUT,
    max_attempts: int = HTTP_MAX_ATTEMPTS,
    request_headers: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(
                url,
                timeout=timeout,
                headers=request_headers or {"User-Agent": "Mozilla/5.0"},
            )

            if _is_retryable_status(resp.status_code):
                raise requests.exceptions.HTTPError(
                    f"retryable_http_status: {resp.status_code}",
                    response=resp,
                )

            resp.raise_for_status()
            return resp.text, None

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = f"network_error: {type(exc).__name__}"
            retryable = True
        except requests.exceptions.HTTPError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = bool(status_code and _is_retryable_status(status_code))
            if status_code:
                last_error = f"http_error: {status_code}"
            else:
                last_error = f"http_error: {type(exc).__name__}"
        except requests.exceptions.RequestException as exc:
            last_error = f"request_error: {type(exc).__name__}"
            retryable = False
        except Exception as exc:
            last_error = f"error: {str(exc)[:120]}"
            retryable = False

        if not retryable or attempt >= max_attempts:
            break

        time.sleep(_backoff_delay(attempt))

    return None, last_error


# ---------------------------------------------------------------------------
# TradingEconomics stream briefs (short-form, low-context auxiliary source)
# ---------------------------------------------------------------------------

def _te_norm_text(text: str) -> str:
    cleaned = unescape(str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _te_stream_title(text: str) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    if sentence:
        return sentence[:160]
    words = text.split()
    return " ".join(words[:12]).strip() or "TradingEconomics Stream Brief"


def _te_symbol_density(text: str) -> float:
    if not text:
        return 1.0
    symbol_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return symbol_count / max(len(text), 1)


def _te_numeric_density(text: str) -> float:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return 1.0
    numeric_like = sum(
        1
        for tok in tokens
        if re.fullmatch(r"[+\-]?\$?[0-9.,:%/]+[kmb]?", tok.lower())
    )
    return numeric_like / max(len(tokens), 1)


def _te_is_stock_blurb(text: str) -> bool:
    lower = text.lower()
    has_stock_term = any(term in lower for term in _TE_STREAM_STOCK_BLURB_TERMS)
    has_macro_term = any(term in lower for term in _TE_STREAM_MACRO_TERMS)
    return has_stock_term and not has_macro_term


def is_valid_te_stream_item(text: str) -> bool:
    # Base filter: keep broad stream coverage and let downstream ranking score quality.
    clean = _te_norm_text(text)
    if not clean:
        return False

    words = re.findall(r"[A-Za-z0-9$%.-]+", clean)
    word_count = len(words)
    if word_count < TRADINGECONOMICS_MIN_WORDS or word_count > TRADINGECONOMICS_MAX_WORDS:
        return False

    lower = clean.lower()
    if any(nav in lower for nav in _TE_STREAM_NAV_TERMS):
        nav_hits = sum(1 for nav in _TE_STREAM_NAV_TERMS if nav in lower)
        if nav_hits >= 4:
            return False

    alpha_tokens = sum(1 for tok in words if re.search(r"[A-Za-z]", tok))
    if alpha_tokens / max(word_count, 1) < 0.35:
        return False

    if not TRADINGECONOMICS_STRICT_FILTER:
        return True

    # Optional strict filter for users who only want macro-heavy, low-noise briefs.
    if _te_symbol_density(clean) > 0.24:
        return False
    if _te_numeric_density(clean) > 0.55:
        return False

    if _te_is_stock_blurb(clean):
        return False

    if re.search(r"\b(open|high|low|close|bid|ask|volume|change)\b", lower) and re.search(r"\b\d", lower):
        return False
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:up|down)\s*\d+(?:\.\d+)?%?\b", lower):
        return False

    return True


def _te_parse_stream_date(raw: str, *, now_utc: datetime) -> datetime | None:
    clean = _te_norm_text(raw)
    if not clean:
        return None

    lower = clean.lower()
    if lower in {"now", "just now", "moments ago"}:
        return now_utc
    if lower == "yesterday":
        return now_utc - timedelta(days=1)

    rel_match = re.search(
        r"\b(?:(an?|one)|(\d+))\s*(second|sec|minute|min|hour|hr|day|week|month|year)s?\s+ago\b",
        lower,
    )
    if rel_match:
        qty = 1 if (rel_match.group(1) or "").strip() else int(rel_match.group(2))
        unit = rel_match.group(3)
        if unit in {"second", "sec"}:
            delta = timedelta(seconds=qty)
        elif unit in {"minute", "min"}:
            delta = timedelta(minutes=qty)
        elif unit in {"hour", "hr"}:
            delta = timedelta(hours=qty)
        elif unit == "day":
            delta = timedelta(days=qty)
        elif unit == "week":
            delta = timedelta(days=7 * qty)
        elif unit == "month":
            delta = timedelta(days=30 * qty)
        else:
            delta = timedelta(days=365 * qty)
        return now_utc - delta

    iso_raw = clean.replace("Z", "+00:00")
    try:
        parsed_iso = datetime.fromisoformat(iso_raw)
        if parsed_iso.tzinfo is None:
            parsed_iso = parsed_iso.replace(tzinfo=timezone.utc)
        return parsed_iso.astimezone(timezone.utc)
    except ValueError:
        pass

    compact = clean.replace("-", "").replace(":", "").replace("Z", "")
    parsed_alpha = _parse_alpha_datetime(compact)
    if parsed_alpha is not None:
        return parsed_alpha

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%b %d, %Y %I:%M %p",
        "%b %d, %Y %H:%M",
        "%d %b %Y %H:%M",
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(clean, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _te_extract_published(node, *, now_utc: datetime) -> datetime | None:
    if node is None:
        return None

    stream_date_tag = node.select_one(".te-stream-date")
    if stream_date_tag is not None:
        raw_stream_date = (stream_date_tag.get_text(" ", strip=True) or "").strip()
        parsed_stream_date = _te_parse_stream_date(raw_stream_date, now_utc=now_utc)
        if parsed_stream_date is not None:
            return parsed_stream_date

    time_tag = node.find("time")
    if time_tag is not None:
        raw_time = (time_tag.get("datetime") or time_tag.get_text(" ", strip=True) or "").strip()
        parsed_time = _te_parse_stream_date(raw_time, now_utc=now_utc)
        if parsed_time is not None:
            return parsed_time
    return None


def _te_container_is_excluded(node) -> bool:
    attrs = " ".join(
        str(val)
        for attr in ("id", "class", "role", "data-module", "data-widget")
        for val in ([node.get(attr)] if node.get(attr) is not None else [])
    ).lower()
    return any(hint in attrs for hint in _TE_STREAM_EXCLUDE_HINTS)


def _te_extract_regex_candidates(
    html: str,
    *,
    now_utc: datetime,
) -> list[tuple[str, datetime | None]]:
    candidates: list[tuple[str, datetime | None]] = []

    li_blocks = re.findall(
        r"<li[^>]*class=[\"'][^\"']*te-stream-item[^\"']*[\"'][^>]*>(.*?)</li>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block in li_blocks:
        desc_match = re.search(
            r"<(?:span|div|p)[^>]*class=[\"'][^\"']*te-stream-item-description[^\"']*[\"'][^>]*>(.*?)</(?:span|div|p)>",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not desc_match:
            continue
        desc_raw = re.sub(r"<[^>]+>", " ", desc_match.group(1))
        text = _te_norm_text(desc_raw)
        if not text:
            continue

        published_dt: datetime | None = None
        date_match = re.search(
            r"<(?:small|span|div)[^>]*class=[\"'][^\"']*te-stream-date[^\"']*[\"'][^>]*>(.*?)</(?:small|span|div)>",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if date_match:
            date_raw = _te_norm_text(re.sub(r"<[^>]+>", " ", date_match.group(1)))
            published_dt = _te_parse_stream_date(date_raw, now_utc=now_utc)

        candidates.append((text, published_dt))

    if candidates:
        return candidates

    for desc_match in re.finditer(
        r"<(?:span|div|p)[^>]*class=[\"'][^\"']*te-stream-item-description[^\"']*[\"'][^>]*>(.*?)</(?:span|div|p)>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        desc_raw = re.sub(r"<[^>]+>", " ", desc_match.group(1))
        text = _te_norm_text(desc_raw)
        if text:
            candidates.append((text, None))
    return candidates


def _te_extract_script_candidates(
    soup,
    *,
    now_utc: datetime,
) -> list[tuple[str, datetime | None]]:
    candidates: list[tuple[str, datetime | None]] = []
    for script in soup.find_all("script"):
        raw = script.string or script.get_text(" ", strip=False) or ""
        if not raw:
            continue
        lower_raw = raw.lower()
        if "te-stream-item-description" not in lower_raw and "indc_news_stream" not in lower_raw:
            continue
        normalized = unescape(raw).replace('\\"', '"').replace("\\/", "/")
        candidates.extend(_te_extract_regex_candidates(normalized, now_utc=now_utc))
    return candidates


def _te_fetch_stream_api_records(
    session: requests.Session,
    *,
    now_utc: datetime,
) -> tuple[list[dict[str, Any]], dict[str, int], str | None]:
    parsed = urlparse(TRADINGECONOMICS_STREAM_URL)
    if not parsed.scheme or not parsed.netloc:
        return [], {"invalid_stream_url": 1}, "invalid_stream_url"

    stream_api_url = f"{parsed.scheme}://{parsed.netloc}/ws/stream.ashx"
    payload, error, _ = fetch_json_with_backoff(
        session=session,
        url=stream_api_url,
        params={"start": 0, "size": max(TRADINGECONOMICS_MAX_ITEMS, 20)},
        timeout=REQUEST_TIMEOUT,
        max_attempts=HTTP_MAX_ATTEMPTS,
        request_headers=_TE_STREAM_REQUEST_HEADERS,
    )
    if error:
        return [], {"api_request_error": 1}, error
    if not isinstance(payload, list):
        return [], {"unexpected_api_payload": 1}, "unexpected_api_payload"

    records: list[dict[str, Any]] = []
    reject_stats: defaultdict[str, int] = defaultdict(int)
    seen_urls: set[str] = set()
    max_age = timedelta(hours=max(TRADINGECONOMICS_MAX_AGE_HOURS, 1))

    for row in payload:
        if len(records) >= max(TRADINGECONOMICS_MAX_ITEMS, 1):
            break
        if not isinstance(row, dict):
            reject_stats["invalid_row_type"] += 1
            continue

        desc_raw = row.get("description") or row.get("html") or ""
        text = _te_norm_text(re.sub(r"<[^>]+>", " ", str(desc_raw)))
        if not text:
            reject_stats["empty_description"] += 1
            continue
        if not is_valid_te_stream_item(text):
            reject_stats["invalid_text_filter"] += 1
            continue

        published_dt: datetime | None = None
        date_raw = _te_norm_text(str(row.get("date") or ""))
        if date_raw:
            published_dt = _te_parse_stream_date(date_raw, now_utc=now_utc)
        if published_dt is None:
            diff_raw = _te_norm_text(str(row.get("diff") or ""))
            if diff_raw:
                published_dt = _te_parse_stream_date(diff_raw, now_utc=now_utc)
        if published_dt is None:
            reject_stats["missing_or_unparseable_date"] += 1
            continue
        if published_dt > now_utc + timedelta(minutes=5):
            reject_stats["future_timestamp"] += 1
            continue
        if (now_utc - published_dt) > max_age:
            reject_stats["older_than_24h"] += 1
            continue

        row_url = _te_norm_text(str(row.get("url") or ""))
        row_id = _te_norm_text(str(row.get("ID") or ""))
        if row_url:
            canonical_url = row_url if row_url.startswith(("http://", "https://")) else urljoin(TRADINGECONOMICS_STREAM_URL, row_url)
            if row_id and "/news/" not in canonical_url:
                canonical_url = canonical_url.rstrip("/") + f"/news/{row_id}"
        else:
            digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:14]
            canonical_url = f"{TRADINGECONOMICS_STREAM_URL}#brief-{digest}"

        if canonical_url in seen_urls:
            reject_stats["deduplicated"] += 1
            continue
        seen_urls.add(canonical_url)

        title = _te_norm_text(str(row.get("title") or "")) or _te_stream_title(text)
        records.append(
            {
                "url": canonical_url,
                "source": "TradingEconomics",
                "source_rss": "https://tradingeconomics.com/stream",
                "status": "ok",
                "error": None,
                "title": title[:160],
                "published": published_dt.isoformat(),
                "text": text,
                "author": None,
                "source_provider": "stream",
            }
        )

    return records, dict(reject_stats), None


def fetch_tradingeconomics_stream() -> list[dict[str, Any]]:
    if not TRADINGECONOMICS_STREAM_URL:
        return []

    now_utc = datetime.now(timezone.utc)
    with requests.Session() as session:
        api_records, api_rejects, api_error = _te_fetch_stream_api_records(
            session=session,
            now_utc=now_utc,
        )
        if api_records:
            if TRADINGECONOMICS_STREAM_DEBUG:
                print(
                    "[te_stream] API stream accepted: "
                    f"{len(api_records)} (rejects: {api_rejects})"
                )
            return api_records
        if TRADINGECONOMICS_STREAM_DEBUG:
            print(
                "[te_stream] API stream empty; falling back to HTML parse "
                f"(error={api_error}, rejects={api_rejects})"
            )

        html, error = fetch_text_with_backoff(
            session=session,
            url=TRADINGECONOMICS_STREAM_URL,
            timeout=REQUEST_TIMEOUT,
            request_headers=_TE_STREAM_REQUEST_HEADERS,
        )
    if error or not html:
        print(f"[te_stream] fetch skipped: {error or 'no_html'}")
        return []
    if BeautifulSoup is None:
        print("[te_stream] BeautifulSoup unavailable; HTML fallback skipped")
        return []
    if TRADINGECONOMICS_DEBUG_HTML_PATH:
        try:
            with open(TRADINGECONOMICS_DEBUG_HTML_PATH, "w", encoding="utf-8") as debug_f:
                debug_f.write(html)
            print(f"[te_stream] debug html saved: {TRADINGECONOMICS_DEBUG_HTML_PATH}")
        except Exception as exc:
            print(f"[te_stream] debug html save failed: {type(exc).__name__}")

    soup = BeautifulSoup(html, "html.parser")
    candidate_nodes: list[tuple[Any | None, str, datetime | None]] = []
    seen_candidate_keys: set[str] = set()

    def _push_candidate(node: Any | None, text: str, published_dt: datetime | None = None) -> None:
        clean_text = _te_norm_text(text)
        if not clean_text:
            return
        if node is None:
            key = f"text:{hashlib.md5(clean_text.encode('utf-8')).hexdigest()[:16]}"
        else:
            key = f"node:{id(node)}"
        if key in seen_candidate_keys:
            return
        seen_candidate_keys.add(key)
        candidate_nodes.append((node, clean_text, published_dt))

    # Primary path: explicit TradingEconomics stream classes from the live DOM.
    for node in soup.select("#stream li.indc_news_stream, li.indc_news_stream, li[class*='indc_news_stream']"):
        if _te_container_is_excluded(node):
            continue
        desc_node = node.select_one("span.te-stream-item-description, .te-stream-item-description")
        text = (
            desc_node.get_text(" ", strip=True)
            if desc_node is not None
            else node.get_text(" ", strip=True)
        )
        _push_candidate(node, text, _te_extract_published(node, now_utc=now_utc))

    # Primary path: explicit TradingEconomics stream classes from the live DOM.
    for desc_node in soup.select(".te-stream-item-description"):
        owner = (
            desc_node.find_parent("li")
            or desc_node.find_parent("article")
            or desc_node.find_parent("div")
            or desc_node
        )
        if _te_container_is_excluded(owner):
            continue
        _push_candidate(
            owner,
            desc_node.get_text(" ", strip=True),
            _te_extract_published(owner, now_utc=now_utc),
        )

    # Secondary path: explicit stream item rows, even if description class is missing.
    for node in soup.select("#stream li.te-stream-item, li.te-stream-item, #stream li"):
        if _te_container_is_excluded(node):
            continue
        desc_node = node.select_one(".te-stream-item-description")
        text = (
            desc_node.get_text(" ", strip=True)
            if desc_node is not None
            else node.get_text(" ", strip=True)
        )
        _push_candidate(node, text, _te_extract_published(node, now_utc=now_utc))

    if TRADINGECONOMICS_STREAM_DEBUG:
        print(
            "[te_stream] selector counts: "
            f"desc={len(soup.select('.te-stream-item-description'))}, "
            f"items={len(soup.select('#stream li.te-stream-item, li.te-stream-item, #stream li'))}, "
            f"candidates={len(candidate_nodes)}"
        )

    # Fallback: looser container scan for resilience when TE tweaks class names.
    if not candidate_nodes:
        candidate_containers = []
        for node in soup.find_all(["section", "div", "main", "article", "ul", "ol"]):
            attrs = " ".join(
                str(val)
                for attr in ("id", "class", "data-module", "data-widget")
                for val in ([node.get(attr)] if node.get(attr) is not None else [])
            ).lower()
            if not attrs:
                continue
            if not any(hint in attrs for hint in _TE_STREAM_ITEM_HINTS):
                continue
            if _te_container_is_excluded(node):
                continue
            candidate_containers.append(node)

        for container in candidate_containers:
            for node in container.find_all(["li", "article", "div"], recursive=True):
                if _te_container_is_excluded(node):
                    continue
                desc_node = node.select_one(".te-stream-item-description")
                text = (
                    desc_node.get_text(" ", strip=True)
                    if desc_node is not None
                    else node.get_text(" ", strip=True)
                )
                _push_candidate(node, text, _te_extract_published(node, now_utc=now_utc))

    if not candidate_nodes:
        for text, published_dt in _te_extract_regex_candidates(html, now_utc=now_utc):
            _push_candidate(None, text, published_dt)

    if not candidate_nodes:
        for text, published_dt in _te_extract_script_candidates(soup, now_utc=now_utc):
            _push_candidate(None, text, published_dt)

    if not candidate_nodes:
        html_lower = html.lower()
        print(
            "[te_stream] ambiguous parse; no item candidates "
            f"(html_markers: desc={html_lower.count('te-stream-item-description')}, "
            f"item={html_lower.count('te-stream-item')}, len={len(html)})"
        )
        return []

    records: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    max_age = timedelta(hours=max(TRADINGECONOMICS_MAX_AGE_HOURS, 1))
    valid_count = 0
    reject_stats: defaultdict[str, int] = defaultdict(int)
    for node, raw_text, parsed_published_dt in candidate_nodes:
        if len(records) >= max(TRADINGECONOMICS_MAX_ITEMS, 1):
            break
        text = _te_norm_text(raw_text)
        if not is_valid_te_stream_item(text):
            reject_stats["invalid_text_filter"] += 1
            continue
        published_dt = parsed_published_dt
        if published_dt is None and node is not None:
            published_dt = _te_extract_published(node, now_utc=now_utc)
        if published_dt is None:
            reject_stats["missing_or_unparseable_date"] += 1
            continue
        if published_dt > now_utc + timedelta(minutes=5):
            reject_stats["future_timestamp"] += 1
            continue
        if (now_utc - published_dt) > max_age:
            reject_stats["older_than_24h"] += 1
            continue

        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:14]
        synthetic_url = f"{TRADINGECONOMICS_STREAM_URL}#brief-{digest}"
        if synthetic_url in seen_urls:
            reject_stats["deduplicated"] += 1
            continue
        seen_urls.add(synthetic_url)

        valid_count += 1
        records.append(
            {
                "url": synthetic_url,
                "source": "TradingEconomics",
                "source_rss": "https://tradingeconomics.com/stream",
                "status": "ok",
                "error": None,
                "title": _te_stream_title(text),
                "published": published_dt.isoformat(),
                "text": text,
                "author": None,
                "source_provider": "stream",
            }
        )

    if valid_count == 0:
        print(
            "[te_stream] no valid stream brief items after filtering "
            f"(rejects: {dict(reject_stats)})"
        )
        return []

    return records


# ---------------------------------------------------------------------------
# Discovery providers
# ---------------------------------------------------------------------------

def discover_links_primary_alpha(
    session: requests.Session,
    *,
    api_key: str = ALPHA_VANTAGE_API_KEY,
    topics: str = ALPHA_NEWS_TOPICS,
    target_limit: int = ALPHA_NEWS_LIMIT,
    daily_budget: int = ALPHA_DAILY_BUDGET,
    time_window_hours: int = ALPHA_TIME_WINDOW_HOURS,
    sort: str = ALPHA_SORT,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    topic_queries = _split_topics(topics)
    if not topic_queries:
        topic_queries = [""]

    meta: dict[str, Any] = {
        "provider": "alpha",
        "failed": False,
        "quota_limited": False,
        "calls_made": 0,
        "daily_budget": daily_budget,
        "topics": topics,
        "topic_queries": topic_queries,
        "target_limit": target_limit,
    }

    if not api_key:
        meta["failed"] = True
        meta["error"] = "missing_api_key"
        return [], {}, meta

    now_utc = datetime.now(timezone.utc)
    time_from = now_utc - timedelta(hours=max(time_window_hours, 1))

    target_limit = max(1, target_limit)
    daily_budget = max(1, daily_budget)
    page_limit = max(1, min(ALPHA_PAGE_LIMIT, target_limit))

    links: list[str] = []
    seen_urls: set[str] = set()
    source_by_url: dict[str, dict[str, Any]] = {}

    for topic in topic_queries:
        topic_time_to = now_utc

        while len(links) < target_limit and meta["calls_made"] < daily_budget:
            remaining = target_limit - len(links)
            this_page_limit = min(page_limit, remaining)

            params = {
                "function": ALPHA_SENTIMENT_FUNCTION,
                "apikey": api_key,
                "sort": sort,
                "limit": this_page_limit,
                "time_from": _alpha_time(time_from),
                "time_to": _alpha_time(topic_time_to),
            }
            if topic:
                params["topics"] = topic

            payload, error, calls = fetch_json_with_backoff(
                session=session,
                url=ALPHA_NEWS_ENDPOINT,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            meta["calls_made"] += calls

            if error:
                meta["failed"] = True
                meta["error"] = error
                break

            if _alpha_quota_or_rate_limit(payload):
                meta["quota_limited"] = True
                meta["error"] = "alpha_quota_or_rate_limit"
                break

            feed = payload.get("feed") if isinstance(payload, dict) else None
            if not isinstance(feed, list) or not feed:
                break

            oldest_dt: datetime | None = None
            new_links = 0

            for item in feed:
                if not isinstance(item, dict):
                    continue
                normalized = normalize_alpha_feed_item(item)
                if not normalized:
                    continue

                url, source_meta = normalized
                if url in seen_urls:
                    continue

                seen_urls.add(url)
                links.append(url)
                source_by_url[url] = source_meta
                new_links += 1

                item_dt = _parse_alpha_datetime(item.get("time_published"))
                if item_dt and (oldest_dt is None or item_dt < oldest_dt):
                    oldest_dt = item_dt

                if len(links) >= target_limit:
                    break

            if meta.get("failed") or meta.get("quota_limited"):
                break
            if new_links == 0:
                break
            if oldest_dt is None:
                break
            if len(feed) < this_page_limit:
                break

            next_time_to = oldest_dt - timedelta(minutes=1)
            if next_time_to <= time_from:
                break
            topic_time_to = next_time_to

        if meta.get("failed") or meta.get("quota_limited") or len(links) >= target_limit:
            break

    if meta["calls_made"] >= daily_budget and len(links) < target_limit:
        meta["quota_limited"] = True
        meta.setdefault("error", "alpha_daily_budget_exhausted")

    return links, source_by_url, meta


def discover_links_fallback_rss(
    session: requests.Session,
    sources: list[dict[str, str]] = RSS_SOURCES,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    links: list[str] = []
    seen: set[str] = set()
    source_by_url: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for source in sources:
        raw_text, error = fetch_text_with_backoff(session, source["url"], timeout=10)
        if error:
            errors.append(f"{source['name']}: {error}")
            continue

        resp = feedparser.parse(raw_text)
        if resp.bozo and resp.bozo_exception:
            errors.append(f"{source['name']}: parse_error: {resp.bozo_exception}")
            continue

        content_from_feed = source.get("content_from_feed", False)

        for entry in resp.entries:
            link = (entry.get("link") or "").strip()
            if not link or link in seen:
                continue
            seen.add(link)
            links.append(link)

            # Extract body text directly from feed entry if flagged
            feed_text = ""
            if content_from_feed:
                # Try content first, then summary, then description
                if entry.get("content"):
                    feed_text = entry["content"][0].get("value", "")
                elif entry.get("summary"):
                    feed_text = entry.get("summary", "")
                elif entry.get("description"):
                    feed_text = entry.get("description", "")
                
                # Strip HTML tags if present
                feed_text = re.sub(r"<[^>]+>", " ", feed_text)
                feed_text = re.sub(r"\s+", " ", feed_text).strip()

            source_by_url[link] = {
                "name": source.get("name"),
                "url": source.get("url"),
                "provider": "rss",
                "published": entry.get("published") or entry.get("updated"),
                "title_hint": entry.get("title"),
                # NEW: pre-extracted text bypasses page scraping
                "feed_text": feed_text if content_from_feed else "",
            }

    meta: dict[str, Any] = {
        "provider": "rss",
        "failed": len(links) == 0,
        "errors": errors,
    }
    return links, source_by_url, meta


def merge_discovered_links(
    primary_links: list[str],
    primary_source_by_url: dict[str, dict[str, Any]],
    fallback_links: list[str],
    fallback_source_by_url: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    merged_links: list[str] = []
    merged_source_by_url: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()

    for links, mapping in (
        (primary_links, primary_source_by_url),
        (fallback_links, fallback_source_by_url),
    ):
        for url in links:
            if url in seen:
                continue
            seen.add(url)
            merged_links.append(url)
            merged_source_by_url[url] = mapping.get(url, {})

    return merged_links, merged_source_by_url


def discover_links(
    session: requests.Session,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    metadata: dict[str, Any] = {
        "primary_source": NEWS_PRIMARY_SOURCE,
        "fallback_source": NEWS_FALLBACK_SOURCE,
        "fallback_used": False,
        "providers": [],
        "provider_meta": {},
    }

    if NEWS_PRIMARY_SOURCE == "alpha":
        primary_links, primary_map, primary_meta = discover_links_primary_alpha(session)
    elif NEWS_PRIMARY_SOURCE == "rss":
        primary_links, primary_map, primary_meta = discover_links_fallback_rss(session)
    else:
        primary_links, primary_map, primary_meta = [], {}, {
            "provider": NEWS_PRIMARY_SOURCE,
            "failed": True,
            "error": "unsupported_primary_source",
        }

    metadata["providers"].append(primary_meta.get("provider", NEWS_PRIMARY_SOURCE))
    metadata["provider_meta"][primary_meta.get("provider", NEWS_PRIMARY_SOURCE)] = primary_meta

    fallback_needed = (
        not primary_links
        or bool(primary_meta.get("failed"))
        or bool(primary_meta.get("quota_limited"))
    )

    if fallback_needed and NEWS_FALLBACK_SOURCE and NEWS_FALLBACK_SOURCE != NEWS_PRIMARY_SOURCE:
        if NEWS_FALLBACK_SOURCE == "rss":
            fallback_links, fallback_map, fallback_meta = discover_links_fallback_rss(session)
        elif NEWS_FALLBACK_SOURCE == "alpha":
            fallback_links, fallback_map, fallback_meta = discover_links_primary_alpha(session)
        else:
            fallback_links, fallback_map, fallback_meta = [], {}, {
                "provider": NEWS_FALLBACK_SOURCE,
                "failed": True,
                "error": "unsupported_fallback_source",
            }

        metadata["fallback_used"] = True
        metadata["providers"].append(fallback_meta.get("provider", NEWS_FALLBACK_SOURCE))
        metadata["provider_meta"][fallback_meta.get("provider", NEWS_FALLBACK_SOURCE)] = fallback_meta

        merged_links, merged_map = merge_discovered_links(
            primary_links,
            primary_map,
            fallback_links,
            fallback_map,
        )
        return merged_links, merged_map, metadata

    return primary_links, primary_map, metadata


# ---------------------------------------------------------------------------
# Trafilatura-based extraction
# ---------------------------------------------------------------------------

def extract_with_trafilatura(html: str, url: str) -> dict[str, Any]:
    if trafilatura is None:
        return {
            "title": None,
            "published": None,
            "text": "",
            "author": None,
        }

    result = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_precision=False,
        output_format="txt",
        with_metadata=True,
    )

    if result is None:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            no_fallback=False,
            output_format="txt",
        )
        return {
            "title": None,
            "published": None,
            "text": text or "",
            "author": None,
        }

    if isinstance(result, dict):
        return {
            "title": result.get("title"),
            "published": result.get("date"),
            "text": result.get("text", ""),
            "author": result.get("author"),
        }

    return {
        "title": None,
        "published": None,
        "text": result or "",
        "author": None,
    }


# ---------------------------------------------------------------------------
# Async scraping
# ---------------------------------------------------------------------------

async def fetch_html(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str | None, str | None]:
    async with semaphore:
        for attempt in range(1, ARTICLE_FETCH_MAX_ATTEMPTS + 1):
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                    headers={"User-Agent": "Mozilla/5.0"},
                ) as resp:
                    if _is_retryable_status(resp.status):
                        raise aiohttp.ClientResponseError(
                            request_info=resp.request_info,
                            history=resp.history,
                            status=resp.status,
                            message="retryable_status",
                            headers=resp.headers,
                        )

                    resp.raise_for_status()
                    html = await resp.text()
                    return (url, html, None)

            except asyncio.TimeoutError:
                error = "timeout"
                retryable = True
            except aiohttp.ClientResponseError as exc:
                error = f"client_response_error: {exc.status}"
                retryable = _is_retryable_status(exc.status)
            except aiohttp.ClientError as exc:
                error = f"client_error: {type(exc).__name__}"
                retryable = True
            except Exception as exc:
                error = f"error: {str(exc)[:100]}"
                retryable = False

            if not retryable or attempt >= ARTICLE_FETCH_MAX_ATTEMPTS:
                return (url, None, error)

            await asyncio.sleep(_backoff_delay(attempt))

    return (url, None, "unknown_error")


async def scrape_articles_async(
    urls: list[str],
    source_by_url: dict[str, dict[str, Any]],
    max_concurrent: int = CONCURRENT_REQUESTS,
) -> list[dict[str, Any]]:
    if aiohttp is None:
        raise RuntimeError("aiohttp is required for async scraping. Install: pip install aiohttp")

    semaphore = asyncio.Semaphore(max_concurrent)
    articles: list[dict[str, Any]] = []

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_html(session, url, semaphore) for url in urls]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Scraping articles"):
            url, html, error = await coro
            source_meta = source_by_url.get(url, {})

            article = {
                "url": url,
                "source": source_meta.get("name"),
                "source_rss": source_meta.get("url"),
                "status": "ok",
                "error": None,
                "title": None,
                "published": None,
                "text": None,
                "author": None,
                "source_provider": source_meta.get("provider"),
            }
            feed_text = source_meta.get("feed_text", "")
            if feed_text and len(feed_text.split()) >= MIN_FEED_WORDS:
                article["text"] = feed_text
                article["title"] = source_meta.get("title_hint")
                article["published"] = source_meta.get("published")
                articles.append(article)
                continue

            if error:
                article["status"] = "error"
                article["error"] = error
            elif html:
                extracted = extract_with_trafilatura(html, url)
                article.update(extracted)

                if not article.get("title") and source_meta.get("title_hint"):
                    article["title"] = source_meta.get("title_hint")
                if not article.get("published") and source_meta.get("published"):
                    article["published"] = source_meta.get("published")

                if not extracted.get("text") or len(extracted["text"].strip()) < 50:
                    article["status"] = "empty_text"
            else:
                article["status"] = "error"
                article["error"] = "no_html"

            articles.append(article)

    return articles


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _article_id(url: str, published_str: str | None) -> str:
    """Stable ID: MD5 of url::YYYY-MM-DD (matches tgrag_setup._article_id)."""
    try:
        dt = datetime.fromisoformat(published_str) if published_str else None
    except ValueError:
        dt = None
    date_str = dt.strftime("%Y-%m-%d") if dt else "nodate"
    return hashlib.md5(f"{url}::{date_str}".encode()).hexdigest()


def _content_hash(text: str) -> str:
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


_TRUST_TIER_1_DOMAINS = (
    "federalreserve.gov",
    "bls.gov",
    "treasury.gov",
    "imf.org",
    "worldbank.org",
    "cnbc.com",
    "bbc.com",
    "reuters.com",
)

_TRUST_TIER_2_DOMAINS = (
    "nasdaq.com",
    "cbsnews.com",
    "oilprice.com",
    "marketwatch.com",
)

_BLOCKED_SOURCE_DOMAINS = (
    "youtube.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "x.com",
    "twitter.com",
)

_OFFICIAL_SOURCE_HINTS = (
    "federal reserve",
    "us treasury",
    "bureau of labor statistics",
    "bls",
)


def _host_from_url(url: str) -> str:
    host = urlparse(url or "").netloc.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def _domain_matches(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == dom or host.endswith(f".{dom}") for dom in domains)


def _classify_source_trust_tier(source_name: str | None, url: str) -> str:
    return _classify_source_trust_tier_with_provider(
        source_name=source_name,
        url=url,
        source_provider=None,
    )


def _classify_source_trust_tier_with_provider(
    *,
    source_name: str | None,
    url: str,
    source_provider: str | None,
) -> str:
    host = _host_from_url(url)
    source_lower = (source_name or "").strip().lower()
    provider_lower = (source_provider or "").strip().lower()

    if source_lower == "tradingeconomics" and provider_lower == "stream":
        return "tier_2"

    if _domain_matches(host, _BLOCKED_SOURCE_DOMAINS):
        return "blocked"

    if (
        _domain_matches(host, _TRUST_TIER_1_DOMAINS)
        or host.endswith(".gov")
        or any(hint in source_lower for hint in _OFFICIAL_SOURCE_HINTS)
    ):
        return "tier_1"

    if _domain_matches(host, _TRUST_TIER_2_DOMAINS):
        return "tier_2"

    return "tier_3"


def _classify_content_class(
    *,
    title: str | None,
    text: str,
    url: str,
    source_name: str | None,
    source_provider: str | None = None,
) -> tuple[str, list[str]]:
    """Deterministic, inspectable content classification."""
    clean_text = " ".join((text or "").split())
    lower_text = clean_text.lower()
    lower_title = (title or "").strip().lower()
    lower_url = (url or "").strip().lower()
    source_lower = (source_name or "").strip().lower()
    provider_lower = (source_provider or "").strip().lower()
    word_count = len(lower_text.split())
    flags: list[str] = []

    # TradingEconomics stream briefs are short-form updates, not normal articles.
    if source_lower == "tradingeconomics" and provider_lower == "stream":
        flags.append("stream_brief")
        return "stream_brief", flags

    if word_count < 25:
        flags.append("very_short_text")
        return "junk", flags

    video_markers = ("video", "/video/", "/videos/", "watch:")
    if any(marker in lower_url for marker in video_markers) or (
        "video" in lower_title and word_count < 220
    ):
        flags.append("video_like_page")
        return "video_stub", flags

    if any(marker in lower_url for marker in ("/quote/", "/quotes/", "quote.aspx")):
        flags.append("quote_url_pattern")
        return "quote_page", flags

    if any(marker in lower_url for marker in ("/symbol/", "/stocks/", "/stock/", "ticker=", "symbol=")):
        flags.append("ticker_url_pattern")
        return "ticker_page", flags

    nav_markers = ("/search", "/topic/", "/topics/", "/tag/", "/categories/", "/section/")
    if _is_boilerplate_article(clean_text, lower_url) or any(marker in lower_url for marker in nav_markers):
        flags.append("navigation_or_boilerplate")
        return "navigation_page", flags

    official_markers = (
        "for immediate release",
        "press release",
        "statement by",
        "minutes of the federal open market committee",
        "news release",
    )
    if (
        _host_from_url(lower_url).endswith(".gov")
        or "federal reserve" in source_lower
        or "treasury" in source_lower
        or any(marker in lower_text[:1200] for marker in official_markers)
    ):
        flags.append("official_source_or_release_language")
        return "official_release", flags

    evergreen_markers = (
        "what is ",
        "explainer",
        "how to ",
        "guide to ",
        "why it matters",
    )
    if any(marker in lower_title for marker in evergreen_markers):
        flags.append("evergreen_title_pattern")
        return "evergreen_explainer", flags

    analysis_markers = ("analysis", "opinion", "column", "outlook", "forecast")
    if any(marker in lower_title for marker in analysis_markers):
        flags.append("analysis_title_pattern")
        return "analysis", flags

    return "news_report", flags


def _score_article_quality(
    *,
    title: str | None,
    text: str,
    published: str | None,
    url: str,
    source_tier: str,
    content_class: str,
    source_name: str | None = None,
    source_provider: str | None = None,
) -> tuple[float, list[str]]:
    """
    Deterministic quality scoring in [0, 100].
    Rules are intentionally simple so rows remain easy to audit.
    """
    words = len((text or "").split())
    flags: list[str] = []

    if (
        (source_name or "").strip().lower() == "tradingeconomics"
        and (source_provider or "").strip().lower() == "stream"
    ) or content_class == "stream_brief":
        return 0.6, ["stream_brief", "short_form_source"]

    score = 50.0

    if words < MIN_ARTICLE_WORDS:
        score -= 20.0
        flags.append("short_article")
    else:
        score += 12.0
    if words >= 350:
        score += 8.0
    if words >= 800:
        score += 5.0

    if not (title or "").strip():
        score -= 8.0
        flags.append("missing_title")
    else:
        score += 4.0

    if not (published or "").strip():
        score -= 5.0
        flags.append("missing_published_at")
    else:
        score += 3.0

    tier_delta = {
        "tier_1": 15.0,
        "tier_2": 8.0,
        "tier_3": 0.0,
        "blocked": -40.0,
    }.get(source_tier, 0.0)
    score += tier_delta
    if source_tier == "blocked":
        flags.append("blocked_source")

    class_delta = {
        "news_report": 8.0,
        "analysis": 6.0,
        "official_release": 8.0,
        "stream_brief": -18.0,
        "evergreen_explainer": 2.0,
        "ticker_page": -15.0,
        "navigation_page": -35.0,
        "video_stub": -25.0,
        "quote_page": -18.0,
        "junk": -40.0,
    }.get(content_class, 0.0)
    score += class_delta
    if content_class in {"ticker_page", "navigation_page", "video_stub", "quote_page", "junk"}:
        flags.append(f"low_signal_class:{content_class}")

    if _is_boilerplate_article(text, url):
        score -= 15.0
        flags.append("boilerplate_detected")

    score = max(0.0, min(100.0, score))
    return round(score, 2), sorted(set(flags))


def _is_boilerplate_article(text: str, url: str = "") -> bool:
    """
    Heuristic filter for navigation-heavy pages (menus, video index pages).
    """
    compact = " ".join((text or "").split())
    if not compact:
        return True

    lowered = compact.lower()
    words = lowered.split()
    if len(words) < 80:
        return True

    nav_markers = (
        "skip to content",
        "home",
        "news",
        "sport",
        "business",
        "technology",
        "health",
        "culture",
        "travel",
        "audio",
        "video",
        "live",
        "weather",
        "newsletters",
        "share",
        "save",
    )
    marker_hits = sum(lowered.count(marker) for marker in nav_markers)
    marker_density = marker_hits / len(words)
    unique_ratio = len(set(words)) / len(words)
    is_video_page = "/videos/" in (url or "").lower()

    return marker_density > 0.08 or unique_ratio < 0.35 or is_video_page


def _save_to_sqlite(articles: list[dict], scraped_at_utc: str) -> None:
    """Persist ok articles to SQLite and keep raw rows for audit."""
    from create_sql_db import create_database, connect_sqlite, ensure_migrations
    create_database(SQLITE_DB)
    ensure_migrations(SQLITE_DB)
    conn = connect_sqlite(SQLITE_DB)
    existing_ids = {
        row[0]
        for row in conn.execute("SELECT article_id FROM articles").fetchall()
    }
    inserted = updated = skipped = 0
    for art in articles:
        if art.get("status") != "ok" or not art.get("text"):
            continue

        url = art.get("url", "")
        source_provider = (art.get("source_provider") or "").strip().lower() or None
        aid = _article_id(art.get("url", ""), art.get("published"))
        content_class, class_flags = _classify_content_class(
            title=art.get("title"),
            text=art["text"],
            url=url,
            source_name=art.get("source"),
            source_provider=source_provider,
        )
        source_trust_tier = _classify_source_trust_tier_with_provider(
            source_name=art.get("source"),
            url=url,
            source_provider=source_provider,
        )
        article_quality_score, score_flags = _score_article_quality(
            title=art.get("title"),
            text=art["text"],
            published=art.get("published"),
            url=url,
            source_tier=source_trust_tier,
            content_class=content_class,
            source_name=art.get("source"),
            source_provider=source_provider,
        )
        quality_flags = sorted(set(class_flags + score_flags))
        if content_class == "stream_brief":
            quality_flags = sorted(set(quality_flags + ["stream_brief", "short_form_source"]))
        quality_flags_json = json.dumps(
            {
                "flags": quality_flags,
                "word_count": len(art["text"].split()),
                "source_provider": art.get("source_provider"),
            },
            separators=(",", ":"),
            sort_keys=True,
        )

        row = (
            aid,
            url,
            art.get("title"),
            art.get("source"),
            art.get("source_rss"),
            art.get("source_provider"),
            art.get("published"),
            scraped_at_utc,
            _content_hash(art["text"]),
            art.get("status"),
            art["text"],
            source_trust_tier,
            content_class,
            article_quality_score,
            quality_flags_json,
        )
        if aid in existing_ids:
            conn.execute(
                "UPDATE articles SET "
                "source_provider = ?, "
                "source_trust_tier = ?, "
                "content_class = ?, "
                "article_quality_score = ?, "
                "quality_flags_json = ?, "
                "processing_state = CASE "
                "  WHEN processing_state IS NULL OR processing_state = 'ingested' THEN 'classified' "
                "  ELSE processing_state "
                "END "
                "WHERE article_id = ?",
                (
                    art.get("source_provider"),
                    source_trust_tier,
                    content_class,
                    article_quality_score,
                    quality_flags_json,
                    aid,
                ),
            )
            updated += 1
        else:
            conn.execute(
                "INSERT INTO articles "
                "(article_id,url,title,source,source_rss,source_provider,published_at,"
                "scraped_at_utc,content_hash,status,raw_text,"
                "source_trust_tier,content_class,article_quality_score,quality_flags_json,processing_state) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                row + ("classified",),
            )
            existing_ids.add(aid)
            inserted += 1
    conn.commit()
    conn.close()
    print(
        f"  SQLite ({SQLITE_DB}): {inserted} new articles saved, "
        f"{updated} existing articles refreshed, {skipped} skipped"
    )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _reset_sqlite_db() -> None:
    """Delete and recreate the SQLite DB file."""
    from create_sql_db import create_database
    if os.path.exists(SQLITE_DB):
        os.remove(SQLITE_DB)
        print(f"Reset SQLite database file: {SQLITE_DB}")
    else:
        print(f"SQLite database file not found, creating new DB: {SQLITE_DB}")
    create_database(SQLITE_DB)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(reset_db: bool = False) -> None:
    print("=" * 60)
    print("News Scraper v2 (alpha-primary + rss fallback + async + trafilatura)")
    print("=" * 60)
    if reset_db:
        print("\n[sqlite] --reset-db enabled: wiping SQLite database before scrape.")
        _reset_sqlite_db()

    print("\nStep 1: Discovering article links ...")
    with requests.Session() as discovery_session:
        links, source_by_url, discovery_meta = discover_links(discovery_session)

    print(f"Found {len(links)} unique links")
    print(f"Discovery providers: {', '.join(discovery_meta.get('providers', [])) or 'none'}")
    if discovery_meta.get("fallback_used"):
        print("Fallback provider was used during discovery")

    if not links and not ENABLE_TRADINGECONOMICS_STREAM:
        print("No links found. Exiting.")
        return

    print("\nStep 2: Scraping articles (async) ...")
    start_time = time.time()

    if links:
        if os.name == "nt" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        articles = asyncio.run(scrape_articles_async(links, source_by_url))
    else:
        print("No RSS/alpha links found; continuing with TradingEconomics stream only.")
        articles = []
    te_articles: list[dict[str, Any]] = []
    if ENABLE_TRADINGECONOMICS_STREAM:
        print("\nStep 2b: Scraping TradingEconomics stream briefs ...")
        te_articles = fetch_tradingeconomics_stream()
        if te_articles:
            print(f"  TradingEconomics stream briefs collected: {len(te_articles)}")
            articles.extend(te_articles)
        else:
            print("  TradingEconomics stream briefs collected: 0")
    scrape_duration = time.time() - start_time
    print(f"Scraped {len(articles)} articles in {scrape_duration:.1f} seconds")
    if scrape_duration > 0:
        print(f"  ({len(articles) / scrape_duration:.1f} articles/second)")

    status_counts: defaultdict[str, int] = defaultdict(int)
    for article in articles:
        status_counts[article["status"]] += 1

    print("\nScrape status breakdown:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    print("\nFinal scrape status breakdown:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    payload = {
        "source_rss": [source["url"] for source in RSS_SOURCES] + (
            ["https://tradingeconomics.com/stream"] if te_articles else []
        ),
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(articles),
        "articles": articles,
        "metadata": {
            "scraper_version": "v2",
            "extraction_method": "trafilatura",
            "async_enabled": True,
            "dedup_authority": "ingestion_pipeline",
            "scrape_duration_seconds": scrape_duration,
            "discovery": discovery_meta,
            "tradingeconomics_stream_enabled": ENABLE_TRADINGECONOMICS_STREAM,
            "tradingeconomics_stream_items": len(te_articles),
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(articles)} articles to {OUTPUT_JSON}")
    print(f"  OK: {status_counts.get('ok', 0)} articles ready for T-GRAG pipeline")
    print(f"  Empty text: {status_counts.get('empty_text', 0)}")
    print(f"  Errors: {status_counts.get('error', 0)}")

    _save_to_sqlite(articles, payload["scraped_at_utc"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News Scraper v2")
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Delete and recreate SQLite DB before scraping and ingesting",
    )
    args = parser.parse_args()
    main(reset_db=args.reset_db)
