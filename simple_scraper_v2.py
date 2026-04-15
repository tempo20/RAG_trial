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
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    import aiohttp
except ImportError:  # pragma: no cover - exercised in dependency-light envs
    aiohttp = None

import feedparser
import requests
from dotenv import load_dotenv
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
) -> tuple[dict[str, Any] | None, str | None, int]:
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
                headers={"User-Agent": "Mozilla/5.0"},
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
) -> tuple[str | None, str | None]:
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"},
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
    """Persist ok articles to SQLite, silently skipping duplicates."""
    from create_sql_db import create_database
    create_database(SQLITE_DB)
    conn = sqlite3.connect(SQLITE_DB)
    inserted = skipped = skipped_boilerplate = 0
    for art in articles:
        if art.get("status") != "ok" or not art.get("text"):
            continue
        if _is_boilerplate_article(art["text"], art.get("url", "")):
            skipped_boilerplate += 1
            continue
        aid = _article_id(art.get("url", ""), art.get("published"))
        row = (
            aid,
            art.get("url", ""),
            art.get("title"),
            art.get("source"),
            art.get("source_rss"),
            art.get("published"),
            scraped_at_utc,
            _content_hash(art["text"]),
            art.get("status"),
            art["text"],
        )
        cur = conn.execute(
            "INSERT OR IGNORE INTO articles "
            "(article_id,url,title,source,source_rss,published_at,"
            "scraped_at_utc,content_hash,status,raw_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            row,
        )
        if cur.rowcount:
            inserted += 1
        else:
            skipped += 1
    conn.commit()
    conn.close()
    print(f"  SQLite ({SQLITE_DB}): {inserted} new articles saved, {skipped} duplicates skipped")
    if skipped_boilerplate:
        print(f"  SQLite ({SQLITE_DB}): {skipped_boilerplate} boilerplate/nav-heavy articles skipped")


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

    if not links:
        print("No links found. Exiting.")
        return

    print("\nStep 2: Scraping articles (async) ...")
    start_time = time.time()

    if os.name == "nt" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    articles = asyncio.run(scrape_articles_async(links, source_by_url))
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
        "source_rss": [source["url"] for source in RSS_SOURCES],
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
