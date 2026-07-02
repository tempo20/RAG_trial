from __future__ import annotations

from collections import Counter
import calendar
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
import hashlib
import re
from typing import Callable, Iterable, Sequence

import feedparser
import pandas as pd
import requests
import yfinance as yf


REDDIT_RSS_FEEDS = [
    "https://www.reddit.com/r/TheRaceTo1Million/new/.rss",
    "https://www.reddit.com/r/wallstreetbets/new/.rss",
]

REDDIT_REQUEST_TIMEOUT_S = 10
REDDIT_USER_AGENT = "RAG_trial-reddit-mentions/1.0 (RSS dashboard)"
REDDIT_CACHE_PROVIDER = "reddit_rss"
YAHOO_VALIDATION_BATCH_SIZE = 50

_CASHTAG_PATTERN = re.compile(
    r"(?<![\w$])\$([A-Za-z]{1,5}(?:[.-][A-Za-z])?)"
    r"(?![A-Za-z0-9-]|\.[A-Za-z])"
)
_BARE_TICKER_PATTERN = re.compile(
    r"(?<![\w$])([A-Z]{2,5}(?:[.-][A-Z])?)"
    r"(?![A-Za-z0-9-]|\.[A-Za-z])"
)

# These are common prose/Reddit tokens that are also valid ticker-shaped text.
# An explicit cashtag (for example, $DD) bypasses this filter.
_BARE_TICKER_STOPWORDS = frozenset(
    {
        "AH",
        "AI",
        "AM",
        "ATH",
        "ATL",
        "BUY",
        "CALL",
        "CALLS",
        "CEO",
        "CFO",
        "COO",
        "CTO",
        "DD",
        "EOD",
        "EPS",
        "ER",
        "ETF",
        "FDA",
        "FED",
        "FOMO",
        "HODL",
        "HOLD",
        "IMO",
        "IMHO",
        "IPO",
        "MOON",
        "NEWS",
        "OP",
        "PE",
        "PM",
        "PUT",
        "PUTS",
        "SEC",
        "SELL",
        "TA",
        "TLDR",
        "USA",
        "USD",
        "WSB",
        "YOLO",
    }
)


@dataclass(frozen=True)
class RedditMentionPost:
    post_id: str
    url: str
    title: str
    source: str
    published_at: str
    tickers: tuple[str, ...]

    def as_cache_article(self) -> dict[str, object]:
        return {
            "url": self.url,
            "title": self.title,
            "source": self.source,
            "publishedDate": self.published_at,
            "symbols": list(self.tickers),
        }


@dataclass(frozen=True)
class RedditMentionReport:
    counts: dict[str, int]
    posts_scanned: int
    feed_errors: dict[str, str]
    fetched_at_utc: str
    posts: tuple[RedditMentionPost, ...] = ()
    validation_error: str | None = None


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)


def _strip_html(value: object) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(unescape(str(value or "")))
        parser.close()
    except Exception:
        return re.sub(r"<[^>]+>", " ", unescape(str(value or "")))
    return " ".join(parser.parts)


def _canonical_ticker(value: str) -> str:
    return value.strip().upper().replace(".", "-")


def extract_ticker_candidates(text: str) -> set[str]:
    """Extract map-free ticker-shaped tokens from Reddit post text."""
    cashtags = {
        _canonical_ticker(match.group(1))
        for match in _CASHTAG_PATTERN.finditer(text or "")
    }
    bare = {
        _canonical_ticker(match.group(1))
        for match in _BARE_TICKER_PATTERN.finditer(text or "")
        if _canonical_ticker(match.group(1)) not in _BARE_TICKER_STOPWORDS
    }
    return cashtags | bare


def _frame_has_market_data(frame: pd.DataFrame | pd.Series) -> bool:
    if frame is None or frame.empty:
        return False
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if isinstance(numeric, pd.Series):
        return bool(numeric.notna().any())
    return bool(numeric.notna().any(axis=None))


def _valid_tickers_from_download(
    downloaded: pd.DataFrame,
    candidates: Sequence[str],
) -> set[str]:
    if downloaded is None or downloaded.empty:
        return set()

    if not isinstance(downloaded.columns, pd.MultiIndex):
        return {candidates[0]} if len(candidates) == 1 and _frame_has_market_data(downloaded) else set()

    valid: set[str] = set()
    for ticker in candidates:
        for level in range(downloaded.columns.nlevels):
            if ticker not in set(downloaded.columns.get_level_values(level)):
                continue
            frame = downloaded.xs(ticker, axis=1, level=level, drop_level=True)
            if _frame_has_market_data(frame):
                valid.add(ticker)
                break
    return valid


def validate_ticker_candidates(candidates: Iterable[str]) -> tuple[set[str], str | None]:
    """Retain only candidates for which Yahoo returns recent market data."""
    normalized = sorted({_canonical_ticker(candidate) for candidate in candidates if candidate})
    if not normalized:
        return set(), None

    valid: set[str] = set()
    errors: list[str] = []
    for start in range(0, len(normalized), YAHOO_VALIDATION_BATCH_SIZE):
        batch = normalized[start : start + YAHOO_VALIDATION_BATCH_SIZE]
        try:
            downloaded = yf.download(
                tickers=batch,
                period="5d",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
                timeout=10,
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {str(exc)[:160]}")
            continue
        valid.update(_valid_tickers_from_download(downloaded, batch))

    error = "; ".join(errors) if errors else None
    return valid, error


def _entry_text(entry: object) -> str:
    title = str(entry.get("title") or "")
    content = entry.get("content") or []
    body = " ".join(
        str(item.get("value") or "")
        for item in content
        if hasattr(item, "get")
    )
    if not body:
        body = str(entry.get("summary") or entry.get("description") or "")
    return re.sub(r"\s+", " ", f"{title} {_strip_html(body)}").strip()


def _normalized_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _entry_published_at(entry: object, fallback: str) -> str:
    for field in ("published_parsed", "updated_parsed"):
        parsed = entry.get(field)
        if parsed:
            try:
                value = datetime.fromtimestamp(calendar.timegm(parsed), timezone.utc)
                return _normalized_datetime(value)
            except (OverflowError, TypeError, ValueError):
                pass

    for field in ("published", "updated"):
        raw = str(entry.get(field) or "").strip()
        if not raw:
            continue
        iso_value = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        try:
            return _normalized_datetime(datetime.fromisoformat(iso_value))
        except ValueError:
            pass
        try:
            return _normalized_datetime(parsedate_to_datetime(raw))
        except (TypeError, ValueError):
            pass
    return fallback


def _feed_source(feed_url: str) -> str:
    match = re.search(r"/r/([^/]+)", feed_url, flags=re.IGNORECASE)
    return f"r/{match.group(1)}" if match else feed_url


def _entry_key(entry: object) -> str:
    identity = str(
        entry.get("id")
        or entry.get("guid")
        or entry.get("link")
        or ""
    ).strip()
    if identity:
        return identity
    fallback = "\n".join(
        str(entry.get(field) or "")
        for field in ("title", "summary", "published", "updated")
    )
    return hashlib.sha256(fallback.encode("utf-8")).hexdigest()


def _fetch_feed_entries(
    session: requests.Session,
    feed_url: str,
    timeout_s: int,
) -> tuple[list[object], str | None]:
    try:
        response = session.get(feed_url, timeout=timeout_s)
        response.raise_for_status()
    except requests.RequestException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        detail = f"HTTP {status_code}" if status_code else type(exc).__name__
        return [], detail
    except Exception as exc:
        return [], f"{type(exc).__name__}: {str(exc)[:160]}"

    parsed = feedparser.parse(response.content)
    if parsed.bozo and not parsed.entries:
        return [], f"RSS parse error: {str(parsed.bozo_exception)[:160]}"
    return list(parsed.entries), None


TickerValidator = Callable[[Iterable[str]], tuple[set[str], str | None]]


def collect_reddit_mentions(
    feed_urls: Sequence[str] = REDDIT_RSS_FEEDS,
    *,
    timeout_s: int = REDDIT_REQUEST_TIMEOUT_S,
    session: requests.Session | None = None,
    ticker_validator: TickerValidator = validate_ticker_candidates,
) -> RedditMentionReport:
    """Fetch Reddit RSS feeds and count validated tickers by unique post."""
    fetched_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    own_session = session is None
    http = session or requests.Session()
    http.headers.update({"User-Agent": REDDIT_USER_AGENT, "Accept": "application/atom+xml, application/rss+xml;q=0.9"})

    feed_errors: dict[str, str] = {}
    seen_posts: set[str] = set()
    collected_posts: list[dict[str, object]] = []
    try:
        for feed_url in feed_urls:
            entries, error = _fetch_feed_entries(http, feed_url, timeout_s)
            if error:
                feed_errors[feed_url] = error
                continue
            for entry in entries:
                key = _entry_key(entry)
                if key in seen_posts:
                    continue
                seen_posts.add(key)
                collected_posts.append(
                    {
                        "post_id": key,
                        "url": str(entry.get("link") or "").strip(),
                        "title": str(entry.get("title") or "").strip(),
                        "source": _feed_source(feed_url),
                        "published_at": _entry_published_at(entry, fetched_at_utc),
                        "candidates": extract_ticker_candidates(_entry_text(entry)),
                    }
                )
    finally:
        if own_session:
            http.close()

    all_candidates = (
        set().union(
            *(post["candidates"] for post in collected_posts)
        )
        if collected_posts
        else set()
    )
    valid_tickers, validation_error = ticker_validator(all_candidates)
    counter: Counter[str] = Counter()
    posts: list[RedditMentionPost] = []
    for post in collected_posts:
        tickers = tuple(sorted(post["candidates"] & valid_tickers))
        counter.update(tickers)
        posts.append(
            RedditMentionPost(
                post_id=str(post["post_id"]),
                url=str(post["url"]),
                title=str(post["title"]),
                source=str(post["source"]),
                published_at=str(post["published_at"]),
                tickers=tickers,
            )
        )

    ordered_counts = dict(
        sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    )
    return RedditMentionReport(
        counts=ordered_counts,
        posts_scanned=len(seen_posts),
        feed_errors=feed_errors,
        fetched_at_utc=fetched_at_utc,
        posts=tuple(posts),
        validation_error=validation_error,
    )
