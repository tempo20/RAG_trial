from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import reddit_mentions as reddit  # noqa: E402
import ta_dashboard as dashboard  # noqa: E402


def _atom_feed(entries: list[dict[str, str]]) -> bytes:
    entry_xml = "".join(
        (
            "<entry>"
            f"<id>{entry['id']}</id>"
            f"<title>{entry['title']}</title>"
            f"<link href=\"{entry['link']}\" />"
            f"<updated>{entry.get('updated', '')}</updated>"
            f"<content type=\"html\">{entry.get('content', '')}</content>"
            "</entry>"
        )
        for entry in entries
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        f"{entry_xml}</feed>"
    ).encode("utf-8")


class _FakeResponse:
    def __init__(self, content: bytes = b"", status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {self.status_code}",
                response=self,
            )


class _FakeSession:
    def __init__(self, responses: dict[str, _FakeResponse]) -> None:
        self.responses = responses
        self.headers: dict[str, str] = {}

    def get(self, url: str, timeout: int) -> _FakeResponse:
        assert timeout == reddit.REDDIT_REQUEST_TIMEOUT_S
        return self.responses[url]


def test_extracts_unknown_ticker_tokens_without_a_reference_map() -> None:
    candidates = reddit.extract_ticker_candidates(
        "CEO likes RKLB, $ionq, and BRK.B. DD is solid, but $DD is explicit."
    )

    assert candidates == {"RKLB", "IONQ", "BRK-B", "DD"}
    assert "CEO" not in candidates


def test_yahoo_validation_retains_only_symbols_with_market_data(monkeypatch) -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("IONQ", "Close"),
            ("RKLB", "Close"),
            ("NOPE", "Close"),
        ]
    )
    downloaded = pd.DataFrame(
        [[10.0, 20.0, float("nan")]],
        columns=columns,
    )
    monkeypatch.setattr(reddit.yf, "download", lambda **kwargs: downloaded)

    valid, error = reddit.validate_ticker_candidates({"NOPE", "RKLB", "IONQ"})

    assert valid == {"RKLB", "IONQ"}
    assert error is None


def test_collect_counts_unique_posts_and_deduplicates_across_feeds() -> None:
    feed_one = _atom_feed(
        [
            {
                "id": "post-1",
                "title": "RKLB RKLB launch",
                "link": "https://reddit.test/post-1",
                "updated": "2026-06-29T12:30:00Z",
                "content": "&lt;p&gt;$IONQ is mentioned too&lt;/p&gt;",
            },
            {
                "id": "post-2",
                "title": "IONQ earnings",
                "link": "https://reddit.test/post-2",
            },
        ]
    )
    feed_two = _atom_feed(
        [
            {
                "id": "post-1",
                "title": "Duplicate RKLB post",
                "link": "https://reddit.test/post-1",
            },
            {
                "id": "post-3",
                "title": "NOPE is invalid",
                "link": "https://reddit.test/post-3",
            },
        ]
    )
    session = _FakeSession(
        {
            "feed-one": _FakeResponse(feed_one),
            "feed-two": _FakeResponse(feed_two),
        }
    )

    report = reddit.collect_reddit_mentions(
        ["feed-one", "feed-two"],
        session=session,
        ticker_validator=lambda candidates: ({"RKLB", "IONQ"}, None),
    )

    assert report.posts_scanned == 3
    assert report.counts == {"IONQ": 2, "RKLB": 1}
    assert report.feed_errors == {}
    assert report.posts[0] == reddit.RedditMentionPost(
        post_id="post-1",
        url="https://reddit.test/post-1",
        title="RKLB RKLB launch",
        source="feed-one",
        published_at="2026-06-29T12:30:00+00:00",
        tickers=("IONQ", "RKLB"),
    )
    assert report.posts[1].published_at == report.fetched_at_utc


def test_collect_preserves_successful_feed_when_another_is_rate_limited() -> None:
    good_feed = _atom_feed(
        [
            {
                "id": "post-1",
                "title": "$RKLB update",
                "link": "https://reddit.test/post-1",
            }
        ]
    )
    session = _FakeSession(
        {
            "good-feed": _FakeResponse(good_feed),
            "limited-feed": _FakeResponse(status_code=429),
        }
    )

    report = reddit.collect_reddit_mentions(
        ["good-feed", "limited-feed"],
        session=session,
        ticker_validator=lambda candidates: (set(candidates), None),
    )

    assert report.counts == {"RKLB": 1}
    assert report.posts_scanned == 1
    assert report.feed_errors == {"limited-feed": "HTTP 429"}


def test_reddit_posts_persist_idempotently_and_filter_by_provider_and_date(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "ta_cache.db"
    now = datetime.now(timezone.utc).replace(microsecond=0)
    report = reddit.RedditMentionReport(
        counts={"IONQ": 1, "RKLB": 1},
        posts_scanned=2,
        feed_errors={},
        fetched_at_utc=now.isoformat(),
        posts=(
            reddit.RedditMentionPost(
                post_id="recent",
                url="https://reddit.test/recent",
                title="IONQ and RKLB",
                source="r/wallstreetbets",
                published_at=now.isoformat(),
                tickers=("IONQ", "RKLB"),
            ),
            reddit.RedditMentionPost(
                post_id="old",
                url="https://reddit.test/old",
                title="Old RKLB post",
                source="r/wallstreetbets",
                published_at=(now - timedelta(days=8)).isoformat(),
                tickers=("RKLB",),
            ),
        ),
    )

    dashboard.persist_reddit_mentions_report(report, db_path=db_path)
    dashboard.persist_reddit_mentions_report(report, db_path=db_path)
    dashboard.ta_cache.upsert_articles(
        [
            {
                "url": "https://news.test/ionq",
                "title": "Financial news IONQ",
                "source": "news",
                "publishedDate": now.isoformat(),
                "symbol": "IONQ",
            }
        ],
        provider="fmp_stock_news",
        db_path=db_path,
    )

    rows = dashboard.load_cached_reddit_mentions(7, db_path=db_path)

    assert [(row["symbol"], row["url"]) for row in rows] == [
        ("IONQ", "https://reddit.test/recent"),
        ("RKLB", "https://reddit.test/recent"),
    ]
    assert dashboard.reddit_mentions_df(rows).to_dict("records") == [
        {"ticker": "IONQ", "post_count": 1},
        {"ticker": "RKLB", "post_count": 1},
    ]
    assert dashboard.reddit_cache_has_recent_posts(db_path=db_path) is True


def test_cached_reddit_rows_follow_the_selected_day_window(tmp_path: Path) -> None:
    db_path = tmp_path / "ta_cache.db"
    now = datetime.now(timezone.utc).replace(microsecond=0)
    posts = tuple(
        reddit.RedditMentionPost(
            post_id=f"post-{days_ago}",
            url=f"https://reddit.test/post-{days_ago}",
            title=f"RKLB post from {days_ago} days ago",
            source="r/wallstreetbets",
            published_at=(now - timedelta(days=days_ago)).isoformat(),
            tickers=("RKLB",),
        )
        for days_ago in (0, 2, 6)
    )
    report = reddit.RedditMentionReport(
        counts={"RKLB": 3},
        posts_scanned=3,
        feed_errors={},
        fetched_at_utc=now.isoformat(),
        posts=posts,
    )
    dashboard.persist_reddit_mentions_report(report, db_path=db_path)

    assert len(dashboard.load_cached_reddit_mentions(1, db_path=db_path)) == 1
    assert len(dashboard.load_cached_reddit_mentions(3, db_path=db_path)) == 2
    assert len(dashboard.load_cached_reddit_mentions(7, db_path=db_path)) == 3


def test_dashboard_does_not_fetch_before_load_click(monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(dashboard.st, "session_state", {})
    monkeypatch.setattr(dashboard.st, "selectbox", lambda *args, **kwargs: 7)
    monkeypatch.setattr(dashboard.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        dashboard.st,
        "info",
        lambda message, *args, **kwargs: messages.append(message),
    )

    def unexpected_fetch() -> reddit.RedditMentionReport:
        raise AssertionError("Reddit feeds must be button-gated")

    monkeypatch.setattr(dashboard, "load_reddit_mentions_report", unexpected_fetch)
    monkeypatch.setattr(
        dashboard,
        "reddit_cache_has_recent_posts",
        lambda: False,
    )

    dashboard.render_reddit_mentions_tab()

    assert messages == [
        "Select Load Reddit mentions to fetch the configured RSS feeds."
    ]


def test_dashboard_displays_cached_rows_without_fetching(monkeypatch) -> None:
    cached_rows = [
        {
            "article_id": "post-1",
            "publishedDate": "2026-06-30T00:00:00+00:00",
            "symbol": "IONQ",
            "site": "r/wallstreetbets",
            "title": "IONQ post",
            "url": "https://reddit.test/post-1",
        },
        {
            "article_id": "post-2",
            "publishedDate": "2026-06-30T00:00:00+00:00",
            "symbol": "RKLB",
            "site": "r/TheRaceTo1Million",
            "title": "RKLB post",
            "url": "https://reddit.test/post-2",
        },
    ]
    displayed: list[pd.DataFrame] = []
    monkeypatch.setattr(dashboard.st, "session_state", {})
    monkeypatch.setattr(dashboard.st, "selectbox", lambda *args, **kwargs: 7)
    monkeypatch.setattr(dashboard.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(dashboard.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        dashboard.st,
        "dataframe",
        lambda frame, *args, **kwargs: displayed.append(frame.copy()),
    )
    monkeypatch.setattr(
        dashboard,
        "reddit_cache_has_recent_posts",
        lambda: True,
    )
    monkeypatch.setattr(
        dashboard,
        "load_cached_reddit_mentions",
        lambda news_days: cached_rows,
    )

    def unexpected_fetch() -> reddit.RedditMentionReport:
        raise AssertionError("existing SQLite cache must prevent an RSS fetch")

    monkeypatch.setattr(dashboard, "load_reddit_mentions_report", unexpected_fetch)

    dashboard.render_reddit_mentions_tab()

    assert displayed[0].to_dict("records") == [
        {"ticker": "IONQ", "post_count": 1},
        {"ticker": "RKLB", "post_count": 1},
    ]
    assert displayed[1].to_dict("records")[0] == {
        "published": "2026-06-30T00:00:00+00:00",
        "ticker": "IONQ",
        "source": "r/wallstreetbets",
        "title": "IONQ post",
        "url": "https://reddit.test/post-1",
    }


def test_dashboard_loads_empty_cache_then_displays_persisted_rows(monkeypatch) -> None:
    report = reddit.RedditMentionReport(
        counts={"RKLB": 1},
        posts_scanned=1,
        feed_errors={},
        fetched_at_utc="2026-06-30T00:00:00+00:00",
    )
    cached_rows = [
        {
            "article_id": "post-1",
            "publishedDate": "2026-06-30T00:00:00+00:00",
            "symbol": "RKLB",
            "site": "r/wallstreetbets",
            "title": "RKLB post",
            "url": "https://reddit.test/post-1",
        }
    ]
    cache_states = iter([False, True])
    persisted: list[reddit.RedditMentionReport] = []
    displayed: list[pd.DataFrame] = []
    session_state: dict[str, object] = {}
    monkeypatch.setattr(dashboard.st, "session_state", session_state)
    monkeypatch.setattr(dashboard.st, "selectbox", lambda *args, **kwargs: 7)
    monkeypatch.setattr(dashboard.st, "button", lambda *args, **kwargs: True)
    monkeypatch.setattr(dashboard.st, "spinner", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(dashboard.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        dashboard.st,
        "dataframe",
        lambda frame, *args, **kwargs: displayed.append(frame.copy()),
    )
    monkeypatch.setattr(dashboard, "load_reddit_mentions_report", lambda: report)
    monkeypatch.setattr(
        dashboard,
        "persist_reddit_mentions_report",
        lambda fetched_report: persisted.append(fetched_report),
    )
    monkeypatch.setattr(
        dashboard,
        "reddit_cache_has_recent_posts",
        lambda: next(cache_states),
    )
    monkeypatch.setattr(
        dashboard,
        "load_cached_reddit_mentions",
        lambda news_days: cached_rows,
    )

    dashboard.render_reddit_mentions_tab()

    assert persisted == [report]
    assert session_state["reddit_mentions_fetch_report"] == report
    assert displayed[0].to_dict("records") == [
        {"ticker": "RKLB", "post_count": 1},
    ]


def test_failed_refresh_preserves_cached_rows_and_surfaces_warning(monkeypatch) -> None:
    report = reddit.RedditMentionReport(
        counts={},
        posts_scanned=0,
        feed_errors={"wallstreetbets": "HTTP 429"},
        fetched_at_utc="2026-06-30T00:00:00+00:00",
    )
    cached_rows = [
        {
            "article_id": "post-1",
            "publishedDate": "2026-06-30T00:00:00+00:00",
            "symbol": "RKLB",
            "site": "r/wallstreetbets",
            "title": "Cached RKLB post",
            "url": "https://reddit.test/post-1",
        }
    ]
    warnings: list[str] = []
    displayed: list[pd.DataFrame] = []
    monkeypatch.setattr(dashboard.st, "session_state", {})
    monkeypatch.setattr(dashboard.st, "selectbox", lambda *args, **kwargs: 7)
    monkeypatch.setattr(dashboard.st, "button", lambda *args, **kwargs: True)
    monkeypatch.setattr(dashboard.st, "spinner", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(dashboard.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        dashboard.st,
        "warning",
        lambda message, *args, **kwargs: warnings.append(message),
    )
    monkeypatch.setattr(dashboard.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        dashboard.st,
        "dataframe",
        lambda frame, *args, **kwargs: displayed.append(frame.copy()),
    )
    monkeypatch.setattr(dashboard, "load_reddit_mentions_report", lambda: report)
    monkeypatch.setattr(dashboard, "persist_reddit_mentions_report", lambda report: 0)
    monkeypatch.setattr(dashboard, "reddit_cache_has_recent_posts", lambda: True)
    monkeypatch.setattr(
        dashboard,
        "load_cached_reddit_mentions",
        lambda news_days: cached_rows,
    )

    dashboard.render_reddit_mentions_tab()

    assert warnings == ["wallstreetbets: HTTP 429"]
    assert displayed[0].to_dict("records") == [
        {"ticker": "RKLB", "post_count": 1}
    ]
