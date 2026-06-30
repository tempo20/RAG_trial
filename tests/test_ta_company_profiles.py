from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rag_trial.db import ta_cache  # noqa: E402
from ta_dashboard import load_company_profiles  # noqa: E402


def _profile(
    ticker: str,
    *,
    description: str | None = "Description",
    sector: str | None = "Sector",
    industry: str | None = "Industry",
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "description": description,
        "sector": sector,
        "industry": industry,
    }


def test_company_profile_cache_creates_normalizes_and_updates_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "ta_cache.db"

    written = ta_cache.upsert_company_profiles(
        {
            " aapl ": _profile("aapl", sector="Technology"),
            "EMPTY": _profile("EMPTY", description=None, sector=None, industry=None),
        },
        fetched_at_utc="2026-06-01T00:00:00+00:00",
        db_path=db_path,
    )

    assert written == 1
    assert ta_cache.load_company_profiles(("aapl", "AAPL", "missing"), db_path=db_path) == {
        "AAPL": {
            "ticker": "AAPL",
            "description": "Description",
            "sector": "Technology",
            "industry": "Industry",
            "fetched_at_utc": "2026-06-01T00:00:00+00:00",
        }
    }

    assert ta_cache.upsert_company_profiles(
        {"AAPL": _profile("AAPL", sector="Updated Technology")},
        fetched_at_utc="2026-06-02T00:00:00+00:00",
        db_path=db_path,
    ) == 1
    updated = ta_cache.load_company_profiles(("AAPL",), db_path=db_path)["AAPL"]
    assert updated["sector"] == "Updated Technology"
    assert updated["fetched_at_utc"] == "2026-06-02T00:00:00+00:00"


def test_dashboard_fetches_only_missing_profiles_and_persists_them(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "ta_cache.db"
    ta_cache.upsert_company_profiles(
        {"AAPL": _profile("AAPL", description="Cached Apple")},
        db_path=db_path,
    )
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    factory_calls: list[tuple[list[str], str]] = []

    class FakeToolkit:
        def get_profile(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "MSFT": {
                        "Description": "Fetched Microsoft",
                        "Sector": "Technology",
                        "Industry": "Software",
                    }
                }
            )

    def factory(tickers: list[str], *, api_key: str) -> FakeToolkit:
        factory_calls.append((tickers, api_key))
        return FakeToolkit()

    profiles = load_company_profiles(
        ("aapl", "msft"),
        db_path=db_path,
        toolkit_factory=factory,
    )

    assert factory_calls == [(["MSFT"], "test-key")]
    assert profiles["AAPL"]["description"] == "Cached Apple"
    assert profiles["MSFT"]["description"] == "Fetched Microsoft"
    assert ta_cache.load_company_profiles(("MSFT",), db_path=db_path)["MSFT"][
        "industry"
    ] == "Software"

    def unexpected_factory(*args, **kwargs):
        raise AssertionError("FinanceToolkit must not run for complete cache hits")

    cached = load_company_profiles(
        ("AAPL", "MSFT"),
        db_path=db_path,
        toolkit_factory=unexpected_factory,
    )
    assert set(cached) == {"AAPL", "MSFT"}


def test_missing_api_key_preserves_hits_and_marks_only_misses(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "ta_cache.db"
    ta_cache.upsert_company_profiles(
        {"AAPL": _profile("AAPL", description="Cached Apple")},
        db_path=db_path,
    )
    monkeypatch.delenv("FMP_API_KEY", raising=False)

    profiles = load_company_profiles(("AAPL", "MSFT"), db_path=db_path)

    assert profiles["AAPL"]["description"] == "Cached Apple"
    assert profiles["AAPL"].get("error") is None
    assert profiles["MSFT"]["error"] == "FMP_API_KEY is not set."


def test_empty_and_failed_provider_responses_are_not_persisted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "ta_cache.db"
    monkeypatch.setenv("FMP_API_KEY", "test-key")

    class EmptyToolkit:
        def get_profile(self) -> pd.DataFrame:
            return pd.DataFrame()

    empty = load_company_profiles(
        ("EMPTY",),
        db_path=db_path,
        toolkit_factory=lambda *args, **kwargs: EmptyToolkit(),
    )
    assert "no profile data" in empty["EMPTY"]["error"].lower()
    assert ta_cache.load_company_profiles(("EMPTY",), db_path=db_path) == {}

    class FailedToolkit:
        def get_profile(self) -> pd.DataFrame:
            raise RuntimeError("provider failed with test-key")

    failed = load_company_profiles(
        ("FAIL",),
        db_path=db_path,
        toolkit_factory=lambda *args, **kwargs: FailedToolkit(),
    )
    assert "provider failed with <redacted>" in failed["FAIL"]["error"]
    assert ta_cache.load_company_profiles(("FAIL",), db_path=db_path) == {}
