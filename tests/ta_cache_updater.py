from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from financetoolkit import Toolkit

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (_PROJECT_ROOT, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rag_trial.chat import chatter
from rag_trial.db import ta_cache
import ta_dashboard
from ta_pipe import PipelineConfig, SignalResult, run_pipeline


def _require_update_environment() -> None:
    ta_cache.require_turso_configured()
    os.environ["TA_CACHE_READ_ONLY"] = "0"


def _successful_fundamental_result(result: dict[str, Any]) -> bool:
    trace = result.get("retrieval_trace") or {}
    return (
        str(result.get("decision") or "").lower() == "answer"
        and result.get("route_type") == "single_ticker_financial"
        and bool(trace.get("finance_context_present"))
        and bool(str(result.get("answer") or "").strip())
    )


def _run_single_ticker_analysis(ticker: str, company_name: str | None = None) -> dict[str, Any]:
    prompt = chatter.SINGLE_TICKER_FINANCIAL_PROMPT_TEMPLATE.format(
        date_min="N/A",
        date_max="N/A",
    )
    return chatter.run_single_ticker_fundamental_route(
        query=ta_dashboard._fundamental_query(ticker),
        ticker=ticker,
        company_name=company_name,
        gen_client=chatter.create_generation_client(),
        base_single_ticker_financial_prompt=prompt,
        dump_query_contexts=False,
    )


def _persist_fundamental_result(
    ticker: str,
    company_name: str | None,
    result: dict[str, Any],
    *,
    db_path: str | Path | None = None,
) -> None:
    trace = result.get("retrieval_trace") or {}
    assessment, score = ta_dashboard._extract_fundamental_fields(result.get("answer") or "")
    resolved_target = result.get("resolved_target") or {}
    ta_cache.upsert_fundamental_analysis(
        ticker=ticker,
        company_name=resolved_target.get("display_name") or company_name,
        query=result.get("query") or ta_dashboard._fundamental_query(ticker),
        answer=result.get("answer") or "",
        decision=result.get("decision"),
        route_type=result.get("route_type"),
        fundamental_assessment=assessment,
        fundamental_score=score,
        finance_context_present=bool(trace.get("finance_context_present")),
        news_context_present=bool(trace.get("news_context_present")),
        news_query=trace.get("news_query") or "",
        news_item_count=int(trace.get("news_item_count") or 0),
        retrieval_trace_json=json.dumps(trace, ensure_ascii=False),
        logs_json=json.dumps(result.get("logs") or [], ensure_ascii=False),
        db_path=db_path,
    )


def refresh_fundamentals(
    ranked: list[SignalResult],
    *,
    top_n: int = ta_dashboard.FUNDAMENTAL_TOP_N,
    route_runner=_run_single_ticker_analysis,
    db_path: str | Path | None = None,
) -> int:
    written = 0
    for result in ranked[:top_n]:
        ticker = result.ticker.upper()
        cached = ta_cache.load_fresh_fundamental_analysis(ticker, db_path=db_path)
        if cached is not None:
            continue
        route_result = route_runner(ticker, None)
        if not _successful_fundamental_result(route_result):
            continue
        _persist_fundamental_result(ticker, None, route_result, db_path=db_path)
        written += 1
    return written


def _clean_profile_value(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "n/a"}:
        return None
    return text


def fetch_company_profiles(
    tickers: list[str] | tuple[str, ...],
    *,
    toolkit_factory=Toolkit,
) -> dict[str, dict[str, Any]]:
    normalized = ta_dashboard._normalized_profile_tickers(tuple(tickers))
    if not normalized:
        return {}

    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("FMP_API_KEY is required to refresh company profiles")

    toolkit = toolkit_factory(list(normalized), api_key=api_key)
    frame = toolkit.get_profile()
    if frame is None or frame.empty:
        return {}

    profiles: dict[str, dict[str, Any]] = {}
    for ticker in normalized:
        profiles[ticker] = {
            "ticker": ticker,
            "description": _clean_profile_value(
                ta_dashboard._profile_value(frame, "Description", ticker)
            ),
            "sector": _clean_profile_value(ta_dashboard._profile_value(frame, "Sector", ticker)),
            "industry": _clean_profile_value(ta_dashboard._profile_value(frame, "Industry", ticker)),
        }
    return profiles


def refresh_company_profiles(
    ranked: list[SignalResult],
    *,
    top_n: int = ta_dashboard.FUNDAMENTAL_TOP_N,
    toolkit_factory=Toolkit,
    db_path: str | Path | None = None,
) -> int:
    tickers = [result.ticker for result in ranked[:top_n]]
    profiles = fetch_company_profiles(tickers, toolkit_factory=toolkit_factory)
    return ta_cache.upsert_company_profiles(profiles, db_path=db_path)


def _history_for_ticker(downloaded: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    return ta_dashboard._history_for_stock_pick_ticker(downloaded, ticker)


def _history_frame_to_bar_rows(ticker: str, hist: pd.DataFrame | None) -> list[dict[str, Any]]:
    return ta_dashboard._history_frame_to_stock_pick_bar_rows(ticker, hist)


def backfill_stock_pick_return_bars(
    tickers: list[str],
    *,
    start_date: str = ta_dashboard.STOCK_PICK_HISTORY_START_DATE,
    end_date: str | None = None,
    toolkit_factory=Toolkit,
    db_path: str | Path | None = None,
) -> int:
    normalized = list(ta_dashboard._normalized_profile_tickers(tuple(tickers)))
    if not normalized:
        return 0

    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("FMP_API_KEY is required to backfill stock-pick return bars")

    end = end_date or datetime.now(timezone.utc).date().isoformat()
    toolkit = toolkit_factory(
        normalized,
        api_key=api_key,
        start_date=start_date,
        end_date=end,
    )
    downloaded = toolkit.get_historical_data()
    written = 0
    for ticker in normalized:
        hist = _history_for_ticker(downloaded, ticker)
        rows = _history_frame_to_bar_rows(ticker, hist)
        if rows:
            written += ta_cache.upsert_daily_bars(
                rows,
                provider="financetoolkit",
                db_path=db_path,
            )
    return written


def save_signal_snapshots(
    ranked: list[SignalResult],
    cfg: PipelineConfig,
    *,
    snapshot_date: str,
    db_path: str | Path | None = None,
) -> int:
    cfg_payload = ta_dashboard._pipeline_config_to_snapshot(cfg)
    written = 0
    for result in ranked:
        ta_cache.upsert_signal_snapshot(
            snapshot_date,
            result.ticker,
            ta_dashboard._serialize_signal_result(result),
            cfg_payload,
            db_path=db_path,
        )
        written += 1
    return written


def update_turso_cache(
    *,
    cfg: PipelineConfig | None = None,
    snapshot_date: str | None = None,
    skip_fundamentals: bool = False,
    skip_profiles: bool = False,
    skip_return_bars: bool = False,
    db_path: str | Path | None = None,
) -> dict[str, int | str]:
    _require_update_environment()
    previous_force_turso = os.environ.get("TA_CACHE_FORCE_TURSO")
    os.environ["TA_CACHE_FORCE_TURSO"] = "1"
    try:
        ta_cache.ensure_cache_db(db_path)

        run_cfg = cfg or ta_dashboard.dashboard_pipeline_config()
        ranked = run_pipeline(run_cfg)
        date_key = snapshot_date or date.today().isoformat()

        snapshot_day = date.fromisoformat(date_key)
        ta_dashboard.save_stock_pick_snapshot(ranked, run_cfg, today=snapshot_day, db_path=db_path)
        ta_dashboard.save_bullish_candidate_snapshot(ranked, run_cfg, today=snapshot_day, db_path=db_path)
        signals = save_signal_snapshots(ranked, run_cfg, snapshot_date=date_key, db_path=db_path)

        tickers = [result.ticker for result in ranked[: run_cfg.top_n]]
        fundamentals = 0 if skip_fundamentals else refresh_fundamentals(ranked, db_path=db_path)
        profiles = 0 if skip_profiles else refresh_company_profiles(ranked, db_path=db_path)
        return_bars = (
            0
            if skip_return_bars
            else backfill_stock_pick_return_bars(tickers, db_path=db_path)
        )

        return {
            "snapshot_date": date_key,
            "ranked": len(ranked),
            "signal_snapshots": signals,
            "fundamentals_written": fundamentals,
            "profiles_written": profiles,
            "return_bars_written": return_bars,
        }
    finally:
        if previous_force_turso is None:
            os.environ.pop("TA_CACHE_FORCE_TURSO", None)
        else:
            os.environ["TA_CACHE_FORCE_TURSO"] = previous_force_turso


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh TA dashboard cache rows in Turso")
    parser.add_argument("--snapshot-date", default=None, help="Snapshot date in YYYY-MM-DD format")
    parser.add_argument("--skip-fundamentals", action="store_true")
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--skip-return-bars", action="store_true")
    args = parser.parse_args()

    summary = update_turso_cache(
        snapshot_date=args.snapshot_date,
        skip_fundamentals=args.skip_fundamentals,
        skip_profiles=args.skip_profiles,
        skip_return_bars=args.skip_return_bars,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
