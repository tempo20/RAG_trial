from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Iterable

import pandas as pd
from dotenv import load_dotenv
from financetoolkit import Toolkit

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (_PROJECT_ROOT, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rag_trial.db import ta_cache
from rag_trial.paths import TA_SQLITE_CACHE_DB_PATH

log = logging.getLogger(__name__)

LOCAL_TO_TURSO_TABLES = (
    "ta_articles",
    "ta_article_symbols",
    "ta_historical_daily_bars",
    "ta_stock_pick_snapshots",
    "ta_fundamental_analyses",
    "ta_bullish_candidate_snapshots",
    "ta_signal_snapshots",
    "ta_company_profiles",
)
JSON_HEAVY_TURSO_TABLES = {
    "ta_bullish_candidate_snapshots",
    "ta_signal_snapshots",
}

ta_dashboard = None
PipelineConfig = None
SignalResult = None
run_pipeline = None


def _load_pipeline_modules() -> None:
    global PipelineConfig, SignalResult, run_pipeline, ta_dashboard
    if (
        ta_dashboard is not None
        and PipelineConfig is not None
        and SignalResult is not None
        and run_pipeline is not None
    ):
        return

    import ta_dashboard as dashboard_module
    from ta_pipe import PipelineConfig as pipeline_config_cls
    from ta_pipe import SignalResult as signal_result_cls
    from ta_pipe import run_pipeline as run_pipeline_fn

    if ta_dashboard is None:
        ta_dashboard = dashboard_module
    if PipelineConfig is None:
        PipelineConfig = pipeline_config_cls
    if SignalResult is None:
        SignalResult = signal_result_cls
    if run_pipeline is None:
        run_pipeline = run_pipeline_fn


def _resolved_local_db_path(db_path: str | Path | None = None) -> Path:
    return Path(db_path) if db_path is not None else TA_SQLITE_CACHE_DB_PATH


def _set_or_clear_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


@contextmanager
def _temporary_env(values: dict[str, str | None]):
    previous = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            _set_or_clear_env(name, value)
        yield
    finally:
        for name, value in previous.items():
            _set_or_clear_env(name, value)


@contextmanager
def _local_cache_environment(db_path: str | Path):
    path = Path(db_path)
    with _temporary_env(
        {
            "TA_SQLITE_CACHE_DB": str(path),
            "TA_CACHE_READ_ONLY": "0",
            "TA_CACHE_FORCE_TURSO": "0",
        }
    ):
        yield


@contextmanager
def _turso_cache_environment():
    with _temporary_env(
        {
            "TA_SQLITE_CACHE_DB": None,
            "TA_CACHE_READ_ONLY": "0",
            "TA_CACHE_FORCE_TURSO": "1",
        }
    ):
        yield


def _successful_fundamental_result(result: dict[str, Any]) -> bool:
    trace = result.get("retrieval_trace") or {}
    return (
        str(result.get("decision") or "").lower() == "answer"
        and result.get("route_type") == "single_ticker_financial"
        and bool(trace.get("finance_context_present"))
        and bool(str(result.get("answer") or "").strip())
    )


def _run_single_ticker_analysis(ticker: str, company_name: str | None = None) -> dict[str, Any]:
    _load_pipeline_modules()
    from rag_trial.chat import chatter

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
    _load_pipeline_modules()
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
    top_n: int | None = None,
    route_runner=_run_single_ticker_analysis,
    db_path: str | Path | None = None,
) -> int:
    _load_pipeline_modules()
    if top_n is None:
        top_n = ta_dashboard.FUNDAMENTAL_TOP_N

    written = 0
    targets = ranked[:top_n]
    log.info("Refreshing fundamentals for up to %d ranked tickers", len(targets))
    for index, result in enumerate(targets, 1):
        ticker = result.ticker.upper()
        cached = ta_cache.load_fresh_fundamental_analysis(ticker, db_path=db_path)
        if cached is not None:
            log.info("Fundamentals [%d/%d] %s cached", index, len(targets), ticker)
            continue
        log.info("Fundamentals [%d/%d] %s refreshing", index, len(targets), ticker)
        route_result = route_runner(ticker, None)
        if not _successful_fundamental_result(route_result):
            log.info(
                "Fundamentals [%d/%d] %s skipped: route produced no answer",
                index,
                len(targets),
                ticker,
            )
            continue
        _persist_fundamental_result(ticker, None, route_result, db_path=db_path)
        written += 1
        log.info("Fundamentals [%d/%d] %s written", index, len(targets), ticker)
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
    _load_pipeline_modules()
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
    top_n: int | None = None,
    toolkit_factory=Toolkit,
    db_path: str | Path | None = None,
) -> int:
    _load_pipeline_modules()
    if top_n is None:
        top_n = ta_dashboard.FUNDAMENTAL_TOP_N

    tickers = [result.ticker for result in ranked[:top_n]]
    log.info("Refreshing company profiles for %d tickers", len(tickers))
    profiles = fetch_company_profiles(tickers, toolkit_factory=toolkit_factory)
    return ta_cache.upsert_company_profiles(profiles, db_path=db_path)


def _history_for_ticker(downloaded: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    _load_pipeline_modules()
    return ta_dashboard._history_for_stock_pick_ticker(downloaded, ticker)


def _history_frame_to_bar_rows(ticker: str, hist: pd.DataFrame | None) -> list[dict[str, Any]]:
    _load_pipeline_modules()
    return ta_dashboard._history_frame_to_stock_pick_bar_rows(ticker, hist)


def backfill_stock_pick_return_bars(
    tickers: list[str],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    toolkit_factory=Toolkit,
    db_path: str | Path | None = None,
) -> int:
    _load_pipeline_modules()
    if start_date is None:
        start_date = ta_dashboard.STOCK_PICK_HISTORY_START_DATE

    normalized = list(ta_dashboard._normalized_profile_tickers(tuple(tickers)))
    if not normalized:
        return 0

    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("FMP_API_KEY is required to backfill stock-pick return bars")

    end = end_date or datetime.now(timezone.utc).date().isoformat()
    log.info(
        "Backfilling stock-pick return bars for %d tickers from %s to %s",
        len(normalized),
        start_date,
        end,
    )
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
    _load_pipeline_modules()
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


def update_local_cache(
    *,
    cfg: PipelineConfig | None = None,
    snapshot_date: str | None = None,
    skip_fundamentals: bool = False,
    skip_profiles: bool = False,
    skip_return_bars: bool = False,
    db_path: str | Path | None = None,
) -> dict[str, int | str]:
    _load_pipeline_modules()
    local_db_path = _resolved_local_db_path(db_path)

    with _local_cache_environment(local_db_path):
        ta_cache.ensure_cache_db(local_db_path)

        run_cfg = cfg or ta_dashboard.dashboard_pipeline_config()
        log.info(
            "Running TA pipeline with news_max_pages=%d news_limit=%d news_days=%d",
            run_cfg.news_max_pages,
            run_cfg.news_limit,
            run_cfg.news_days,
        )
        ranked = run_pipeline(run_cfg)
        date_key = snapshot_date or date.today().isoformat()

        snapshot_day = date.fromisoformat(date_key)
        log.info("Saving dashboard snapshots for %s with %d ranked candidates", date_key, len(ranked))
        ta_dashboard.save_stock_pick_snapshot(ranked, run_cfg, today=snapshot_day, db_path=local_db_path)
        ta_dashboard.save_bullish_candidate_snapshot(
            ranked,
            run_cfg,
            today=snapshot_day,
            db_path=local_db_path,
        )
        signals = save_signal_snapshots(
            ranked,
            run_cfg,
            snapshot_date=date_key,
            db_path=local_db_path,
        )
        log.info("Saved %d signal snapshots", signals)

        tickers = [result.ticker for result in ranked[: run_cfg.top_n]]
        fundamentals = (
            0
            if skip_fundamentals
            else refresh_fundamentals(ranked, db_path=local_db_path)
        )
        if skip_fundamentals:
            log.info("Skipping fundamentals refresh")
        profiles = 0 if skip_profiles else refresh_company_profiles(ranked, db_path=local_db_path)
        if skip_profiles:
            log.info("Skipping company profile refresh")
        return_bars = (
            0
            if skip_return_bars
            else backfill_stock_pick_return_bars(tickers, db_path=local_db_path)
        )
        if skip_return_bars:
            log.info("Skipping stock-pick return bar backfill")

    return {
        "cache_db_path": str(local_db_path),
        "snapshot_date": date_key,
        "ranked": len(ranked),
        "signal_snapshots": signals,
        "fundamentals_written": fundamentals,
        "profiles_written": profiles,
        "return_bars_written": return_bars,
    }


def update_turso_cache(
    *,
    cfg: PipelineConfig | None = None,
    snapshot_date: str | None = None,
    skip_fundamentals: bool = False,
    skip_profiles: bool = False,
    skip_return_bars: bool = False,
    db_path: str | Path | None = None,
) -> dict[str, int | str]:
    log.warning("update_turso_cache() is deprecated; writing local ta_cache.db instead")
    return update_local_cache(
        cfg=cfg,
        snapshot_date=snapshot_date,
        skip_fundamentals=skip_fundamentals,
        skip_profiles=skip_profiles,
        skip_return_bars=skip_return_bars,
        db_path=db_path,
    )


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
    )


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")]


def _is_turso_stream_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "stream not found" in text or ("hrana" in text and "404" in text)


def _close_quietly(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass


def _execute_turso_batch(
    dst,
    sql: str,
    params: tuple[Any, ...],
    *,
    table: str,
    copied: int,
    max_attempts: int = 5,
):
    for attempt in range(max_attempts):
        try:
            dst.execute(sql, params)
            dst.commit()
            return dst
        except Exception as exc:
            if attempt + 1 < max_attempts and _is_turso_stream_error(exc):
                sleep_s = min(2**attempt, 10)
                log.warning(
                    (
                        "%s: Turso stream expired after %d copied; "
                        "reconnecting and retrying batch (%d/%d) after %ds"
                    ),
                    table,
                    copied,
                    attempt + 1,
                    max_attempts,
                    sleep_s,
                )
                _close_quietly(dst)
                time.sleep(sleep_s)
                dst = ta_cache.connect()
                continue
            raise
    return dst


def _copy_table_to_turso(
    src: sqlite3.Connection,
    table: str,
    *,
    batch_size: int,
) -> int | None:
    if not _table_exists(src, table):
        log.info("%s: skipped", table)
        return None

    cols = _table_columns(src, table)
    if not cols:
        log.info("%s: skipped, no columns", table)
        return 0

    cursor = src.execute(f"SELECT {', '.join(cols)} FROM {table}")
    copied = 0
    effective_batch_size = 1 if table in JSON_HEAVY_TURSO_TABLES else batch_size
    row_placeholder = "(" + ", ".join(["?"] * len(cols)) + ")"
    column_list = ", ".join(cols)

    dst = ta_cache.connect()
    try:
        while True:
            batch = cursor.fetchmany(effective_batch_size)
            if not batch:
                break

            sql = (
                f"INSERT OR REPLACE INTO {table} ({column_list}) VALUES "
                + ", ".join([row_placeholder] * len(batch))
            )
            params: list[Any] = []
            for row in batch:
                params.extend(row[col] for col in cols)

            dst = _execute_turso_batch(
                dst,
                sql,
                tuple(params),
                table=table,
                copied=copied,
            )

            copied += len(batch)
            log.info("%s: copied %d", table, copied)
    finally:
        _close_quietly(dst)

    log.info("%s: done %d", table, copied)
    return copied


def copy_local_cache_to_turso(
    local_path: str | Path | None = None,
    *,
    tables: Iterable[str] = LOCAL_TO_TURSO_TABLES,
    batch_size: int = 250,
) -> dict[str, int | str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)
    ta_cache.require_turso_configured()

    local_db_path = _resolved_local_db_path(local_path)
    ta_cache.ensure_cache_db(local_db_path)

    copied: dict[str, int | str] = {}
    src = sqlite3.connect(local_db_path)
    src.row_factory = sqlite3.Row
    try:
        with _turso_cache_environment():
            ta_cache.ensure_cache_db()
            for table in tables:
                copied_count = _copy_table_to_turso(
                    src,
                    table,
                    batch_size=batch_size,
                )
                copied[table] = "skipped" if copied_count is None else copied_count
    finally:
        src.close()

    return copied


def _build_config_from_args(args: argparse.Namespace) -> PipelineConfig | None:
    if args.news_max_pages is None and args.news_sleep_s is None:
        return None

    if args.news_max_pages is not None and args.news_max_pages <= 0:
        raise ValueError("--news-max-pages must be greater than 0")
    if args.news_sleep_s is not None and args.news_sleep_s < 0:
        raise ValueError("--news-sleep-s must be 0 or greater")

    _load_pipeline_modules()
    cfg = ta_dashboard.dashboard_pipeline_config()
    if args.news_max_pages is not None:
        cfg.news_max_pages = args.news_max_pages
    if args.news_sleep_s is not None:
        cfg.news_sleep_s = args.news_sleep_s
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh TA dashboard rows in local SQLite ta_cache.db")
    parser.add_argument("--snapshot-date", default=None, help="Snapshot date in YYYY-MM-DD format")
    parser.add_argument("--skip-fundamentals", action="store_true")
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--skip-return-bars", action="store_true")
    parser.add_argument(
        "--db-path",
        default=str(TA_SQLITE_CACHE_DB_PATH),
        help="Local SQLite TA cache path to refresh",
    )
    parser.add_argument(
        "--push-turso",
        action="store_true",
        help="Copy refreshed local cache rows into Turso after the local update succeeds",
    )
    parser.add_argument(
        "--push-turso-only",
        action="store_true",
        help="Only copy existing local cache rows into Turso; do not refresh local cache",
    )
    parser.add_argument(
        "--push-batch-size",
        type=int,
        default=250,
        help="Rows per INSERT batch when copying local cache rows into Turso",
    )
    parser.add_argument(
        "--news-max-pages",
        type=int,
        default=None,
        help="Override the FMP stock-news page cap for this run",
    )
    parser.add_argument(
        "--news-sleep-s",
        type=float,
        default=None,
        help="Override sleep seconds between FMP stock-news pages for this run",
    )
    args = parser.parse_args()

    if args.push_batch_size <= 0:
        parser.error("--push-batch-size must be greater than 0")

    if args.push_turso and args.push_turso_only:
        parser.error("--push-turso and --push-turso-only cannot be used together")

    if args.push_turso_only:
        refresh_only_args = {
            "--snapshot-date": args.snapshot_date is not None,
            "--skip-fundamentals": args.skip_fundamentals,
            "--skip-profiles": args.skip_profiles,
            "--skip-return-bars": args.skip_return_bars,
            "--news-max-pages": args.news_max_pages is not None,
            "--news-sleep-s": args.news_sleep_s is not None,
        }
        invalid = [name for name, enabled in refresh_only_args.items() if enabled]
        if invalid:
            parser.error(
                "--push-turso-only does not run a local refresh; remove "
                + ", ".join(invalid)
            )

        summary = {
            "cache_db_path": str(_resolved_local_db_path(args.db_path)),
            "turso_copied": copy_local_cache_to_turso(
                args.db_path,
                batch_size=args.push_batch_size,
            ),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    try:
        cfg = _build_config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    summary: dict[str, Any] = update_local_cache(
        cfg=cfg,
        snapshot_date=args.snapshot_date,
        skip_fundamentals=args.skip_fundamentals,
        skip_profiles=args.skip_profiles,
        skip_return_bars=args.skip_return_bars,
        db_path=args.db_path,
    )

    if args.push_turso:
        summary["turso_copied"] = copy_local_cache_to_turso(
            args.db_path,
            batch_size=args.push_batch_size,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
