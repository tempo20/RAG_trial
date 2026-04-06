"""
Load hist_data.parquet (FinanceToolkit wide panel) into Neo4j.

Creates:
  - (:Instrument {ticker}) — one per symbol in the parquet
  - (:MarketBar {bar_uid, bar_date, ...OHLCV metrics}) — one per trading day × ticker
  - (:MarketBar)-[:FOR_INSTRUMENT]->(:Instrument)

Then links the news graph to instruments when an :Entity clearly matches a ticker:
  - (:Entity)-[:ALIASES_TICKER]->(:Instrument)
  when canonical_name = toLower(ticker) OR toUpper(trim(name)) = ticker

Downstream, articles connect to prices via paths like:
  (:Article)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:Entity)-[:ALIASES_TICKER]->(:Instrument)<-[:FOR_INSTRUMENT]-(:MarketBar)

Usage:
  python hist_to_db.py              # FinanceToolkit download, then Neo4j ingest (FMP_API_KEY)
  python hist_to_db.py --no-fetch   # use existing parquet only (no API call)
  python hist_to_db.py --parquet custom.parquet --batch-size 2000
  python hist_to_db.py --skip-link  # only bars/instruments, no Entity links
  python hist_to_db.py --fetch-only # only write hist_data.parquet (FMP_API_KEY)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

DEFAULT_PARQUET = Path("hist_data.parquet")

# Same universe as get_fin.ipynb (FinanceToolkit / FMP).
TOP_100_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "LLY", "AVGO",
    "JPM", "V", "XOM", "UNH", "MA", "COST", "HD", "PG", "JNJ", "ORCL",
    "ADBE", "CRM", "KO", "PEP", "ABBV", "CVX", "MRK", "BAC", "WMT", "MCD",
    "CSCO", "ACN", "INTC", "AMD", "NFLX", "QCOM", "TXN", "LIN", "NEE", "PM",
    "AMGN", "IBM", "HON", "INTU", "UPS", "RTX", "LOW", "CAT", "SPGI", "GS",
    "BLK", "DE", "TMO", "ISRG", "AXP", "SYK", "BKNG", "MDLZ", "GILD", "ADP",
    "LMT", "CB", "MO", "ZTS", "PNC", "CI", "MMC", "PLD", "ELV", "DUK",
    "SO", "VRTX", "REGN", "BDX", "ETN", "APD", "SHW", "ITW", "EOG", "FDX",
]


def fetch_historical_to_parquet(
    parquet_path: Path = DEFAULT_PARQUET,
    tickers: list[str] | None = None,
    start_date: str = "2017-12-31",
    api_key: str | None = None,
) -> None:
    """Download wide-panel historical data (FinanceToolkit) and save like get_fin.ipynb."""
    try:
        from financetoolkit import Toolkit
    except ImportError as e:
        raise SystemExit(
            "financetoolkit is required for the default run and --fetch-only. "
            "Install it (e.g. pip install financetoolkit), or use --no-fetch with an existing parquet."
        ) from e
    key = api_key or os.getenv("FMP_API_KEY")
    if not key:
        raise SystemExit(
            "FMP_API_KEY not set in environment (required for FinanceToolkit download). "
            "Set FMP_API_KEY or run with --no-fetch if parquet already exists."
        )
    universe = tickers if tickers is not None else TOP_100_TICKERS
    companies = Toolkit(universe, api_key=key, start_date=start_date)
    hist_data = companies.get_historical_data()
    hist_data.to_parquet(parquet_path, index=True)


# Wide hist_data columns: metric (level 0) × ticker (level 1)
METRIC_TO_PROP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Return": "return_1d",
    "Volatility": "volatility",
    "Excess Return": "excess_return",
    "Excess Volatility": "excess_volatility",
    "Cumulative Return": "cumulative_return",
}

BAR_FLOAT_PROPS = frozenset(METRIC_TO_PROP.values())


def _to_python_scalar(val):
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (np.floating, np.integer)):
        return val.item()
    return val


def long_frame_from_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(
            "Expected a MultiIndex DataFrame (metric × ticker). "
            "Re-export from FinanceToolkit get_historical_data()."
        )
    df.columns.names = ["metric", "ticker"]
    stacked = df.stack(level="ticker", future_stack=True)
    rename_map = {
        m: METRIC_TO_PROP.get(m, m.lower().replace(" ", "_"))
        for m in stacked.columns
    }
    stacked = stacked.rename(columns=rename_map)
    out = stacked.reset_index()
    dates = out["date"]
    if hasattr(dates.dt, "to_timestamp"):
        dates = dates.dt.to_timestamp()
    out["bar_date"] = pd.to_datetime(dates).dt.strftime("%Y-%m-%d")
    out["bar_uid"] = out["ticker"].astype(str) + "_" + out["bar_date"]
    return out


def row_to_neo_params(row: pd.Series) -> dict:
    """Build a Cypher parameter map; omit NaN so we do not write null noise unless needed."""
    p = {
        "ticker": str(row["ticker"]),
        "bar_uid": str(row["bar_uid"]),
        "bar_date": str(row["bar_date"]),
    }
    for col in BAR_FLOAT_PROPS:
        if col not in row.index:
            continue
        v = row[col]
        if pd.isna(v):
            continue
        p[col] = float(_to_python_scalar(v))
    return p


UPSERT_BARS = """
UNWIND $rows AS row
MERGE (i:Instrument {ticker: row.ticker})
MERGE (b:MarketBar {bar_uid: row.bar_uid})
SET b.bar_date = date(row.bar_date),
    b.ticker = row.ticker,
    b.open = row.open,
    b.high = row.high,
    b.low = row.low,
    b.close = row.close,
    b.adj_close = row.adj_close,
    b.volume = row.volume,
    b.dividends = row.dividends,
    b.return_1d = row.return_1d,
    b.volatility = row.volatility,
    b.excess_return = row.excess_return,
    b.excess_volatility = row.excess_volatility,
    b.cumulative_return = row.cumulative_return
MERGE (b)-[:FOR_INSTRUMENT]->(i)
"""


def rows_batch_to_params(long_df: pd.DataFrame, start: int, end: int) -> list[dict]:
    batch = long_df.iloc[start:end]
    rows = []
    for _, row in batch.iterrows():
        base = row_to_neo_params(row)
        # All metric keys for Cypher SET (explicit query expects them)
        full = {
            "ticker": base["ticker"],
            "bar_uid": base["bar_uid"],
            "bar_date": base["bar_date"],
        }
        for k in BAR_FLOAT_PROPS:
            full[k] = base.get(k)
        rows.append(full)
    return rows


def ensure_constraints(session) -> None:
    session.run(
        """
        CREATE CONSTRAINT instrument_ticker_unique IF NOT EXISTS
        FOR (i:Instrument) REQUIRE i.ticker IS UNIQUE
        """
    )
    session.run(
        """
        CREATE CONSTRAINT market_bar_uid_unique IF NOT EXISTS
        FOR (b:MarketBar) REQUIRE b.bar_uid IS UNIQUE
        """
    )


def link_entities_to_instruments(session) -> int:
    """Return number of Entity–Instrument pairs linked (one ALIASES_TICKER per pair)."""
    result = session.run(
        """
        MATCH (e:Entity), (i:Instrument)
        WHERE e.mapped_ticker IS NOT NULL
          AND e.mapped_ticker = i.ticker
        MERGE (e)-[:ALIASES_TICKER]->(i)
        RETURN count(*) AS cnt
        """
    ).single()
    return int(result["cnt"]) if result and result["cnt"] is not None else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest financial parquet into Neo4j")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET,
        help="Path to hist_data.parquet",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1500,
        help="Rows per UNWIND batch",
    )
    parser.add_argument(
        "--skip-link",
        action="store_true",
        help="Do not create Entity-[:ALIASES_TICKER]->Instrument edges",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip FinanceToolkit; ingest existing parquet only",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Only write parquet (FMP_API_KEY); do not connect to Neo4j",
    )
    parser.add_argument(
        "--start-date",
        default="2017-12-31",
        help="Toolkit start_date for download (default run or --fetch-only); default: 2017-12-31",
    )
    args = parser.parse_args()

    if args.fetch_only and args.no_fetch:
        raise SystemExit("Cannot combine --fetch-only with --no-fetch.")

    if args.fetch_only:
        print(f"Fetching historical data -> {args.parquet.resolve()} ...")
        fetch_historical_to_parquet(
            parquet_path=args.parquet,
            start_date=args.start_date,
        )
        print("Parquet written. Done (fetch-only).")
        return

    if not args.no_fetch:
        print(f"Fetching historical data -> {args.parquet.resolve()} ...")
        fetch_historical_to_parquet(
            parquet_path=args.parquet,
            start_date=args.start_date,
        )

    if not args.parquet.is_file():
        raise SystemExit(f"Parquet not found: {args.parquet.resolve()}")

    print(f"Reading {args.parquet} ...")
    long_df = long_frame_from_parquet(args.parquet)
    print(f"Long format: {len(long_df)} rows (bars)")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    n = len(long_df)
    with driver.session() as session:
        ensure_constraints(session)
        print(f"Upserting MarketBar + Instrument in batches of {args.batch_size} ...")
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            rows = rows_batch_to_params(long_df, start, end)
            session.run(UPSERT_BARS, {"rows": rows})
            print(f"  {end}/{n}")

        if not args.skip_link:
            print("Linking Entity nodes to Instrument (ticker match) ...")
            cnt = link_entities_to_instruments(session)
            print(f"  ALIASES_TICKER relationships (total in graph after merge): {cnt}")

            # Materialize a direct bridge from news text chunks to Instruments.
            # This makes it easy to retrieve market data tied to the same chunks/entities.
            print("Creating Chunk -> Instrument mention edges (MENTIONS_INSTRUMENT) ...")
            session.run(
                """
                MATCH (ch:Chunk)-[:MENTIONS]->(e:Entity)-[:ALIASES_TICKER]->(i:Instrument)
                MERGE (ch)-[:MENTIONS_INSTRUMENT]->(i)
                """
            )

    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
