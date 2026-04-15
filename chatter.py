"""
chatter.py - Hybrid QA chatbot using lean Neo4j plus SQLite evidence.

Neo4j stores structured macro objects only:
  - Period
  - Entity
  - MacroEvent
  - Asset
  - Channel

SQLite remains the source of truth for:
  - articles
  - chunks
  - entity_mentions
  - macro_events and related normalized tables

Retrieval strategy:
  1. Resolve entity / asset intent from the query
  2. Use Neo4j for structured event reasoning
  3. Use SQLite to materialize chunk/article evidence
  4. Fall back to SQLite chunk embedding search for broad news QA
"""

from __future__ import annotations

import json
import importlib
import os
import re
import sqlite3
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from dotenv import load_dotenv
load_dotenv()

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=FutureWarning)

import dateparser
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from graph_schema import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    PERIOD_GRANULARITY,
    asset_key,
    period_key_for,
)
from tgrag_setup import (
    EMBED_MODEL_NAME,
    _canonicalize,
    load_financial_entity_map,
    load_ticker_company_map,
)
from convo_memory import (
    ConversationMemory,
    resolve_coreference,
    resolve_temporal_carryover,
    save_memory,
    load_memory,
    MEMORY_PATH,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEBUG_SKIP_GENERATION = os.getenv("DEBUG_SKIP_GENERATION", "0").strip() in {"1", "true", "yes"}
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TICKER_MAP_PATH = Path(os.getenv("TICKER_MAP_PATH", "ticker_company_map.csv"))
FIN_ENTITY_MAP_PATH = Path(os.getenv("FIN_ENTITY_MAP_PATH", "financial_entity_map.csv"))
SQLITE_DB = os.getenv("SQLITE_DB", "my_database.db")

TOP_K = int(os.getenv("TOP_K", "4"))
EXPANDED_K = int(os.getenv("EXPANDED_K", "8"))
RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "5"))
SQLITE_SEMANTIC_CANDIDATE_LIMIT = int(os.getenv("SQLITE_SEMANTIC_CANDIDATE_LIMIT", "1500"))
MACRO_EVENT_CANDIDATE_LIMIT = int(os.getenv("MACRO_EVENT_CANDIDATE_LIMIT", "600"))
GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "1024"))
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
ALPHA_VANTAGE_MCP_URL = os.getenv("ALPHA_VANTAGE_MCP_URL", "https://mcp.alphavantage.co/mcp").strip()
_alpha_mcp_remote_flag = os.getenv("ENABLE_ALPHA_MCP_REMOTE")
ENABLE_ALPHA_MCP_REMOTE = (
    _alpha_mcp_remote_flag.strip().lower() in {"1", "true", "yes", "on"}
    if _alpha_mcp_remote_flag is not None
    else bool(ALPHA_VANTAGE_API_KEY)
)

ALPHA_MCP_BETA = "mcp-client-2025-11-20"
ALPHA_MCP_SERVER_NAME = "alphavantage"
ALPHA_MCP_ALLOWED_TOOLS = (
    "TOOL_CALL",
    "GLOBAL_QUOTE",
    "TIME_SERIES_DAILY",
    "COMPANY_OVERVIEW",
)
ALPHA_MCP_FALLBACK_NOTE = "Live market data is temporarily unavailable; answer is based on retrieved news context."
try:
    ALPHA_MCP_TIMEOUT_SECONDS = float(os.getenv("ALPHA_MCP_TIMEOUT_SECONDS", "25"))
except ValueError:
    ALPHA_MCP_TIMEOUT_SECONDS = 25.0

SOURCE_KEYWORDS = {
    "bbc": "BBC",
    "bloomberg": "Bloomberg",
    "cnbc": "CNBC",
    "marketwatch": "MarketWatch",
    "nasdaq": "Nasdaq",
    "cbs": "CBS MoneyWatch",
}

MARKET_INTENT_HINTS = (
    "price",
    "quote",
    "trading at",
    "market cap",
    "valuation",
    "p/e",
    "pe ratio",
    "eps",
    "fundamental",
    "balance sheet",
    "cash flow",
    "income statement",
    "company overview",
    "daily close",
    "open price",
    "high",
    "low",
    "volume",
    "performance",
    "return",
)

CAUSAL_ENTITY_TICKER_MAP = {
    # ------------------------------------------------------------------
    # USD / broad dollar
    # ------------------------------------------------------------------
    "usd":                  "DX-Y.NYB",
    "dollar":               "DX-Y.NYB",
    "us-dollar":            "DX-Y.NYB",
    "us-dollar-index":      "DX-Y.NYB",
    "dxy":                  "DX-Y.NYB",

    # ------------------------------------------------------------------
    # U.S. rates / curve
    # ------------------------------------------------------------------
    "us-3m-yield":          "^IRX",      # 13-week bill proxy on Yahoo
    "us-13w-yield":         "^IRX",
    "us-5y-yield":          "^FVX",
    "us-10y-yield":         "^TNX",
    "us-30y-yield":         "^TYX",

    # Better to derive curve spreads in code rather than map them directly
    # e.g. yield_curve_10y_3m = ^TNX - ^IRX
    #      yield_curve_10y_2y = external source or explicit chosen proxy
    "yield-curve":          "^TNX",      # placeholder anchor only
    "real-yields":          "TIP",       # practical proxy
    "treasuries":           "TLT",       # long-duration Treasury proxy
    "fed":                  "ZQ=F",      # Fed funds futures proxy
    "fed-funds-rate":       "ZQ=F",

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------
    "oil":                  "CL=F",
    "crude":                "CL=F",
    "wti":                  "CL=F",
    "wti-crude":            "CL=F",
    "brent":                "BZ=F",
    "brent-crude":          "BZ=F",
    "natural-gas":          "NG=F",
    "gas":                  "NG=F",
    "gasoline":             "RB=F",
    "heating-oil":          "HO=F",

    # ------------------------------------------------------------------
    # Metals / commodities
    # ------------------------------------------------------------------
    "gold":                 "GC=F",
    "silver":               "SI=F",
    "copper":               "HG=F",
    "platinum":             "PL=F",
    "palladium":            "PA=F",

    # ------------------------------------------------------------------
    # U.S. equity / volatility / risk sentiment
    # ------------------------------------------------------------------
    "sp500":                "SPY",
    "s&p500":               "SPY",
    "s-and-p-500":          "SPY",
    "nasdaq":               "QQQ",
    "nasdaq-100":           "QQQ",
    "dow":                  "DIA",
    "russell-2000":         "IWM",
    "small-caps":           "IWM",
    "vix":                  "^VIX",
    "equities":             "SPY",
    "stocks":               "SPY",

    # ------------------------------------------------------------------
    # Credit / spread proxies
    # ------------------------------------------------------------------
    "credit":               "LQD",
    "investment-grade":     "LQD",
    "high-yield":           "HYG",
    "junk-bonds":           "HYG",
    "em-credit":            "EMB",

    # ------------------------------------------------------------------
    # Major FX
    # ------------------------------------------------------------------
    "eur-usd":              "EURUSD=X",
    "euro-dollar":          "EURUSD=X",
    "usd-jpy":              "USDJPY=X",
    "dollar-yen":           "USDJPY=X",
    "gbp-usd":              "GBPUSD=X",
    "sterling-dollar":      "GBPUSD=X",
    "aud-usd":              "AUDUSD=X",
    "aussie-dollar":        "AUDUSD=X",
    "nzd-usd":              "NZDUSD=X",
    "kiwi-dollar":          "NZDUSD=X",
    "usd-chf":              "USDCHF=X",
    "dollar-franc":         "USDCHF=X",
    "usd-cad":              "USDCAD=X",
    "dollar-loonie":        "USDCAD=X",
    "usd-cnh":              "CNH=X",
    "offshore-yuan":        "CNH=X",
    "usd-cny":              "CNY=X",
    "dollar-yuan":          "CNH=X",     # prefer offshore for trading sensitivity
    "eur-jpy":              "EURJPY=X",
    "gbp-jpy":              "GBPJPY=X",
    "aud-jpy":              "AUDJPY=X",

    # ------------------------------------------------------------------
    # Region / country / macro proxies
    # ------------------------------------------------------------------
    "eurozone":             "FXE",
    "euro":                 "FXE",
    "japan":                "EWJ",
    "yen":                  "FXY",
    "uk":                   "FXB",
    "british-pound":        "FXB",
    "china":                "FXI",
    "hong-kong":            "EWH",
    "emerging-markets":     "EEM",
    "em":                   "EEM",

    # ------------------------------------------------------------------
    # Commodity equity proxies
    # ------------------------------------------------------------------
    "energy-stocks":        "XLE",
    "oil-stocks":           "XLE",
    "gold-miners":          "GDX",
    "miners":               "XME",

    # ------------------------------------------------------------------
    # Inflation / commodity basket proxies
    # ------------------------------------------------------------------
    "inflation":            "TIP",
    "commodities":          "DBC",
    "commodity-index":      "DBC",

    # ------------------------------------------------------------------
    # Crypto / alternative risk sentiment
    # ------------------------------------------------------------------
    "bitcoin":              "BTC-USD",
    "btc":                  "BTC-USD",
    "ethereum":             "ETH-USD",
    "eth":                  "ETH-USD",

    # ------------------------------------------------------------------
    # Safe havens / defensive
    # ------------------------------------------------------------------
    "safe-havens":          "GC=F",
    "gold-proxy":           "GC=F",
    "yen-safe-haven":       "USDJPY=X",
    "treasury-safe-haven":  "TLT",
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def strip_think_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def anthropic_text(response) -> str:
    parts = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip()


def connect_sqlite(db_path: str = SQLITE_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_embedding_json(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _sqlite_row_to_chunk_dict(row: sqlite3.Row) -> dict:
    return {
        "chunk_uid": row["chunk_id"],
        "text": row["text"],
        "embedding": _parse_embedding_json(row["embedding_json"]),
        "published_date": row["published_date"],
        "period_key": row["period_key"],
        "title": row["title"],
        "url": row["url"],
        "source": row["source"],
        "article_id": row["article_id"],
    }


def _fetch_chunk_rows_by_ids(
    conn: sqlite3.Connection,
    chunk_ids: list[str],
    date_start: str | None = None,
    date_end: str | None = None,
    source_filter: str | None = None,
) -> dict[str, dict]:
    if not chunk_ids:
        return {}

    placeholders = ",".join("?" for _ in chunk_ids)
    sql = f"""
        SELECT
            c.chunk_id,
            c.article_id,
            c.text,
            c.published_date,
            c.period_key,
            c.embedding_json,
            a.title,
            a.url,
            a.source
        FROM chunks c
        JOIN articles a ON a.article_id = c.article_id
        WHERE c.chunk_id IN ({placeholders})
    """
    params: list[Any] = list(chunk_ids)
    if date_start:
        sql += " AND c.published_date >= ?"
        params.append(date_start)
    if date_end:
        sql += " AND c.published_date <= ?"
        params.append(date_end)
    if source_filter:
        sql += " AND a.source = ?"
        params.append(source_filter)

    rows = conn.execute(sql, params).fetchall()
    return {row["chunk_id"]: _sqlite_row_to_chunk_dict(row) for row in rows}


def _get_sqlite_date_range(conn: sqlite3.Connection) -> tuple[str, str]:
    row = conn.execute(
        """
        SELECT
            MIN(COALESCE(chunks.published_date, substr(articles.published_at, 1, 10))) AS earliest,
            MAX(COALESCE(chunks.published_date, substr(articles.published_at, 1, 10))) AS latest
        FROM articles
        LEFT JOIN chunks ON chunks.article_id = articles.article_id
        """
    ).fetchone()
    date_min = str(row["earliest"]) if row and row["earliest"] else "unknown"
    date_max = str(row["latest"]) if row and row["latest"] else "unknown"
    return date_min, date_max


def extract_source_filter(query: str) -> str | None:
    q = query.lower()
    for kw, label in SOURCE_KEYWORDS.items():
        if kw in q:
            return label
    return None


def is_market_data_intent(query: str) -> bool:
    q = query.lower()
    return any(hint in q for hint in MARKET_INTENT_HINTS)

CAUSAL_INTENT_HINTS = (
    "affect", "impact", "effect", "influence", "cause", "drive",
    "mean for", "implication", "what does", "how does", "why is",
    "because of", "result of", "due to", "respond to", "reaction",
    "stronger", "weaker", "rise", "fall", "rally", "selloff",
)

CAUSAL_USD_ANCHORS = (
    "dollar", "usd", "currency", "forex", "fx", "rate", "fed",
    "inflation", "oil", "gold", "macro", "economy", "trade",
    "tariff", "yield", "bond", "iran", "china", "russia", "war",
    "sanction", "geopolit",
)

def is_causal_analysis_intent(query: str) -> bool:
    q = query.lower()
    has_causal_verb = any(hint in q for hint in CAUSAL_INTENT_HINTS)
    has_usd_anchor = any(anchor in q for anchor in CAUSAL_USD_ANCHORS)
    return has_causal_verb and has_usd_anchor

CAUSAL_SYSTEM_PROMPT_TEMPLATE = (
    "You are a macro-financial analyst. "
    "Answer in 3-5 sentences maximum. "
    "State the directional impact on the USD, the single most important transmission mechanism, "
    "and your confidence level. "
    "Label claims from the retrieved news as [EVIDENCE] and claims from macro theory as [THEORY]. "
    "At the end, give me a concise summary based on the evidence and theory. "
    "Do not fabricate prices or statistics not present in the chunks. "
    "If the chunks are insufficient, say so in one sentence. "
    "Your database covers articles from {date_min} to {date_max}."
)

# ---------------------------------------------------------------------------
# Causal chain decomposition + multi-hop retrieval
# ---------------------------------------------------------------------------

CHAIN_DECOMPOSE_PROMPT = """\
You are a macro-financial analyst. Given a question about how an event affects the U.S. Dollar, \
identify the transmission chain as a list of entity canonical names that must be retrieved to \
answer it fully.

Rules:
- Return ONLY valid JSON, no markdown.
- Use only these canonical names (or close matches): iran, russia, china, saudi-arabia, \
israel, ukraine, middle-east, opec, wti-crude, brent-crude, natural-gas, gold, silver, \
copper, federal-reserve, fed-funds-rate, us-cpi, us-pce, us-gdp, us-jobs, us-10y-yield, \
us-2y-yield, yield-curve, usd, us-dollar-index, eur-usd, usd-jpy, sp500, vix, \
us-trade-balance, tariffs, petrodollar, us-debt, us-treasury, us-sanctions
- Include 2 to 4 hops. Always include the trigger entity and usd or us-dollar-index as the last hop.
- Order hops from trigger → intermediate → usd.

Output format:
{{"hops": ["entity1", "entity2", "entity3"]}}

Question: {query}"""


def decompose_causal_chain(query: str, gen_client: Any) -> list[str]:
    """
    Ask the LLM to decompose a causal query into ordered transmission hops.
    Returns a list of canonical entity names, e.g. ["iran", "wti-crude", "usd"].
    Falls back to [query_entity, "usd"] on any failure.
    """
    prompt = CHAIN_DECOMPOSE_PROMPT.format(query=query)
    try:
        out = gen_client.messages.create(
            model=GEN_MODEL_NAME,
            max_tokens=128,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = strip_think_tags(anthropic_text(out))
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            hops = data.get("hops", [])
            if isinstance(hops, list) and len(hops) >= 2:
                return [str(h).strip() for h in hops if h]
    except Exception as exc:
        print(f"  [causal chain] decomposition failed: {exc}")
    return []


def retrieve_causal_chain(
    query: str,
    hops: list[str],
    embed_model: SentenceTransformer,
    driver,
    sqlite_conn: sqlite3.Connection,
    alias_to_ticker: dict[str, str],
    ticker_to_canonical: dict[str, str],
    alias_to_fin_entity: dict[str, dict[str, Any]],
    source_filter: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    chunks_per_hop: int = 4,
) -> tuple[list[dict], QueryTarget]:
    """
    Run retrieve() for each hop entity and merge the results.
    Returns (all_chunks, hop_labels_per_chunk, primary_target).

    Each chunk gets an 'expansion_kind' label showing which hop it came from,
    so build_context() surfaces this in the context passed to the LLM.
    """
    all_chunks: list[dict] = []
    seen_uids: set[str] = set()
    primary_target: QueryTarget | None = None

    for hop in hops:
        try:
            # Reuse the full retrieve() stack — entity resolution + 3-layer retrieval
            hop_chunks, hop_target = retrieve(
                query=hop,          # retrieve by canonical entity name, not the full query
                embed_model=embed_model,
                driver=driver,
                sqlite_conn=sqlite_conn,
                alias_to_ticker=alias_to_ticker,
                ticker_to_canonical=ticker_to_canonical,
                alias_to_fin_entity=alias_to_fin_entity,
                top_k=chunks_per_hop,
                expanded_k=chunks_per_hop * 2,
                recency_half_life_days=RECENCY_HALF_LIFE_DAYS,
                source_filter=source_filter,
                date_start=date_start,
                date_end=date_end,
            )
        except Exception as e:
            print(f"  [causal hop: {hop} | retrieval failed: {e}")
            continue

        if primary_target is None and hop_target.canonical_name:
            primary_target = hop_target

        for ch in hop_chunks[:chunks_per_hop]:
            uid = ch.get("chunk_uid")
            if uid and uid not in seen_uids:
                seen_uids.add(uid)
                ch["expansion_kind"] = f"causal_hop:{hop}"
                all_chunks.append(ch)

        print(f"  [causal hop: {hop} | chunks: {len([c for c in hop_chunks[:chunks_per_hop] if c.get('chunk_uid') not in seen_uids - {c.get('chunk_uid')}])}]")

    # Fall back to a general semantic search on the original query
    # in case hop-entity retrieval returned nothing useful
    if not all_chunks:
        try:
            sem_chunks, sem_target = retrieve(
                query=query,
                embed_model=embed_model,
                driver=driver,
                sqlite_conn=sqlite_conn,
                alias_to_ticker=alias_to_ticker,
                ticker_to_canonical=ticker_to_canonical,
                alias_to_fin_entity=alias_to_fin_entity,
                source_filter=source_filter,
                date_start=date_start,
                date_end=date_end,
            )
            all_chunks = sem_chunks
            if primary_target is None:
                primary_target = sem_target
        except Exception as exc:
            print(f"  [causal fallback failed: {exc}]")

    # Safety net — always return a valid QueryTarget
    if primary_target is None:
        primary_target = QueryTarget(
            query_type=QUERY_TYPE_GENERAL,
            canonical_name=None,
            display_name="general",
            ticker=None,
            entity_type=None,
            confidence=0.0,
        )

    return all_chunks, primary_target

def fetch_market_context(
    hops: list[str],
    date_start: str | None,
    date_end: str | None,
    lookback_days: int = 7,
) -> str:
    """
    For any hop in the causal chain that maps to a known market symbol,
    fetch recent historical price data and format it as a context string
    for injection into the generation prompt.

    Returns an empty string if FMP_API_KEY is not set or financetoolkit
    is not installed.
    """
    api_key = os.getenv("FMP_API_KEY", "").strip()
    if not api_key:
        return ""

    try:
        from financetoolkit import Toolkit
    except ImportError:
        return ""

    # Resolve which hops have known symbols
    symbols = []
    symbol_to_hop = {}
    for hop in hops:
        symbol = CAUSAL_ENTITY_TICKER_MAP.get(hop)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
            symbol_to_hop[symbol] = hop

    if not symbols:
        return ""

    # Determine date range — use query dates if present, else last N days
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    if date_end:
        end_dt = date_end
    else:
        end_dt = now.strftime("%Y-%m-%d")
    if date_start:
        start_dt = date_start
    else:
        start_dt = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    lines = ["MARKET DATA (fetched via FMP):"]

    try:
        toolkit = Toolkit(symbols, api_key=api_key, start_date=start_dt, end_date=end_dt)
        hist = toolkit.get_historical_data()
        print(f"  [market debug] columns type: {type(hist.columns).__name__}, shape: {hist.shape}")
        print(f"  [market debug] top-level keys: {hist.columns.get_level_values(0).unique().tolist()[:5]}")

        # hist is a DataFrame with MultiIndex columns (symbol, OHLCV fields)
        # or a flat DataFrame if single symbol — handle both
        for symbol in symbols:
            hop = symbol_to_hop[symbol]
            try:
                # financetoolkit MultiIndex: top level = field, second level = symbol
                # e.g. hist["Close"]["CL=F"] or hist["Close"] if single symbol
                if hasattr(hist.columns, "levels"):
                    # MultiIndex — fields are top level, symbols are second level
                    top_level = hist.columns.get_level_values(0).unique().tolist()
                    if "Close" not in top_level:
                        lines.append(f"[{hop.upper()}] No Close data for {symbol}")
                        continue
                    close_col = hist["Close"]
                    if symbol not in close_col.columns:
                        lines.append(f"[{hop.upper()}] Symbol {symbol} not in results")
                        continue
                    df = hist.xs(symbol, axis=1, level=1)
                else:
                    # Flat DataFrame — single symbol case
                    df = hist

                if df is None or df.empty:
                    lines.append(f"[{hop.upper()}] No data returned for {symbol}")
                    continue

                lines.append(f"\n[{hop.upper()} | {symbol}]")
                close_series = df["Close"].tail(5).dropna()

                for i, (date_idx, close_val) in enumerate(close_series.items()):
                    date_str = str(date_idx)[:10]
                    close_fmt = f"{close_val:.2f}"

                    if i == 0:
                        ret_str = ""
                    else:
                        prev_val = close_series.iloc[i - 1]
                        ret = (close_val - prev_val) / prev_val * 100
                        ret_str = f"({ret:+.2f}%)"

                    lines.append(f"  {date_str}: {close_fmt} {ret_str}".strip())

            except Exception as e:
                lines.append(f"[{hop.upper()}] Parse error: {e}")

    except Exception as e:
        print(f"  [market data] fetch failed: {e}")
        return ""

    return "\n".join(lines)

def build_alpha_mcp_url(base_url: str, api_key: str) -> str:
    parsed = urlparse(base_url)
    query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_params["apikey"] = api_key
    return urlunparse(parsed._replace(query=urlencode(query_params)))


def _mcp_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""

    texts: list[str] = []
    if isinstance(content, list):
        blocks = content
    else:
        try:
            blocks = list(content)
        except TypeError:
            blocks = []

    for block in blocks:
        if isinstance(block, dict):
            if block.get("type") == "text" and block.get("text"):
                texts.append(str(block["text"]))
            continue
        text = getattr(block, "text", None)
        if text:
            texts.append(str(text))
    return "\n".join(texts).strip()


def inspect_mcp_response(response: Any) -> dict[str, Any]:
    """
    FIX 1: Only check for rate limits in tool_result blocks marked as errors,
    not in all result text (which may include API documentation).
    """
    tool_calls: list[str] = []
    has_result_block = False
    has_error = False
    rate_limited = False
 
    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "mcp_tool_use":
            name = getattr(block, "name", None)
            if name:
                tool_calls.append(str(name))
        elif block_type == "mcp_tool_result":
            has_result_block = True
            # FIX: Only check is_error flag first
            if bool(getattr(block, "is_error", False)):
                has_error = True
                # FIX: Only check rate limit text if there's an actual error
                result_text = _mcp_result_text(getattr(block, "content", None)).lower()
                if any(marker in result_text for marker in (
                    "rate limit",
                    "limit reached",
                    "too many requests",
                    "call frequency",
                    "quota exceeded",
                )):
                    rate_limited = True
 
    unique_tool_calls = list(dict.fromkeys(tool_calls))
    return {
        "tool_calls": unique_tool_calls,
        "has_result_block": has_result_block,
        "has_error": has_error,
        "rate_limited": rate_limited,
    }


def _has_market_tool_execution(tool_calls: list[str]) -> bool:
    """
    FIX 2: Accept TOOL_CALL as a valid market tool execution,
    since it's used for discovery and may precede actual data tools.
    """
    # TOOL_CALL is the discovery tool and is legitimate
    market_tools = {"TOOL_CALL", "GLOBAL_QUOTE", "TIME_SERIES_DAILY", "COMPANY_OVERVIEW"}
    return any(call in market_tools for call in tool_calls)


def _is_planning_only_text(text: str) -> bool:
    t = text.lower().strip()
    planning_markers = (
        "i'll retrieve",
        "i will retrieve",
        "now i'll",
        "now i will",
        "get the parameter schema",
        "i'm going to",
        "i am going to",
    )
    return any(marker in t for marker in planning_markers)


# ---------------------------------------------------------------------------
# Temporal Query Decomposition
# ---------------------------------------------------------------------------

TIME_WORDS = re.compile(
    r"\b(yesterday|last\s+week|this\s+week|last\s+month|this\s+month|"
    r"today|ago|since|until|between|"
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}|"
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{4}|"
    r"\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"\d{4}[-/]\d{2}|\d{1,2}/\d{1,2})\b",
    re.IGNORECASE,
)

MULTI_TIME_WORDS = re.compile(r"\b(compare|vs\.?|versus|differ)\b", re.IGNORECASE)
FROM_TO_PATTERN = re.compile(r"from\s+(.+?)\s+to\s+(.+?)[\?\.\!,]?\s*$", re.IGNORECASE)

DECOMPOSE_PROMPT = """\
Analyze this question and decompose it into separate sub-queries for each time period.

Rules:
- Today is {today}.
- Use ISO format YYYY-MM-DD for dates.
- Each sub-query should target one time period.

Output ONLY valid JSON (no markdown):
{{"sub_queries": [{{"query": "...", "time_start": "YYYY-MM-DD or null", "time_end": "YYYY-MM-DD or null"}}]}}

Question: {query}"""


def _resolve_time_phrase(phrase: str) -> tuple[str | None, str | None]:
    now = datetime.now(timezone.utc)
    p = phrase.strip().lower()

    if p == "yesterday":
        dt = dateparser.parse("yesterday", settings={"RETURN_AS_TIMEZONE_AWARE": True})
        d = dt.strftime("%Y-%m-%d") if dt else None
        return d, d
    if "last week" in p:
        s = dateparser.parse("7 days ago", settings={"RETURN_AS_TIMEZONE_AWARE": True})
        e = dateparser.parse("1 day ago", settings={"RETURN_AS_TIMEZONE_AWARE": True})
        return (s.strftime("%Y-%m-%d") if s else None, e.strftime("%Y-%m-%d") if e else None)
    if "this week" in p:
        days_back = now.weekday()
        s = dateparser.parse(f"{days_back} days ago", settings={"RETURN_AS_TIMEZONE_AWARE": True})
        return s.strftime("%Y-%m-%d") if s else None, now.strftime("%Y-%m-%d")
    if "last month" in p:
        s = dateparser.parse("1 month ago", settings={"RETURN_AS_TIMEZONE_AWARE": True})
        return s.strftime("%Y-%m-%d") if s else None, now.strftime("%Y-%m-%d")
    if "this month" in p:
        return now.strftime("%Y-%m-01"), now.strftime("%Y-%m-%d")
    if p == "today":
        d = now.strftime("%Y-%m-%d")
        return d, d

    dt = dateparser.parse(p, settings={"RETURN_AS_TIMEZONE_AWARE": True})
    if dt:
        d = dt.strftime("%Y-%m-%d")
        return d, d
    return None, None


def _extract_single_time_range(query: str) -> dict:
    matches = TIME_WORDS.findall(query)
    if not matches:
        return {"query": query, "time_start": None, "time_end": None}
    best = max(matches, key=lambda m: (1 if re.search(r"\d{4}", m) else 0, len(m)))
    start, end = _resolve_time_phrase(best)
    return {"query": query, "time_start": start, "time_end": end}


def decompose_query(query: str, gen_client: Any) -> list[dict]:
    if not TIME_WORDS.search(query):
        return [{"query": query, "time_start": None, "time_end": None}]

    ft = FROM_TO_PATTERN.search(query)
    if ft:
        a_start, a_end = _resolve_time_phrase(ft.group(1).strip())
        b_start, b_end = _resolve_time_phrase(ft.group(2).strip())
        if a_start or b_start:
            return [
                {"query": query, "time_start": a_start, "time_end": a_end},
                {"query": query, "time_start": b_start, "time_end": b_end},
            ]

    if re.search(r"\blast\s+week\b", query, re.IGNORECASE) and \
       re.search(r"\bthis\s+week\b", query, re.IGNORECASE):
        ls, le = _resolve_time_phrase("last week")
        ts, te = _resolve_time_phrase("this week")
        if ls or ts:
            return [
                {"query": query, "time_start": ls, "time_end": le},
                {"query": query, "time_start": ts, "time_end": te},
            ]

    if re.search(r"\blast\s+month\b", query, re.IGNORECASE) and \
       re.search(r"\bthis\s+month\b", query, re.IGNORECASE):
        ls, le = _resolve_time_phrase("last month")
        ts, te = _resolve_time_phrase("this month")
        if ls or ts:
            return [
                {"query": query, "time_start": ls, "time_end": le},
                {"query": query, "time_start": ts, "time_end": te},
            ]

    if not MULTI_TIME_WORDS.search(query):
        return [_extract_single_time_range(query)]

    # True multi-period — fall back to LLM decomposition
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = DECOMPOSE_PROMPT.format(today=today, query=query)
    out = gen_client.messages.create(
        model=GEN_MODEL_NAME,
        max_tokens=GEN_MAX_TOKENS,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = strip_think_tags(anthropic_text(out))
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            valid = [
                {"query": sq["query"], "time_start": sq.get("time_start"), "time_end": sq.get("time_end")}
                for sq in data.get("sub_queries", [])
                if isinstance(sq, dict) and "query" in sq
            ]
            if valid:
                return valid
        except (json.JSONDecodeError, TypeError):
            pass

    return [_extract_single_time_range(query)]


def _date_to_period_keys(date_str: str | None) -> list[str]:
    """Convert a date string to one or more period keys for graph filtering."""
    if not date_str:
        return []
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return [period_key_for(dt, PERIOD_GRANULARITY)]
    except ValueError:
        return []


def _date_range_to_period_keys(start: str | None, end: str | None) -> list[str]:
    """Expand a date range into all period keys it spans."""
    if not start and not end:
        return []
    keys = set()
    try:
        s = datetime.strptime(start or end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        e = datetime.strptime(end or start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        current = s
        while current <= e:
            keys.add(period_key_for(current, PERIOD_GRANULARITY))
            # Advance by roughly one period
            if PERIOD_GRANULARITY == "week":
                current += timedelta(weeks=1)
            elif PERIOD_GRANULARITY == "month":
                # Move to next month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
            else:  # quarter
                current += timedelta(days=92)
    except ValueError:
        pass
    return sorted(keys)


# ---------------------------------------------------------------------------
# Entity Resolution
# ---------------------------------------------------------------------------

COMPARE_WORDS = re.compile(r"\b(compare|vs\.?|versus|against)\b", re.IGNORECASE)

QUERY_TYPE_SINGLE = "single_entity"
QUERY_TYPE_GENERAL = "general"


@dataclass
class QueryTarget:
    query_type: str
    canonical_name: Optional[str]   # stable entity id from SQLite / Neo4j
    display_name: Optional[str]     # human-readable, for context header
    ticker: Optional[str]           # non-None only for ORG entities
    entity_type: Optional[str]      # ORG / PER / LOC
    confidence: float
    candidates: list[tuple[str, str]] = field(default_factory=list)  # (canonical, display)


def _normalize_for_matching(query: str) -> str:
    """Strip query boilerplate before entity matching."""
    q = _canonicalize(query)
    stop_phrases = [
        "what is the latest news on", "what is the latest news about",
        "latest news on", "latest news about", "news on", "news about",
        "what happened to", "tell me about", "latest on",
        "what do you know about", "give me an update on",
    ]
    for phrase in stop_phrases:
        q = q.replace(_canonicalize(phrase), " ")
    return re.sub(r"\s+", " ", q).strip()


def _lookup_entity_in_sqlite(conn: sqlite3.Connection, q_norm: str) -> dict | None:
    rows = conn.execute(
        """
        SELECT
            canonical_entity_id,
            display_name,
            entity_type,
            ticker,
            MAX(COALESCE(confidence, 0)) AS confidence
        FROM entity_mentions
        WHERE canonical_entity_id = ?
           OR lower(display_name) = lower(?)
        GROUP BY canonical_entity_id, display_name, entity_type, ticker
        ORDER BY confidence DESC, canonical_entity_id
        LIMIT 1
        """,
        (q_norm, q_norm),
    ).fetchall()
    return dict(rows[0]) if rows else None


def _resolve_asset_target(
    conn: sqlite3.Connection,
    query: str,
    target: QueryTarget | None = None,
) -> dict | None:
    if target and target.ticker:
        return {
            "asset_key": asset_key("ticker", target.ticker),
            "target_type": "ticker",
            "target_id": target.ticker,
            "display_name": target.ticker,
        }

    q_norm = _normalize_for_matching(query)
    exact = conn.execute(
        """
        SELECT DISTINCT target_type, target_id
        FROM asset_impacts
        WHERE lower(target_id) = lower(?)
        LIMIT 1
        """,
        (q_norm,),
    ).fetchone()
    if exact:
        return {
            "asset_key": asset_key(exact["target_type"], exact["target_id"]),
            "target_type": exact["target_type"],
            "target_id": exact["target_id"],
            "display_name": exact["target_id"],
        }

    rows = conn.execute(
        """
        SELECT DISTINCT target_type, target_id
        FROM asset_impacts
        WHERE target_id IS NOT NULL
          AND target_id <> ''
        """
    ).fetchall()
    phrase_hits: list[tuple[str, sqlite3.Row]] = []
    for row in rows:
        target_norm = _canonicalize(str(row["target_id"]))
        if target_norm and re.search(r"\b" + re.escape(target_norm) + r"\b", q_norm):
            phrase_hits.append((target_norm, row))

    if not phrase_hits:
        return None

    phrase_hits.sort(key=lambda item: len(item[0]), reverse=True)
    match = phrase_hits[0][1]
    return {
        "asset_key": asset_key(match["target_type"], match["target_id"]),
        "target_type": match["target_type"],
        "target_id": match["target_id"],
        "display_name": match["target_id"],
    }


def resolve_query_target(
    query: str,
    alias_to_ticker: dict[str, str],
    ticker_to_canonical: dict[str, str],
    driver,
    sqlite_conn: sqlite3.Connection | None = None,
    alias_to_fin_entity: dict[str, dict] | None = None,  # ADD
) -> QueryTarget:
    """
    Resolve the primary entity in a query to a QueryTarget.

    Priority:
      Tier 1 — explicit uppercase ticker token in query (e.g. "AAPL", "NVDA")
      Tier 2 — exact alias match in alias_to_ticker
      Tier 3 — longest alias phrase found inside normalized query
      Tier 4 — SQLite entity_mentions canonical_entity_id / display_name match
      Tier 5 — general (no entity resolved)
    """
    q_norm = _normalize_for_matching(query)

    # Tier 1: explicit ticker tokens (1-5 uppercase letters)
    ticker_candidates = re.findall(r"\b([A-Z]{1,5})\b", query)
    for tok in ticker_candidates:
        if tok in ticker_to_canonical:
            display = ticker_to_canonical[tok]
            return QueryTarget(
                query_type=QUERY_TYPE_SINGLE,
                canonical_name=tok,        # canonical_name IS the ticker for ORG
                display_name=display,
                ticker=tok,
                entity_type="ORG",
                confidence=0.99,
                candidates=[(tok, display)],
            )

    # Tier 2: exact alias match
    if q_norm in alias_to_ticker:
        ticker = alias_to_ticker[q_norm]
        display = ticker_to_canonical.get(ticker, q_norm)
        return QueryTarget(
            query_type=QUERY_TYPE_SINGLE,
            canonical_name=ticker,
            display_name=display,
            ticker=ticker,
            entity_type="ORG",
            confidence=0.98,
            candidates=[(ticker, display)],
        )
    # Tier 2b: financial entity map — catches Fed, ECB, FOMC, indices, etc.
    if alias_to_fin_entity:
        from tgrag_setup import link_financial_entity
        linked = link_financial_entity(q_norm, "ORG", alias_to_fin_entity)
        if linked:
            return QueryTarget(
                query_type=QUERY_TYPE_SINGLE,
                canonical_name=linked["canonical_name"],
                display_name=linked["display_name"],
                ticker=linked.get("ticker"),
                entity_type=linked["entity_type"],
                confidence=0.97,
                candidates=[(linked["canonical_name"], linked["display_name"])],
            )

    # Tier 3: longest alias phrase found inside normalized query
    phrase_hits: list[tuple[str, str, str]] = []  # (alias, ticker, display)
    for alias, ticker in alias_to_ticker.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", q_norm):
            display = ticker_to_canonical.get(ticker, alias)
            phrase_hits.append((alias, ticker, display))

    if phrase_hits:
        # Longest alias wins (most specific)
        phrase_hits.sort(key=lambda x: len(x[0]), reverse=True)
        alias, ticker, display = phrase_hits[0]
        return QueryTarget(
            query_type=QUERY_TYPE_SINGLE,
            canonical_name=ticker,
            display_name=display,
            ticker=ticker,
            entity_type="ORG",
            confidence=0.95,
            candidates=[(ticker, display)],
        )

    if alias_to_fin_entity:
        fin_phrase_hits: list[tuple[str, dict[str, Any]]] = []
        for alias, entity in alias_to_fin_entity.items():
            if re.search(r"\b" + re.escape(alias) + r"\b", q_norm):
                fin_phrase_hits.append((alias, entity))

        if fin_phrase_hits:
            fin_phrase_hits.sort(key=lambda x: len(x[0]), reverse=True)
            _, entity = fin_phrase_hits[0]
            canonical_name = str(entity["canonical_name"])
            display_name = str(entity.get("display_name") or canonical_name)
            return QueryTarget(
                query_type=QUERY_TYPE_SINGLE,
                canonical_name=canonical_name,
                display_name=display_name,
                ticker=entity.get("ticker"),
                entity_type=entity.get("entity_type") or "ORG",
                confidence=0.94,
                candidates=[(canonical_name, display_name)],
            )

    # Tier 4: SQLite entity_mentions for PER/LOC/ORG fallback
    if sqlite_conn is not None:
        r = _lookup_entity_in_sqlite(sqlite_conn, q_norm)
    else:
        r = None

    if r:
        return QueryTarget(
            query_type=QUERY_TYPE_SINGLE,
            canonical_name=r["canonical_entity_id"],
            display_name=r.get("display_name") or r["canonical_entity_id"],
            ticker=r.get("ticker"),
            entity_type=r.get("entity_type"),
            confidence=max(0.75, float(r.get("confidence") or 0.0)),
            candidates=[(r["canonical_entity_id"], r.get("display_name") or r["canonical_entity_id"])],
        )

    # Tier 5: no entity resolved
    return QueryTarget(
        query_type=QUERY_TYPE_GENERAL,
        canonical_name=None,
        display_name=None,
        ticker=None,
        entity_type=None,
        confidence=0.0,
    )


def retrieve_entity_chunks(
    sqlite_conn: sqlite3.Connection,
    target: QueryTarget,
    period_keys: list[str] | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    source_filter: str | None = None,
    top_k: int = 80,
) -> list[dict]:
    """Direct SQLite entity-mention retrieval for broad QA coverage."""
    if not target.canonical_name:
        return []

    sql = """
        SELECT
            c.chunk_id,
            c.article_id,
            c.text,
            c.published_date,
            c.period_key,
            c.embedding_json,
            a.title,
            a.url,
            a.source,
            em.display_name AS entity_display,
            em.confidence AS mention_confidence
        FROM entity_mentions em
        JOIN chunks c ON c.chunk_id = em.chunk_id
        JOIN articles a ON a.article_id = c.article_id
        WHERE em.canonical_entity_id = ?
    """
    params: list[Any] = [target.canonical_name]
    if period_keys:
        placeholders = ",".join("?" for _ in period_keys)
        sql += f" AND c.period_key IN ({placeholders})"
        params.extend(period_keys)
    if date_start:
        sql += " AND c.published_date >= ?"
        params.append(date_start)
    if date_end:
        sql += " AND c.published_date <= ?"
        params.append(date_end)
    if source_filter:
        sql += " AND a.source = ?"
        params.append(source_filter)
    sql += " ORDER BY c.published_date DESC, c.chunk_index ASC LIMIT ?"
    params.append(max(top_k, 1) * 10)

    rows = sqlite_conn.execute(sql, params).fetchall()
    out: list[dict] = []
    for row in rows:
        item = _sqlite_row_to_chunk_dict(row)
        item["retrieval_kind"] = "entity_mentions"
        item["entity_display"] = row["entity_display"]
        item["mention_confidence"] = row["mention_confidence"]
        out.append(item)
    return out


def retrieve_graph_event_chunks(
    driver,
    sqlite_conn: sqlite3.Connection,
    target: QueryTarget,
    period_keys: list[str] | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    source_filter: str | None = None,
    top_k: int = 24,
) -> list[dict]:
    """Fetch macro events linked to the resolved entity, then materialize SQLite evidence."""
    if not target.canonical_name:
        return []

    period_filter = "AND m.period_key IN $period_keys" if period_keys else ""
    cypher = f"""
    MATCH (e:Entity {{entity_id: $entity_id}})<-[:INVOLVES]-(m:MacroEvent)
    WHERE 1=1
      {period_filter}
    RETURN
        m.macro_event_id AS macro_event_id,
        m.chunk_id       AS chunk_id,
        m.article_id     AS article_id,
        m.summary        AS macro_summary,
        m.event_type     AS event_type,
        m.region         AS region,
        m.time_horizon   AS time_horizon,
        m.confidence     AS macro_confidence,
        m.evidence_text  AS evidence_text
    ORDER BY coalesce(m.confidence, 0.0) DESC
    LIMIT $top_k
    """
    params: dict[str, Any] = {"entity_id": target.canonical_name, "top_k": top_k}
    if period_keys:
        params["period_keys"] = period_keys

    with driver.session() as session:
        rows = session.run(cypher, params).data()

    chunk_lookup = _fetch_chunk_rows_by_ids(
        sqlite_conn,
        [str(row["chunk_id"]) for row in rows if row.get("chunk_id")],
        date_start=date_start,
        date_end=date_end,
        source_filter=source_filter,
    )
    out: list[dict] = []
    for row in rows:
        chunk_id = row.get("chunk_id")
        base = chunk_lookup.get(chunk_id)
        if not base:
            continue
        out.append(
            {
                **base,
                "retrieval_kind": "macro_event_entity",
                "macro_event_id": row["macro_event_id"],
                "macro_summary": row["macro_summary"],
                "event_type": row["event_type"],
                "region": row["region"],
                "time_horizon": row["time_horizon"],
                "macro_confidence": row["macro_confidence"],
                "evidence_text": row["evidence_text"],
                "semantic_score": float(row.get("macro_confidence") or 0.55),
            }
        )
    return out


def retrieve_asset_chunks(
    driver,
    sqlite_conn: sqlite3.Connection,
    asset_target: dict | None,
    period_keys: list[str] | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    source_filter: str | None = None,
    top_k: int = 20,
) -> list[dict]:
    """Fetch macro events that impact an asset, then materialize SQLite evidence."""
    if not asset_target:
        return []

    period_filter = "AND m.period_key IN $period_keys" if period_keys else ""
    cypher = f"""
    MATCH (a:Asset {{asset_key: $asset_key}})<-[r:IMPACTS]-(m:MacroEvent)
    WHERE 1=1
      {period_filter}
    RETURN
        m.macro_event_id AS macro_event_id,
        m.chunk_id       AS chunk_id,
        m.summary        AS macro_summary,
        m.event_type     AS event_type,
        m.confidence     AS macro_confidence,
        r.direction      AS impact_direction,
        r.strength       AS impact_strength,
        r.horizon        AS impact_horizon,
        r.rationale      AS impact_rationale
    ORDER BY coalesce(m.confidence, 0.0) DESC
    LIMIT $top_k
    """
    params: dict[str, Any] = {"asset_key": asset_target["asset_key"], "top_k": top_k}
    if period_keys:
        params["period_keys"] = period_keys

    with driver.session() as session:
        rows = session.run(cypher, params).data()

    chunk_lookup = _fetch_chunk_rows_by_ids(
        sqlite_conn,
        [str(row["chunk_id"]) for row in rows if row.get("chunk_id")],
        date_start=date_start,
        date_end=date_end,
        source_filter=source_filter,
    )
    out: list[dict] = []
    for row in rows:
        chunk_id = row.get("chunk_id")
        base = chunk_lookup.get(chunk_id)
        if not base:
            continue
        out.append(
            {
                **base,
                "retrieval_kind": "macro_event_asset",
                "macro_event_id": row["macro_event_id"],
                "macro_summary": row["macro_summary"],
                "event_type": row["event_type"],
                "macro_confidence": row["macro_confidence"],
                "asset_target_id": asset_target["target_id"],
                "asset_target_type": asset_target["target_type"],
                "impact_direction": row["impact_direction"],
                "impact_strength": row["impact_strength"],
                "impact_horizon": row["impact_horizon"],
                "impact_rationale": row["impact_rationale"],
                "semantic_score": float(row.get("macro_confidence") or 0.55),
            }
        )
    return out


def retrieve_macro_semantic_chunks(
    driver,
    sqlite_conn: sqlite3.Connection,
    embed_model: SentenceTransformer,
    query: str,
    period_keys: list[str] | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    source_filter: str | None = None,
    top_k: int = 12,
) -> list[dict]:
    """Semantic search over MacroEvent summaries, then fetch SQLite evidence."""
    period_filter = "WHERE m.period_key IN $period_keys" if period_keys else ""
    cypher = f"""
    MATCH (m:MacroEvent)
    {period_filter}
    RETURN
        m.macro_event_id AS macro_event_id,
        m.chunk_id       AS chunk_id,
        m.summary        AS macro_summary,
        m.event_type     AS event_type,
        m.region         AS region,
        m.time_horizon   AS time_horizon,
        m.confidence     AS macro_confidence
    LIMIT $limit
    """
    params: dict[str, Any] = {"limit": MACRO_EVENT_CANDIDATE_LIMIT}
    if period_keys:
        params["period_keys"] = period_keys

    with driver.session() as session:
        rows = session.run(cypher, params).data()

    if not rows:
        return []

    summaries = [str(row.get("macro_summary") or "") for row in rows]
    qvec = embed_model.encode([query], normalize_embeddings=True)[0]
    svecs = embed_model.encode(summaries, normalize_embeddings=True)

    scored: list[tuple[float, dict]] = []
    for row, svec in zip(rows, svecs):
        score = cosine_sim(qvec, np.array(svec, dtype=np.float32))
        scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    top_rows = [row for score, row in scored[:top_k] if score > 0.20]

    chunk_lookup = _fetch_chunk_rows_by_ids(
        sqlite_conn,
        [str(row["chunk_id"]) for row in top_rows if row.get("chunk_id")],
        date_start=date_start,
        date_end=date_end,
        source_filter=source_filter,
    )

    out: list[dict] = []
    for score, row in scored[:top_k]:
        chunk_id = row.get("chunk_id")
        base = chunk_lookup.get(chunk_id)
        if not base:
            continue
        out.append(
            {
                **base,
                "retrieval_kind": "macro_event_semantic",
                "macro_event_id": row["macro_event_id"],
                "macro_summary": row["macro_summary"],
                "event_type": row["event_type"],
                "region": row["region"],
                "time_horizon": row["time_horizon"],
                "macro_confidence": row["macro_confidence"],
                "semantic_score": score,
            }
        )
    return out


def retrieve_cooccurrence_chunks(
    sqlite_conn: sqlite3.Connection,
    target: QueryTarget,
    period_keys: list[str] | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    source_filter: str | None = None,
    max_neighbors: int = 3,
    max_chunks: int = 4,
) -> list[dict]:
    """SQLite substitute for co-occurrence expansion using shared chunks."""
    if not target.canonical_name:
        return []

    neighbor_rows = sqlite_conn.execute(
        """
        SELECT
            em2.canonical_entity_id AS neighbor_id,
            MAX(NULLIF(em2.display_name, '')) AS neighbor_display,
            COUNT(*) AS cooccur_count
        FROM entity_mentions em1
        JOIN entity_mentions em2
          ON em1.chunk_id = em2.chunk_id
        WHERE em1.canonical_entity_id = ?
          AND em2.canonical_entity_id <> em1.canonical_entity_id
        GROUP BY em2.canonical_entity_id
        ORDER BY cooccur_count DESC, neighbor_id
        LIMIT ?
        """,
        (target.canonical_name, max_neighbors),
    ).fetchall()
    if not neighbor_rows:
        return []

    neighbor_ids = [row["neighbor_id"] for row in neighbor_rows]
    placeholders = ",".join("?" for _ in neighbor_ids)
    sql = f"""
        SELECT
            c.chunk_id,
            c.article_id,
            c.text,
            c.published_date,
            c.period_key,
            c.embedding_json,
            a.title,
            a.url,
            a.source,
            em.canonical_entity_id AS neighbor_id,
            em.display_name AS neighbor_display
        FROM entity_mentions em
        JOIN chunks c ON c.chunk_id = em.chunk_id
        JOIN articles a ON a.article_id = c.article_id
        WHERE em.canonical_entity_id IN ({placeholders})
    """
    params: list[Any] = list(neighbor_ids)
    if period_keys:
        period_placeholders = ",".join("?" for _ in period_keys)
        sql += f" AND c.period_key IN ({period_placeholders})"
        params.extend(period_keys)
    if date_start:
        sql += " AND c.published_date >= ?"
        params.append(date_start)
    if date_end:
        sql += " AND c.published_date <= ?"
        params.append(date_end)
    if source_filter:
        sql += " AND a.source = ?"
        params.append(source_filter)
    sql += " ORDER BY c.published_date DESC LIMIT ?"
    params.append(max_chunks)

    rows = sqlite_conn.execute(sql, params).fetchall()
    out: list[dict] = []
    for row in rows:
        item = _sqlite_row_to_chunk_dict(row)
        item["expansion_kind"] = "cooccurrence"
        item["neighbor_display"] = row["neighbor_display"] or row["neighbor_id"]
        out.append(item)
    return out


def retrieve_semantic_chunks(
    sqlite_conn: sqlite3.Connection,
    embed_model: SentenceTransformer,
    query: str,
    period_keys: list[str] | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    source_filter: str | None = None,
    top_k: int = 6,
) -> list[dict]:
    """SQLite semantic fallback over chunk embeddings."""
    sql = """
        SELECT
            c.chunk_id,
            c.article_id,
            c.text,
            c.published_date,
            c.period_key,
            c.embedding_json,
            a.title,
            a.url,
            a.source
        FROM chunks c
        JOIN articles a ON a.article_id = c.article_id
        WHERE c.embedding_json IS NOT NULL
    """
    params: list[Any] = []
    if period_keys:
        placeholders = ",".join("?" for _ in period_keys)
        sql += f" AND c.period_key IN ({placeholders})"
        params.extend(period_keys)
    if date_start:
        sql += " AND c.published_date >= ?"
        params.append(date_start)
    if date_end:
        sql += " AND c.published_date <= ?"
        params.append(date_end)
    if source_filter:
        sql += " AND a.source = ?"
        params.append(source_filter)
    sql += " ORDER BY c.published_date DESC LIMIT ?"
    params.append(max(SQLITE_SEMANTIC_CANDIDATE_LIMIT, top_k))

    rows = sqlite_conn.execute(sql, params).fetchall()
    if not rows:
        return []

    qvec = embed_model.encode([query], normalize_embeddings=True)[0]
    scored: list[dict] = []
    for row in rows:
        item = _sqlite_row_to_chunk_dict(row)
        emb = item.get("embedding")
        if emb is None:
            continue
        item["semantic_score"] = cosine_sim(qvec, np.array(emb, dtype=np.float32))
        item["retrieval_kind"] = "sqlite_semantic"
        scored.append(item)

    scored.sort(key=lambda item: item["semantic_score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Scoring and ranking
# ---------------------------------------------------------------------------

def _published_date_to_ts(published_date) -> int | None:
    """Convert Neo4j Date object or ISO string to unix timestamp."""
    if published_date is None:
        return None
    try:
        if hasattr(published_date, "year"):
            # Neo4j Date object
            dt = datetime(published_date.year, published_date.month, published_date.day,
                          tzinfo=timezone.utc)
        else:
            dt = datetime.strptime(str(published_date), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def score_and_rank(
    rows: list[dict],
    query_vec: np.ndarray,
    recency_half_life_days: float = RECENCY_HALF_LIFE_DAYS,
    semantic_weight: float = 0.75,
    recency_weight: float = 0.25,
) -> list[dict]:
    """
    Score each chunk: weighted combination of cosine similarity and recency decay.
    Deduplicates by chunk_uid, keeping highest score.
    """
    now_ts = int(datetime.now(timezone.utc).timestamp())
    half_life_s = recency_half_life_days * 86400.0

    seen: dict[str, dict] = {}
    for r in rows:
        uid = r.get("chunk_uid")
        if not uid:
            continue

        emb = r.get("embedding")
        if emb is None:
            sim = float(r.get("semantic_score") or r.get("knowledge_score") or 0.5)
        else:
            sim = cosine_sim(query_vec, np.array(emb, dtype=np.float32))

        ts = _published_date_to_ts(r.get("published_date"))
        if ts is None:
            decay = 0.8
        else:
            age = max(0, now_ts - ts)
            decay = float(np.exp(-np.log(2) * age / half_life_s))

        score = semantic_weight * sim + recency_weight * decay
        r = {**r, "score": score}

        if uid not in seen or score > seen[uid]["score"]:
            seen[uid] = r

    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)


# ---------------------------------------------------------------------------
# Main retrieval orchestrator
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    embed_model: SentenceTransformer,
    driver,
    sqlite_conn: sqlite3.Connection,
    alias_to_ticker: dict[str, str],
    ticker_to_canonical: dict[str, str],
    alias_to_fin_entity: dict[str, dict[str, Any]],
    top_k: int = TOP_K,
    expanded_k: int = EXPANDED_K,
    recency_half_life_days: float = RECENCY_HALF_LIFE_DAYS,
    source_filter: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> tuple[list[dict], QueryTarget]:
    """
    Hybrid retrieval for one sub-query.
    Returns (ranked_chunks, target).
    """
    target = resolve_query_target(
        query,
        alias_to_ticker,
        ticker_to_canonical,
        driver,
        sqlite_conn=sqlite_conn,
        alias_to_fin_entity=alias_to_fin_entity,
    )
    period_keys = _date_range_to_period_keys(date_start, date_end) or None

    query_vec = embed_model.encode([query], normalize_embeddings=True)[0]

    if target.canonical_name:
        entity_rows = retrieve_entity_chunks(
            sqlite_conn,
            target,
            period_keys=period_keys,
            date_start=date_start,
            date_end=date_end,
            source_filter=source_filter,
        )
        macro_rows = retrieve_graph_event_chunks(
            driver,
            sqlite_conn,
            target,
            period_keys=period_keys,
            date_start=date_start,
            date_end=date_end,
            source_filter=source_filter,
        )
        asset_rows = retrieve_asset_chunks(
            driver,
            sqlite_conn,
            _resolve_asset_target(sqlite_conn, query, target),
            period_keys=period_keys,
            date_start=date_start,
            date_end=date_end,
            source_filter=source_filter,
            top_k=max(4, expanded_k),
        )
        all_rows = entity_rows + macro_rows + asset_rows
        ranked = score_and_rank(all_rows, query_vec, recency_half_life_days)

        if len(ranked) < top_k:
            cooccur_rows = retrieve_cooccurrence_chunks(
                sqlite_conn,
                target,
                period_keys=period_keys,
                date_start=date_start,
                date_end=date_end,
                source_filter=source_filter,
                max_neighbors=3,
                max_chunks=4,
            )
            all_rows += cooccur_rows
            ranked = score_and_rank(all_rows, query_vec, recency_half_life_days)

        if len(ranked) < top_k:
            semantic_rows = retrieve_semantic_chunks(
                sqlite_conn,
                embed_model,
                query,
                period_keys=period_keys,
                date_start=date_start,
                date_end=date_end,
                source_filter=source_filter,
                top_k=expanded_k,
            )
            all_rows += semantic_rows
            ranked = score_and_rank(all_rows, query_vec, recency_half_life_days)

    else:
        asset_target = _resolve_asset_target(sqlite_conn, query)
        asset_rows = retrieve_asset_chunks(
            driver,
            sqlite_conn,
            asset_target,
            period_keys=period_keys,
            date_start=date_start,
            date_end=date_end,
            source_filter=source_filter,
            top_k=max(4, expanded_k),
        )
        macro_rows = retrieve_macro_semantic_chunks(
            driver,
            sqlite_conn,
            embed_model,
            query,
            period_keys=period_keys,
            date_start=date_start,
            date_end=date_end,
            source_filter=source_filter,
            top_k=max(4, expanded_k),
        )
        sem_rows = retrieve_semantic_chunks(
            sqlite_conn,
            embed_model,
            query,
            period_keys=period_keys,
            date_start=date_start,
            date_end=date_end,
            source_filter=source_filter,
            top_k=expanded_k,
        )
        ranked = score_and_rank(asset_rows + macro_rows + sem_rows, query_vec, recency_half_life_days)

    return ranked[:expanded_k], target


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(
    query: str,
    target: QueryTarget,
    chunks: list[dict],
) -> str:
    lines = []
    lines.append(f"TARGET ENTITY : {target.display_name or 'unknown'}")
    lines.append(f"TARGET CANONICAL: {target.canonical_name or 'unknown'}")
    lines.append(f"TARGET TICKER : {target.ticker or 'N/A'}")
    lines.append(f"ENTITY TYPE   : {target.entity_type or 'unknown'}")
    lines.append(f"CONFIDENCE    : {target.confidence:.2f}")
    lines.append("")

    if not chunks:
        lines.append("No relevant chunks found.")
        return "\n".join(lines)

    lines.append("NEWS EVIDENCE:")
    for ch in chunks:
        lines.append(f"[SOURCE: {ch.get('source', '?')}]")
        lines.append(f"[TITLE: {ch.get('title', '?')}]")
        lines.append(f"[URL: {ch.get('url', '?')}]")
        lines.append(f"[PERIOD: {ch.get('period_key', '?')}]")
        if ch.get("retrieval_kind"):
            lines.append(f"[RETRIEVAL: {ch['retrieval_kind']}]")
        if ch.get("expansion_kind"):
            lines.append(f"[EXPANSION: {ch['expansion_kind']}]")
        if ch.get("macro_summary"):
            lines.append(f"[MACRO SUMMARY: {ch['macro_summary']}]")
        if ch.get("event_type"):
            lines.append(f"[EVENT TYPE: {ch['event_type']}]")
        if ch.get("asset_target_id"):
            lines.append(f"[ASSET TARGET: {ch['asset_target_type']}::{ch['asset_target_id']}]")
        if ch.get("impact_direction") or ch.get("impact_strength"):
            lines.append(
                f"[IMPACT: direction={ch.get('impact_direction', '?')}, "
                f"strength={ch.get('impact_strength', '?')}, horizon={ch.get('impact_horizon', '?')}]"
            )
        lines.append(ch.get("text", ""))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = (
    "You are a financial news assistant. "
    "Answer ONLY using the exact facts stated in the provided news chunks below. "
    "Do NOT add any information not explicitly present in the chunks. "
    "Do NOT use your own knowledge about companies, valuations, or events. "
    "If the chunks do not contain enough information to answer, say: "
    "'The retrieved articles do not contain sufficient information to answer this.' "
    "Your database covers articles from {date_min} to {date_max}. "
    "Do not output <think> tags or chain-of-thought reasoning."
)


def build_system_prompt(base: str, memory: "ConversationMemory") -> str:
    """
    Inject conversation history into the system prompt when memory is non-empty.
    The history block is appended after the base instructions so it never
    overrides the grounding constraints.
    """
    ctx = memory.context_for_prompt()
    if not ctx:
        return base
    return base + f"\n\n{ctx}"


def generate_answer(query: str, context: str, gen_client: Any, system_prompt: str) -> str:
    out = gen_client.messages.create(
        model=GEN_MODEL_NAME,
        max_tokens=GEN_MAX_TOKENS,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            }
        ],
    )
    raw = anthropic_text(out)
    cleaned = strip_think_tags(raw).strip()
    return cleaned or "I could not generate a grounded answer from the retrieved context."


def generate_answer_with_remote_mcp(
    query: str,
    context: str,
    ticker: str,
    gen_client: Any,
    system_prompt: str,
) -> tuple[str | None, dict[str, Any]]:
    """
    FIX 3: Remove timeout parameter and relax validation logic.
    """
    if not ENABLE_ALPHA_MCP_REMOTE:
        return None, {"failed": True, "reason": "remote mcp disabled"}
    if not ALPHA_VANTAGE_API_KEY:
        return None, {"failed": True, "reason": "alpha vantage api key missing"}
 
    mcp_url = build_alpha_mcp_url(ALPHA_VANTAGE_MCP_URL, ALPHA_VANTAGE_API_KEY)
    mcp_system_prompt = (
        system_prompt
        + "\n\nFor this turn, you may use Alpha Vantage MCP tool results as grounded evidence in addition to "
          "the retrieved news context. Do not use outside knowledge."
    )
 
    toolset_config = {
        "type": "mcp_toolset",
        "mcp_server_name": ALPHA_MCP_SERVER_NAME,
        "default_config": {"enabled": False},
        "configs": {tool_name: {"enabled": True} for tool_name in ALPHA_MCP_ALLOWED_TOOLS},
    }
 
    mcp_servers = [
        {
            "type": "url",
            "name": ALPHA_MCP_SERVER_NAME,
            "url": mcp_url,
        }
    ]
 
    # FIX: More direct prompt that doesn't force specific tool order
    primary_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Resolved ticker: {ticker}\n\n"
        "Use the available Alpha Vantage tools to get current market data for this ticker. "
        "Then answer the question using both the retrieved news context and the live market data."
    )
 
    def _run_once(prompt: str) -> tuple[str | None, dict[str, Any]]:
        try:
            # FIX 3a: Remove timeout parameter - let the SDK handle it
            out = gen_client.beta.messages.create(
                model=GEN_MODEL_NAME,
                max_tokens=GEN_MAX_TOKENS,
                temperature=0,
                system=mcp_system_prompt,
                messages=[{"role": "user", "content": prompt}],
                betas=[ALPHA_MCP_BETA],
                mcp_servers=mcp_servers,
                tools=[toolset_config],
                # timeout parameter removed
            )
        except Exception as exc:
            return None, {"failed": True, "reason": f"mcp request failed: {exc}"}
 
        text = strip_think_tags(anthropic_text(out)).strip()
        mcp_meta = inspect_mcp_response(out)
 
        # FIX 3b: Simplified validation - only fail on actual errors
        if mcp_meta["has_error"]:
            return None, {"failed": True, "reason": "mcp tool returned an error", **mcp_meta}
        if mcp_meta["rate_limited"]:
            return None, {"failed": True, "reason": "mcp tool appears rate-limited", **mcp_meta}
        
        # FIX 3c: Accept response if we got ANY tool execution with results
        if mcp_meta["has_result_block"] and text:
            return text, {"failed": False, **mcp_meta}
        
        # Only fail if we got nothing useful
        if not mcp_meta["tool_calls"]:
            return None, {"failed": True, "reason": "no tools called", **mcp_meta}
        if not text:
            return None, {"failed": True, "reason": "empty assistant text after mcp call", **mcp_meta}
        
        # Still got something - accept it
        return text, {"failed": False, **mcp_meta}
 
    # FIX 3d: Single attempt instead of retry logic
    # The retry was trying to force a specific tool pattern that may not be optimal
    return _run_once(primary_prompt)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Loading generation model...")
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in environment.")
    try:
        anthropic_module = importlib.import_module("anthropic")
    except ImportError as exc:
        raise RuntimeError("anthropic package is not installed. Run: pip install anthropic") from exc
    Anthropic = getattr(anthropic_module, "Anthropic")
    gen_client = Anthropic(api_key=ANTHROPIC_API_KEY)

    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Connecting to SQLite...")
    sqlite_conn = connect_sqlite(SQLITE_DB)

    print("Loading ticker map...")
    alias_to_ticker, ticker_to_canonical = load_ticker_company_map(TICKER_MAP_PATH)
    alias_to_fin_entity = load_financial_entity_map(FIN_ENTITY_MAP_PATH, TICKER_MAP_PATH)

    # Get article date range for system prompt base from SQLite source-of-truth
    date_min, date_max = _get_sqlite_date_range(sqlite_conn)
    # Base prompt — conversation history is injected per-turn via build_system_prompt()
    base_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(date_min=date_min, date_max=date_max)
    base_causal_system_prompt = CAUSAL_SYSTEM_PROMPT_TEMPLATE.format(date_min=date_min, date_max=date_max)

    # ── Memory: load from disk if available ──────────────────────────────────
    memory = load_memory(MEMORY_PATH)
    if memory.turn_count > 0:
        print(f"  [memory] Resuming session '{memory.session_id}' "
              f"({memory.turn_count} prior turns)")
    else:
        print(f"  [memory] New session '{memory.session_id}'")

    print(f"\n--- T-GRAG Chatbot ready ---")
    print(f"Articles from {date_min} to {date_max}")
    print("Type your question (or 'quit' to exit).\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query or query.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        t0 = time.perf_counter()

        # ── Hook 1: Coreference resolution ───────────────────────────────────
        query, was_rewritten = resolve_coreference(query, memory)
        market_data_intent = is_market_data_intent(query)
        causal_intent = is_causal_analysis_intent(query)
        if was_rewritten:
            print(f"  [memory] Coreference resolved → \"{query}\"")

        source_filter = extract_source_filter(query)
        if source_filter:
            print(f"  [source filter: {source_filter}]")

        # ── Hook 2: Temporal decomposition ───────────────────────────────────
        sub_queries = decompose_query(query, gen_client)
        if len(sub_queries) > 1:
            print(f"  [decomposed into {len(sub_queries)} sub-queries]")

        # ── Hook 3: Temporal carryover from memory ────────────────────────────
        sub_queries = resolve_temporal_carryover(sub_queries, memory)

        all_contexts: list[str] = []
        all_urls: list[str] = []

        # Track the primary target across sub-queries (first resolved entity wins)
        primary_target = None
        primary_date_start = None
        primary_date_end = None
        all_chunks: list[dict] = []
        market_ctx = ""
        for sq in sub_queries:
            date_start = sq.get("time_start")
            date_end = sq.get("time_end")

            if date_start or date_end:
                print(f"  [time filter: {date_start} → {date_end}]")

            if causal_intent and not market_data_intent:
                # ── Causal path: decompose into transmission hops, retrieve each ──
                hops = decompose_causal_chain(sq["query"], gen_client)
                if hops:
                    print(f"  [causal chain: {' → '.join(hops)}]")
                    chunks, target = retrieve_causal_chain(
                        query=sq["query"],
                        hops=hops,
                        embed_model=embed_model,
                        driver=driver,
                        sqlite_conn=sqlite_conn,
                        alias_to_ticker=alias_to_ticker,
                        ticker_to_canonical=ticker_to_canonical,
                        alias_to_fin_entity=alias_to_fin_entity,
                        source_filter=source_filter,
                        date_start=date_start,
                        date_end=date_end,
                    )
                    market_ctx = fetch_market_context(
                        hops=hops,
                        date_start=date_start,
                        date_end=date_end,
                    )
                    print(f"  [market data: {len([l for l in market_ctx.splitlines() if l.strip().startswith('20')])} price points fetched]")
                    if market_ctx:
                        print(f"  [market data fetched for causal hops]")
                else:
                    # Decomposition returned nothing — fall back to standard retrieve
                    print("  [causal chain] decomposition empty, falling back to standard retrieve")
                    chunks, target = retrieve(
                        query=sq["query"],
                        embed_model=embed_model,
                        driver=driver,
                        sqlite_conn=sqlite_conn,
                        alias_to_ticker=alias_to_ticker,
                        ticker_to_canonical=ticker_to_canonical,
                        alias_to_fin_entity=alias_to_fin_entity,
                        top_k=TOP_K,
                        expanded_k=EXPANDED_K,
                        recency_half_life_days=RECENCY_HALF_LIFE_DAYS,
                        source_filter=source_filter,
                        date_start=date_start,
                        date_end=date_end,
                    )
            else:
                # ── Standard path: unchanged ──────────────────────────────────────
                chunks, target = retrieve(
                    query=sq["query"],
                    embed_model=embed_model,
                    driver=driver,
                    sqlite_conn=sqlite_conn,
                    alias_to_ticker=alias_to_ticker,
                    ticker_to_canonical=ticker_to_canonical,
                    alias_to_fin_entity=alias_to_fin_entity,
                    top_k=TOP_K,
                    expanded_k=EXPANDED_K,
                    recency_half_life_days=RECENCY_HALF_LIFE_DAYS,
                    source_filter=source_filter,
                    date_start=date_start,
                    date_end=date_end,
                )

            print(f"  [entity: {target.display_name or 'general'} | "
                  f"canonical: {target.canonical_name or '—'} | "
                  f"confidence: {target.confidence:.2f} | "
                  f"chunks: {len(chunks)}]")

            # Capture primary target from the first sub-query that resolves an entity
            if primary_target is None or (
                target.canonical_name and not primary_target.canonical_name
            ):
                primary_target = target
                primary_date_start = date_start
                primary_date_end = date_end

            all_chunks.extend(chunks)

            period_label = f"[{date_start} → {date_end}] " if (date_start or date_end) else ""
            ctx = build_context(sq["query"], target, chunks)
            if market_ctx:
                ctx = ctx + "\n\n" + market_ctx
            all_contexts.append(period_label + ctx)

            for ch in chunks:
                url = ch.get("url")
                if url and url not in all_urls:
                    all_urls.append(url)

        if not all_contexts:
            print("Assistant: No relevant chunks found.\n")
            continue

        # ── Hook 4: Inject conversation memory into system prompt ─────────────
        system_prompt = build_system_prompt(base_system_prompt, memory)

        # Route causal queries to the analyst prompt instead of the factual prompt
        if causal_intent and not market_data_intent:
            causal_system_prompt = build_system_prompt(base_causal_system_prompt, memory)
            print("  [causal analysis mode]")
        else:
            causal_system_prompt = None

        merged_context = "\n\n---\n\n".join(all_contexts)

        if DEBUG_SKIP_GENERATION:
            continue

        final = ""
        market_data_note: str | None = None

        should_try_mcp = (
            market_data_intent
            and ENABLE_ALPHA_MCP_REMOTE
            and bool(ALPHA_VANTAGE_API_KEY)
            and primary_target is not None
            and bool(primary_target.ticker)
        )

        if should_try_mcp:
            print(f"  [alpha mcp: querying remote server (timeout={ALPHA_MCP_TIMEOUT_SECONDS:.0f}s)]")
            mcp_answer, mcp_meta = generate_answer_with_remote_mcp(
                query=query,
                context=merged_context,
                ticker=primary_target.ticker or "",
                gen_client=gen_client,
                system_prompt=system_prompt,
            )
            if mcp_answer is not None:
                final = mcp_answer
                tools_used = ", ".join(mcp_meta.get("tool_calls", [])) if isinstance(mcp_meta, dict) else ""
                if tools_used:
                    print(f"  [alpha mcp tools: {tools_used}]")
            else:
                reason = mcp_meta.get("reason", "unknown") if isinstance(mcp_meta, dict) else "unknown"
                print(f"  [alpha mcp fallback: {reason}]")
                # Use causal prompt on MCP fallback too if applicable
                active_prompt = causal_system_prompt if causal_system_prompt else system_prompt
                final = generate_answer(query, merged_context, gen_client, active_prompt)
                market_data_note = ALPHA_MCP_FALLBACK_NOTE
        else:
            if market_data_intent and ENABLE_ALPHA_MCP_REMOTE and (primary_target is None or not primary_target.ticker):
                print("  [alpha mcp skipped: no resolved ticker]")
            active_prompt = causal_system_prompt if causal_system_prompt else system_prompt
            final = generate_answer(query, merged_context, gen_client, active_prompt)

        if market_data_note:
            final = f"{final}\n\nNote: {market_data_note}"

        elapsed = time.perf_counter() - t0
        print(f"\nAssistant: {final}")
        if all_urls:
            print("\nSources:")
            for url in all_urls:
                print(f"  - {url}")
        print(f"  [{elapsed:.1f}s]\n")

        # ── Hook 5: Record turn to memory ─────────────────────────────────────
        memory.record_turn(
            query=query,
            target=primary_target,
            date_start=primary_date_start,
            date_end=primary_date_end,
            answer=final,
            chunks=all_chunks,
            source_urls=all_urls,
        )

        # ── Hook 6: Compress if session is getting long ───────────────────────
        compressed = memory.maybe_compress(
            gen_client=gen_client,
            gen_model=GEN_MODEL_NAME,
        )
        if compressed:
            print(f"  [memory] Session compressed to {memory.turn_count} recent turns + summary")

        # ── Hook 7: Persist to disk ───────────────────────────────────────────
        save_memory(memory, MEMORY_PATH)

    sqlite_conn.close()
    driver.close()


if __name__ == "__main__":
    main()
