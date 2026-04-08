"""
Temporal-Graph RAG Chatbot

Connects to an already-populated Neo4j graph (built by tgrag_setup.py)
and runs interactive retrieval + generation in a terminal loop.

Features:
  - Temporal Query Decomposition (TQD): splits multi-temporal queries
  - Three-layer retrieval: temporal filter -> entity match -> semantic ranking
  - Sub-answer aggregation for comparative / multi-period questions
  - Default 7-day look-back for market data when no time window is given

Usage:
    python chatter.py
"""

import json
import os
import re
import time
import warnings
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
load_dotenv()

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=FutureWarning)

import dateparser
import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from transformers import pipeline
from dataclasses import dataclass
from typing import Optional

from tgrag_setup import EMBED_MODEL_NAME, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, load_ticker_company_map, canonicalize
from pathlib import Path
ticker_lookup = load_ticker_company_map(Path("ticker_company_map.csv"))
GEN_MODEL_NAME = "Qwen/Qwen3-0.6B"

# Default look-back window for market bars when the user gives no time filter.
# Set to None to disable the default and always use unbounded queries.
DEFAULT_MARKET_LOOKBACK_DAYS: int | None = 7

MIN_STRICT_CHUNKS_BEFORE_EXPANSION = 3

SOURCE_KEYWORDS = {
    "bbc": "BBC",
    "bloomberg": "Bloomberg",
    "cnbc": "CNBC",
    "marketwatch": "MarketWatch",
    "nasdaq": "Nasdaq",
    "stockbiz": "Stockbiz",
    "marketbeat": "MarketBeat",
    "cbs": "CBS MoneyWatch",
}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def extract_source_filter(query: str) -> str | None:
    q = query.lower()
    for keyword, label in SOURCE_KEYWORDS.items():
        if keyword in q:
            return label
    return None


def strip_think_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


# Temporal Query Decomposition

TIME_WORDS = re.compile(
    r"\b(yesterday|last\s+week|this\s+week|last\s+month|this\s+month|"
    r"today|ago|before\s+\w+day|after\s+\w+day|since|until|between|"
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}|"
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{4}|"
    r"\d{4}[-/]\d{2}[-/]\d{2}|"
    r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"\d{4}[-/]\d{2}|\d{1,2}/\d{1,2})\b",
    re.IGNORECASE,
)

MULTI_TIME_WORDS = re.compile(
    r"\b(compare|vs\.?|versus|differ)",
    re.IGNORECASE,
)

FROM_TO_PATTERN = re.compile(
    r"from\s+(.+?)\s+to\s+(.+?)[\?\.\!,]?\s*$",
    re.IGNORECASE,
)

DECOMPOSE_PROMPT = """\
Analyze this question and decompose it into separate sub-queries for each time period.

Rules:
- Today is {today}.
- Use ISO format YYYY-MM-DD for dates.
- Each sub-query should target one time period.

Output ONLY valid JSON (no markdown):
{{"sub_queries": [{{"query": "...", "time_start": "YYYY-MM-DD or null", "time_end": "YYYY-MM-DD or null"}}]}}

Question: {query}"""


def resolve_date(date_str: str | None) -> int | None:
    """Convert a date string to unix timestamp via dateparser."""
    if not date_str or date_str.lower() == "null":
        return None
    dt = dateparser.parse(
        date_str, settings={"RETURN_AS_TIMEZONE_AWARE": True}
    )
    if dt:
        return int(dt.timestamp())
    return None


def ts_to_datestr(ts: int | None) -> str | None:
    """Convert unix timestamp seconds to YYYY-MM-DD (UTC)."""
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()


def _resolve_time_phrase(phrase: str) -> tuple[str | None, str | None]:
    """Resolve a single time phrase to (start_date, end_date) strings."""
    now = datetime.now(timezone.utc)
    p = phrase.strip().lower()

    if p == "yesterday":
        dt = dateparser.parse("yesterday", settings={"RETURN_AS_TIMEZONE_AWARE": True})
        d = dt.strftime("%Y-%m-%d") if dt else None
        return d, d
    if "last week" in p:
        s = dateparser.parse("7 days ago", settings={"RETURN_AS_TIMEZONE_AWARE": True})
        e = dateparser.parse("1 day ago", settings={"RETURN_AS_TIMEZONE_AWARE": True})
        return (
            s.strftime("%Y-%m-%d") if s else None,
            e.strftime("%Y-%m-%d") if e else None,
        )
    if "this week" in p:
        days_back = now.weekday()
        s = dateparser.parse(
            f"{days_back} days ago", settings={"RETURN_AS_TIMEZONE_AWARE": True}
        )
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
    """Use dateparser directly to resolve time expressions without LLM."""
    time_matches = TIME_WORDS.findall(query)
    if not time_matches:
        return {"query": query, "time_start": None, "time_end": None}

    def _match_score(m: str) -> tuple[int, int]:
        year_bonus = 1 if re.search(r"\d{4}", m) else 0
        return (year_bonus, len(m))

    best = max(time_matches, key=_match_score)
    start, end = _resolve_time_phrase(best)
    return {"query": query, "time_start": start, "time_end": end}


def decompose_query(query: str, pipe) -> list[dict]:
    """Resolve temporal references. Uses regex whenever possible, LLM only as last resort."""
    has_time = TIME_WORDS.search(query)

    if not has_time:
        return [{"query": query, "time_start": None, "time_end": None}]

    # Handle "from X to Y" pattern with regex (no LLM needed)
    ft = FROM_TO_PATTERN.search(query)
    if ft:
        period_a, period_b = ft.group(1).strip(), ft.group(2).strip()
        a_start, a_end = _resolve_time_phrase(period_a)
        b_start, b_end = _resolve_time_phrase(period_b)
        if a_start or b_start:
            return [
                {"query": query, "time_start": a_start, "time_end": a_end},
                {"query": query, "time_start": b_start, "time_end": b_end},
            ]

    # Common multi-period phrasing without requiring "vs/compare".
    if re.search(r"\blast\s+week\b", query, flags=re.IGNORECASE) and re.search(
        r"\bthis\s+week\b", query, flags=re.IGNORECASE
    ):
        last_start, last_end = _resolve_time_phrase("last week")
        this_start, this_end = _resolve_time_phrase("this week")
        if last_start or this_start:
            return [
                {"query": query, "time_start": last_start, "time_end": last_end},
                {"query": query, "time_start": this_start, "time_end": this_end},
            ]
    if re.search(r"\blast\s+month\b", query, flags=re.IGNORECASE) and re.search(
        r"\bthis\s+month\b", query, flags=re.IGNORECASE
    ):
        last_start, last_end = _resolve_time_phrase("last month")
        this_start, this_end = _resolve_time_phrase("this month")
        if last_start or this_start:
            return [
                {"query": query, "time_start": last_start, "time_end": last_end},
                {"query": query, "time_start": this_start, "time_end": this_end},
            ]

    has_multi = MULTI_TIME_WORDS.search(query)
    if not has_multi:
        return [_extract_single_time_range(query)]

    # True multi-period query (compare/vs) -- use LLM to decompose
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = DECOMPOSE_PROMPT.format(today=today, query=query)
    messages = [{"role": "user", "content": prompt}]
    out = pipe(messages, max_new_tokens=1024, do_sample=False)
    raw = out[0]["generated_text"][-1]["content"]
    raw = strip_think_tags(raw)

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            sub_queries = data.get("sub_queries", [])
            if isinstance(sub_queries, list):
                valid = []
                for sq in sub_queries:
                    if isinstance(sq, dict) and "query" in sq:
                        valid.append({
                            "query": sq["query"],
                            "time_start": sq.get("time_start"),
                            "time_end": sq.get("time_end"),
                        })
                if valid:
                    return valid
        except (json.JSONDecodeError, TypeError):
            pass

    return [_extract_single_time_range(query)]


# Three-Layer Retrieval

def _fmt_num(x) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


def _fetch_market_bars(
    session,
    ticker: str,
    dstart: str | None,
    dend: str | None,
    limit: int = 7,
) -> list[dict]:
    """
    Fetch OHLC bars for a ticker in [dstart, dend].
    Falls back to the nearest previous trading day if an exact day has no bar.
    """
    is_exact_day = dstart is not None and dend is not None and dstart == dend

    bars = session.run(
        """
        MATCH (i:Instrument {ticker: $ticker})<-[:FOR_INSTRUMENT]-(b:MarketBar)
        WHERE ($dstart IS NULL OR b.bar_date >= date($dstart))
          AND ($dend   IS NULL OR b.bar_date <= date($dend))
        RETURN b.bar_date AS bar_date,
               b.open     AS open,
               b.high     AS high,
               b.low      AS low,
               b.close    AS close
        ORDER BY b.bar_date DESC
        LIMIT $limit
        """,
        {"ticker": ticker, "dstart": dstart, "dend": dend, "limit": limit},
    ).data()

    # Weekend / holiday fallback for exact-day queries
    if not bars and is_exact_day:
        bars = session.run(
            """
            MATCH (i:Instrument {ticker: $ticker})<-[:FOR_INSTRUMENT]-(b:MarketBar)
            WHERE b.bar_date < date($dstart)
            RETURN b.bar_date AS bar_date,
                   b.open     AS open,
                   b.high     AS high,
                   b.low      AS low,
                   b.close    AS close
            ORDER BY b.bar_date DESC
            LIMIT 1
            """,
            {"ticker": ticker, "dstart": dstart},
        ).data()

    return bars


def _build_market_summary(ticker: str, bars: list[dict], dstart: str | None) -> str:
    """Turn a list of bar dicts into a human-readable market summary string."""
    bars_sorted = sorted(bars, key=lambda r: r["bar_date"])
    start_close = bars_sorted[0].get("close")
    end_close = bars_sorted[-1].get("close")
    start_bd = bars_sorted[0]["bar_date"]
    end_bd = bars_sorted[-1]["bar_date"]
    start_bd_s = start_bd.isoformat() if hasattr(start_bd, "isoformat") else str(start_bd)
    end_bd_s = end_bd.isoformat() if hasattr(end_bd, "isoformat") else str(end_bd)
    is_exact_day = dstart is not None and start_bd_s != dstart

    pct_change = None
    if start_close is not None and end_close is not None:
        try:
            pct_change = (float(end_close) - float(start_close)) / float(start_close) * 100.0
        except Exception:
            pass

    lows  = [b.get("low")  for b in bars_sorted if b.get("low")  is not None]
    highs = [b.get("high") for b in bars_sorted if b.get("high") is not None]
    min_low  = min(float(x) for x in lows)  if lows  else None
    max_high = max(float(x) for x in highs) if highs else None

    summary = (
        f"[MARKET DATA] {ticker} — {start_bd_s} to {end_bd_s} | "
        f"start close={_fmt_num(start_close)}, end close={_fmt_num(end_close)}"
    )
    if is_exact_day:
        summary += " (nearest previous trading day used)"
    if pct_change is not None:
        direction = "▲" if pct_change >= 0 else "▼"
        summary += f" | period change={direction}{abs(pct_change):.2f}%"
    if min_low is not None:
        summary += f" | range low={_fmt_num(min_low)}, high={_fmt_num(max_high)}"

    line_items = []
    for b in bars:
        bd_s = b["bar_date"].isoformat() if hasattr(b["bar_date"], "isoformat") else str(b["bar_date"])
        line_items.append(
            f"  {bd_s}: O={_fmt_num(b.get('open'))} H={_fmt_num(b.get('high'))} "
            f"L={_fmt_num(b.get('low'))} C={_fmt_num(b.get('close'))}"
        )

    return summary + "\n" + "\n".join(line_items)

QUERY_TYPE_SINGLE = "single_entity_company"
QUERY_TYPE_MULTI = "multi_entity_company"
QUERY_TYPE_GENERAL = "general"

COMPARE_WORDS = re.compile(r"\b(compare|vs\.?|versus|against)\b", re.IGNORECASE)
QUERY_INTENT_GENERAL = "general_company_news"
QUERY_INTENT_COMPETITION = "competition"
QUERY_INTENT_MNA = "mna"
QUERY_INTENT_PARTNERSHIP = "partnership"
QUERY_INTENT_INVESTMENT = "investment"
QUERY_INTENT_PRODUCT = "product"
QUERY_INTENT_EARNINGS = "earnings"
QUERY_INTENT_SUPPLY = "supply_chain"

INTENT_PATTERNS = {
    QUERY_INTENT_COMPETITION: re.compile(
        r"\b(compete|competitor|competition|rival|rivals)\b", re.IGNORECASE
    ),
    QUERY_INTENT_MNA: re.compile(
        r"\b(acquire|acquired|acquisition|buy|bought|purchase|purchased|merger|merge)\b",
        re.IGNORECASE,
    ),
    QUERY_INTENT_PARTNERSHIP: re.compile(
        r"\b(partner|partnership|collaborat|collaboration|teamed up|joined forces)\b",
        re.IGNORECASE,
    ),
    QUERY_INTENT_INVESTMENT: re.compile(
        r"\b(invest|investment|stake|backed|funding|fundraise|raised)\b",
        re.IGNORECASE,
    ),
    QUERY_INTENT_PRODUCT: re.compile(
        r"\b(product|launch|launched|release|released|unveil|unveiled|device|chip|model|platform)\b",
        re.IGNORECASE,
    ),
    QUERY_INTENT_EARNINGS: re.compile(
        r"\b(earnings|revenue|profit|guidance|forecast|results|reported|quarter|q1|q2|q3|q4)\b",
        re.IGNORECASE,
    ),
    QUERY_INTENT_SUPPLY: re.compile(
        r"\b(supply|supplier|supplied|providing|manufactur|foundry|shipment|deliver)\b",
        re.IGNORECASE,
    ),
}

@dataclass
class QueryTarget:
    query_type: str
    entity_canonical: Optional[str]
    display_name: Optional[str]
    ticker: Optional[str]
    confidence: float
    ambiguous: bool
    candidates: list[tuple[str, str, str]]


def classify_query_type(query: str) -> str:
    if COMPARE_WORDS.search(query):
        return QUERY_TYPE_MULTI
    return QUERY_TYPE_SINGLE

def classify_query_intent(query: str) -> str:
    for intent, pattern in INTENT_PATTERNS.items():
        if pattern.search(query):
            return intent
    return QUERY_INTENT_GENERAL

SEMANTIC_POLICY_BY_INTENT = {
    QUERY_INTENT_GENERAL: {
        "allowed_relations": {
            "ACQUIRED",
            "PARTNERED_WITH",
            "INVESTED_IN",
            "LAUNCHED",
            "SUPPLIED",
            "REPORTED",
        },
        "allowed_neighbor_types": {"ORG", "PRODUCT", "EVENT", "STOCK"},
        "min_relation_confidence": 0.50,
        "max_neighbor_entities": 5,
        "max_extra_chunks": 4,
    },
    QUERY_INTENT_COMPETITION: {
        "allowed_relations": {"COMPETES_WITH"},
        "allowed_neighbor_types": {"ORG", "PRODUCT", "STOCK"},
        "min_relation_confidence": 0.50,
        "max_neighbor_entities": 6,
        "max_extra_chunks": 5,
    },
    QUERY_INTENT_MNA: {
        "allowed_relations": {"ACQUIRED"},
        "allowed_neighbor_types": {"ORG", "STOCK", "EVENT"},
        "min_relation_confidence": 0.60,
        "max_neighbor_entities": 4,
        "max_extra_chunks": 4,
    },
    QUERY_INTENT_PARTNERSHIP: {
        "allowed_relations": {"PARTNERED_WITH"},
        "allowed_neighbor_types": {"ORG", "PRODUCT", "EVENT", "STOCK"},
        "min_relation_confidence": 0.55,
        "max_neighbor_entities": 4,
        "max_extra_chunks": 4,
    },
    QUERY_INTENT_INVESTMENT: {
        "allowed_relations": {"INVESTED_IN"},
        "allowed_neighbor_types": {"ORG", "STOCK", "EVENT"},
        "min_relation_confidence": 0.60,
        "max_neighbor_entities": 4,
        "max_extra_chunks": 4,
    },
    QUERY_INTENT_PRODUCT: {
        "allowed_relations": {"LAUNCHED", "SUPPLIED"},
        "allowed_neighbor_types": {"PRODUCT", "ORG", "EVENT"},
        "min_relation_confidence": 0.55,
        "max_neighbor_entities": 5,
        "max_extra_chunks": 5,
    },
    QUERY_INTENT_EARNINGS: {
        "allowed_relations": {"REPORTED"},
        "allowed_neighbor_types": {"EVENT", "ORG", "STOCK"},
        "min_relation_confidence": 0.60,
        "max_neighbor_entities": 4,
        "max_extra_chunks": 4,
    },
    QUERY_INTENT_SUPPLY: {
        "allowed_relations": {"SUPPLIED"},
        "allowed_neighbor_types": {"ORG", "PRODUCT", "EVENT"},
        "min_relation_confidence": 0.60,
        "max_neighbor_entities": 4,
        "max_extra_chunks": 4,
    },
}

def get_semantic_policy_for_query(query: str) -> dict:
    intent = classify_query_intent(query)
    return {
        "intent": intent,
        **SEMANTIC_POLICY_BY_INTENT.get(intent, SEMANTIC_POLICY_BY_INTENT[QUERY_INTENT_GENERAL]),
    }

def normalize_query_for_matching(query: str) -> str:
    q = canonicalize(query)
    stop_phrases = [
        "latest news on", "latest news about", "news on", "news about",
        "what is the latest news on", "what is the latest news about",
        "what happened to", "tell me about", "latest on"
    ]
    for phrase in stop_phrases:
        q = q.replace(canonicalize(phrase), " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q


def get_entity_name_map(driver) -> dict[str, str]:
    with driver.session() as session:
        rows = session.run(
            "MATCH (e:Entity) RETURN e.canonical_name AS cname, e.name AS name"
        ).data()
    out = {}
    for r in rows:
        cname = r.get("cname")
        if cname:
            out[cname] = r.get("name") or cname
    return out


def resolve_query_target(query: str, ticker_lookup: dict[str, str], driver) -> QueryTarget:
    query_type = classify_query_type(query)
    q_norm = normalize_query_for_matching(query)
    entity_name_map = get_entity_name_map(driver)

    # Tier 1: explicit ticker in query
    explicit_tickers = sorted({t.upper() for t in re.findall(r"\b[A-Z]{1,5}\b", query)})
    if explicit_tickers:
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (i:Instrument)
                WHERE i.ticker IN $tickers
                RETURN i.ticker AS ticker
                """,
                {"tickers": explicit_tickers},
            ).data()
        valid = [r["ticker"] for r in rows if r.get("ticker")]
        if valid:
            ticker = valid[0]
            canonicals = [c for c, tk in ticker_lookup.items() if tk == ticker]
            canonical = max(canonicals, key=len) if canonicals else None
            display = entity_name_map.get(canonical, canonical or ticker)
            return QueryTarget(
                query_type=query_type,
                entity_canonical=canonical,
                display_name=display,
                ticker=ticker,
                confidence=0.99,
                ambiguous=False,
                candidates=[(canonical or "", display or "", ticker)],
            )

    # Tier 2: exact alias/company match
    exact_hits = []
    for canonical, ticker in ticker_lookup.items():
        if canonical == q_norm:
            exact_hits.append((canonical, entity_name_map.get(canonical, canonical), ticker))
    if len(exact_hits) == 1:
        c, d, t = exact_hits[0]
        return QueryTarget(query_type, c, d, t, 0.98, False, [(c, d, t)])

    # Tier 3: exact canonical phrase in query
    phrase_hits = []
    for canonical, ticker in ticker_lookup.items():
        if re.search(r"\b" + re.escape(canonical) + r"\b", q_norm):
            phrase_hits.append((canonical, entity_name_map.get(canonical, canonical), ticker))

    if phrase_hits:
        phrase_hits.sort(key=lambda x: len(x[0]), reverse=True)
        best = phrase_hits[0]
        ambiguous = len({t for _, _, t in phrase_hits}) > 1
        return QueryTarget(
            query_type=query_type,
            entity_canonical=best[0],
            display_name=best[1],
            ticker=best[2],
            confidence=0.95,
            ambiguous=ambiguous,
            candidates=phrase_hits[:5],
        )

    # Tier 4: exact entity canonical in DB
    if q_norm in entity_name_map:
        return QueryTarget(
            query_type=QUERY_TYPE_SINGLE,
            entity_canonical=q_norm,
            display_name=entity_name_map[q_norm],
            ticker=ticker_lookup.get(q_norm),
            confidence=0.85,
            ambiguous=False,
            candidates=[(q_norm, entity_name_map[q_norm], ticker_lookup.get(q_norm, ""))],
        )

    return QueryTarget(
        query_type=QUERY_TYPE_GENERAL,
        entity_canonical=None,
        display_name=None,
        ticker=None,
        confidence=0.0,
        ambiguous=False,
        candidates=[],
    )

def retrieve_target_anchored_chunks(
    driver,
    target: QueryTarget,
    source_filter: str | None = None,
    time_start: int | None = None,
    time_end: int | None = None,
) -> list[dict]:
    if not target.entity_canonical and not target.ticker:
        return []

    params = {
        "target_entity": target.entity_canonical,
        "target_ticker": target.ticker,
        "source_filter": source_filter,
    }

    time_clause = ""
    if time_start is not None:
        time_clause += " AND c.published_ts >= $ts_start"
        params["ts_start"] = time_start
    if time_end is not None:
        time_clause += " AND c.published_ts <= $ts_end"
        params["ts_end"] = time_end

    query = f"""
    MATCH (c:Chunk)<-[:HAS_CHUNK]-(a:Article)
    WHERE ($source_filter IS NULL OR c.source = $source_filter)
      {time_clause}
      AND (
        ($target_entity IS NOT NULL AND EXISTS {{
            MATCH (c)-[:MENTIONS]->(:Entity {{canonical_name: $target_entity}})
        }})
        OR
        ($target_ticker IS NOT NULL AND EXISTS {{
            MATCH (c)-[:MENTIONS_INSTRUMENT]->(:Instrument {{ticker: $target_ticker}})
        }})
      )
    RETURN DISTINCT
        c.chunk_uid    AS chunk_uid,
        c.text         AS text,
        c.embedding    AS embedding,
        c.published_ts AS published_ts,
        c.source       AS source,
        a.title        AS title,
        a.url          AS url,
        c.article_id   AS article_id,
        c.chunk_id     AS chunk_id
    """
    with driver.session() as session:
        return session.run(query, params).data()

def retrieve_semantic_expansion_chunks(
    driver,
    target: QueryTarget,
    source_filter: str | None = None,
    time_start: int | None = None,
    time_end: int | None = None,
    allowed_relations: set[str] | None = None,
    allowed_neighbor_types: set[str] | None = None,
    min_relation_confidence: float = 0.0,
    max_neighbor_entities: int = 5,
    max_extra_chunks: int = 6,
) -> list[dict]:
    if not target.entity_canonical:
        return []

    if allowed_relations is None:
        allowed_relations = {
            "ACQUIRED",
            "PARTNERED_WITH",
            "INVESTED_IN",
            "LAUNCHED",
            "SUPPLIED",
            "REPORTED",
            "COMPETES_WITH",
        }

    if allowed_neighbor_types is None:
        allowed_neighbor_types = {"ORG", "PRODUCT", "EVENT", "STOCK"}

    params = {
        "target_entity": target.entity_canonical,
        "source_filter": source_filter,
        "allowed_relations": list(allowed_relations),
        "allowed_neighbor_types": list(allowed_neighbor_types),
        "min_relation_confidence": float(min_relation_confidence),
        "max_neighbor_entities": max_neighbor_entities,
        "max_extra_chunks": max_extra_chunks,
    }

    time_clause = ""
    if time_start is not None:
        time_clause += " AND c.published_ts >= $ts_start"
        params["ts_start"] = time_start
    if time_end is not None:
        time_clause += " AND c.published_ts <= $ts_end"
        params["ts_end"] = time_end

    query = f"""
    MATCH (:Entity {{canonical_name: $target_entity}})-[r]-(n:Entity)
    WHERE type(r) IN $allowed_relations
        AND n.type IN $allowed_neighbor_types
        AND n.canonical_name <> $target_entity
        AND coalesce(r.confidence, 0.0) >= $min_relation_confidence
        AND r.source_chunk_uid IS NOT NULL
    WITH
        n,
        type(r) AS relation_type,
        coalesce(r.confidence, 0.0) AS relation_confidence,
        r.source_chunk_uid AS relation_source_chunk_uid,
        r.extractor AS relation_extractor
    ORDER BY relation_confidence DESC, n.canonical_name
    LIMIT $max_neighbor_entities

    MATCH (c:Chunk)
    WHERE c.chunk_uid = relation_source_chunk_uid
    MATCH (c)<-[:HAS_CHUNK]-(a:Article)
    WHERE ($source_filter IS NULL OR c.source = $source_filter)
    {time_clause}
    RETURN DISTINCT
        c.chunk_uid                AS chunk_uid,
        c.text                     AS text,
        c.embedding                AS embedding,
        c.published_ts             AS published_ts,
        c.source                   AS source,
        a.title                    AS title,
        a.url                      AS url,
        c.article_id               AS article_id,
        c.chunk_id                 AS chunk_id,
        relation_type              AS relation_type,
        relation_confidence        AS relation_confidence,
        relation_source_chunk_uid  AS relation_source_chunk_uid,
        relation_extractor         AS relation_extractor
    LIMIT $max_extra_chunks
    """
    with driver.session() as session:
        return session.run(query, params).data()

def retrieve_controlled_expansion_chunks(
    driver,
    target: QueryTarget,
    source_filter: str | None = None,
    time_start: int | None = None,
    time_end: int | None = None,
    min_edge_weight: int = 2,
    allowed_neighbor_types: set[str] | None = None,
    max_neighbor_entities: int = 5,
    max_extra_chunks: int = 6,
) -> list[dict]:
    if not target.entity_canonical:
        return []

    if allowed_neighbor_types is None:
        allowed_neighbor_types = {"ORG", "PRODUCT", "EVENT", "STOCK"}

    params = {
        "target_entity": target.entity_canonical,
        "target_ticker": target.ticker,
        "source_filter": source_filter,
        "min_edge_weight": min_edge_weight,
        "allowed_neighbor_types": list(allowed_neighbor_types),
        "max_neighbor_entities": max_neighbor_entities,
        "max_extra_chunks": max_extra_chunks,
    }

    time_clause = ""
    if time_start is not None:
        time_clause += " AND c.published_ts >= $ts_start"
        params["ts_start"] = time_start
    if time_end is not None:
        time_clause += " AND c.published_ts <= $ts_end"
        params["ts_end"] = time_end

    query = f"""
    MATCH (:Entity {{canonical_name: $target_entity}})-[r:CO_OCCURS_CHUNK]-(n:Entity)
    WHERE coalesce(r.weight, 0) >= $min_edge_weight
        AND n.type IN $allowed_neighbor_types
        AND n.canonical_name <> $target_entity
    WITH n, r.weight AS edge_weight
    ORDER BY edge_weight DESC, n.canonical_name
    LIMIT $max_neighbor_entities

    MATCH (c:Chunk)-[:MENTIONS]->(n)
    MATCH (c)<-[:HAS_CHUNK]-(a:Article)
    WHERE ($source_filter IS NULL OR c.source = $source_filter)
        {time_clause}
      AND NOT EXISTS {{
          MATCH (c)-[:MENTIONS]->(:Entity {{canonical_name: $target_entity}})
      }}
      AND (
          $target_ticker IS NULL OR
          NOT EXISTS {{
              MATCH (c)-[:MENTIONS_INSTRUMENT]->(:Instrument {{ticker: $target_ticker}})
          }}
      )
    RETURN DISTINCT
        c.chunk_uid    AS chunk_uid,
        c.text         AS text,
        c.embedding    AS embedding,
        c.published_ts AS published_ts,
        c.source       AS source,
        a.title        AS title,
        a.url          AS url,
        c.article_id   AS article_id,
        c.chunk_id     AS chunk_id,
        edge_weight    AS edge_weight
    LIMIT $max_extra_chunks
    """
    with driver.session() as session:
        return session.run(query, params).data()

def fetch_market_text_for_target(
    driver,
    target: QueryTarget,
    time_start: int | None = None,
    time_end: int | None = None,
) -> list[str]:
    if not target.ticker:
        return []

    now = datetime.now(timezone.utc)
    if time_start is None and time_end is None and DEFAULT_MARKET_LOOKBACK_DAYS:
        market_dstart = (now - timedelta(days=DEFAULT_MARKET_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        market_dend = now.strftime("%Y-%m-%d")
    else:
        market_dstart = ts_to_datestr(time_start)
        market_dend = ts_to_datestr(time_end)

    with driver.session() as session:
        bars = _fetch_market_bars(session, target.ticker, market_dstart, market_dend, limit=7)
        if not bars:
            return []
        return [_build_market_summary(target.ticker, bars, market_dstart)]

def three_layer_retrieve(
    query: str,
    embed_model: SentenceTransformer,
    driver,
    ticker_lookup: dict,
    top_k: int = 3,
    expanded_k: int = 6,
    recency_half_life_days: float = 7.0,
    source_filter: str | None = None,
    time_start: int | None = None,
    time_end: int | None = None,
):
    # STEP 1: resolve one target from the query
    target = resolve_query_target(query, ticker_lookup, driver)
    semantic_policy = get_semantic_policy_for_query(query)

    # Strict path for single-company queries
    if target.query_type == QUERY_TYPE_SINGLE and target.entity_canonical:
        # STEP 2: only retrieve chunks directly anchored to target entity/ticker
        anchored_rows = retrieve_target_anchored_chunks(
            driver=driver,
            target=target,
            source_filter=source_filter,
            time_start=time_start,
            time_end=time_end,
        )

        # score only the anchored rows
        qvec = embed_model.encode([query], normalize_embeddings=True)[0]
        now_ts = int(datetime.now(timezone.utc).timestamp())
        half_life_seconds = recency_half_life_days * 86400.0

        scored = []
        for r in anchored_rows:
            emb = np.array(r["embedding"], dtype=np.float32)
            sim = cosine_sim(qvec, emb)

            ts = r.get("published_ts")
            if ts is None:
                recency_weight = 0.85
            else:
                age = max(0, now_ts - int(ts))
                recency_weight = float(np.exp(-np.log(2) * age / half_life_seconds))

            score = (0.80 * sim) + (0.20 * recency_weight)
            r["score"] = score
            scored.append(r)

        scored.sort(key=lambda x: x["score"], reverse=True)
        seeds = scored[:top_k]

        expanded = {s["chunk_uid"]: s for s in seeds}

        for s in expanded.values():
            s["expansion_kind"] = "direct"

        # STEP 7A: semantic expansion first
        if len(expanded) < MIN_STRICT_CHUNKS_BEFORE_EXPANSION:
            semantic_rows = retrieve_semantic_expansion_chunks(
                driver=driver,
                target=target,
                source_filter=source_filter,
                time_start=time_start,
                time_end=time_end,
                allowed_relations=semantic_policy["allowed_relations"],
                allowed_neighbor_types=semantic_policy["allowed_neighbor_types"],
                min_relation_confidence=semantic_policy["min_relation_confidence"],
                max_neighbor_entities=semantic_policy["max_neighbor_entities"],
                max_extra_chunks=semantic_policy["max_extra_chunks"],
            )

            for r in semantic_rows:
                uid = r.get("chunk_uid")
                if not uid or uid in expanded:
                    continue

                emb = np.array(r["embedding"], dtype=np.float32)
                sim = cosine_sim(qvec, emb)

                ts = r.get("published_ts")
                if ts is None:
                    recency_weight = 0.85
                else:
                    age = max(0, now_ts - int(ts))
                    recency_weight = float(np.exp(-np.log(2) * age / half_life_seconds))

                relation_confidence = float(r.get("relation_confidence", 0.0) or 0.0)
                relation_bonus = min(max(relation_confidence, 0.0), 1.0) * 0.10

                score = ((0.72 * sim) + (0.13 * recency_weight) + relation_bonus) * 0.80
                r["score"] = score
                r["expansion_kind"] = "semantic"
                expanded[uid] = r

        # STEP 7B: co-occurrence only if still too few chunks
        if len(expanded) < MIN_STRICT_CHUNKS_BEFORE_EXPANSION:
            extra_rows = retrieve_controlled_expansion_chunks(
                driver=driver,
                target=target,
                source_filter=source_filter,
                time_start=time_start,
                time_end=time_end,
                min_edge_weight=3,
                allowed_neighbor_types={"ORG", "PRODUCT", "EVENT", "STOCK"},
                max_neighbor_entities=3,
                max_extra_chunks=3,
            )

            for r in extra_rows:
                uid = r.get("chunk_uid")
                if not uid or uid in expanded:
                    continue

                emb = np.array(r["embedding"], dtype=np.float32)
                sim = cosine_sim(qvec, emb)

                ts = r.get("published_ts")
                if ts is None:
                    recency_weight = 0.85
                else:
                    age = max(0, now_ts - int(ts))
                    recency_weight = float(np.exp(-np.log(2) * age / half_life_seconds))

                edge_weight = float(r.get("edge_weight", 0))
                edge_bonus = min(edge_weight / 5.0, 1.0) * 0.05

                score = ((0.70 * sim) + (0.15 * recency_weight) + edge_bonus) * 0.60
                r["score"] = score
                r["expansion_kind"] = "cooccurrence"
                expanded[uid] = r

        # local same-article chunk expansion remains okay
        with driver.session() as session:
            for s in seeds:
                neighbors = session.run(
                    """
                    MATCH (c:Chunk {chunk_uid: $chunk_uid})
                    OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(n1:Chunk)
                    OPTIONAL MATCH (c)<-[:NEXT_CHUNK]-(p1:Chunk)
                    WITH collect(DISTINCT n1) + collect(DISTINCT p1) AS nearby
                    UNWIND nearby AS n
                    WITH n
                    WHERE n IS NOT NULL
                      AND ($source_filter IS NULL OR n.source = $source_filter)
                    RETURN collect(n) AS nearby
                    """,
                    {
                        "chunk_uid": s["chunk_uid"],
                        "source_filter": source_filter,
                    },
                ).single()

                if neighbors and neighbors["nearby"]:
                    for n in neighbors["nearby"]:
                        uid = n.get("chunk_uid")
                        if uid and uid not in expanded:
                            expanded[uid] = {
                                "chunk_uid": uid,
                                "text": n.get("text", ""),
                                "source": n.get("source"),
                                "title": s["title"],
                                "url": s["url"],
                                "article_id": n.get("article_id"),
                                "chunk_id": n.get("chunk_id"),
                                "score": s["score"] * 0.95,
                                "expansion_kind": "adjacent_chunk",
                            }

        expanded_values = list(expanded.values())
        
        expanded_values = [
            x for x in expanded_values
            if x.get("expansion_kind") == "direct" or x.get("score", 0.0) >= 0.35
        ]

        expanded_list = sorted(
            expanded_values,
            key=lambda x: x.get("score", 0),
            reverse=True,
        )

        market_texts = fetch_market_text_for_target(
            driver=driver,
            target=target,
            time_start=time_start,
            time_end=time_end,
        )

        return expanded_list[:expanded_k], market_texts, target

    # fallback path for non-single-entity queries
    return [], [], target


# Generation

SYSTEM_PROMPT_TEMPLATE = (
    "You are a finance assistant. Answer the user's question using only the "
    "provided context (news chunks and, when available, market data snippets). "
    "The articles in your database span from {date_min} to {date_max}. "
    "Use only the target entity and target ticker shown in the context. "
    "Do not use market data for any other company. "
    "Prioritize directly anchored news evidence over general background. "
    "When market data is present in the context, describe the price trend "
    "(direction, percentage change, high/low range) and connect it to relevant "
    "news developments. "
    "If the context is ambiguous, off-target, or insufficient, say so clearly. "
    "Do not output <think> tags, hidden reasoning, or chain-of-thought. "
    "Return only a concise final answer."
)

def build_structured_context(
    query: str,
    target: QueryTarget,
    retrieved_chunks: list[dict],
    market_texts: list[str],
) -> str:
    lines = []
    semantic_policy = get_semantic_policy_for_query(query)

    lines.append(f"TARGET ENTITY: {target.display_name or 'unknown'}")
    lines.append(f"TARGET CANONICAL: {target.entity_canonical or 'unknown'}")
    lines.append(f"TARGET TICKER: {target.ticker or 'unknown'}")
    lines.append(f"TARGET CONFIDENCE: {target.confidence:.2f}")
    lines.append(f"QUERY INTENT: {semantic_policy['intent']}")
    lines.append("")

    lines.append("PRIMARY NEWS EVIDENCE:")
    if not retrieved_chunks:
        lines.append("No directly anchored news chunks found.")
    else:
        for ch in retrieved_chunks:
            url = ch.get("url", "NO URL")
            title = ch.get("title", "NO TITLE")
            lines.append(f"[TITLE: {title}]")
            lines.append(f"[ARTICLE URL: {url}]")

            expansion_kind = ch.get("expansion_kind", "unknown")
            lines.append(f"[EVIDENCE TYPE: {expansion_kind}]")

            if ch.get("relation_type"):
                lines.append(f"[SEMANTIC RELATION: {ch['relation_type']}]")
            if ch.get("relation_confidence") is not None:
                try:
                    lines.append(f"[RELATION CONFIDENCE: {float(ch['relation_confidence']):.2f}]")
                except Exception:
                    pass
            if ch.get("relation_source_chunk_uid"):
                lines.append(f"[RELATION SOURCE CHUNK: {ch['relation_source_chunk_uid']}]")
            if ch.get("edge_weight") is not None:
                lines.append(f"[COOCCUR WEIGHT: {ch['edge_weight']}]")

            lines.append(ch.get("text", ""))
            lines.append("")

    lines.append("MARKET DATA:")
    if not market_texts:
        lines.append("No market data available.")
    else:
        for mt in market_texts:
            lines.append(mt)
            lines.append("")

    return "\n".join(lines)

def generate_answer(query: str, context: str, pipe, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
            ),
        },
    ]

    out = pipe(messages, max_new_tokens=512, do_sample=False)
    generated = out[0]["generated_text"]

    # Handle chat-template output
    if isinstance(generated, list):
        raw = generated[-1].get("content", "")
    else:
        raw = str(generated)

    cleaned = strip_think_tags(raw).strip()

    # Fallback: if think-tag stripping removed everything, keep raw text
    if not cleaned:
        cleaned = raw.strip()

    # Final fallback so you never print a blank answer
    if not cleaned:
        cleaned = "I could not generate a grounded answer from the retrieved context."

    return cleaned


TEST_QUERIES = [
    # Exact company names
    {"query": "What is the latest news on Google?", "entity": "google", "ticker": "GOOGL"},
    {"query": "What is the latest news on Nvidia?", "entity": "nvidia", "ticker": "NVDA"},
    {"query": "What is the latest news on Apple?", "entity": "apple", "ticker": "AAPL"},
    {"query": "What is the latest news on Visa?", "entity": "visa", "ticker": "V"},
    {"query": "What is the latest news on Microsoft?", "entity": "microsoft", "ticker": "MSFT"},
    {"query": "What is the latest news on Amazon?", "entity": "amazon", "ticker": "AMZN"},
    {"query": "What is the latest news on Meta?", "entity": "meta", "ticker": "META"},
    {"query": "What is the latest news on Oracle?", "entity": "oracle", "ticker": "ORCL"},

    # Official company names / aliases
    {"query": "What is the latest news on Alphabet?", "entity": "alphabet", "ticker": "GOOGL"},
    {"query": "What is the latest news on Meta Platforms?", "entity": "meta platforms", "ticker": "META"},
    {"query": "What is the latest news on Berkshire Hathaway?", "entity": "berkshire hathaway", "ticker": None},
    {"query": "What is the latest news on JPMorgan?", "entity": "jpmorgan", "ticker": "JPM"},

    # Direct ticker queries
    {"query": "What is the latest news on GOOGL?", "entity": "google", "ticker": "GOOGL"},
    {"query": "What is the latest news on NVDA?", "entity": "nvidia", "ticker": "NVDA"},
    {"query": "What is the latest news on AAPL?", "entity": "apple", "ticker": "AAPL"},
    {"query": "What is the latest news on V?", "entity": "visa", "ticker": "V"},

    # Time-filtered single-entity queries
    {"query": "What happened to Google this week?", "entity": "google", "ticker": "GOOGL"},
    {"query": "What happened to Nvidia this week?", "entity": "nvidia", "ticker": "NVDA"},
    {"query": "What happened to Apple this month?", "entity": "apple", "ticker": "AAPL"},

    # Source-filter flavored queries
    {"query": "What is the latest CNBC news on Google?", "entity": "google", "ticker": "GOOGL"},
    {"query": "What is the latest CNBC news on Nvidia?", "entity": "nvidia", "ticker": "NVDA"},

    # Lower-coverage / harder names
    {"query": "What is the latest news on AMD?", "entity": "amd", "ticker": "AMD"},
    {"query": "What is the latest news on IBM?", "entity": "ibm", "ticker": "IBM"},
    {"query": "What is the latest news on Costco?", "entity": "costco", "ticker": "COST"},
]

def normalize_name(x):
    if not x:
        return x
    return x.replace(" inc", "").replace(" corporation", "").strip()

def run_retrieval_eval(embed_model, driver, ticker_lookup):
    results = []

    for item in TEST_QUERIES:
        query = item["query"]

        target = resolve_query_target(query, ticker_lookup, driver)

        chunks = retrieve_target_anchored_chunks(
            driver=driver,
            target=target,
            source_filter=None,
            time_start=None,
            time_end=None,
        )

        market_texts = fetch_market_text_for_target(
            driver=driver,
            target=target,
            time_start=None,
            time_end=None,
        )

        result = {
            "query": query,
            "expected_entity": item["entity"],
            "expected_ticker": item["ticker"],
            "resolved_entity": target.entity_canonical,
            "resolved_ticker": target.ticker,
            "entity_ok": normalize_name(target.entity_canonical) == normalize_name(item["entity"]),
            "ticker_ok": target.ticker == item["ticker"] if item["ticker"] is not None else True,
            "anchored_chunk_count": len(chunks),
            "market_count": len(market_texts),
            "market_ok": (
                len(market_texts) <= 1 and
                (
                    len(market_texts) == 0
                    or item["ticker"] is None
                    or target.ticker == item["ticker"]
                )
            ),
        }

        results.append(result)

    return results

# Main loop

def main():
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Loading generation model...")
    pipe = pipeline("text-generation", model=GEN_MODEL_NAME)

    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    with driver.session() as session:
        date_range = session.run(
            """
            MATCH (a:Article)
            WHERE a.published IS NOT NULL
            RETURN min(a.published) AS earliest, max(a.published) AS latest
            """
        ).single()
    date_min = date_range["earliest"] or "unknown"
    date_max = date_range["latest"] or "unknown"
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        date_min=date_min, date_max=date_max
    )

    ticker_lookup = load_ticker_company_map(Path("ticker_company_map.csv"))

    # # Uncomment to run retrieval evaluation
    # eval_results = run_retrieval_eval(embed_model, driver, ticker_lookup)
    # print("\n=== RETRIEVAL EVAL RESULTS ===")
    # for r in eval_results:
    #     print(
    #         f"Query: {r['query']}\n"
    #         f"  expected_entity={r['expected_entity']} | resolved_entity={r['resolved_entity']} | entity_ok={r['entity_ok']}\n"
    #         f"  expected_ticker={r['expected_ticker']} | resolved_ticker={r['resolved_ticker']} | ticker_ok={r['ticker_ok']}\n"
    #         f"  anchored_chunk_count={r['anchored_chunk_count']} | market_count={r['market_count']} | market_ok={r['market_ok']}\n"
    #     )
    # print("=== END RETRIEVAL EVAL ===\n")

    print(f"\n--- TG-RAG Chatbot ready ---")
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

        source_filter = extract_source_filter(query)
        if source_filter:
            print(f"  [source filter: {source_filter}]")

        # Step 1: Temporal Query Decomposition
        sub_queries = decompose_query(query, pipe)
        if len(sub_queries) > 1:
            print(f"  [decomposed into {len(sub_queries)} sub-queries]")

        # Step 2: Three-layer retrieval per sub-query
        all_contexts: list[str] = []
        all_urls: list[str] = []
        for sq in sub_queries:
            ts_start = resolve_date(sq["time_start"])
            ts_end = resolve_date(sq["time_end"])

            if sq["time_start"] or sq["time_end"]:
                print(
                    f"  [time filter: "
                    f"{sq.get('time_start', '?')} -> "
                    f"{sq.get('time_end', '?')}]"
                )

            retrieved_chunks, market_texts, target = three_layer_retrieve(
                query=sq["query"],
                embed_model=embed_model,
                driver=driver,
                ticker_lookup=ticker_lookup,
                top_k=4, # prev 3
                expanded_k=8, #prev 6
                recency_half_life_days=5,
                source_filter=source_filter,
                time_start=ts_start,
                time_end=ts_end,
            )

            period_label = ""
            if sq.get("time_start") and sq.get("time_end"):
                period_label = f"[{sq['time_start']} to {sq['time_end']}] "

            context = build_structured_context(
                query=sq["query"],
                target=target,
                retrieved_chunks=retrieved_chunks,
                market_texts=market_texts,
            )

            print(context)

            all_contexts.append(period_label + context)

            for x in retrieved_chunks:
                url = x.get("url")
                if url and url not in all_urls:
                    all_urls.append(url)


        # Step 3: Single generation from merged context
        if not all_contexts:
            print("Assistant: No relevant chunks found.\n")
            continue

        context = "\n\n".join(all_contexts)
        sources_block = "Sources used:\n" + "\n".join(f"- {u}" for u in all_urls)
        print("\n--- CONTEXT BEING SENT TO LLM ---")
        print(context)
        print("--- END CONTEXT ---\n")
        final = generate_answer(query, context, pipe, system_prompt)
        elapsed = time.perf_counter() - t0
        print(f"\nAssistant: {final}")
        print(sources_block)
        print(f"  [{elapsed:.1f}s]\n")

    driver.close()


if __name__ == "__main__":
    main()