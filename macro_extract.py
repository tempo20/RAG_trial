"""
macro_extract.py - Macro event extraction from news chunks.

Loads chunks from SQLite that have not yet been processed, calls Claude
to extract structured macro events, stores the raw JSON, then normalises
into macro_events / macro_event_shock_types / macro_channels /
asset_impacts / evidence_spans tables.

Usage:
    python macro_extract.py                    # process all unprocessed chunks
    python macro_extract.py --limit 100        # process at most N chunks
    python macro_extract.py --reprocess        # reprocess failed runs only
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SQLITE_DB         = os.getenv("SQLITE_DB", "my_database.db")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEN_MODEL_NAME    = os.getenv("GEN_MODEL_NAME", "claude-haiku-4-5-20251001")

PROMPT_VERSION        = "v1"
SCHEMA_VERSION        = "v1"
MIN_TOKENS            = int(os.getenv("MACRO_MIN_TOKENS", "40"))
MAX_TOKENS_OUT        = int(os.getenv("MACRO_MAX_TOKENS_OUT", "4096"))
CHUNK_SCORE_THRESHOLD = int(os.getenv("MACRO_CHUNK_SCORE_THRESHOLD", "2"))

# ---------------------------------------------------------------------------
# Canonical vocabularies
# ---------------------------------------------------------------------------

SHOCK_TYPES = [
    "energy_supply_disruption",
    "commodity_supply_disruption",
    "geopolitical_risk_escalation",
    "central_bank_hawkish_shift",
    "central_bank_dovish_shift",
    "inflation_upside_surprise",
    "inflation_downside_surprise",
    "growth_upside_surprise",
    "growth_downside_surprise",
    "trade_restriction_escalation",
    "banking_stress",
    "credit_tightening",
    "fiscal_expansion",
    "fiscal_contraction",
    "currency_intervention",
    "sovereign_risk_event",
]

MACRO_CHANNELS = [
    "risk_off_flow",
    "liquidity_demand",
    "rate_differentials",
    "real_yield_shift",
    "inflation_expectations_shift",
    "growth_differentials",
    "terms_of_trade",
    "commodity_price_shock",
    "policy_expectations_repricing",
    "credit_spread_widening",
    "supply_disruption",
    "demand_repricing",
]

DIRECTIONS = ["up", "down", "mixed", "unclear"]
STRENGTHS  = ["weak", "moderate", "strong"]
HORIZONS   = ["intraday", "near_term", "medium_term", "long_term"]

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

MACRO_EXTRACTION_PROMPT = """\
You are a macro-economic analyst. Read the news excerpt below and extract \
all macro-economic events described or implied.

Return ONLY a JSON object (no markdown, no prose) with the following structure:

{{
  "events": [
    {{
      "event_type": "<one of: {shock_types}>",
      "summary": "<one sentence describing the event>",
      "region": "<country, region, or 'global'>",
      "time_horizon": "<one of: {horizons}>",
      "shock_types": ["<subset of: {shock_types}>"],
      "channels": [
        {{
          "channel_name": "<one of: {channels}>",
          "direction": "<one of: {directions}>",
          "strength": "<one of: {strengths}>"
        }}
      ],
      "asset_impacts": [
        {{
          "target_type": "<'ticker' | 'asset_class' | 'currency' | 'commodity'>",
          "target_id": "<ticker symbol, asset class name, currency code, or commodity name>",
          "direction": "<one of: {directions}>",
          "strength": "<one of: {strengths}>",
          "horizon": "<one of: {horizons}>",
          "rationale": "<brief explanation>"
        }}
      ],
      "evidence_spans": ["<exact quoted phrase from the text>"],
      "confidence": <float 0.0-1.0>
    }}
  ]
}}

If the text contains no macro-economic events, return {{"events": []}}.
Use ONLY the allowed enum values listed above — do not invent new labels.

NEWS EXCERPT:
{text}
"""

# ---------------------------------------------------------------------------
# Enum enforcement helpers
# ---------------------------------------------------------------------------

def _snap(value: str, allowed: list[str], label: str) -> str | None:
    """
    Return value if it is in allowed, otherwise find the closest match
    via difflib. Returns None if no close match found (score < 0.6).
    """
    if value in allowed:
        return value
    matches = difflib.get_close_matches(value, allowed, n=1, cutoff=0.6)
    if matches:
        print(f"  [enum] '{value}' snapped to '{matches[0]}' for {label}")
        return matches[0]
    print(f"  [enum] '{value}' has no close match in {label} — dropped")
    return None


def _enforce_enums(events: list[dict]) -> list[dict]:
    """Validate and snap all enum fields in a parsed events list."""
    clean = []
    for ev in events:
        # event_type / shock_types
        ev["event_type"] = _snap(ev.get("event_type", ""), SHOCK_TYPES, "SHOCK_TYPES") or ""
        ev["shock_types"] = [
            s for raw in ev.get("shock_types", [])
            if (s := _snap(raw, SHOCK_TYPES, "shock_types")) is not None
        ]
        ev["time_horizon"] = _snap(ev.get("time_horizon", ""), HORIZONS, "HORIZONS") or ""

        # channels
        clean_channels = []
        for ch in ev.get("channels", []):
            ch["channel_name"] = _snap(ch.get("channel_name", ""), MACRO_CHANNELS, "MACRO_CHANNELS")
            ch["direction"]    = _snap(ch.get("direction", ""), DIRECTIONS, "DIRECTIONS")
            ch["strength"]     = _snap(ch.get("strength", ""), STRENGTHS, "STRENGTHS")
            if all(ch.get(k) for k in ("channel_name", "direction", "strength")):
                clean_channels.append(ch)
        ev["channels"] = clean_channels

        # asset_impacts
        clean_impacts = []
        for imp in ev.get("asset_impacts", []):
            imp["direction"] = _snap(imp.get("direction", ""), DIRECTIONS, "DIRECTIONS")
            imp["strength"]  = _snap(imp.get("strength", ""), STRENGTHS, "STRENGTHS")
            imp["horizon"]   = _snap(imp.get("horizon", ""), HORIZONS, "HORIZONS")
            if imp.get("target_id") and all(imp.get(k) for k in ("direction", "strength", "horizon")):
                clean_impacts.append(imp)
        ev["asset_impacts"] = clean_impacts

        clean.append(ev)
    return clean

# ---------------------------------------------------------------------------
# Prefilter — zero-cost relevance screening before any API call
# ---------------------------------------------------------------------------

# Sources whose content is always macro-relevant — skip scoring entirely
HARD_INCLUDE_SOURCES: frozenset[str] = frozenset({
    "Federal Reserve",
    "BLS",
    "US Treasury",
    "OilPrice.com",
})

# High-signal macro terms; each hit scores +2
_HARD_MACRO_TERMS: list[str] = [
    "inflation", "deflation", "stagflation",
    "interest rate", "rate hike", "rate cut", "rate decision",
    "federal reserve", "central bank", "ecb", "bank of england", "boj", "pboc",
    "fed funds", "quantitative easing", "quantitative tightening",
    "gdp", "gross domestic product",
    "cpi", "pce", "consumer price index",
    "yield", "treasury yield", "bond yield", "10-year yield",
    "tariff", "trade war", "trade deficit", "trade surplus",
    "crude oil", "brent", "wti", "opec",
    "fiscal policy", "monetary policy",
    "recession", "economic growth", "economic slowdown",
    "unemployment", "nonfarm payroll", "jobless claims",
    "exchange rate", "dollar index", "dxy",
    "sanctions", "export controls",
]

# Directional/causal verbs that amplify macro signal; each hit scores +1
_DIRECTIONAL_TERMS: list[str] = [
    "surged", "plunged", "spiked", "tumbled",
    "tightened", "eased", "cut rates", "hiked rates",
    "accelerated", "contracted", "expanded",
    "widened", "narrowed",
]

# Company/earnings/product noise that suppresses score; each hit scores -2
_NOISE_TERMS: list[str] = [
    "quarterly earnings", "earnings per share", "eps beat",
    "product launch", "iphone", "android",
    "stock buyback", "share repurchase",
    "advertising revenue", "monthly active users",
]

# Article title keywords that trigger hard include regardless of source
_TITLE_MACRO_KEYWORDS: list[str] = [
    "inflation", "interest rate", "fed ", "federal reserve",
    "gdp", "tariff", "recession", "yield", "unemployment",
    "cpi", "opec", "oil price", "monetary", "fiscal",
    "treasury", "deficit", "currency", "trade war", "sanctions",
    "rate hike", "rate cut", "central bank",
]


def _article_is_hard_include(source: str, title: str) -> bool:
    """Return True if every chunk from this article should always be sent to Claude."""
    if source in HARD_INCLUDE_SOURCES:
        return True
    title_lower = (title or "").lower()
    return any(kw in title_lower for kw in _TITLE_MACRO_KEYWORDS)


def _chunk_macro_score(text: str) -> int:
    """
    Score a chunk for macro relevance without any API call.
    Each hard macro term:   +2
    Each directional term:  +1
    Each noise term:        -2
    """
    low = text.lower()
    score = 0
    score += sum(2 for t in _HARD_MACRO_TERMS if t in low)
    score += sum(1 for t in _DIRECTIONAL_TERMS if t in low)
    score -= sum(2 for t in _NOISE_TERMS if t in low)
    return score


def _should_process_chunk(chunk: dict) -> bool:
    """Return True if this chunk deserves a Claude call."""
    if _article_is_hard_include(chunk.get("source", ""), chunk.get("title", "")):
        return True
    return _chunk_macro_score(chunk["text"]) >= CHUNK_SCORE_THRESHOLD


# ---------------------------------------------------------------------------
# Claude client
# ---------------------------------------------------------------------------

def _build_client() -> Any:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in environment.")
    try:
        anthropic = importlib.import_module("anthropic")
    except ImportError as exc:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic") from exc
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _repair_json(raw: str) -> str | None:
    """
    Attempt to salvage valid JSON from a truncated response by finding
    the last position where the brace depth returns to zero.
    Returns the repaired string, or None if no complete object was found.
    """
    depth = 0
    last_valid = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(raw):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_valid = i + 1
    return raw[:last_valid] if last_valid else None


def _call_claude(client: Any, chunk_text: str) -> tuple[bool, str, str | None]:
    """
    Call Claude and return (success, raw_json, error_text).
    Uses assistant prefill of '{' to force a JSON object response.
    Falls back to _repair_json if the response is truncated mid-JSON.
    """
    prompt = MACRO_EXTRACTION_PROMPT.format(
        shock_types=", ".join(SHOCK_TYPES),
        horizons=", ".join(HORIZONS),
        channels=", ".join(MACRO_CHANNELS),
        directions=", ".join(DIRECTIONS),
        strengths=", ".join(STRENGTHS),
        text=chunk_text,
    )
    try:
        response = client.messages.create(
            model=GEN_MODEL_NAME,
            max_tokens=MAX_TOKENS_OUT,
            temperature=0,
            messages=[
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": "{"},
            ],
        )
        raw_json = "{" + response.content[0].text

        # Try parsing as-is first
        try:
            json.loads(raw_json)
            return True, raw_json, None
        except json.JSONDecodeError:
            pass

        # Response was likely truncated — attempt repair
        repaired = _repair_json(raw_json)
        if repaired:
            try:
                json.loads(repaired)
                print("  [repair] truncated JSON salvaged")
                return True, repaired, None
            except json.JSONDecodeError:
                pass

        return False, "", f"json_decode_error: could not parse or repair response"

    except Exception as exc:
        return False, "", str(exc)

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_unprocessed_chunks(
    conn: sqlite3.Connection,
    reprocess_failed: bool = False,
) -> list[dict]:
    """
    Return chunks that have no macro_extraction_runs row (or only failed
    rows if reprocess_failed=True).  Also pulls article title and source so
    the prefilter can use them without an extra query.
    """
    if reprocess_failed:
        sql = """
            SELECT c.chunk_id, c.article_id, c.text, c.token_count,
                   a.title, a.source
            FROM chunks c
            JOIN articles a ON a.article_id = c.article_id
            LEFT JOIN macro_extraction_runs r
                   ON r.chunk_id = c.chunk_id AND r.success = 1
            WHERE r.run_id IS NULL
              AND c.token_count >= ?
        """
    else:
        sql = """
            SELECT c.chunk_id, c.article_id, c.text, c.token_count,
                   a.title, a.source
            FROM chunks c
            JOIN articles a ON a.article_id = c.article_id
            LEFT JOIN macro_extraction_runs r ON r.chunk_id = c.chunk_id
            WHERE r.run_id IS NULL
              AND c.token_count >= ?
        """
    rows = conn.execute(sql, (MIN_TOKENS,)).fetchall()
    return [
        {
            "chunk_id":    r[0],
            "article_id":  r[1],
            "text":        r[2],
            "token_count": r[3],
            "title":       r[4] or "",
            "source":      r[5] or "",
        }
        for r in rows
    ]


def _write_run(
    conn: sqlite3.Connection,
    run_id: str,
    chunk: dict,
    success: bool,
    raw_json: str | None,
    error_text: str | None,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO macro_extraction_runs "
        "(run_id, article_id, chunk_id, model_provider, model_name, "
        "prompt_version, schema_version, created_at, success, raw_json, error_text) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            chunk["article_id"],
            chunk["chunk_id"],
            "anthropic",
            GEN_MODEL_NAME,
            PROMPT_VERSION,
            SCHEMA_VERSION,
            _now_utc(),
            1 if success else 0,
            raw_json or None,
            error_text,
        ),
    )
    conn.commit()


def _write_normalized(conn: sqlite3.Connection, run_id: str, chunk: dict, events: list[dict]) -> None:
    """Write normalized macro rows for one run."""
    for idx, ev in enumerate(events):
        macro_event_id = _md5(f"{run_id}::{idx}")

        # macro_events
        conn.execute(
            "INSERT OR IGNORE INTO macro_events "
            "(macro_event_id, run_id, article_id, chunk_id, event_type, summary, "
            "region, time_horizon, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                macro_event_id,
                run_id,
                chunk["article_id"],
                chunk["chunk_id"],
                ev.get("event_type", ""),
                ev.get("summary", ""),
                ev.get("region", ""),
                ev.get("time_horizon", ""),
                ev.get("confidence"),
            ),
        )

        # macro_event_shock_types
        for shock in ev.get("shock_types", []):
            conn.execute(
                "INSERT OR IGNORE INTO macro_event_shock_types (macro_event_id, shock_type) VALUES (?, ?)",
                (macro_event_id, shock),
            )

        # macro_channels
        for ch in ev.get("channels", []):
            channel_id = _md5(f"{macro_event_id}::{ch['channel_name']}")
            conn.execute(
                "INSERT OR IGNORE INTO macro_channels "
                "(macro_channel_id, macro_event_id, channel_name, direction, strength, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    channel_id,
                    macro_event_id,
                    ch["channel_name"],
                    ch["direction"],
                    ch["strength"],
                    None,
                ),
            )

        # asset_impacts
        for imp in ev.get("asset_impacts", []):
            impact_id = _md5(f"{macro_event_id}::{imp.get('target_id', '')}")
            conn.execute(
                "INSERT OR IGNORE INTO asset_impacts "
                "(impact_id, macro_event_id, target_type, target_id, direction, "
                "strength, horizon, confidence, rationale) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    impact_id,
                    macro_event_id,
                    imp.get("target_type", ""),
                    imp.get("target_id", ""),
                    imp["direction"],
                    imp["strength"],
                    imp["horizon"],
                    ev.get("confidence"),
                    imp.get("rationale", ""),
                ),
            )

        # evidence_spans
        for span in ev.get("evidence_spans", []):
            evidence_id = _md5(f"{run_id}::{span}")
            conn.execute(
                "INSERT OR IGNORE INTO evidence_spans "
                "(evidence_id, run_id, article_id, chunk_id, parent_kind, parent_id, evidence_text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    evidence_id,
                    run_id,
                    chunk["article_id"],
                    chunk["chunk_id"],
                    "macro_event",
                    macro_event_id,
                    span,
                ),
            )

    conn.commit()

# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def run_extraction(
    db_path: str = SQLITE_DB,
    limit: int | None = None,
    reprocess_failed: bool = False,
) -> None:
    client = _build_client()
    from create_sql_db import connect_sqlite
    conn = connect_sqlite(db_path)

    chunks = _load_unprocessed_chunks(conn, reprocess_failed=reprocess_failed)
    if limit:
        chunks = chunks[:limit]

    total = len(chunks)

    # Prefilter — fast, zero-cost pass before any API call
    surviving = [c for c in chunks if _should_process_chunk(c)]
    prefiltered_out = total - len(surviving)
    print(
        f"[macro_extract] {total} unprocessed chunks — "
        f"{prefiltered_out} skipped by prefilter, "
        f"{len(surviving)} sent to Claude"
    )

    ok = failed = skipped_empty = 0

    for i, chunk in enumerate(surviving, 1):
        print(f"  [{i}/{len(surviving)}] chunk {chunk['chunk_id'][:16]}...", end=" ")

        run_id = str(uuid.uuid4())
        success, raw_json, error_text = _call_claude(client, chunk["text"])

        _write_run(conn, run_id, chunk, success, raw_json if success else None, error_text)

        if not success:
            print(f"FAILED: {error_text}")
            failed += 1
            continue

        try:
            parsed = json.loads(raw_json)
            events: list[dict] = parsed.get("events", [])
        except (json.JSONDecodeError, AttributeError) as exc:
            print(f"PARSE_ERROR: {exc}")
            failed += 1
            continue

        if not events:
            print("no events")
            skipped_empty += 1
            continue

        events = _enforce_enums(events)
        _write_normalized(conn, run_id, chunk, events)
        print(f"{len(events)} event(s)")
        ok += 1

    conn.close()
    print(
        f"\n[macro_extract] done — {ok} chunks with events, "
        f"{skipped_empty} empty, {failed} failed, "
        f"{prefiltered_out} prefiltered (no API call)"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Macro event extraction from news chunks")
    parser.add_argument("--db", default=SQLITE_DB, help="Path to SQLite DB")
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to process")
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess chunks that previously failed (success=0)",
    )
    args = parser.parse_args()
    run_extraction(db_path=args.db, limit=args.limit, reprocess_failed=args.reprocess)
