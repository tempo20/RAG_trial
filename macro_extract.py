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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SQLITE_DB         = os.getenv("SQLITE_DB", "my_database.db")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEN_MODEL_NAME    = os.getenv("GEN_MODEL_NAME", "claude-haiku-4-5-20251001")
PROMPT_TEMPLATES_PATH = Path(os.getenv("PROMPT_TEMPLATES_PATH", "prompt_templates.json"))

PROMPT_VERSION              = "v2"
SCHEMA_VERSION              = "v1"
MIN_TOKENS                  = int(os.getenv("MACRO_MIN_TOKENS", "40"))
MAX_TOKENS_OUT              = int(os.getenv("MACRO_MAX_TOKENS_OUT", "1200"))
CHUNK_SCORE_THRESHOLD       = int(os.getenv("MACRO_CHUNK_SCORE_THRESHOLD", "3"))
MACRO_MAX_CHUNKS_PER_ARTICLE = int(os.getenv("MACRO_MAX_CHUNKS_PER_ARTICLE", "2"))

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

DEFAULT_MACRO_EXTRACTION_PROMPT = """\
You are a macro-economic analyst. Read the news excerpt below and extract \
macro-economic events that are explicitly stated or directly and unambiguously \
supported by the text.

Rules:
- Extract AT MOST 2 events. If you find more, keep only the 2 most significant.
- Include an event ONLY if the text contains clear, direct evidence for it.
- Do NOT infer weak, speculative, or ambiguous implications. If in doubt, omit.
- Focus on the single dominant macro mechanism per event.
- For evidence_spans: include 1-2 short phrases (under 15 words each) copied \
verbatim from the text that directly support the event. Do not pad with extra quotes.
- Keep asset_impacts to the 1-2 most directly affected assets.
- Keep channels to the 1-2 most directly relevant transmission mechanisms.
- Set confidence high (>= 0.7) only when the text is unambiguous.

Return ONLY a JSON object (no markdown, no prose):

{{
  "events": [
    {{
      "event_type": "<one of: {shock_types}>",
      "summary": "<one concise sentence>",
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
          "target_id": "<symbol or name>",
          "direction": "<one of: {directions}>",
          "strength": "<one of: {strengths}>",
          "horizon": "<one of: {horizons}>",
          "rationale": "<one short phrase>"
        }}
      ],
      "evidence_spans": ["<short verbatim phrase from text>"],
      "confidence": <float 0.0-1.0>
    }}
  ]
}}

If the text contains no clear macro-economic events, return {{"events": []}}.
Use ONLY the allowed enum values listed — do not invent new labels.

NEWS EXCERPT:
{text}
"""


def _coerce_template_value(value: Any, key: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "\n".join(value)
    raise ValueError(
        f"Prompt template '{key}' must be either a string or a list of strings."
    )


def _load_macro_extraction_prompt(path: Path) -> str:
    """
    Load the macro extraction prompt from prompt_templates.json.
    Falls back to DEFAULT_MACRO_EXTRACTION_PROMPT when unavailable.
    """
    if not path.exists():
        return DEFAULT_MACRO_EXTRACTION_PROMPT
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_MACRO_EXTRACTION_PROMPT

    value = data.get("MACRO_EXTRACTION_PROMPT")
    if value is None:
        return DEFAULT_MACRO_EXTRACTION_PROMPT
    return _coerce_template_value(value, "MACRO_EXTRACTION_PROMPT")


MACRO_EXTRACTION_PROMPT = _load_macro_extraction_prompt(PROMPT_TEMPLATES_PATH)

# ---------------------------------------------------------------------------
# Enum enforcement helpers
# ---------------------------------------------------------------------------

def _snap(value: str, allowed: list[str], label: str) -> tuple[str | None, str]:
    """
    Return value if it is in allowed, otherwise find the closest match
    via difflib. Returns None if no close match found (score < 0.6).
    """
    if value in allowed:
        return value, "kept"
    matches = difflib.get_close_matches(value, allowed, n=1, cutoff=0.6)
    if matches:
        print(f"  [enum] '{value}' snapped to '{matches[0]}' for {label}")
        return matches[0], "snapped"
    print(f"  [enum] '{value}' has no close match in {label} — dropped")
    return None, "dropped"


def _enforce_enums(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """Validate and snap all enum fields in a parsed events list."""
    clean = []
    audits: list[dict] = []
    for idx, ev in enumerate(events):
        # event_type / shock_types
        raw_event_type = ev.get("event_type", "")
        normalized_event_type, action = _snap(raw_event_type, SHOCK_TYPES, "SHOCK_TYPES")
        audits.append(
            {
                "macro_event_index": idx,
                "parent_kind": "macro_event",
                "field_label": "event_type",
                "raw_value": raw_event_type,
                "normalized_value": normalized_event_type,
                "action": action,
            }
        )
        ev["event_type"] = normalized_event_type or ""

        clean_shocks = []
        for raw in ev.get("shock_types", []):
            normalized, action = _snap(raw, SHOCK_TYPES, "shock_types")
            audits.append(
                {
                    "macro_event_index": idx,
                    "parent_kind": "macro_event",
                    "field_label": "shock_types",
                    "raw_value": raw,
                    "normalized_value": normalized,
                    "action": action,
                }
            )
            if normalized is not None:
                clean_shocks.append(normalized)
        ev["shock_types"] = clean_shocks

        raw_horizon = ev.get("time_horizon", "")
        normalized_horizon, action = _snap(raw_horizon, HORIZONS, "HORIZONS")
        audits.append(
            {
                "macro_event_index": idx,
                "parent_kind": "macro_event",
                "field_label": "time_horizon",
                "raw_value": raw_horizon,
                "normalized_value": normalized_horizon,
                "action": action,
            }
        )
        ev["time_horizon"] = normalized_horizon or ""

        # channels
        clean_channels = []
        for ch in ev.get("channels", []):
            raw_name = ch.get("channel_name", "")
            normalized_name, action = _snap(raw_name, MACRO_CHANNELS, "MACRO_CHANNELS")
            audits.append(
                {
                    "macro_event_index": idx,
                    "parent_kind": "channel",
                    "field_label": "channel_name",
                    "raw_value": raw_name,
                    "normalized_value": normalized_name,
                    "action": action,
                }
            )
            ch["channel_name"] = normalized_name

            raw_direction = ch.get("direction", "")
            normalized_direction, action = _snap(raw_direction, DIRECTIONS, "DIRECTIONS")
            audits.append(
                {
                    "macro_event_index": idx,
                    "parent_kind": "channel",
                    "field_label": "direction",
                    "raw_value": raw_direction,
                    "normalized_value": normalized_direction,
                    "action": action,
                }
            )
            ch["direction"] = normalized_direction

            raw_strength = ch.get("strength", "")
            normalized_strength, action = _snap(raw_strength, STRENGTHS, "STRENGTHS")
            audits.append(
                {
                    "macro_event_index": idx,
                    "parent_kind": "channel",
                    "field_label": "strength",
                    "raw_value": raw_strength,
                    "normalized_value": normalized_strength,
                    "action": action,
                }
            )
            ch["strength"] = normalized_strength
            if all(ch.get(k) for k in ("channel_name", "direction", "strength")):
                clean_channels.append(ch)
        ev["channels"] = clean_channels

        # asset_impacts
        clean_impacts = []
        for imp in ev.get("asset_impacts", []):
            raw_direction = imp.get("direction", "")
            normalized_direction, action = _snap(raw_direction, DIRECTIONS, "DIRECTIONS")
            audits.append(
                {
                    "macro_event_index": idx,
                    "parent_kind": "asset_impact",
                    "field_label": "direction",
                    "raw_value": raw_direction,
                    "normalized_value": normalized_direction,
                    "action": action,
                }
            )
            imp["direction"] = normalized_direction

            raw_strength = imp.get("strength", "")
            normalized_strength, action = _snap(raw_strength, STRENGTHS, "STRENGTHS")
            audits.append(
                {
                    "macro_event_index": idx,
                    "parent_kind": "asset_impact",
                    "field_label": "strength",
                    "raw_value": raw_strength,
                    "normalized_value": normalized_strength,
                    "action": action,
                }
            )
            imp["strength"] = normalized_strength

            raw_horizon = imp.get("horizon", "")
            normalized_horizon, action = _snap(raw_horizon, HORIZONS, "HORIZONS")
            audits.append(
                {
                    "macro_event_index": idx,
                    "parent_kind": "asset_impact",
                    "field_label": "horizon",
                    "raw_value": raw_horizon,
                    "normalized_value": normalized_horizon,
                    "action": action,
                }
            )
            imp["horizon"] = normalized_horizon
            if imp.get("target_id") and all(imp.get(k) for k in ("direction", "strength", "horizon")):
                clean_impacts.append(imp)
        ev["asset_impacts"] = clean_impacts

        clean.append(ev)
    return clean, audits

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

def _count_term_hits(text: str, terms: list[str]) -> int:
    low = text.lower()
    return sum(1 for t in terms if t in low)

def _chunk_macro_score(text: str) -> int:
    low = text.lower()
    hard_hits = _count_term_hits(low, _HARD_MACRO_TERMS)
    directional_hits = _count_term_hits(low, _DIRECTIONAL_TERMS)
    noise_hits = _count_term_hits(low, _NOISE_TERMS)
    return 2 * hard_hits + directional_hits - 2 * noise_hits


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


RETRY_QUEUE_NAME = "retry_failed"
REVIEW_QUEUE_NAME = "review_suspicious"


def _raw_excerpt(raw_json: str | None, max_chars: int = 400) -> str | None:
    if not raw_json:
        return None
    compact = " ".join(raw_json.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _load_chunks_by_ids(conn: sqlite3.Connection, chunk_ids: list[str]) -> list[dict]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = conn.execute(
        f"""
        SELECT c.chunk_id, c.article_id, c.text, c.token_count, a.title, a.source
        FROM chunks c
        JOIN articles a ON a.article_id = c.article_id
        WHERE c.chunk_id IN ({placeholders})
        ORDER BY c.published_date DESC, c.chunk_index ASC
        """,
        chunk_ids,
    ).fetchall()
    return [
        {
            "chunk_id": r[0],
            "article_id": r[1],
            "text": r[2],
            "token_count": r[3],
            "title": r[4] or "",
            "source": r[5] or "",
        }
        for r in rows
    ]


def _write_processing_audit(
    conn: sqlite3.Connection,
    *,
    chunk: dict,
    stage: str,
    status: str,
    run_id: str | None = None,
    failure_reason: str | None = None,
    queue_name: str | None = None,
    chunk_macro_score: int | None = None,
    was_hard_include: bool = False,
    event_count: int | None = None,
    suspicious: bool = False,
    review_reasons: list[str] | None = None,
    raw_json: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO macro_processing_audit (
            audit_id, run_id, article_id, chunk_id, created_at, stage, status,
            failure_reason, queue_name, chunk_macro_score, was_hard_include,
            event_count, suspicious, review_reasons_json, raw_response_excerpt
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            run_id,
            chunk["article_id"],
            chunk["chunk_id"],
            _now_utc(),
            stage,
            status,
            failure_reason,
            queue_name,
            chunk_macro_score,
            1 if was_hard_include else 0,
            event_count,
            1 if suspicious else 0,
            json.dumps(review_reasons or []),
            _raw_excerpt(raw_json),
        ),
    )
    conn.commit()


def _write_enum_audits(conn: sqlite3.Connection, run_id: str, audits: list[dict]) -> None:
    if not audits:
        return
    rows = [
        (
            str(uuid.uuid4()),
            run_id,
            audit.get("macro_event_index"),
            audit["parent_kind"],
            audit["field_label"],
            audit.get("raw_value"),
            audit.get("normalized_value"),
            audit["action"],
            _now_utc(),
        )
        for audit in audits
    ]
    conn.executemany(
        """
        INSERT INTO macro_enum_audit (
            audit_id, run_id, macro_event_index, parent_kind, field_label,
            raw_value, normalized_value, action, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def _review_reasons_for_output(
    chunk: dict,
    *,
    chunk_score: int,
    was_hard_include: bool,
    events: list[dict],
    enum_audits: list[dict],
) -> list[str]:
    reasons: list[str] = []
    if not events and (was_hard_include or chunk_score >= CHUNK_SCORE_THRESHOLD + 3):
        reasons.append("suspicious_empty_output")
    if any(audit.get("action") == "dropped" for audit in enum_audits):
        reasons.append("enum_value_dropped")
    if any(audit.get("action") == "snapped" for audit in enum_audits):
        reasons.append("enum_value_snapped")
    if events and any(not ev.get("evidence_spans") for ev in events):
        reasons.append("missing_evidence_spans")
    if events and any(not ev.get("asset_impacts") for ev in events):
        reasons.append("event_without_asset_impacts")
    if events and any(not ev.get("channels") for ev in events):
        reasons.append("event_without_channels")
    return reasons


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
    chunk_ids: list[str] | None = None,
) -> None:
    client = _build_client()
    from create_sql_db import create_database, connect_sqlite
    create_database(db_path)
    conn = connect_sqlite(db_path)

    chunks = _load_chunks_by_ids(conn, chunk_ids) if chunk_ids else _load_unprocessed_chunks(
        conn,
        reprocess_failed=reprocess_failed,
    )
    if limit:
        chunks = chunks[:limit]

    total = len(chunks)

    # Prefilter — fast, zero-cost pass before any API call
    surviving = []
    for chunk in chunks:
        was_hard_include = _article_is_hard_include(chunk.get("source", ""), chunk.get("title", ""))
        chunk_score = _chunk_macro_score(chunk["text"])
        chunk["_hard_macro_hits"] = _count_term_hits(chunk["text"], _HARD_MACRO_TERMS)
        chunk["_directional_hits"] = _count_term_hits(chunk["text"], _DIRECTIONAL_TERMS)
        chunk["_noise_hits"] = _count_term_hits(chunk["text"], _NOISE_TERMS)
        chunk["_title_macro_hits"] = _count_term_hits(chunk.get("title", ""), _TITLE_MACRO_KEYWORDS)
        should_process = was_hard_include or chunk_score >= CHUNK_SCORE_THRESHOLD
        chunk["_macro_score"] = chunk_score
        chunk["_was_hard_include"] = was_hard_include
        if should_process:
            surviving.append(chunk)
        else:
            _write_processing_audit(
                conn,
                chunk=chunk,
                stage="prefilter",
                status="prefilter_skipped",
                chunk_macro_score=chunk_score,
                was_hard_include=was_hard_include,
                review_reasons=[],
            )
    prefiltered_out = total - len(surviving)

    # ------------------------------------------------------------------
    # Article-level redundancy control — keep only the top-N scoring
    # chunks per article to avoid paying for redundant API calls across
    # chunks of the same article.
    # ------------------------------------------------------------------
    _article_buckets: dict[str, list[dict]] = defaultdict(list)
    for chunk in surviving:
        _article_buckets[chunk["article_id"]].append(chunk)

    dedup_surviving: list[dict] = []
    article_dedup_skipped = 0
    for _art_chunks in _article_buckets.values():
        article_is_hard_include = any(bool(c.get("_was_hard_include")) for c in _art_chunks)

        if article_is_hard_include:
            dedup_surviving.extend(_art_chunks)
            continue

        _art_chunks.sort(
            key=lambda c: (
                c.get("_macro_score", 0),
                c.get("_title_macro_hits", 0),
                c.get("_hard_macro_hits", 0),
                c.get("_directional_hits", 0),
            ),
            reverse=True,
        )

        kept = _art_chunks[:MACRO_MAX_CHUNKS_PER_ARTICLE]
        dropped = _art_chunks[MACRO_MAX_CHUNKS_PER_ARTICLE:]
        dedup_surviving.extend(kept)

        for sk in dropped:
            article_dedup_skipped += 1
            _write_processing_audit(
                conn,
                chunk=sk,
                stage="article_dedup",
                status="article_dedup_skipped",
                chunk_macro_score=sk.get("_macro_score"),
                was_hard_include=bool(sk.get("_was_hard_include")),
                review_reasons=["article_level_redundancy_skip"],
            )
        dedup_surviving.extend(kept)
        for sk in dropped:
            article_dedup_skipped += 1
            _write_processing_audit(
                conn,
                chunk=sk,
                stage="article_dedup",
                status="article_dedup_skipped",
                chunk_macro_score=sk.get("_macro_score"),
                was_hard_include=bool(sk.get("_was_hard_include")),
                review_reasons=["article_level_redundancy_skip"],
            )
    surviving = dedup_surviving

    print(
        f"[macro_extract] {total} unprocessed chunks — "
        f"{prefiltered_out} skipped by prefilter, "
        f"{article_dedup_skipped} skipped by article dedup, "
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
            _write_processing_audit(
                conn,
                chunk=chunk,
                run_id=run_id,
                stage="model_call",
                status="api_failed",
                failure_reason=error_text,
                queue_name=RETRY_QUEUE_NAME,
                chunk_macro_score=chunk.get("_macro_score"),
                was_hard_include=bool(chunk.get("_was_hard_include")),
                suspicious=True,
                review_reasons=["api_failed"],
                raw_json=raw_json,
            )
            failed += 1
            continue

        try:
            parsed = json.loads(raw_json)
            events: list[dict] = parsed.get("events", [])
        except (json.JSONDecodeError, AttributeError) as exc:
            print(f"PARSE_ERROR: {exc}")
            _write_processing_audit(
                conn,
                chunk=chunk,
                run_id=run_id,
                stage="parse",
                status="parse_failed",
                failure_reason=str(exc),
                queue_name=RETRY_QUEUE_NAME,
                chunk_macro_score=chunk.get("_macro_score"),
                was_hard_include=bool(chunk.get("_was_hard_include")),
                suspicious=True,
                review_reasons=["parse_failed"],
                raw_json=raw_json,
            )
            failed += 1
            continue

        if not events:
            print("no events")
            review_reasons = _review_reasons_for_output(
                chunk,
                chunk_score=int(chunk.get("_macro_score") or 0),
                was_hard_include=bool(chunk.get("_was_hard_include")),
                events=[],
                enum_audits=[],
            )
            suspicious = bool(review_reasons)
            _write_processing_audit(
                conn,
                chunk=chunk,
                run_id=run_id,
                stage="normalize",
                status="empty_success",
                queue_name=REVIEW_QUEUE_NAME if suspicious else None,
                chunk_macro_score=chunk.get("_macro_score"),
                was_hard_include=bool(chunk.get("_was_hard_include")),
                event_count=0,
                suspicious=suspicious,
                review_reasons=review_reasons,
                raw_json=raw_json,
            )
            skipped_empty += 1
            continue

        events, enum_audits = _enforce_enums(events)
        _write_enum_audits(conn, run_id, enum_audits)
        review_reasons = _review_reasons_for_output(
            chunk,
            chunk_score=int(chunk.get("_macro_score") or 0),
            was_hard_include=bool(chunk.get("_was_hard_include")),
            events=events,
            enum_audits=enum_audits,
        )
        suspicious = bool(review_reasons)
        _write_normalized(conn, run_id, chunk, events)
        _write_processing_audit(
            conn,
            chunk=chunk,
            run_id=run_id,
            stage="normalize",
            status="events_written",
            queue_name=REVIEW_QUEUE_NAME if suspicious else None,
            chunk_macro_score=chunk.get("_macro_score"),
            was_hard_include=bool(chunk.get("_was_hard_include")),
            event_count=len(events),
            suspicious=suspicious,
            review_reasons=review_reasons,
            raw_json=raw_json,
        )
        print(f"{len(events)} event(s)")
        ok += 1

    conn.close()
    print(
        f"\n[macro_extract] done — {ok} chunks with events, "
        f"{skipped_empty} empty, {failed} failed, "
        f"{prefiltered_out} prefiltered, "
        f"{article_dedup_skipped} article-deduped (no API call)"
    )


def _queue_rows(
    conn: sqlite3.Connection,
    *,
    queue_name: str,
    limit: int = 20,
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            audit_id,
            run_id,
            article_id,
            chunk_id,
            stage,
            status,
            failure_reason,
            queue_name,
            chunk_macro_score,
            was_hard_include,
            event_count,
            suspicious,
            review_reasons_json,
            raw_response_excerpt,
            created_at
        FROM macro_processing_audit
        WHERE queue_name = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (queue_name, limit),
    ).fetchall()


def print_queue(db_path: str = SQLITE_DB, *, queue_name: str, limit: int = 20) -> None:
    from create_sql_db import create_database, connect_sqlite

    create_database(db_path)
    conn = connect_sqlite(db_path)
    conn.row_factory = sqlite3.Row
    rows = _queue_rows(conn, queue_name=queue_name, limit=limit)
    print(f"[macro_extract] queue={queue_name} rows={len(rows)}")
    for row in rows:
        reasons = json.loads(row["review_reasons_json"] or "[]")
        print(
            f"- chunk={row['chunk_id']} run={row['run_id'] or '—'} status={row['status']} "
            f"stage={row['stage']} score={row['chunk_macro_score']} reasons={','.join(reasons) or '—'}"
        )
        if row["failure_reason"]:
            print(f"  failure: {row['failure_reason']}")
        if row["raw_response_excerpt"]:
            print(f"  excerpt: {row['raw_response_excerpt']}")
    conn.close()


def inspect_run(
    *,
    db_path: str = SQLITE_DB,
    run_id: str | None = None,
    chunk_id: str | None = None,
) -> None:
    if not run_id and not chunk_id:
        raise ValueError("inspect_run requires run_id or chunk_id")

    from create_sql_db import connect_sqlite
    from create_sql_db import create_database

    create_database(db_path)
    conn = connect_sqlite(db_path)
    conn.row_factory = sqlite3.Row
    if run_id:
        run = conn.execute(
            """
            SELECT run_id, article_id, chunk_id, model_provider, model_name, created_at,
                   success, raw_json, error_text
            FROM macro_extraction_runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
    else:
        run = conn.execute(
            """
            SELECT run_id, article_id, chunk_id, model_provider, model_name, created_at,
                   success, raw_json, error_text
            FROM macro_extraction_runs
            WHERE chunk_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (chunk_id,),
        ).fetchone()
    if not run:
        print("[macro_extract] no matching run found")
        conn.close()
        return

    print(
        f"[macro_extract] run={run['run_id']} chunk={run['chunk_id']} "
        f"success={run['success']} created_at={run['created_at']}"
    )
    if run["error_text"]:
        print(f"  error: {run['error_text']}")
    audit_rows = conn.execute(
        """
        SELECT stage, status, queue_name, suspicious, review_reasons_json, raw_response_excerpt, created_at
        FROM macro_processing_audit
        WHERE run_id = ?
        ORDER BY created_at DESC
        """,
        (run["run_id"],),
    ).fetchall()
    for row in audit_rows:
        reasons = json.loads(row["review_reasons_json"] or "[]")
        print(
            f"  audit stage={row['stage']} status={row['status']} queue={row['queue_name'] or '—'} "
            f"suspicious={row['suspicious']} reasons={','.join(reasons) or '—'}"
        )
        if row["raw_response_excerpt"]:
            print(f"    excerpt: {row['raw_response_excerpt']}")

    enum_rows = conn.execute(
        """
        SELECT macro_event_index, parent_kind, field_label, raw_value, normalized_value, action
        FROM macro_enum_audit
        WHERE run_id = ?
        ORDER BY macro_event_index, parent_kind, field_label
        """,
        (run["run_id"],),
    ).fetchall()
    if enum_rows:
        print("  enum_audit:")
        for row in enum_rows[:20]:
            print(
                f"    idx={row['macro_event_index']} kind={row['parent_kind']} field={row['field_label']} "
                f"action={row['action']} raw={row['raw_value']!r} normalized={row['normalized_value']!r}"
            )
    conn.close()


def retry_queue(
    db_path: str = SQLITE_DB,
    *,
    queue_name: str = RETRY_QUEUE_NAME,
    limit: int | None = None,
) -> None:
    from create_sql_db import create_database, connect_sqlite

    create_database(db_path)
    conn = connect_sqlite(db_path)
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT DISTINCT chunk_id
        FROM macro_processing_audit
        WHERE queue_name = ?
        ORDER BY created_at DESC
    """
    params: list[object] = [queue_name]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    chunk_ids = [row["chunk_id"] for row in rows]
    if not chunk_ids:
        print(f"[macro_extract] queue {queue_name} is empty")
        return
    print(f"[macro_extract] retrying {len(chunk_ids)} queued chunk(s) from {queue_name}")
    run_extraction(db_path=db_path, limit=limit, chunk_ids=chunk_ids)


def report_diagnostics(db_path: str = SQLITE_DB, limit: int = 20) -> None:
    from create_sql_db import create_database, connect_sqlite

    create_database(db_path)
    conn = connect_sqlite(db_path)
    conn.row_factory = sqlite3.Row
    enum_rows = conn.execute(
        """
        SELECT action, field_label, raw_value, normalized_value, COUNT(*) AS n
        FROM macro_enum_audit
        WHERE action IN ('dropped', 'snapped')
        GROUP BY action, field_label, raw_value, normalized_value
        ORDER BY n DESC, action, field_label
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    malformed_rows = conn.execute(
        """
        SELECT status, failure_reason, COUNT(*) AS n
        FROM macro_processing_audit
        WHERE status IN ('api_failed', 'parse_failed', 'empty_success')
           OR suspicious = 1
        GROUP BY status, failure_reason
        ORDER BY n DESC, status
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()

    print("[macro_extract] enum issues:")
    for row in enum_rows:
        print(
            f"- {row['action']} field={row['field_label']} raw={row['raw_value']!r} "
            f"normalized={row['normalized_value']!r} count={row['n']}"
        )
    print("[macro_extract] malformed / suspicious outputs:")
    for row in malformed_rows:
        print(f"- status={row['status']} failure={row['failure_reason'] or '—'} count={row['n']}")


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
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract", help="Run macro extraction")
    extract_parser.add_argument("--db", default=SQLITE_DB, help="Path to SQLite DB")
    extract_parser.add_argument("--limit", type=int, default=None, help="Max chunks to process")
    extract_parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess chunks that previously failed (success=0)",
    )

    queue_parser = subparsers.add_parser("queue", help="Print retry or review queue")
    queue_parser.add_argument("--db", default=SQLITE_DB, help="Path to SQLite DB")
    queue_parser.add_argument(
        "--name",
        choices=[RETRY_QUEUE_NAME, REVIEW_QUEUE_NAME],
        default=RETRY_QUEUE_NAME,
        help="Queue to print",
    )
    queue_parser.add_argument("--limit", type=int, default=20, help="Rows to print")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect one macro run")
    inspect_parser.add_argument("--db", default=SQLITE_DB, help="Path to SQLite DB")
    inspect_parser.add_argument("--run-id", default=None, help="Run id to inspect")
    inspect_parser.add_argument("--chunk-id", default=None, help="Chunk id to inspect")

    retry_parser = subparsers.add_parser("retry-queue", help="Retry all chunks from a queue")
    retry_parser.add_argument("--db", default=SQLITE_DB, help="Path to SQLite DB")
    retry_parser.add_argument(
        "--name",
        choices=[RETRY_QUEUE_NAME, REVIEW_QUEUE_NAME],
        default=RETRY_QUEUE_NAME,
        help="Queue to retry",
    )
    retry_parser.add_argument("--limit", type=int, default=None, help="Max queued chunks to retry")

    report_parser = subparsers.add_parser("report", help="Report malformed outputs and enum issues")
    report_parser.add_argument("--db", default=SQLITE_DB, help="Path to SQLite DB")
    report_parser.add_argument("--limit", type=int, default=20, help="Rows to print per report")

    args = parser.parse_args()
    command = args.command or "extract"
    if command == "extract":
        run_extraction(
            db_path=getattr(args, "db", SQLITE_DB),
            limit=getattr(args, "limit", None),
            reprocess_failed=getattr(args, "reprocess", False),
        )
    elif command == "queue":
        print_queue(
            db_path=getattr(args, "db", SQLITE_DB),
            queue_name=getattr(args, "name", RETRY_QUEUE_NAME),
            limit=getattr(args, "limit", 20),
        )
    elif command == "inspect":
        inspect_run(
            db_path=getattr(args, "db", SQLITE_DB),
            run_id=getattr(args, "run_id", None),
            chunk_id=getattr(args, "chunk_id", None),
        )
    elif command == "retry-queue":
        retry_queue(
            db_path=getattr(args, "db", SQLITE_DB),
            queue_name=getattr(args, "name", RETRY_QUEUE_NAME),
            limit=getattr(args, "limit", None),
        )
    elif command == "report":
        report_diagnostics(
            db_path=getattr(args, "db", SQLITE_DB),
            limit=getattr(args, "limit", 20),
        )
