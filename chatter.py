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
import uuid
import os
import re
import sqlite3
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

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
SINGLE_TICKER_OUTLOOK_ENFORCEMENT_VERSION = "v1"
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TICKER_MAP_PATH = Path(os.getenv("TICKER_MAP_PATH", "ticker_company_map.csv"))
FIN_ENTITY_MAP_PATH = Path(os.getenv("FIN_ENTITY_MAP_PATH", "financial_entity_map.csv"))
SQLITE_DB = os.getenv("SQLITE_DB", "my_database.db")
PROMPT_TEMPLATES_PATH = Path(os.getenv("PROMPT_TEMPLATES_PATH", "prompt_templates.json"))

SHOW_PROVENANCE = os.getenv("SHOW_PROVENANCE", "0").strip().lower() in {"1", "true", "yes", "on"}
TOP_K = int(os.getenv("TOP_K", "3"))
EXPANDED_K = int(os.getenv("EXPANDED_K", "6"))
RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "5"))

SUMMARY_TOP_K = int(os.getenv("SUMMARY_TOP_K", "12"))
SUMMARY_EXPANDED_K = int(os.getenv("SUMMARY_EXPANDED_K", "24"))
SUMMARY_RECENCY_HALF_LIFE_DAYS = float(os.getenv("SUMMARY_RECENCY_HALF_LIFE_DAYS", "2"))
SUMMARY_MAX_CHUNKS_PER_SOURCE = int(os.getenv("SUMMARY_MAX_CHUNKS_PER_SOURCE", "1"))
SUMMARY_DUPLICATE_SIM_THRESHOLD = float(os.getenv("SUMMARY_DUPLICATE_SIM_THRESHOLD", "0.9"))
SUMMARY_MIN_UNIQUE_SOURCES = int(os.getenv("SUMMARY_MIN_UNIQUE_SOURCES", "3"))
SUMMARY_CANDIDATE_LIMIT = int(os.getenv("SUMMARY_CANDIDATE_LIMIT", "300"))
SUMMARY_STREAM_BRIEF_MAX_SHARE = float(os.getenv("SUMMARY_STREAM_BRIEF_MAX_SHARE", "0.55"))
SUMMARY_MIN_FULL_CONTEXT_CHUNKS = int(os.getenv("SUMMARY_MIN_FULL_CONTEXT_CHUNKS", "2"))
SUMMARY_TERMS = (
    "summary",
    "summarise",
    "summarize",
    "recap",
    "brief",
    "wrap-up",
    "wrap up",
    "overview",
)
SUMMARY_TIME_TERMS = (
    "today",
    "today's",
    "todays",
    "daily",
    "yesterday",
    "yesterday's",
)
SUMMARY_DOMAIN_TERMS = (
    "macro",
    "news",
    "market",
    "markets",
)
SQLITE_SEMANTIC_CANDIDATE_LIMIT = int(os.getenv("SQLITE_SEMANTIC_CANDIDATE_LIMIT", "0"))
MACRO_EVENT_CANDIDATE_LIMIT = int(os.getenv("MACRO_EVENT_CANDIDATE_LIMIT", "600"))
GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "650"))
SINGLE_TICKER_GEN_MAX_TOKENS = int(os.getenv("SINGLE_TICKER_GEN_MAX_TOKENS", "1000"))
GEN_CONTINUATION_MAX_ROUNDS = int(os.getenv("GEN_CONTINUATION_MAX_ROUNDS", "1"))
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ENABLE_CROSS_ENCODER_RERANK = os.getenv("ENABLE_CROSS_ENCODER_RERANK", "1").strip().lower() in {"1", "true", "yes", "on"}
RERANK_CANDIDATE_LIMIT = int(os.getenv("RERANK_CANDIDATE_LIMIT", "10"))
RERANK_WEIGHT = float(os.getenv("RERANK_WEIGHT", "0.85"))
STREAM_BRIEF_DAILY_SUMMARY_MULTIPLIER = float(os.getenv("STREAM_BRIEF_DAILY_SUMMARY_MULTIPLIER", "1.30"))
STREAM_BRIEF_CONFIDENCE_PENALTY_THRESHOLD = float(os.getenv("STREAM_BRIEF_CONFIDENCE_PENALTY_THRESHOLD", "0.50"))
STREAM_BRIEF_DOMINANT_THRESHOLD = float(os.getenv("STREAM_BRIEF_DOMINANT_THRESHOLD", "0.85"))
SOURCE_KEYWORDS = {
    "bbc": "BBC",
    "bloomberg": "Bloomberg",
    "cnbc": "CNBC",
    "marketwatch": "MarketWatch",
    "nasdaq": "Nasdaq",
    "cbs": "CBS MoneyWatch",
}

ROUTE_TYPES = (
    "signal_discovery",
    "latest_news",
    "daily_summary",
    "macro_causal",
    "entity_profile",
    "single_ticker_financial",
    "live_market_data",
    "broad_exploration",
    "ambiguous",
)

SCORING_COMPONENT_ORDER = (
    "semantic_score",
    "cross_encoder_score",
    "keyword_overlap_score",
    "target_match_score",
    "source_quality_score",
    "recency_score",
    "graph_relevance_score",
    "event_support_score",
    "duplicate_penalty",
    "ambiguity_penalty",
)

UNIFIED_SCORING_WEIGHTS: dict[str, float] = {
    "semantic_score": float(os.getenv("W1_SEMANTIC_SCORE", "0.24")),
    "cross_encoder_score": float(os.getenv("W2_CROSS_ENCODER_SCORE", "0.16")),
    "keyword_overlap_score": float(os.getenv("W3_KEYWORD_OVERLAP_SCORE", "0.08")),
    "target_match_score": float(os.getenv("W4_TARGET_MATCH_SCORE", "0.12")),
    "source_quality_score": float(os.getenv("W5_SOURCE_QUALITY_SCORE", "0.10")),
    "recency_score": float(os.getenv("W6_RECENCY_SCORE", "0.10")),
    "graph_relevance_score": float(os.getenv("W7_GRAPH_RELEVANCE_SCORE", "0.08")),
    "event_support_score": float(os.getenv("W8_EVENT_SUPPORT_SCORE", "0.12")),
    "duplicate_penalty": float(os.getenv("W9_DUPLICATE_PENALTY", "0.07")),
    "ambiguity_penalty": float(os.getenv("W10_AMBIGUITY_PENALTY", "0.07")),
}

ROUTE_WEIGHT_MULTIPLIERS: dict[str, dict[str, float]] = {
    "signal_discovery": {
        "semantic_score": 1.10,
        "target_match_score": 1.15,
        "source_quality_score": 1.20,
        "recency_score": 1.30,
        "graph_relevance_score": 1.20,
        "event_support_score": 1.25,
        "duplicate_penalty": 1.20,
    },
    "latest_news": {"recency_score": 1.35, "source_quality_score": 1.15},
    "daily_summary": {"source_quality_score": 1.15, "duplicate_penalty": 1.25},
    "macro_causal": {"event_support_score": 1.35, "graph_relevance_score": 1.25},
    "entity_profile": {"target_match_score": 1.35, "recency_score": 0.85},
    "single_ticker_financial": {},
    "live_market_data": {"recency_score": 1.25, "target_match_score": 1.20},
    "broad_exploration": {"semantic_score": 1.10, "keyword_overlap_score": 1.10},
    "ambiguous": {"ambiguity_penalty": 1.50, "target_match_score": 0.85},
}

ROUTE_PROFILES: dict[str, dict[str, Any]] = {
    "signal_discovery": {
        "top_k": max(5, TOP_K + 2),
        "expanded_k": max(12, EXPANDED_K + 4),
        "recency_half_life_days": max(1.0, RECENCY_HALF_LIFE_DAYS * 0.6),
        "candidate_cap": max(24, EXPANDED_K * 4),
        "answer_strictness": "high",
        "allowed_content_classes": ("news_report", "analysis", "official_release", "stream_brief"),
    },
    "latest_news": {
        "top_k": max(4, TOP_K),
        "expanded_k": max(EXPANDED_K, TOP_K + 4),
        "recency_half_life_days": max(1.0, RECENCY_HALF_LIFE_DAYS * 0.8),
        "candidate_cap": max(30, EXPANDED_K * 8),
        "answer_strictness": "high",
        "allowed_content_classes": ("news_report", "analysis", "official_release", "stream_brief"),
    },
    "daily_summary": {
        "top_k": SUMMARY_TOP_K,
        "expanded_k": SUMMARY_EXPANDED_K,
        "recency_half_life_days": SUMMARY_RECENCY_HALF_LIFE_DAYS,
        "candidate_cap": max(SUMMARY_CANDIDATE_LIMIT, SUMMARY_EXPANDED_K * 4),
        "answer_strictness": "high",
        "allowed_content_classes": ("news_report", "analysis", "official_release", "stream_brief"),
    },
    "macro_causal": {
        "top_k": max(6, TOP_K),
        "expanded_k": max(EXPANDED_K + 4, TOP_K + 8),
        "recency_half_life_days": max(1.0, RECENCY_HALF_LIFE_DAYS),
        "candidate_cap": max(48, EXPANDED_K * 10),
        "answer_strictness": "high",
        "allowed_content_classes": ("news_report", "analysis", "official_release", "stream_brief"),
    },
    "entity_profile": {
        "top_k": max(5, TOP_K),
        "expanded_k": max(EXPANDED_K, TOP_K + 6),
        "recency_half_life_days": max(2.0, RECENCY_HALF_LIFE_DAYS * 1.2),
        "candidate_cap": max(36, EXPANDED_K * 8),
        "answer_strictness": "medium",
        "allowed_content_classes": ("news_report", "analysis", "official_release", "stream_brief", "evergreen_explainer"),
    },
    "single_ticker_financial": {
        "top_k": 0,
        "expanded_k": 0,
        "recency_half_life_days": max(1.0, RECENCY_HALF_LIFE_DAYS),
        "candidate_cap": 0,
        "answer_strictness": "very_high",
        "allowed_content_classes": (),
    },
    "live_market_data": {
        "top_k": max(4, TOP_K),
        "expanded_k": max(EXPANDED_K, TOP_K + 4),
        "recency_half_life_days": max(1.0, RECENCY_HALF_LIFE_DAYS * 0.7),
        "candidate_cap": max(28, EXPANDED_K * 6),
        "answer_strictness": "high",
        "allowed_content_classes": ("news_report", "analysis", "official_release", "stream_brief", "ticker_page"),
    },
    "broad_exploration": {
        "top_k": max(6, TOP_K),
        "expanded_k": max(EXPANDED_K + 4, TOP_K + 8),
        "recency_half_life_days": max(2.0, RECENCY_HALF_LIFE_DAYS * 1.1),
        "candidate_cap": max(60, EXPANDED_K * 12),
        "answer_strictness": "medium",
        "allowed_content_classes": ("news_report", "analysis", "official_release", "stream_brief", "evergreen_explainer"),
    },
    "ambiguous": {
        "top_k": max(5, TOP_K),
        "expanded_k": max(EXPANDED_K, TOP_K + 6),
        "recency_half_life_days": max(1.5, RECENCY_HALF_LIFE_DAYS),
        "candidate_cap": max(40, EXPANDED_K * 8),
        "answer_strictness": "very_high",
        "allowed_content_classes": ("news_report", "analysis", "official_release", "stream_brief"),
    },
}

SOURCE_TRUST_SCORES = {
    "tier_1": 1.0,
    "tier_2": 0.82,
    "tier_3": 0.62,
    "blocked": 0.0,
}

CONTENT_CLASS_SCORES = {
    "news_report": 1.0,
    "analysis": 0.9,
    "official_release": 0.95,
    "stream_brief": 0.50,
    "evergreen_explainer": 0.68,
    "ticker_page": 0.45,
    "navigation_page": 0.2,
    "video_stub": 0.25,
    "quote_page": 0.22,
    "junk": 0.0,
}

FULL_CONTEXT_CONTENT_CLASSES = frozenset({"news_report", "analysis", "official_release"})

SOURCE_QUALITY_FALLBACK = {
    "reuters": 0.95,
    "bloomberg": 0.93,
    "wsj": 0.91,
    "ft": 0.91,
    "cnbc": 0.80,
    "bbc": 0.80,
    "marketwatch": 0.72,
    "nasdaq": 0.70,
}

AMBIGUITY_ROUTE_TERMS = (
    "which",
    "what about",
    "is it",
    "they",
    "them",
    "this company",
    "that company",
    "or",
    "vs",
    "versus",
)

SIGNAL_DISCOVERY_HINTS = (
    "what matters today",
    "what changed",
    "top signals",
    "knowledge arbitrage",
    "emerging narratives",
    "market-moving",
    "market moving",
    "what should i watch",
    "top risks",
    "top opportunities",
)

ROUTE_HINTS = {
    "signal_discovery": SIGNAL_DISCOVERY_HINTS,
    "latest_news": ("latest", "recent", "new", "update", "today", "yesterday"),
    "daily_summary": SUMMARY_TERMS + SUMMARY_TIME_TERMS,
    "macro_causal": (
        "affect", "impact", "effect", "influence", "cause", "drive",
        "mean for", "implication", "what does", "how does", "why is",
        "because of", "result of", "due to", "respond to", "reaction",
        "stronger", "weaker", "rise", "fall", "rally", "selloff",
    ),
    "entity_profile": ("profile", "overview", "who is", "what is", "background", "exposure"),
    "live_market_data": ("price", "quote", "trading", "market", "valuation", "live", "real-time", "intraday"),
    "broad_exploration": ("explain", "landscape", "themes", "broad", "overview", "drivers"),
}

SUMMARY_THEME_KEYWORDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "geopolitics": ("Geopolitics", ("war", "sanction", "geopolit", "conflict", "election", "diplomatic", "tariff")),
    "energy": ("Energy", ("oil", "gas", "lng", "energy", "opec", "refinery", "brent", "wti")),
    "central_banks": ("Central Banks", ("fed", "fomc", "ecb", "boj", "rate", "hike", "cut", "monetary")),
    "equities": ("Equities", ("equity", "stock", "shares", "index", "earnings", "valuation", "sp500", "nasdaq")),
    "inflation_growth": ("Inflation/Growth", ("inflation", "cpi", "gdp", "growth", "recession", "employment", "labor")),
    "fx_rates": ("FX/Rates", ("usd", "dollar", "fx", "currency", "yield", "treasury", "bond")),
    "commodities": ("Commodities", ("gold", "silver", "copper", "commodity", "wheat", "corn")),
}

ANSWER_CONFIDENCE_WEIGHTS = {
    "relevant_chunks": 0.22,
    "source_diversity": 0.15,
    "retrieval_margin": 0.15,
    "verifier_support": 0.16,
    "recency_coverage": 0.12,
    "ambiguity_penalty": 0.10,
    "contradiction_penalty": 0.10,
}

RESOLUTION_DISAMBIGUATION_THRESHOLD = float(os.getenv("RESOLUTION_DISAMBIGUATION_THRESHOLD", "0.55"))

DOMAIN_CANONICAL_ALIASES: dict[str, dict[str, Any]] = {
    "united states": {"canonical_name": "country:united_states", "display_name": "United States", "entity_type": "COUNTRY", "category": "countries"},
    "us": {"canonical_name": "country:united_states", "display_name": "United States", "entity_type": "COUNTRY", "category": "countries"},
    "china": {"canonical_name": "country:china", "display_name": "China", "entity_type": "COUNTRY", "category": "countries"},
    "japan": {"canonical_name": "country:japan", "display_name": "Japan", "entity_type": "COUNTRY", "category": "countries"},
    "eurozone": {"canonical_name": "region:eurozone", "display_name": "Eurozone", "entity_type": "REGION", "category": "countries"},
    "fed": {"canonical_name": "cb:fed", "display_name": "Federal Reserve", "entity_type": "CENTRAL_BANK", "category": "central_banks"},
    "federal reserve": {"canonical_name": "cb:fed", "display_name": "Federal Reserve", "entity_type": "CENTRAL_BANK", "category": "central_banks"},
    "ecb": {"canonical_name": "cb:ecb", "display_name": "European Central Bank", "entity_type": "CENTRAL_BANK", "category": "central_banks"},
    "boj": {"canonical_name": "cb:boj", "display_name": "Bank of Japan", "entity_type": "CENTRAL_BANK", "category": "central_banks"},
    "gold": {"canonical_name": "commodity:gold", "display_name": "Gold", "entity_type": "COMMODITY", "category": "commodities"},
    "oil": {"canonical_name": "commodity:oil", "display_name": "Crude Oil", "entity_type": "COMMODITY", "category": "commodities"},
    "brent": {"canonical_name": "commodity:brent", "display_name": "Brent Crude", "entity_type": "COMMODITY", "category": "commodities"},
    "wti": {"canonical_name": "commodity:wti", "display_name": "WTI Crude", "entity_type": "COMMODITY", "category": "commodities"},
    "sp500": {"canonical_name": "index:sp500", "display_name": "S&P 500", "entity_type": "INDEX", "category": "indices"},
    "s&p 500": {"canonical_name": "index:sp500", "display_name": "S&P 500", "entity_type": "INDEX", "category": "indices"},
    "nasdaq": {"canonical_name": "index:nasdaq", "display_name": "NASDAQ Composite", "entity_type": "INDEX", "category": "indices"},
    "qqq": {"canonical_name": "etf:qqq", "display_name": "Invesco QQQ Trust", "entity_type": "ETF", "category": "etfs", "ticker": "QQQ"},
    "spy": {"canonical_name": "etf:spy", "display_name": "SPDR S&P 500 ETF", "entity_type": "ETF", "category": "etfs", "ticker": "SPY"},
    "inflation": {"canonical_name": "macro:inflation", "display_name": "Inflation", "entity_type": "MACRO_CONCEPT", "category": "macro_concepts"},
    "growth": {"canonical_name": "macro:growth", "display_name": "Economic Growth", "entity_type": "MACRO_CONCEPT", "category": "macro_concepts"},
    "yield curve": {"canonical_name": "macro:yield_curve", "display_name": "Yield Curve", "entity_type": "MACRO_CONCEPT", "category": "macro_concepts"},
}

_RERANKER = None

def _coerce_prompt_template(value: Any, key: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "\n".join(value)
    raise ValueError(
        f"Prompt template '{key}' must be either a string or a list of strings."
    )

def load_prompt_templates(path: Path) -> tuple[str, str, str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt templates file not found: {path}. "
            "Create it or set PROMPT_TEMPLATES_PATH."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in prompt templates file {path}: {exc}") from exc

    required = ("SYSTEM_PROMPT_TEMPLATE", "CAUSAL_SYSTEM_PROMPT_TEMPLATE")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Prompt templates file {path} is missing keys: {', '.join(missing)}")

    system_prompt = _coerce_prompt_template(data["SYSTEM_PROMPT_TEMPLATE"], "SYSTEM_PROMPT_TEMPLATE")
    causal_prompt = _coerce_prompt_template(data["CAUSAL_SYSTEM_PROMPT_TEMPLATE"], "CAUSAL_SYSTEM_PROMPT_TEMPLATE")

    daily_summary_raw = data.get("DAILY_SUMMARY_PROMPT_TEMPLATE")
    if daily_summary_raw is None:
        daily_summary_prompt = (
            "You are a grounded macro-financial news summarizer. "
            "Answer using only the provided context. "
            "Group related stories into 3-5 major themes. "
            "Focus on macro-relevant developments and cite factual sentences as [S1], [S2], etc. "
            "Do not use outside knowledge. "
            "Your article database covers {date_min} to {date_max}."
        )
    else:
        daily_summary_prompt = _coerce_prompt_template(
            daily_summary_raw,
            "DAILY_SUMMARY_PROMPT_TEMPLATE",
        )

    single_ticker_financial_raw = data.get("SINGLE_TICKER_FINANCIAL_PROMPT_TEMPLATE")
    if single_ticker_financial_raw is None:
        single_ticker_financial_prompt = (
            "You are a grounded single-ticker financial analyst. "
            "Use only the FINANCIAL DATA [F] block provided in the context. "
            "Do not use outside knowledge or article/news evidence. "
            "Structure your response with these exact section headers: Answer, Evidence, Outlook, Outlook Confidence (0-1), Theory. "
            "Every factual sentence in Answer and Evidence must include [F]. "
            "Outlook must be exactly one of: Bullish, Bearish, or Neutral. "
            "Outlook Confidence (0-1) must be a single number from 0.00 to 1.00, where 0.00 is fully bearish and 1.00 is fully bullish; use 0.50 for neutral/mixed. "
            "Theory may contain cautious inference from [F] and must not contain citations. "
            "If FINANCIAL DATA [F] is missing or unusable, state that there is insufficient financial data [F] to answer. "
            "Your article database covers {date_min} to {date_max}."
        )
    else:
        single_ticker_financial_prompt = _coerce_prompt_template(
            single_ticker_financial_raw,
            "SINGLE_TICKER_FINANCIAL_PROMPT_TEMPLATE",
        )

    return system_prompt, causal_prompt, daily_summary_prompt, single_ticker_financial_prompt

MARKET_INTENT_HINTS = (
    "price",
    "quote",
    "trading at",
    "market cap",
    "company overview",
    "daily close",
    "open price",
    "high",
    "low",
    "volume",
)

SINGLE_TICKER_FINANCIAL_INTENT_HINTS = (
    "performance",
    "trend",
    "valuation",
    "fundamental",
    "balance sheet",
    "cash flow",
    "income statement",
    "return",
    "eps",
    "earnings per share",
    "p/e",
    "pe ratio",
)

UNMAPPED_TICKER_CONTEXT_HINTS = (
    "stock",
    "ticker",
    "share",
    "shares",
    "equity",
    "valuation",
    "earnings",
    "eps",
    "income statement",
    "balance sheet",
    "cash flow",
    "financial performance",
    "market cap",
    "price target",
)

UNMAPPED_TICKER_BLOCKLIST = {
    "US",
    "USA",
    "UK",
    "EU",
    "UAE",
    "UN",
    "IMF",
    "CPI",
    "PPI",
    "PCE",
    "GDP",
    "PMI",
    "ISM",
    "FOMC",
    "ECB",
    "BOE",
    "BOJ",
    "SNB",
    "RBA",
    "RBNZ",
    "OPEC",
    "OECD",
    "BIS",
    "NATO",
    "ASEAN",
    "EPS",
    "PE",
}

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
    keys = set(row.keys())
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
        "source_trust_tier": row["source_trust_tier"] if "source_trust_tier" in keys else None,
        "content_class": row["content_class"] if "content_class" in keys else None,
        "article_quality_score": row["article_quality_score"] if "article_quality_score" in keys else None,
        "quality_flags_json": row["quality_flags_json"] if "quality_flags_json" in keys else None,
    }


def _keyword_overlap_score(query: str, text: str) -> float:
    query_terms = {
        token
        for token in re.findall(r"[a-zA-Z0-9]{3,}", query.lower())
        if token not in {"what", "when", "where", "which", "with", "from", "that", "this", "have"}
    }
    if not query_terms:
        return 0.0
    text_terms = set(re.findall(r"[a-zA-Z0-9]{3,}", (text or "").lower()))
    if not text_terms:
        return 0.0
    return len(query_terms & text_terms) / max(len(query_terms), 1)


def _safe_json_loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-value)))


def _resolve_route_profile(route_type: str | None) -> dict[str, Any]:
    route = route_type if route_type in ROUTE_PROFILES else "broad_exploration"
    profile = ROUTE_PROFILES[route]
    return {
        "route_type": route,
        "top_k": int(profile["top_k"]),
        "expanded_k": int(profile["expanded_k"]),
        "recency_half_life_days": float(profile["recency_half_life_days"]),
        "candidate_cap": int(profile["candidate_cap"]),
        "answer_strictness": str(profile["answer_strictness"]),
        "allowed_content_classes": tuple(profile.get("allowed_content_classes") or ()),
    }


def _resolve_scoring_weights(route_type: str | None) -> dict[str, float]:
    weights = dict(UNIFIED_SCORING_WEIGHTS)
    route = route_type if route_type in ROUTE_WEIGHT_MULTIPLIERS else None
    if route:
        for component, multiplier in ROUTE_WEIGHT_MULTIPLIERS[route].items():
            if component in weights:
                weights[component] = float(weights[component]) * float(multiplier)
    return weights


def classify_query_route(
    *,
    query: str,
    summary_mode: bool,
    causal_intent: bool,
    market_data_intent: bool,
    target: QueryTarget | None = None,
    financial_intent: bool = False,
    explicit_latest_news_intent: bool = False,
) -> str:
    q = (query or "").strip().lower()
    if summary_mode:
        return "daily_summary"
    if causal_intent:
        return "macro_causal"
    if target and target.needs_disambiguation:
        return "ambiguous"
    signal_discovery_intent = has_signal_discovery_intent(query)
    if (
        target
        and target.query_type == QUERY_TYPE_SINGLE
        and target.ticker
        and financial_intent
        and not explicit_latest_news_intent
        and not signal_discovery_intent
    ):
        return "single_ticker_financial"
    if market_data_intent:
        return "live_market_data"
    if signal_discovery_intent:
        return "signal_discovery"
    if any(h in q for h in ROUTE_HINTS["latest_news"]):
        return "latest_news"
    if target and target.canonical_name and any(h in q for h in ROUTE_HINTS["entity_profile"]):
        return "entity_profile"
    if any(h in q for h in AMBIGUITY_ROUTE_TERMS):
        return "ambiguous"
    return "broad_exploration"


def _route_allows_content_class(route_profile: dict[str, Any], content_class: str | None) -> bool:
    allowed = route_profile.get("allowed_content_classes") or ()
    if not allowed:
        return True
    if not content_class:
        return True
    return str(content_class).strip().lower() in {c.lower() for c in allowed}


def _is_stream_brief_chunk(row: dict[str, Any]) -> bool:
    return (row.get("content_class") or "").strip().lower() == "stream_brief"


def _is_full_context_chunk(row: dict[str, Any]) -> bool:
    return (row.get("content_class") or "").strip().lower() in FULL_CONTEXT_CONTENT_CLASSES


def _source_quality_component(row: dict[str, Any], route_type: str | None = None) -> float:
    trust_tier = (row.get("source_trust_tier") or "").strip().lower()
    trust_score = SOURCE_TRUST_SCORES.get(trust_tier, 0.65)
    content_class = (row.get("content_class") or "").strip().lower()
    class_score = CONTENT_CLASS_SCORES.get(content_class, 0.60)
    if content_class == "stream_brief" and route_type == "daily_summary":
        class_score = _clamp01(class_score * STREAM_BRIEF_DAILY_SUMMARY_MULTIPLIER)
    quality_raw = _safe_float(row.get("article_quality_score"), 60.0)
    quality_score = _clamp01(quality_raw / 100.0 if quality_raw > 1.0 else quality_raw)
    source_name = (row.get("source") or "").strip().lower()
    source_fallback = SOURCE_QUALITY_FALLBACK.get(source_name, 0.68)
    return _clamp01(0.35 * trust_score + 0.25 * class_score + 0.25 * quality_score + 0.15 * source_fallback)


def _target_match_component(row: dict[str, Any], target: QueryTarget | None) -> float:
    if target is None or not target.canonical_name:
        return 0.40
    kind = str(row.get("retrieval_kind") or "")
    score = 0.35
    if kind in {"entity_mentions", "macro_event_entity"}:
        score += 0.35
    if kind in {"macro_event_asset"} and target.ticker:
        score += 0.20
    text = (row.get("text") or "").lower()
    display_name = (target.display_name or "").lower()
    canonical = (target.canonical_name or "").lower()
    ticker = (target.ticker or "").lower()
    if display_name and display_name in text:
        score += 0.18
    if canonical and canonical in text:
        score += 0.16
    if ticker and ticker in text:
        score += 0.10
    return _clamp01(score)


def _graph_relevance_component(row: dict[str, Any]) -> float:
    kind = str(row.get("retrieval_kind") or "")
    if kind == "signal_alert":
        return 0.95
    if kind.startswith("macro_event"):
        return 0.90
    if kind == "entity_mentions":
        return 0.62
    if kind == "summary_diverse_source":
        return 0.52
    if kind == "sqlite_semantic":
        return 0.40
    return 0.35


def _event_support_component(row: dict[str, Any]) -> float:
    support = _safe_float(row.get("support_score"), -1.0)
    if support >= 0.0:
        return _clamp01(support)
    verification_status = (row.get("verification_status") or "").strip().lower()
    if verification_status == "verified":
        return 0.92
    if verification_status == "weak":
        return 0.52
    if verification_status == "rejected":
        return 0.12
    return _clamp01(_safe_float(row.get("macro_confidence"), 0.45))


def _compute_contradiction_signal(chunks: list[dict[str, Any]]) -> float:
    directions = set()
    for chunk in chunks:
        direction = (chunk.get("impact_direction") or "").strip().lower()
        if direction in {"up", "down", "positive", "negative"}:
            directions.add(direction)
    if {"up", "down"} <= directions or {"positive", "negative"} <= directions:
        return 1.0
    return 0.0


def _apply_unified_scoring(
    *,
    query: str,
    rows: list[dict],
    query_vec: np.ndarray,
    target: QueryTarget | None,
    route_type: str,
    recency_half_life_days: float,
) -> list[dict]:
    now_ts = int(datetime.now(timezone.utc).timestamp())
    half_life_s = max(0.1, recency_half_life_days) * 86400.0
    weights = _resolve_scoring_weights(route_type)
    seen_uid: set[str] = set()
    seen_text_fingerprints: set[str] = set()
    scored: list[dict] = []

    for row in rows:
        uid = row.get("chunk_uid") or row.get("candidate_id") or row.get("signal_id") or row.get("cluster_id")
        if not uid:
            continue

        emb = row.get("embedding")
        if emb is None:
            semantic_score = _clamp01(_safe_float(row.get("semantic_score"), _safe_float(row.get("candidate_score"), 0.45)))
        else:
            semantic_score = _clamp01(cosine_sim(query_vec, np.array(emb, dtype=np.float32)))
        candidate_text = row.get("text") or row.get("summary") or row.get("headline") or row.get("title") or ""
        keyword_overlap = _clamp01(_keyword_overlap_score(query, candidate_text))
        target_match = _target_match_component(row, target)
        source_quality = _source_quality_component(row, route_type=route_type)

        ts = _published_date_to_ts(row.get("published_date"))
        recency_score = float(np.exp(-max(0.0, float(now_ts - ts)) / half_life_s)) if ts is not None else 0.0
        recency_score = _clamp01(recency_score)

        graph_relevance = _graph_relevance_component(row)
        event_support = _event_support_component(row)

        text_fingerprint = re.sub(r"\s+", " ", str(candidate_text).strip().lower())[:280]
        duplicate_penalty = 1.0 if (uid in seen_uid or text_fingerprint in seen_text_fingerprints) else 0.0
        ambiguity_penalty = _clamp01(target.ambiguity_score if target else 0.0)

        cross_encoder_score = _clamp01(_safe_float(row.get("cross_encoder_score"), 0.0))
        final_score = (
            weights["semantic_score"] * semantic_score
            + weights["cross_encoder_score"] * cross_encoder_score
            + weights["keyword_overlap_score"] * keyword_overlap
            + weights["target_match_score"] * target_match
            + weights["source_quality_score"] * source_quality
            + weights["recency_score"] * recency_score
            + weights["graph_relevance_score"] * graph_relevance
            + weights["event_support_score"] * event_support
            - weights["duplicate_penalty"] * duplicate_penalty
            - weights["ambiguity_penalty"] * ambiguity_penalty
        )

        scored_row = {
            **row,
            "semantic_score": semantic_score,
            "cross_encoder_score": cross_encoder_score,
            "keyword_overlap_score": keyword_overlap,
            "target_match_score": target_match,
            "source_quality_score": source_quality,
            "recency_score": recency_score,
            "graph_relevance_score": graph_relevance,
            "event_support_score": event_support,
            "duplicate_penalty": duplicate_penalty,
            "ambiguity_penalty": ambiguity_penalty,
            "final_score": float(final_score),
            "score": float(final_score),
            "score_components": {
                "semantic_score": semantic_score,
                "cross_encoder_score": cross_encoder_score,
                "keyword_overlap_score": keyword_overlap,
                "target_match_score": target_match,
                "source_quality_score": source_quality,
                "recency_score": recency_score,
                "graph_relevance_score": graph_relevance,
                "event_support_score": event_support,
                "duplicate_penalty": duplicate_penalty,
                "ambiguity_penalty": ambiguity_penalty,
            },
            "scoring_weights": dict(weights),
        }
        scored.append(scored_row)
        seen_uid.add(uid)
        if text_fingerprint:
            seen_text_fingerprints.add(text_fingerprint)

    scored.sort(key=lambda item: item.get("final_score", item.get("score", 0.0)), reverse=True)
    return scored


def _signal_candidate_text(row: dict[str, Any]) -> str:
    asset_targets = _safe_json_loads(row.get("asset_targets_json"), [])
    top_assets = _safe_json_loads(row.get("top_assets_json"), [])
    asset_terms: list[str] = []
    for payload in list(asset_targets) + list(top_assets):
        if isinstance(payload, dict):
            for key in ("target_id", "display_name", "asset_key", "ticker", "name"):
                value = payload.get(key)
                if value:
                    asset_terms.append(str(value))
        elif payload:
            asset_terms.append(str(payload))
    return " ".join(
        part
        for part in (
            row.get("headline"),
            row.get("signal_summary"),
            row.get("canonical_summary"),
            row.get("event_type"),
            row.get("primary_shock_type"),
            row.get("region"),
            " ".join(asset_terms),
        )
        if part
    ).strip()


def _signal_target_match_component(
    row: dict[str, Any],
    target: QueryTarget | None,
    asset_target: dict[str, Any] | None,
) -> float:
    expected_terms: list[str] = []
    if target:
        for value in (target.display_name, target.canonical_name, target.ticker):
            if value:
                expected_terms.append(str(value).lower())
    if asset_target:
        for value in (asset_target.get("display_name"), asset_target.get("target_id"), asset_target.get("asset_key")):
            if value:
                expected_terms.append(str(value).lower())
    expected_terms = [term for term in expected_terms if term]
    if not expected_terms:
        return 0.45

    text = _signal_candidate_text(row).lower()
    asset_blob = (
        json.dumps(_safe_json_loads(row.get("asset_targets_json"), []), ensure_ascii=False).lower()
        + " "
        + json.dumps(_safe_json_loads(row.get("top_assets_json"), []), ensure_ascii=False).lower()
    )
    score = 0.12
    for term in expected_terms:
        if term in text:
            score += 0.42
        if term in asset_blob:
            score += 0.18
    return _clamp01(score)


def _signal_graph_relevance_component(row: dict[str, Any]) -> float:
    event_count = min(max(_safe_float(row.get("supporting_event_count"), _safe_float(row.get("member_count"), 0.0)), 0.0), 5.0) / 5.0
    source_count = min(max(_safe_float(row.get("supporting_source_count"), _safe_float(row.get("unique_source_count"), 0.0)), 0.0), 5.0) / 5.0
    has_structure = 0.10 if row.get("event_type") else 0.0
    has_shock = 0.08 if row.get("primary_shock_type") else 0.0
    return _clamp01(0.42 + 0.20 * event_count + 0.20 * source_count + has_structure + has_shock)


def _signal_event_support_component(row: dict[str, Any]) -> float:
    base_signal = _clamp01(_safe_float(row.get("base_signal_score"), _safe_float(row.get("cluster_signal_score"), 0.0)))
    confidence_score = _clamp01(_safe_float(row.get("confidence_score"), 0.0))
    event_count = min(max(_safe_float(row.get("supporting_event_count"), _safe_float(row.get("member_count"), 0.0)), 0.0), 4.0) / 4.0
    source_count = min(max(_safe_float(row.get("supporting_source_count"), _safe_float(row.get("unique_source_count"), 0.0)), 0.0), 4.0) / 4.0
    return _clamp01(0.35 * base_signal + 0.25 * confidence_score + 0.20 * event_count + 0.20 * source_count)


def _signal_source_quality_component(row: dict[str, Any]) -> float:
    base_quality = _clamp01(_safe_float(row.get("cluster_source_quality_score"), 0.0))
    unique_sources = min(max(_safe_float(row.get("unique_source_count"), _safe_float(row.get("supporting_source_count"), 0.0)), 0.0), 5.0) / 5.0
    return _clamp01(max(base_quality, 0.35 + 0.45 * base_quality + 0.20 * unique_sources))


def _synthesize_signal_headline(row: dict[str, Any]) -> str:
    if row.get("headline"):
        return str(row["headline"]).strip()
    event_type = (row.get("event_type") or "macro signal").replace("_", " ").strip()
    region = str(row.get("region") or "global").strip()
    summary = str(row.get("signal_summary") or row.get("canonical_summary") or "").strip()
    if summary:
        summary = re.sub(r"\s+", " ", summary)[:120].rstrip(". ")
        return f"{event_type.title()} in {region}: {summary}"
    return f"{event_type.title()} in {region}"


def _load_signal_evidence_chunks(
    sqlite_conn: sqlite3.Connection,
    cluster_ids: list[str],
    *,
    date_start: str | None = None,
    date_end: str | None = None,
    source_filter: str | None = None,
    per_cluster_limit: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    if not cluster_ids:
        return {}

    placeholders = ",".join("?" for _ in cluster_ids)
    sql = f"""
        SELECT
            cm.cluster_id,
            cm.macro_event_id,
            COALESCE(cm.chunk_id, me.chunk_id) AS chunk_id,
            COALESCE(cm.article_id, me.article_id, c.article_id) AS article_id,
            cm.similarity_score,
            me.summary AS macro_summary,
            me.event_type,
            me.region,
            me.time_horizon,
            me.confidence AS macro_confidence,
            me.verification_status,
            me.support_score,
            me.novelty_hint,
            me.urgency,
            me.market_surprise,
            c.text,
            c.embedding_json,
            c.published_date,
            c.period_key,
            a.title,
            a.url,
            a.source,
            a.source_trust_tier,
            a.content_class,
            a.article_quality_score,
            a.quality_flags_json
        FROM cluster_members cm
        JOIN macro_events me
          ON me.macro_event_id = cm.macro_event_id
        LEFT JOIN chunks c
          ON c.chunk_id = COALESCE(cm.chunk_id, me.chunk_id)
        JOIN articles a
          ON a.article_id = COALESCE(cm.article_id, me.article_id, c.article_id)
        WHERE cm.cluster_id IN ({placeholders})
          AND c.chunk_id IS NOT NULL
    """
    params: list[Any] = list(cluster_ids)
    if date_start:
        sql += " AND c.published_date >= ?"
        params.append(date_start)
    if date_end:
        sql += " AND c.published_date <= ?"
        params.append(date_end)
    if source_filter:
        sql += " AND a.source = ?"
        params.append(source_filter)
    sql += """
        ORDER BY
            cm.cluster_id,
            COALESCE(me.support_score, 0.0) DESC,
            COALESCE(cm.similarity_score, 0.0) DESC,
            c.published_date DESC
    """

    rows = sqlite_conn.execute(sql, params).fetchall()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_by_cluster: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        cluster_id = str(row["cluster_id"])
        chunk_id = str(row["chunk_id"])
        if not cluster_id or not chunk_id or chunk_id in seen_by_cluster[cluster_id]:
            continue
        if len(grouped[cluster_id]) >= max(per_cluster_limit, 1):
            continue
        item = _sqlite_row_to_chunk_dict(row)
        item.update(
            {
                "cluster_id": cluster_id,
                "macro_event_id": row["macro_event_id"],
                "retrieval_kind": "signal_cluster_evidence",
                "macro_summary": row["macro_summary"],
                "event_type": row["event_type"],
                "region": row["region"],
                "time_horizon": row["time_horizon"],
                "macro_confidence": row["macro_confidence"],
                "verification_status": row["verification_status"],
                "support_score": row["support_score"],
                "novelty_hint": row["novelty_hint"],
                "urgency": row["urgency"],
                "market_surprise": row["market_surprise"],
                "graph_relevance_score": 0.88,
                "event_support_score": _clamp01(_safe_float(row["support_score"], _safe_float(row["macro_confidence"], 0.55))),
            }
        )
        grouped[cluster_id].append(item)
        seen_by_cluster[cluster_id].add(chunk_id)
    return grouped


def retrieve_top_signals(
    *,
    query: str,
    embed_model: SentenceTransformer,
    sqlite_conn: sqlite3.Connection,
    alias_to_ticker: dict[str, str],
    ticker_to_canonical: dict[str, str],
    alias_to_fin_entity: dict[str, dict[str, Any]],
    driver=None,
    top_k: int | None = None,
    expanded_k: int | None = None,
    recency_half_life_days: float | None = None,
    source_filter: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> tuple[list[dict[str, Any]], QueryTarget, dict[str, Any]]:
    target = resolve_query_target(
        query,
        alias_to_ticker,
        ticker_to_canonical,
        driver,
        sqlite_conn=sqlite_conn,
        alias_to_fin_entity=alias_to_fin_entity,
    )
    route_profile = _resolve_route_profile("signal_discovery")
    top_limit = max(int(top_k or route_profile["top_k"]), 1)
    expanded_limit = max(int(expanded_k or route_profile["expanded_k"]), top_limit)
    recency_half_life = float(recency_half_life_days or route_profile["recency_half_life_days"])
    candidate_limit = max(int(route_profile["candidate_cap"]), expanded_limit)
    asset_target = _resolve_asset_target(sqlite_conn, query, target)
    query_vec = embed_model.encode([query], normalize_embeddings=True)[0]

    def _signal_filter(date_column: str, cluster_ref: str) -> tuple[str, list[Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        if date_start:
            clauses.append(f"{date_column} >= ?")
            params.append(date_start)
        if date_end:
            clauses.append(f"{date_column} <= ?")
            params.append(date_end)
        if source_filter:
            clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM cluster_members cm2
                    JOIN articles a2
                      ON a2.article_id = COALESCE(
                          cm2.article_id,
                          (SELECT me2.article_id FROM macro_events me2 WHERE me2.macro_event_id = cm2.macro_event_id LIMIT 1)
                      )
                    WHERE cm2.cluster_id = {cluster_ref}
                      AND a2.source = ?
                )
                """.strip()
            )
            params.append(source_filter)
        return " AND ".join(clauses), params

    primary_filter, primary_params = _signal_filter("sa.signal_date", "sa.cluster_id")
    primary_sql = f"""
        SELECT
            sa.signal_id,
            sa.cluster_id,
            sa.signal_date,
            sa.rank AS signal_rank,
            sa.signal_score AS base_signal_score,
            sa.headline,
            sa.summary AS signal_summary,
            sa.novelty_hint,
            sa.urgency,
            sa.market_surprise,
            sa.top_assets_json,
            ecs.score_id,
            ecs.score_date,
            ecs.novelty_score,
            ecs.source_quality_score AS cluster_source_quality_score,
            ecs.velocity_score,
            ecs.asset_impact_score,
            ecs.confidence_score,
            ecs.recency_score AS cluster_recency_score,
            ecs.signal_score AS cluster_signal_score,
            ecs.supporting_event_count,
            ecs.supporting_source_count,
            ec.event_type,
            ec.primary_shock_type,
            ec.region,
            ec.canonical_summary,
            ec.first_event_time,
            ec.last_event_time,
            ec.member_count,
            ec.unique_source_count,
            ec.asset_targets_json
        FROM signal_alerts sa
        JOIN event_clusters ec
          ON ec.cluster_id = sa.cluster_id
        LEFT JOIN event_cluster_scores ecs
          ON ecs.score_id = sa.score_id
        WHERE {primary_filter}
          AND COALESCE(sa.status, 'active') <> 'dismissed'
        ORDER BY sa.signal_date DESC, sa.signal_score DESC, COALESCE(sa.rank, 9999) ASC
        LIMIT ?
    """
    raw_rows = [dict(row) for row in sqlite_conn.execute(primary_sql, [*primary_params, candidate_limit]).fetchall()]

    if not raw_rows:
        fallback_filter, fallback_params = _signal_filter("ecs.score_date", "ecs.cluster_id")
        fallback_sql = f"""
            SELECT
                COALESCE(ecs.score_id, ecs.cluster_id || '::' || ecs.score_date) AS signal_id,
                ecs.cluster_id,
                ecs.score_date AS signal_date,
                NULL AS signal_rank,
                ecs.signal_score AS base_signal_score,
                NULL AS headline,
                ec.canonical_summary AS signal_summary,
                NULL AS novelty_hint,
                NULL AS urgency,
                NULL AS market_surprise,
                NULL AS top_assets_json,
                ecs.score_id,
                ecs.score_date,
                ecs.novelty_score,
                ecs.source_quality_score AS cluster_source_quality_score,
                ecs.velocity_score,
                ecs.asset_impact_score,
                ecs.confidence_score,
                ecs.recency_score AS cluster_recency_score,
                ecs.signal_score AS cluster_signal_score,
                ecs.supporting_event_count,
                ecs.supporting_source_count,
                ec.event_type,
                ec.primary_shock_type,
                ec.region,
                ec.canonical_summary,
                ec.first_event_time,
                ec.last_event_time,
                ec.member_count,
                ec.unique_source_count,
                ec.asset_targets_json
            FROM event_cluster_scores ecs
            JOIN event_clusters ec
              ON ec.cluster_id = ecs.cluster_id
            WHERE {fallback_filter}
            ORDER BY ecs.score_date DESC, ecs.signal_score DESC
            LIMIT ?
        """
        raw_rows = [dict(row) for row in sqlite_conn.execute(fallback_sql, [*fallback_params, candidate_limit]).fetchall()]

    if not raw_rows:
        return [], target, {
            "route_type": "signal_discovery",
            "candidate_count": 0,
            "ranked_count": 0,
            "selected_signal_count": 0,
            "date_start": date_start,
            "date_end": date_end,
            "source_filter": source_filter,
            "target": {
                "canonical_name": target.canonical_name,
                "display_name": target.display_name,
                "ticker": target.ticker,
                "ambiguity_score": target.ambiguity_score,
                "needs_disambiguation": target.needs_disambiguation,
                "resolution_mode": target.resolution_mode,
            },
            "ranked_candidates": [],
        }

    texts = [_signal_candidate_text(row) for row in raw_rows]
    embeddings = embed_model.encode(texts, normalize_embeddings=True)
    weights = _resolve_scoring_weights("signal_discovery")
    now_ts = int(datetime.now(timezone.utc).timestamp())
    half_life_s = max(0.1, recency_half_life) * 86400.0
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    scored_rows: list[dict[str, Any]] = []

    for row, emb in zip(raw_rows, embeddings):
        signal_id = str(row.get("signal_id") or "")
        cluster_id = str(row.get("cluster_id") or "")
        if not signal_id or not cluster_id:
            continue
        text = _signal_candidate_text(row)
        fingerprint = re.sub(r"\s+", " ", text.lower())[:280]
        published_ts = _published_date_to_ts(row.get("signal_date"))
        recency_score = float(np.exp(-max(0.0, float(now_ts - published_ts)) / half_life_s)) if published_ts is not None else 0.0
        recency_score = _clamp01(max(recency_score, _safe_float(row.get("cluster_recency_score"), 0.0)))
        semantic_score = _clamp01(cosine_sim(query_vec, np.array(emb, dtype=np.float32)))
        keyword_overlap = _clamp01(_keyword_overlap_score(query, text))
        target_match = _signal_target_match_component(row, target, asset_target)
        source_quality = _signal_source_quality_component(row)
        graph_relevance = _signal_graph_relevance_component(row)
        event_support = _signal_event_support_component(row)
        duplicate_penalty = 1.0 if (signal_id in seen_ids or cluster_id in seen_ids or fingerprint in seen_texts) else 0.0
        ambiguity_penalty = _clamp01(target.ambiguity_score if target else 0.0)
        cross_encoder_score = 0.0
        final_score = (
            weights["semantic_score"] * semantic_score
            + weights["cross_encoder_score"] * cross_encoder_score
            + weights["keyword_overlap_score"] * keyword_overlap
            + weights["target_match_score"] * target_match
            + weights["source_quality_score"] * source_quality
            + weights["recency_score"] * recency_score
            + weights["graph_relevance_score"] * graph_relevance
            + weights["event_support_score"] * event_support
            - weights["duplicate_penalty"] * duplicate_penalty
            - weights["ambiguity_penalty"] * ambiguity_penalty
        )
        signal_score = _clamp01(_safe_float(row.get("base_signal_score"), _safe_float(row.get("cluster_signal_score"), 0.0)))
        scored_rows.append(
            {
                **row,
                "candidate_id": signal_id,
                "candidate_kind": "signal_alert",
                "retrieval_kind": "signal_alert",
                "signal_id": signal_id,
                "cluster_id": cluster_id,
                "headline": _synthesize_signal_headline(row),
                "title": _synthesize_signal_headline(row),
                "summary": row.get("signal_summary") or row.get("canonical_summary") or "",
                "text": text,
                "published_date": row.get("signal_date"),
                "semantic_score": semantic_score,
                "cross_encoder_score": cross_encoder_score,
                "keyword_overlap_score": keyword_overlap,
                "target_match_score": target_match,
                "source_quality_score": source_quality,
                "recency_score": recency_score,
                "graph_relevance_score": graph_relevance,
                "event_support_score": event_support,
                "duplicate_penalty": duplicate_penalty,
                "ambiguity_penalty": ambiguity_penalty,
                "signal_score": signal_score,
                "final_score": float(final_score),
                "score": float(final_score),
                "score_components": {
                    "semantic_score": semantic_score,
                    "cross_encoder_score": cross_encoder_score,
                    "keyword_overlap_score": keyword_overlap,
                    "target_match_score": target_match,
                    "source_quality_score": source_quality,
                    "recency_score": recency_score,
                    "graph_relevance_score": graph_relevance,
                    "event_support_score": event_support,
                    "duplicate_penalty": duplicate_penalty,
                    "ambiguity_penalty": ambiguity_penalty,
                    "signal_score": signal_score,
                    "novelty_score": _clamp01(_safe_float(row.get("novelty_score"), 0.0)),
                    "velocity_score": _clamp01(_safe_float(row.get("velocity_score"), 0.0)),
                    "asset_impact_score": _clamp01(_safe_float(row.get("asset_impact_score"), 0.0)),
                    "confidence_score": _clamp01(_safe_float(row.get("confidence_score"), 0.0)),
                },
                "scoring_weights": dict(weights),
            }
        )
        seen_ids.add(signal_id)
        seen_ids.add(cluster_id)
        if fingerprint:
            seen_texts.add(fingerprint)

    scored_rows.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
    evidence_map = _load_signal_evidence_chunks(
        sqlite_conn,
        [str(row["cluster_id"]) for row in scored_rows],
        date_start=date_start,
        date_end=date_end,
        source_filter=source_filter,
        per_cluster_limit=3,
    )

    selected_signals: list[dict[str, Any]] = []
    for row in scored_rows:
        evidence_chunks = evidence_map.get(str(row["cluster_id"]), [])
        if not evidence_chunks:
            continue
        selected_signals.append({**row, "evidence_chunks": evidence_chunks})
        if len(selected_signals) >= top_limit:
            break

    return selected_signals[:expanded_limit], target, {
        "route_type": "signal_discovery",
        "candidate_count": len(raw_rows),
        "ranked_count": len(scored_rows),
        "selected_signal_count": len(selected_signals),
        "date_start": date_start,
        "date_end": date_end,
        "source_filter": source_filter,
        "target": {
            "canonical_name": target.canonical_name,
            "display_name": target.display_name,
            "ticker": target.ticker,
            "ambiguity_score": target.ambiguity_score,
            "needs_disambiguation": target.needs_disambiguation,
            "resolution_mode": target.resolution_mode,
        },
        "ranked_candidates": scored_rows[:expanded_limit],
    }


def _cluster_summary_themes(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    themed: list[dict[str, Any]] = []
    theme_counts: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        text = f"{chunk.get('title', '')} {chunk.get('text', '')}".lower()
        best_theme = "general"
        best_label = "General"
        best_hits = 0
        for theme_id, (label, keywords) in SUMMARY_THEME_KEYWORDS.items():
            hits = sum(1 for keyword in keywords if keyword in text)
            if hits > best_hits:
                best_hits = hits
                best_theme = theme_id
                best_label = label
        confidence = _clamp01(0.35 + 0.12 * best_hits) if best_hits > 0 else 0.30
        theme_counts[best_theme] += 1
        themed.append(
            {
                **chunk,
                "theme_id": best_theme,
                "theme_label": best_label,
                "theme_confidence": confidence,
            }
        )
    themed.sort(
        key=lambda item: (
            theme_counts.get(str(item.get("theme_id") or "general"), 0),
            _safe_float(item.get("theme_confidence"), 0.0),
            _safe_float(item.get("final_score"), _safe_float(item.get("score"), 0.0)),
        ),
        reverse=True,
    )
    return themed


def _compute_answer_confidence(
    chunks: list[dict],
    target: QueryTarget | None,
    decision_hint: str | None = None,
    route_type: str | None = None,
) -> tuple[float, dict[str, Any]]:
    relevant_chunks = _clamp01(len(chunks) / 8.0)
    unique_sources = len({(chunk.get("source") or "").lower() for chunk in chunks if chunk.get("source")})
    source_diversity = _clamp01(unique_sources / 4.0)

    top_score = _safe_float(chunks[0].get("final_score"), _safe_float(chunks[0].get("score"), 0.0)) if chunks else 0.0
    second_score = _safe_float(chunks[1].get("final_score"), _safe_float(chunks[1].get("score"), 0.0)) if len(chunks) > 1 else 0.0
    retrieval_margin = _clamp01(max(0.0, top_score - second_score))

    verifier_support_vals = [_safe_float(chunk.get("event_support_score"), -1.0) for chunk in chunks]
    verifier_support_vals = [value for value in verifier_support_vals if value >= 0.0]
    verifier_support = _clamp01(sum(verifier_support_vals) / len(verifier_support_vals)) if verifier_support_vals else 0.45

    recency_vals = [_safe_float(chunk.get("recency_score"), -1.0) for chunk in chunks]
    recency_vals = [value for value in recency_vals if value >= 0.0]
    recency_coverage = _clamp01(sum(recency_vals) / len(recency_vals)) if recency_vals else 0.0

    ambiguity_penalty = _clamp01(target.ambiguity_score if target else 0.0)
    contradiction_penalty = _compute_contradiction_signal(chunks)

    score_0_1 = (
        ANSWER_CONFIDENCE_WEIGHTS["relevant_chunks"] * relevant_chunks
        + ANSWER_CONFIDENCE_WEIGHTS["source_diversity"] * source_diversity
        + ANSWER_CONFIDENCE_WEIGHTS["retrieval_margin"] * retrieval_margin
        + ANSWER_CONFIDENCE_WEIGHTS["verifier_support"] * verifier_support
        + ANSWER_CONFIDENCE_WEIGHTS["recency_coverage"] * recency_coverage
        - ANSWER_CONFIDENCE_WEIGHTS["ambiguity_penalty"] * ambiguity_penalty
        - ANSWER_CONFIDENCE_WEIGHTS["contradiction_penalty"] * contradiction_penalty
    )
    confidence = max(0.0, min(100.0, round(100.0 * score_0_1, 1)))

    stream_brief_count = sum(1 for chunk in chunks if _is_stream_brief_chunk(chunk))
    stream_brief_share = (stream_brief_count / len(chunks)) if chunks else 0.0
    if stream_brief_share > STREAM_BRIEF_CONFIDENCE_PENALTY_THRESHOLD:
        over = min(
            1.0,
            (stream_brief_share - STREAM_BRIEF_CONFIDENCE_PENALTY_THRESHOLD)
            / max(1e-6, 1.0 - STREAM_BRIEF_CONFIDENCE_PENALTY_THRESHOLD),
        )
        damp = 1.0 - 0.25 * over
        confidence = max(0.0, round(confidence * damp, 1))
    if stream_brief_share >= STREAM_BRIEF_DOMINANT_THRESHOLD:
        confidence = min(confidence, 32.0)
    elif stream_brief_share >= 0.70:
        confidence = min(confidence, 55.0)

    if decision_hint == "ambiguous":
        confidence = min(confidence, 55.0)

    signals = {
        "relevant_chunks": relevant_chunks,
        "source_diversity": source_diversity,
        "retrieval_margin": retrieval_margin,
        "verifier_support": verifier_support,
        "recency_coverage": recency_coverage,
        "ambiguity_score": ambiguity_penalty,
        "contradiction_signals": contradiction_penalty,
        "stream_brief_count": stream_brief_count,
        "stream_brief_share": stream_brief_share,
        "route_type": route_type,
    }
    return confidence, signals


def _decide_answer_mode(answer_confidence: float) -> str:
    if answer_confidence < 35.0:
        return "abstain"
    if answer_confidence <= 60.0:
        return "cautious_answer"
    return "answer"


def _db_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return set()
    return {str(row["name"] if isinstance(row, sqlite3.Row) else row[1]) for row in rows}


def _insert_row_if_columns_exist(conn: sqlite3.Connection, table: str, payload: dict[str, Any]) -> None:
    cols = _db_columns(conn, table)
    if not cols:
        return
    row = {k: v for k, v in payload.items() if k in cols}
    if not row:
        return
    keys = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    conn.execute(f"INSERT OR REPLACE INTO {table} ({keys}) VALUES ({placeholders})", tuple(row.values()))


def _log_observability(
    *,
    conn: sqlite3.Connection,
    run_id: str,
    query: str,
    route_type: str,
    target: QueryTarget | None,
    candidates: list[dict[str, Any]],
    selected_chunks: list[dict[str, Any]],
    selected_signals: list[dict[str, Any]] | None = None,
    answer_confidence: float,
    decision: str,
    latency_ms: float,
    retrieval_trace: dict[str, Any],
    route_reason: dict[str, Any] | None = None,
    answer_meta: dict[str, Any] | None = None,
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    selected_keys = {
        str(item)
        for item in (
            *(chunk.get("chunk_uid") for chunk in selected_chunks),
            *(chunk.get("candidate_id") for chunk in selected_chunks),
            *(chunk.get("signal_id") for chunk in selected_chunks),
            *(chunk.get("cluster_id") for chunk in selected_chunks),
        )
        if item
    }
    selected_signal_payload = selected_signals or []

    for index, candidate in enumerate(candidates):
        chunk_uid = candidate.get("chunk_uid")
        candidate_id = str(candidate.get("candidate_id") or f"{run_id}::{chunk_uid or 'candidate'}::{index}")
        selected = (
            candidate_id in selected_keys
            or str(candidate.get("signal_id") or "") in selected_keys
            or str(candidate.get("cluster_id") or "") in selected_keys
            or str(chunk_uid or "") in selected_keys
        )
        _insert_row_if_columns_exist(
            conn,
            "retrieval_candidates",
            {
                "run_id": run_id,
                "candidate_id": candidate_id,
                "candidate_kind": candidate.get("retrieval_kind") or "chunk",
                "article_id": candidate.get("article_id"),
                "chunk_id": chunk_uid,
                "macro_event_id": candidate.get("macro_event_id"),
                "cluster_id": candidate.get("cluster_id"),
                "signal_id": candidate.get("signal_id"),
                "semantic_score": candidate.get("semantic_score"),
                "cross_encoder_score": candidate.get("cross_encoder_score"),
                "keyword_overlap_score": candidate.get("keyword_overlap_score"),
                "target_match_score": candidate.get("target_match_score"),
                "source_quality_score": candidate.get("source_quality_score"),
                "recency_score": candidate.get("recency_score"),
                "graph_relevance_score": candidate.get("graph_relevance_score"),
                "event_support_score": candidate.get("event_support_score"),
                "duplicate_penalty": candidate.get("duplicate_penalty"),
                "ambiguity_penalty": candidate.get("ambiguity_penalty"),
                "final_score": candidate.get("final_score"),
                "score_trace_json": json.dumps(
                    {
                        **(candidate.get("score_components") or {}),
                        "signal_score": candidate.get("signal_score"),
                    },
                    ensure_ascii=False,
                ),
                "selected": 1 if selected else 0,
                "created_at": created_at,
            },
        )

    _insert_row_if_columns_exist(
        conn,
            "qa_runs",
            {
                "run_id": run_id,
                "query": query,
                "route_type": route_type,
                "route_reason": json.dumps(route_reason or {}, ensure_ascii=False),
                "route_decision_json": json.dumps(route_reason or {}, ensure_ascii=False),
                "resolved_target_json": json.dumps(
                    {
                        "canonical_name": target.canonical_name if target else None,
                        "display_name": target.display_name if target else None,
                        "ticker": target.ticker if target else None,
                    "entity_type": target.entity_type if target else None,
                    "best_candidate": target.best_candidate if target else None,
                    "candidates": target.candidates if target else [],
                    "ambiguity_score": target.ambiguity_score if target else None,
                    "resolution_mode": target.resolution_mode if target else None,
                    "needs_disambiguation": target.needs_disambiguation if target else None,
                },
                ensure_ascii=False,
            ),
            "retrieval_trace_json": json.dumps(retrieval_trace, ensure_ascii=False),
            "selected_chunks_json": json.dumps(
                [
                    {
                        "chunk_uid": chunk.get("chunk_uid"),
                        "article_id": chunk.get("article_id"),
                        "source": chunk.get("source"),
                        "title": chunk.get("title"),
                        "theme_id": chunk.get("theme_id"),
                        "theme_label": chunk.get("theme_label"),
                        "score": chunk.get("final_score", chunk.get("score")),
                    }
                    for chunk in selected_chunks
                ],
                ensure_ascii=False,
            ),
            "selected_macro_events_json": json.dumps(
                [
                    {
                        "macro_event_id": chunk.get("macro_event_id"),
                        "verification_status": chunk.get("verification_status"),
                        "support_score": chunk.get("support_score"),
                        "confidence_calibrated": chunk.get("confidence_calibrated"),
                    }
                    for chunk in selected_chunks
                    if chunk.get("macro_event_id")
                ],
                ensure_ascii=False,
            ),
            "selected_signals_json": json.dumps(selected_signal_payload, ensure_ascii=False),
            "selected_signal_alerts_json": json.dumps(selected_signal_payload, ensure_ascii=False),
            "answer_confidence": answer_confidence,
            "decision": decision,
            "answer_meta_json": json.dumps(answer_meta or {}, ensure_ascii=False),
            "answer_decision_json": json.dumps(answer_meta or {}, ensure_ascii=False),
            "latency": latency_ms,
            "created_at": created_at,
        },
    )
    conn.commit()


def load_reranker():
    global _RERANKER
    if not ENABLE_CROSS_ENCODER_RERANK:
        return None
    if _RERANKER is not None:
        return _RERANKER
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("  [reranker unavailable: CrossEncoder import failed]")
        _RERANKER = False
        return None
    try:
        print(f"Loading reranker: {RERANKER_MODEL_NAME}")
        _RERANKER = CrossEncoder(RERANKER_MODEL_NAME)
        return _RERANKER
    except Exception as exc:
        print(f"  [reranker unavailable: {exc}]")
        _RERANKER = False
        return None


def apply_reranker(query: str, ranked_rows: list[dict], reranker=None) -> list[dict]:
    if not ranked_rows:
        return ranked_rows
    active_reranker = reranker if reranker is not None else load_reranker()
    if not active_reranker:
        return ranked_rows

    pool_size = min(RERANK_CANDIDATE_LIMIT, len(ranked_rows))
    pool = ranked_rows[:pool_size]
    try:
        scores = active_reranker.predict(
            [(query, row.get("text", "")) for row in pool],
            show_progress_bar=False,
        )
    except TypeError:
        scores = active_reranker.predict([(query, row.get("text", "")) for row in pool])
    reranked = []
    for row, rerank_score in zip(pool, scores):
        raw = float(rerank_score)
        cross_score = _clamp01(_sigmoid(raw / 8.0))
        combined_score = (1.0 - RERANK_WEIGHT) * float(row.get("score") or 0.0) + RERANK_WEIGHT * cross_score
        reranked.append(
            {
                **row,
                "rerank_score_raw": raw,
                "cross_encoder_score": cross_score,
                "score": combined_score,
            }
        )
    reranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return reranked + ranked_rows[pool_size:]


def build_citation_map(chunks: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    next_idx = 1
    for chunk in chunks:
        chunk_uid = chunk.get("chunk_uid")
        if not chunk_uid or chunk_uid in mapping:
            continue
        mapping[chunk_uid] = f"S{next_idx}"
        next_idx += 1
    return mapping


def format_provenance(chunks: list[dict]) -> str:
    if not chunks:
        return "Why this answer: no retrieved evidence."
    citation_map = build_citation_map(chunks)
    lines = [
        "Why this answer:",
        "  [M] = market price data (FMP/Yahoo feed) — NOT from article database",
    ]
    for chunk in chunks:
        chunk_uid = chunk.get("chunk_uid")
        if not chunk_uid:
            continue
        lines.append(
            f"- [{citation_map[chunk_uid]}] {chunk.get('source', '?')} | "
            f"{chunk.get('title', '?')} | retrieval={chunk.get('retrieval_kind', '?')} | chunk={chunk_uid}"
        )
        if chunk.get("macro_summary"):
            lines.append(f"  macro={chunk['macro_summary']}")
        if chunk.get("evidence_text"):
            lines.append(f"  evidence={chunk['evidence_text']}")
    return "\n".join(lines)


def ensure_structured_answer(answer: str, chunks: list[dict]) -> str:
    answer = (answer or "").strip()
    citation_map = build_citation_map(chunks)

    if not answer:
        answer = "The retrieved evidence is insufficient to answer confidently."

    has_answer = bool(re.search(r"(?im)^\s*answer\s*:", answer))
    has_evidence = bool(re.search(r"(?im)^\s*evidence\s*:", answer))
    has_theory = bool(re.search(r"(?im)^\s*theory\s*:", answer))
    if has_answer and has_evidence and has_theory:
        return answer

    def _extract_section(label: str) -> str:
        pattern = rf"(?is)^\s*{label}\s*:(.*?)(?=^\s*(?:Answer|Evidence|Theory)\s*:|\Z)"
        match = re.search(pattern, answer, flags=re.MULTILINE)
        return (match.group(1).strip() if match else "")

    def _cap_sentences(text: str, max_sentences: int) -> str:
        text = re.sub(r"\s+", " ", (text or "").strip())
        if not text:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:max_sentences]).strip()

    def _compress_number_heavy_text(text: str) -> str:
        text = re.sub(r"\s+", " ", (text or "").strip())
        number_count = len(re.findall(r"(?:\b\d+(?:\.\d+)?%?|\$[\d,.]+)", text))
        if number_count >= 5:
            clauses = re.split(r"(?<=[.!?])\s+|;\s+", text)
            text = " ".join(clauses[:2]).strip()
        return text

    def _chunk_citation_line(chunk: dict[str, Any]) -> str | None:
        chunk_uid = chunk.get("chunk_uid")
        if not chunk_uid:
            return None
        citation = citation_map.get(chunk_uid)
        if not citation:
            return None
        source = (chunk.get("source") or "unknown source").strip()
        text = re.sub(r"\s+", " ", (chunk.get("text") or "").strip())
        snippet = _cap_sentences(_compress_number_heavy_text(text), 2)
        snippet = snippet[:180].rstrip()
        if snippet and len(text) > len(snippet):
            snippet += "..."
        if not snippet:
            snippet = (chunk.get("title") or "no snippet available").strip()
        return f"- [{citation}] {source}: {snippet}"

    evidence_lines = [line for line in (_chunk_citation_line(chunk) for chunk in chunks[:3]) if line]
    inline_refs = " ".join(f"[{ref}]" for ref in list(citation_map.values())[:3])

    answer_section = _extract_section("Answer")
    evidence_section = _extract_section("Evidence")
    theory_section = _extract_section("Theory")

    if not (has_answer or has_evidence or has_theory):
        answer_section = answer
    if not answer_section:
        answer_section = "The retrieved evidence does not support a confident conclusion."
    answer_section = _cap_sentences(answer_section, 4)
    if inline_refs and inline_refs not in answer_section:
        answer_section = f"{answer_section} {inline_refs}".strip()

    if not evidence_section:
        if evidence_lines:
            evidence_section = "\n".join(evidence_lines)
        else:
            evidence_section = "No concrete evidence snippets were retrieved."
    else:
        evidence_lines_existing = [line for line in evidence_section.splitlines() if line.strip()]
        if evidence_lines_existing:
            if len(evidence_lines_existing) == 1:
                evidence_section = _cap_sentences(evidence_lines_existing[0], 3)
            else:
                evidence_section = "\n".join(evidence_lines_existing[:3])

    if not theory_section:
        if chunks:
            theory_section = (
                "The conclusion is constrained by the cited evidence; causal interpretation remains tentative."
            )
        else:
            theory_section = "No grounded mechanism can be stated without additional evidence."
    theory_section = _cap_sentences(theory_section, 3)
    if re.sub(r"\W+", " ", theory_section).strip().lower() == re.sub(r"\W+", " ", answer_section).strip().lower():
        theory_section = "No additional inference is needed beyond the cited answer."

    return (
        f"Answer: {answer_section}\n"
        f"Evidence: {evidence_section}\n"
        f"Theory: {theory_section}"
    )


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
            a.source,
            a.source_trust_tier,
            a.content_class,
            a.article_quality_score,
            a.quality_flags_json
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


def has_signal_discovery_intent(query: str) -> bool:
    q = (query or "").strip().lower()
    return any(h in q for h in ROUTE_HINTS["signal_discovery"])


def is_explicit_latest_news_query(query: str) -> bool:
    q = (query or "").strip().lower()
    has_news_term = any(term in q for term in (" news", "news ", "headline", "headlines", "news update"))
    has_latest_term = any(h in q for h in ROUTE_HINTS["latest_news"])
    return has_news_term and has_latest_term


def is_single_ticker_financial_intent(query: str) -> bool:
    q = (query or "").strip().lower()
    return any(hint in q for hint in SINGLE_TICKER_FINANCIAL_INTENT_HINTS)


def is_market_data_intent(query: str) -> bool:
    q = query.lower()
    return any(hint in q for hint in MARKET_INTENT_HINTS)

def is_summary_query(query: str) -> bool:
    q = (query or "").strip().lower()
    has_summary = any(term in q for term in SUMMARY_TERMS)
    has_time = any(term in q for term in SUMMARY_TIME_TERMS)
    has_domain = any(term in q for term in SUMMARY_DOMAIN_TERMS)
    return has_summary and (has_time or has_domain)


def infer_summary_date_range(query: str) -> tuple[str | None, str | None]:
    q = (query or "").strip().lower()
    today = datetime.now(timezone.utc).date()

    has_today = any(w in q for w in ("today", "today's", "todays", "daily"))
    has_yesterday = any(w in q for w in ("yesterday", "yesterday's"))

    # Multi-day query: let decompose_query handle each day's date range separately.
    if has_today and has_yesterday:
        return None, None

    if has_yesterday:
        d = today - timedelta(days=1)
        return d.isoformat(), d.isoformat()

    if has_today:
        return today.isoformat(), today.isoformat()

    return None, None


def _build_summary_date_resolution_block(
    query: str,
    forced_start: str | None,
    forced_end: str | None,
    used_date_ranges: list[tuple[str | None, str | None]],
) -> str:
    """
    Build a DATE RESOLUTION preamble for summary-mode contexts.

    The LLM retrieves chunks that are already filtered to the correct calendar
    dates, but without an explicit binding it will guess what "today" and
    "yesterday" mean from training-data priors and get them wrong.  This block
    pins each relative label to its exact ISO-8601 date so the model uses the
    right dates when writing its answer.
    """
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    q = query.lower()

    has_today = any(w in q for w in ("today", "today's", "todays", "daily"))
    has_yesterday = any(w in q for w in ("yesterday", "yesterday's"))

    lines: list[str] = [
        "DATE RESOLUTION",
        "(These are the exact calendar dates for this query. "
        "Use absolute dates in your answer; do NOT write 'today' or 'yesterday'.)",
    ]
    if has_today:
        lines.append(f'  "today"     = {today.isoformat()}')
    if has_yesterday:
        lines.append(f'  "yesterday" = {yesterday.isoformat()}')

    # Compute the union of all date ranges actually used for retrieval.
    all_starts: list[str] = [s for s, _ in used_date_ranges if s]
    all_ends: list[str] = [e for _, e in used_date_ranges if e]
    if forced_start:
        all_starts.append(forced_start)
    if forced_end:
        all_ends.append(forced_end)

    if all_starts or all_ends:
        cov_start = min(all_starts) if all_starts else "unknown"
        cov_end = max(all_ends) if all_ends else "unknown"
        lines.append(f"  Coverage    : {cov_start} to {cov_end}")

    return "\n".join(lines)


def dedupe_chunks_for_summary(chunks: list[dict], max_per_article: int = 2) -> list[dict]:
    seen_chunk_ids: set[str] = set()
    per_article_counts: dict[str, int] = {}
    out: list[dict] = []

    for ch in chunks:
        chunk_uid = ch.get("chunk_uid")
        article_id = ch.get("article_id")

        if chunk_uid and chunk_uid in seen_chunk_ids:
            continue
        if chunk_uid:
            seen_chunk_ids.add(chunk_uid)

        if article_id:
            count = per_article_counts.get(article_id, 0)
            if count >= max_per_article:
                continue
            per_article_counts[article_id] = count + 1

        out.append(ch)

    return out


def _filter_summary_chunks(
    chunks: list[dict],
    max_per_source: int = SUMMARY_MAX_CHUNKS_PER_SOURCE,
    dup_sim_threshold: float = SUMMARY_DUPLICATE_SIM_THRESHOLD,
) -> list[dict]:
    """
    Post-retrieval quality filter applied once to the globally-merged candidate
    pool in the daily-summary path.

    Pass 1 – near-duplicate removal:
      Iterates chunks in rank order (highest rank first). Any chunk whose
      similarity to an already-kept chunk meets or exceeds dup_sim_threshold
      is dropped; the higher-ranked (earlier) chunk is always kept.
      - When both chunks have embeddings: uses cosine similarity.
      - When either embedding is absent: falls back to word-token Jaccard
        similarity on the chunk text.

    Pass 2 – per-source cap:
      At most max_per_source chunks from each source domain are retained;
      higher-ranked chunks win within each source.

    Debug lines are always printed so they appear in terminal logs.
    """
    if not chunks:
        return chunks

    print(f"  [summary-filter] candidates={len(chunks)}")

    def _jaccard(text_a: str, text_b: str) -> float:
        toks_a = set(re.findall(r"[a-z0-9]{3,}", (text_a or "").lower()))
        toks_b = set(re.findall(r"[a-z0-9]{3,}", (text_b or "").lower()))
        if not toks_a or not toks_b:
            return 0.0
        return len(toks_a & toks_b) / len(toks_a | toks_b)

    # Pass 1: drop near-duplicates -----------------------------------------
    kept_after_dedup: list[dict] = []
    kept_vecs: list[np.ndarray | None] = []
    dropped_dup_uids: list[str] = []

    for ch in chunks:
        uid = ch.get("chunk_uid") or ch.get("article_id") or "?"
        raw_emb = ch.get("embedding")
        vec = np.asarray(raw_emb, dtype=np.float32) if raw_emb is not None else None

        is_dup = False
        for i, prev in enumerate(kept_after_dedup):
            prev_vec = kept_vecs[i]
            if vec is not None and prev_vec is not None:
                sim = cosine_sim(vec, prev_vec)
            else:
                sim = _jaccard(ch.get("text", ""), prev.get("text", ""))
            if sim >= dup_sim_threshold:
                is_dup = True
                break

        if is_dup:
            dropped_dup_uids.append(uid)
            continue

        kept_after_dedup.append(ch)
        kept_vecs.append(vec)

    if dropped_dup_uids:
        print(
            f"  [summary-filter] duplicates dropped "
            f"(sim>={dup_sim_threshold}): {dropped_dup_uids}"
        )

    # Pass 2: per-source cap -----------------------------------------------
    source_counts: dict[str, int] = {}
    source_cap_uids: list[str] = []
    out: list[dict] = []

    for ch in kept_after_dedup:
        src = (ch.get("source") or "unknown").strip().lower()
        count = source_counts.get(src, 0)
        if count >= max_per_source:
            source_cap_uids.append(ch.get("chunk_uid") or ch.get("article_id") or "?")
            continue
        source_counts[src] = count + 1
        out.append(ch)

    if source_cap_uids:
        print(
            f"  [summary-filter] source-cap drops "
            f"(max {max_per_source}/source): {source_cap_uids}"
        )

    # Pass 3: keep stream briefs useful but prevent domination in summaries.
    if out:
        max_stream_briefs = max(
            1,
            int(round(max(0.10, min(0.95, SUMMARY_STREAM_BRIEF_MAX_SHARE)) * len(out))),
        )
        stream_kept = 0
        stream_share_drops: list[str] = []
        capped_out: list[dict] = []
        for ch in out:
            if _is_stream_brief_chunk(ch):
                if stream_kept >= max_stream_briefs:
                    stream_share_drops.append(ch.get("chunk_uid") or ch.get("article_id") or "?")
                    continue
                stream_kept += 1
            capped_out.append(ch)
        if stream_share_drops:
            print(
                f"  [summary-filter] stream-brief share drops "
                f"(max {max_stream_briefs}): {stream_share_drops}"
            )
        out = capped_out

        fuller_available = [c for c in kept_after_dedup if _is_full_context_chunk(c)]
        fuller_in_out = sum(1 for ch in out if _is_full_context_chunk(ch))
        if fuller_available and fuller_in_out < SUMMARY_MIN_FULL_CONTEXT_CHUNKS:
            needed = SUMMARY_MIN_FULL_CONTEXT_CHUNKS - fuller_in_out
            existing_ids = {ch.get("chunk_uid") for ch in out if ch.get("chunk_uid")}
            replacements: list[dict] = []
            for candidate in fuller_available:
                cid = candidate.get("chunk_uid")
                if cid and cid in existing_ids:
                    continue
                replacements.append(candidate)
                if len(replacements) >= needed:
                    break

            if replacements:
                stream_positions = [idx for idx, ch in enumerate(out) if _is_stream_brief_chunk(ch)]
                while replacements and stream_positions:
                    pos = stream_positions.pop()
                    out[pos] = replacements.pop(0)
                while replacements:
                    out.append(replacements.pop(0))
                print(
                    f"  [summary-filter] injected fuller-context chunks "
                    f"to maintain summary balance (target={SUMMARY_MIN_FULL_CONTEXT_CHUNKS})"
                )

    unique_src_list = sorted({(ch.get("source") or "unknown").strip().lower() for ch in out})
    print(
        f"  [summary-filter] selected={len(out)} | "
        f"unique sources ({len(unique_src_list)}): {unique_src_list}"
    )
    return out


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

(
    SYSTEM_PROMPT_TEMPLATE,
    CAUSAL_SYSTEM_PROMPT_TEMPLATE,
    DAILY_SUMMARY_PROMPT_TEMPLATE,
    SINGLE_TICKER_FINANCIAL_PROMPT_TEMPLATE,
) = load_prompt_templates(PROMPT_TEMPLATES_PATH)

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
    reranker=None,
) -> tuple[list[dict], QueryTarget]:
    """
    Run retrieve() for each hop entity and merge the results.
    Returns (all_chunks, hop_labels_per_chunk, primary_target).

    Each chunk gets an 'expansion_kind' label showing which hop it came from,
    so build_context() surfaces this in the context passed to the LLM.
    """
    # Normalize and dedupe hops to avoid repeated full retrieval passes.
    normalized_hops: list[str] = []
    seen_hops: set[str] = set()
    for hop in hops:
        hop_norm = str(hop).strip().lower()
        if not hop_norm or hop_norm in seen_hops:
            continue
        seen_hops.add(hop_norm)
        normalized_hops.append(hop_norm)

    all_chunks: list[dict] = []
    seen_uids: set[str] = set()
    primary_target: QueryTarget | None = None

    for hop in normalized_hops:
        try:
            # Reuse the full retrieve() stack — entity resolution + 3-layer retrieval
            hop_chunks, hop_target, _ = retrieve(
                query=hop,          # retrieve by canonical entity name, not the full query
                embed_model=embed_model,
                reranker=reranker,
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

        added_from_hop = 0
        for ch in hop_chunks[:chunks_per_hop]:
            uid = ch.get("chunk_uid")
            if uid and uid not in seen_uids:
                seen_uids.add(uid)
                ch["expansion_kind"] = f"causal_hop:{hop}"
                all_chunks.append(ch)
                added_from_hop += 1

        print(f"  [causal hop: {hop} | chunks: {added_from_hop}]")

    # Fall back to a general semantic search on the original query
    # in case hop-entity retrieval returned nothing useful
    if not all_chunks:
        try:
            sem_chunks, sem_target, _ = retrieve(
                query=query,
                embed_model=embed_model,
                reranker=reranker,
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
    lookback_days: int = 365,
    query: str | None = None,
) -> str:
    """
    For any hop in the causal chain that maps to a known market symbol,
    fetch FinanceToolkit financial context and format it as supplemental
    [F]-citable context for injection into mixed retrieval prompts.

    Unlike the single-ticker financial route, this function must not tell the
    model to use only [F] data because causal prompts can also include article
    evidence that remains citable as [Sx].
    """
    symbols: list[str] = []
    symbol_to_hop: dict[str, str] = {}
    for hop in hops:
        hop_text = str(hop or "").strip()
        if not hop_text:
            continue
        normalized_hop = hop_text.lower().replace(" ", "-")
        symbol = (
            CAUSAL_ENTITY_TICKER_MAP.get(hop_text)
            or CAUSAL_ENTITY_TICKER_MAP.get(hop_text.lower())
            or CAUSAL_ENTITY_TICKER_MAP.get(normalized_hop)
        )
        if symbol and symbol not in symbols:
            symbols.append(symbol)
            symbol_to_hop[symbol] = hop_text

    if not symbols:
        return ""

    fetched_blocks: list[tuple[str, str, str]] = []
    safe_lookback_days = max(int(lookback_days or 365), 365)
    for symbol in symbols:
        hop = symbol_to_hop[symbol]
        try:
            block = fetch_financial_context(
                ticker=symbol,
                date_start=date_start,
                date_end=date_end,
                lookback_days=safe_lookback_days,
                include_technicals=True,
            )
        except Exception as exc:
            print(f"  [market data] financial context fetch failed for {symbol}: {exc}")
            continue
        if (block or "").strip():
            fetched_blocks.append((hop, symbol, block.strip()))

    if not fetched_blocks:
        return ""

    lines: list[str] = []
    if query and query.strip():
        lines.append(f"QUERY: {query.strip()}")
        lines.append("")
    lines.extend(
        [
            "SUPPLEMENTAL FINANCIAL DATA [F]",
            "Cite facts from these FinanceToolkit blocks as [F]. Continue to cite retrieved article/news evidence as [Sx].",
            "Source: FinancialModelingPrep via FinanceToolkit. These blocks supplement, not replace, retrieved evidence.",
        ]
    )

    for hop, symbol, block in fetched_blocks:
        lines.append("")
        lines.append(f"[{hop.upper()} | {symbol}]")
        lines.append(block)

    return "\n".join(lines)


def _append_statement_rows(
    lines: list[str],
    df,
    ticker: str,
    max_rows: int = 8,
) -> None:
    """
    Extract rows from a financetoolkit financial statement DataFrame and
    append compact formatted lines. Handles both MultiIndex (ticker × period)
    and flat (metric-indexed, period-columned) shapes.
    """
    try:
        import pandas as pd

        sub = df
        if hasattr(df.columns, "levels"):
            top = df.columns.get_level_values(0).unique().tolist()
            if ticker in top:
                sub = df[ticker]

        if sub is None or sub.empty:
            return

        period_col = sub.columns[-1]
        lines.append(f"  Period: {str(period_col)[:10]}")

        count = 0
        for metric, value in sub[period_col].items():
            if count >= max_rows:
                break
            if pd.isna(value):
                continue
            try:
                value_fmt = f"{value:,.4g}" if isinstance(value, float) else str(value)
            except Exception:
                value_fmt = str(value)
            lines.append(f"  {metric}: {value_fmt}")
            count += 1
    except Exception as exc:
        lines.append(f"  (parse error: {exc})")


def _append_latest_technical_indicators(
    lines: list[str],
    indicators_df,
    ticker: str,
    max_indicators: int = 14,
) -> int:
    """
    Append a compact latest snapshot of technical indicators.

    `collect_all_indicators()` usually returns index=date, columns=indicators
    for a single ticker, but this helper also tolerates MultiIndex columns.
    Returns the number of indicator rows appended.
    """
    try:
        import pandas as pd

        if indicators_df is None or indicators_df.empty:
            return 0

        sub = indicators_df

        # Multi-ticker safety: isolate the requested ticker if present in columns.
        if hasattr(sub, "columns") and hasattr(sub.columns, "levels"):
            for level in range(sub.columns.nlevels):
                try:
                    level_values = {str(v) for v in sub.columns.get_level_values(level)}
                except Exception:
                    continue
                if ticker in level_values:
                    try:
                        sub = sub.xs(ticker, axis=1, level=level)
                    except Exception:
                        pass
                    break

        if sub is None or sub.empty:
            return 0

        # Keep only rows with at least one non-null metric and use latest row.
        if hasattr(sub, "dropna"):
            sub = sub.dropna(how="all")
        if sub is None or sub.empty:
            return 0

        latest_idx = sub.index[-1]
        latest_row = sub.iloc[-1]
        if hasattr(latest_row, "empty") and latest_row.empty:
            return 0

        lines.append(f"  As Of: {str(latest_idx)[:10]}")

        appended = 0
        for metric, value in latest_row.items():
            if appended >= max_indicators:
                break
            if pd.isna(value):
                continue
            if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
                value_fmt = f"{float(value):.4g}"
            else:
                value_fmt = str(value)
            lines.append(f"  {metric}: {value_fmt}")
            appended += 1
        return appended
    except Exception as exc:
        lines.append(f"  (parse error: {exc})")
        return 0


# ---------------------------------------------------------------------------
# Financial context v2 — normalization and analytical helpers
# ---------------------------------------------------------------------------

def _normalize_fin_df(df, ticker: str):
    """
    Normalize a FinanceToolkit statement/ratio DataFrame into a flat
    (metric × period) frame with string index and string column labels.

    Handles:
    - None / empty input → returns None
    - MultiIndex columns where level-0 is ticker → slice to ticker sub-frame
    - MultiIndex rows → flatten by joining tuple levels with " | "
    """
    import pandas as pd

    if df is None:
        return None
    try:
        if df.empty:
            return None
    except Exception:
        return None

    sub = df
    try:
        if hasattr(sub.columns, "levels"):
            top = list(sub.columns.get_level_values(0).unique())
            if ticker in top:
                sub = sub[ticker]
            elif len(top) == 1:
                sub = sub[top[0]]
    except Exception:
        pass

    try:
        if hasattr(sub.index, "levels"):
            sub = sub.copy()
            sub.index = [
                " | ".join(str(l) for l in idx) if isinstance(idx, tuple) else str(idx)
                for idx in sub.index
            ]
    except Exception:
        pass

    try:
        if sub.empty:
            return None
    except Exception:
        return None

    return sub


def _safe_metric(df, field: str):
    """
    Return (value, period_label) for the latest non-null column of *field* in
    a normalized statement DataFrame.  Returns (None, None) on any miss.
    """
    import pandas as pd

    if df is None or field not in df.index:
        return None, None
    for col in reversed(list(df.columns)):
        try:
            v = df.loc[field, col]
            if pd.notna(v):
                return v, str(col)[:10]
        except Exception:
            continue
    return None, None


def _fmt_val(val, *, pct: bool = False, scale: str = "auto", decimals: int = 2) -> str:
    """
    Format a numeric metric value for display in the [F] block.

    scale: "auto" | "B" | "M" | "K" | "raw"
    Returns "unavailable" for None / NaN / Inf.
    """
    if val is None:
        return "unavailable"
    try:
        v = float(val)
    except Exception:
        return str(val)
    if v != v or v in (float("inf"), float("-inf")):
        return "unavailable"

    if pct:
        return f"{v:.{decimals}f}%"

    if scale == "auto":
        abs_v = abs(v)
        if abs_v >= 1e9:
            return f"${v / 1e9:.{decimals}f}B"
        if abs_v >= 1e6:
            return f"${v / 1e6:.{decimals}f}M"
        if abs_v >= 1e3:
            return f"${v / 1e3:.{decimals}f}K"
        return f"${v:.{decimals}f}"
    if scale == "B":
        return f"${v / 1e9:.{decimals}f}B"
    if scale == "M":
        return f"${v / 1e6:.{decimals}f}M"
    if scale == "K":
        return f"${v / 1e3:.{decimals}f}K"
    return f"{v:.{decimals}f}"


def _compute_derived_metrics(
    income, balance, cashflow, profile, ticker: str
) -> dict:
    """
    Compute all derived analytical metrics from normalized statement DataFrames.

    Returns: dict[str, (value, period_label, unavail_reason)]
    value is None when unavailable; unavail_reason explains why.
    """
    import pandas as pd

    m: dict = {}

    # --- raw inputs ---------------------------------------------------------
    revenue, rev_p    = _safe_metric(income,   "Revenue")
    cogs,    _        = _safe_metric(income,   "Cost of Goods Sold")
    gross,   _        = _safe_metric(income,   "Gross Profit")
    op_inc,  _        = _safe_metric(income,   "Operating Income")
    net_inc, _        = _safe_metric(income,   "Net Income")
    sga,     _        = _safe_metric(income,   "Selling, General and Administrative Expenses")
    rnd,     _        = _safe_metric(income,   "Research and Development Expenses")

    cur_assets, _     = _safe_metric(balance,  "Total Current Assets")
    cur_liabs,  _     = _safe_metric(balance,  "Total Current Liabilities")
    tot_equity, _     = _safe_metric(balance,  "Total Stockholders Equity")
    cash,       _     = _safe_metric(balance,  "Cash and Cash Equivalents")
    tot_debt,   _     = _safe_metric(balance,  "Total Debt")
    lt_debt,    _     = _safe_metric(balance,  "Long Term Debt")

    ocf,   _          = _safe_metric(cashflow, "Operating Cash Flow")
    capex, _          = _safe_metric(cashflow, "Capital Expenditure")

    # store raw for classification use
    m["_revenue"]           = (revenue,   rev_p, None)
    m["_net_income"]        = (net_inc,   rev_p, None)
    m["_ocf"]               = (ocf,       rev_p, None)
    m["_cur_assets"]        = (cur_assets, rev_p, None)
    m["_cur_liabs"]         = (cur_liabs,  rev_p, None)
    m["_tot_debt"]          = (tot_debt,   rev_p, None)
    m["_cash"]              = (cash,       rev_p, None)

    # --- revenue YoY growth -------------------------------------------------
    if income is not None and "Revenue" in income.index:
        rev_cols = [c for c in income.columns if pd.notna(income.loc["Revenue", c])]
        if len(rev_cols) >= 2:
            r_now  = income.loc["Revenue", rev_cols[-1]]
            r_prev = income.loc["Revenue", rev_cols[-2]]
            if r_prev and r_prev != 0:
                m["revenue_yoy_growth"] = (
                    (r_now - r_prev) / abs(r_prev) * 100,
                    f"{str(rev_cols[-2])[:7]}→{str(rev_cols[-1])[:7]}",
                    None,
                )
            else:
                m["revenue_yoy_growth"] = (None, None, "prior-period revenue is zero")
        else:
            m["revenue_yoy_growth"] = (None, None, "insufficient periods")
    else:
        m["revenue_yoy_growth"] = (None, None, "no Revenue history")

    # --- 3Y revenue CAGR ----------------------------------------------------
    if income is not None and "Revenue" in income.index:
        rev_cols = [c for c in income.columns if pd.notna(income.loc["Revenue", c])]
        if len(rev_cols) >= 4:
            r_end   = income.loc["Revenue", rev_cols[-1]]
            r_start = income.loc["Revenue", rev_cols[-4]]
            if r_start and r_start > 0 and r_end and r_end > 0:
                m["revenue_3y_cagr"] = (
                    ((r_end / r_start) ** (1 / 3) - 1) * 100,
                    f"{str(rev_cols[-4])[:7]}→{str(rev_cols[-1])[:7]}",
                    None,
                )
            else:
                m["revenue_3y_cagr"] = (None, None, "non-positive revenue endpoints")
        else:
            m["revenue_3y_cagr"] = (None, None, "fewer than 4 periods available")
    else:
        m["revenue_3y_cagr"] = (None, None, "no Revenue history")

    # --- margin metrics -----------------------------------------------------
    def margin(numerator, label):
        if revenue is None or revenue == 0:
            return (None, None, "missing Revenue")
        if numerator is None:
            return (None, None, f"missing {label}")
        return (numerator / revenue * 100, rev_p, None)

    if gross is not None:
        m["gross_margin"] = margin(gross, "Gross Profit")
    elif cogs is not None and revenue is not None:
        m["gross_margin"] = margin(revenue - cogs, "Gross Profit")
    else:
        m["gross_margin"] = (None, None, "missing Gross Profit and COGS")

    m["operating_margin"] = margin(op_inc,  "Operating Income")
    m["net_margin"]        = margin(net_inc, "Net Income")
    m["ocf_margin"]        = margin(ocf,     "Operating Cash Flow")

    # --- FCF ----------------------------------------------------------------
    if ocf is not None and capex is not None:
        fcf = ocf - abs(capex)
        m["free_cash_flow"]   = (fcf,  rev_p, None)
        m["fcf_margin"]       = margin(fcf, "FCF")
    else:
        reason = "missing OCF" if ocf is None else "missing CapEx"
        m["free_cash_flow"] = (None, None, reason)
        m["fcf_margin"]     = (None, None, reason)

    # --- expense ratios -----------------------------------------------------
    m["sga_pct_revenue"] = (
        (abs(sga) / revenue * 100, rev_p, None)
        if sga is not None and revenue and revenue != 0
        else (None, None, "missing SG&A" if sga is None else "missing Revenue")
    )
    m["rnd_pct_revenue"] = (
        (abs(rnd) / revenue * 100, rev_p, None)
        if rnd is not None and revenue and revenue != 0
        else (None, None, "missing R&D" if rnd is None else "missing Revenue")
    )

    # --- liquidity ----------------------------------------------------------
    if cur_assets is not None and cur_liabs and cur_liabs != 0:
        m["current_ratio"] = (cur_assets / cur_liabs, rev_p, None)
    else:
        m["current_ratio"] = (None, None, "missing Current Assets or Liabilities")

    if cash is not None and cur_liabs and cur_liabs != 0:
        m["cash_ratio"] = (cash / cur_liabs, rev_p, None)
    else:
        m["cash_ratio"] = (None, None, "missing Cash or Current Liabilities")

    # --- leverage -----------------------------------------------------------
    debt = tot_debt if tot_debt is not None else lt_debt
    if debt is not None and tot_equity and tot_equity != 0:
        m["debt_to_equity"] = (debt / tot_equity, rev_p, None)
    else:
        m["debt_to_equity"] = (None, None, "missing Debt or Equity")

    if cash is not None and debt is not None:
        m["net_cash_position"] = (cash - debt, rev_p, None)
    else:
        m["net_cash_position"] = (None, None, "missing Cash or Debt")

    # --- operating cash flow (explicit) -------------------------------------
    m["operating_cash_flow"] = (ocf, rev_p, None if ocf is not None else "missing OCF")

    # --- cash burn ----------------------------------------------------------
    if ocf is not None:
        if ocf >= 0:
            m["cash_burn_status"] = ("positive OCF — not burning cash", rev_p, None)
        else:
            if cash is not None and ocf != 0:
                months = abs(cash / (ocf / 12))
                m["cash_burn_status"] = (
                    f"burning cash; ~{months:.0f} months runway at current rate",
                    rev_p, None,
                )
            else:
                m["cash_burn_status"] = ("burning cash; runway unknown", rev_p, None)
    else:
        m["cash_burn_status"] = (None, None, "missing OCF")

    # --- market cap, EV, Price/Sales, EV/Sales ------------------------------
    market_cap = None
    if profile is not None:
        try:
            if not profile.empty:
                col = ticker if ticker in profile.columns else profile.columns[0]
                if "Market Capitalization" in profile.index:
                    mc_raw = profile.loc["Market Capitalization", col]
                    if pd.notna(mc_raw):
                        market_cap = float(mc_raw)
        except Exception:
            pass

    if market_cap is not None:
        m["market_cap"] = (market_cap, "current", None)
        if debt is not None and cash is not None:
            m["enterprise_value"] = (market_cap + debt - cash, "current", None)
        else:
            m["enterprise_value"] = (None, None, "missing Debt or Cash for EV")

        if revenue and revenue != 0:
            m["price_to_sales"] = (market_cap / revenue, rev_p, None)
            ev_v = m["enterprise_value"][0]
            m["ev_to_sales"] = (
                (ev_v / revenue, rev_p, None)
                if ev_v is not None
                else (None, None, "missing EV")
            )
        else:
            m["price_to_sales"] = (None, None, "missing Revenue")
            m["ev_to_sales"]    = (None, None, "missing Revenue or EV")
    else:
        for k in ("market_cap", "enterprise_value", "price_to_sales", "ev_to_sales"):
            m[k] = (None, None, "missing Market Cap from profile")

    return m


def _compute_price_trend(hist_df) -> dict:
    """
    Compute price trend metrics from an OHLCV DataFrame (index=date).

    Returns: dict[str, (value, date_label, unavail_reason)]
    """
    import pandas as pd

    _KEYS = [
        "latest_close", "latest_date", "return_1d", "return_5d",
        "return_1m", "return_3m", "return_6m", "return_ytd", "return_1y",
        "high_52w", "low_52w", "dist_52w_high", "dist_52w_low",
        "latest_volume", "avg_volume_20d", "volume_ratio",
    ]

    def _unavail(reason):
        return {k: (None, None, reason) for k in _KEYS}

    if hist_df is None or hist_df.empty:
        return _unavail("no historical data")
    if "Close" not in hist_df.columns:
        return _unavail("Close column missing in historical data")

    close = hist_df["Close"].dropna()
    if close.empty:
        return _unavail("all Close prices are null")

    latest_close = float(close.iloc[-1])
    latest_date  = str(close.index[-1])[:10]
    m: dict = {}
    m["latest_close"] = (latest_close, latest_date, None)
    m["latest_date"]  = (latest_date,  latest_date, None)

    def _ret(n):
        if len(close) > n:
            return (float(close.iloc[-1]) / float(close.iloc[-1 - n]) - 1) * 100
        return None

    for key, n in [
        ("return_1d", 1), ("return_5d", 5), ("return_1m", 21),
        ("return_3m", 63), ("return_6m", 126), ("return_1y", 252),
    ]:
        v = _ret(n)
        m[key] = (v, latest_date, None if v is not None else "insufficient history")

    # YTD
    try:
        yr = pd.to_datetime(close.index[-1]).year
        ytd_slice = close[pd.to_datetime(close.index) >= pd.Timestamp(f"{yr}-01-01")]
        if len(ytd_slice) >= 2:
            m["return_ytd"] = (
                (float(ytd_slice.iloc[-1]) / float(ytd_slice.iloc[0]) - 1) * 100,
                latest_date, None,
            )
        else:
            m["return_ytd"] = (None, None, "insufficient data for YTD")
    except Exception as e:
        m["return_ytd"] = (None, None, f"YTD error: {e}")

    # 52-week window
    w52   = close.tail(252)
    h52   = float(w52.max())
    l52   = float(w52.min())
    m["high_52w"]      = (h52, latest_date, None)
    m["low_52w"]       = (l52, latest_date, None)
    m["dist_52w_high"] = ((latest_close / h52 - 1) * 100, latest_date, None)
    m["dist_52w_low"]  = ((latest_close / l52 - 1) * 100, latest_date, None)

    # Volume
    if "Volume" in hist_df.columns:
        vol = hist_df["Volume"].dropna()
        if not vol.empty:
            lv    = float(vol.iloc[-1])
            avg20 = float(vol.tail(20).mean())
            m["latest_volume"]  = (lv,    latest_date, None)
            m["avg_volume_20d"] = (avg20, latest_date, None)
            m["volume_ratio"]   = (
                (lv / avg20, latest_date, None) if avg20 != 0
                else (None, None, "zero avg volume")
            )
        else:
            for k in ("latest_volume", "avg_volume_20d", "volume_ratio"):
                m[k] = (None, None, "no Volume data")
    else:
        for k in ("latest_volume", "avg_volume_20d", "volume_ratio"):
            m[k] = (None, None, "Volume column missing")

    return m


def _compute_technical_trend(hist_df, indicators_df, ticker: str) -> dict:
    """
    Compute technical trend metrics: SMAs from raw close prices, and
    RSI / MACD / Bollinger position from collect_all_indicators() output.

    Returns: dict[str, (value, date_label, unavail_reason)]
    """
    import pandas as pd

    m: dict = {}

    close: "pd.Series | None" = None
    latest_date = None
    if hist_df is not None and not hist_df.empty and "Close" in hist_df.columns:
        close = hist_df["Close"].dropna()
        if not close.empty:
            latest_date = str(close.index[-1])[:10]

    # SMAs computed from raw close prices
    for period, key in [(20, "sma_20"), (50, "sma_50"), (200, "sma_200")]:
        vs_key = f"price_vs_{key}"
        if close is not None and len(close) >= period:
            sma = float(close.tail(period).mean())
            m[key]    = (sma, latest_date, None)
            m[vs_key] = ((float(close.iloc[-1]) / sma - 1) * 100, latest_date, None)
        else:
            reason = f"fewer than {period} trading days available"
            m[key]    = (None, None, reason)
            m[vs_key] = (None, None, reason)

    # Indicators from collect_all_indicators()
    sub = None
    if indicators_df is not None and not indicators_df.empty:
        try:
            sub = indicators_df
            if hasattr(sub.columns, "levels"):
                for lvl in range(sub.columns.nlevels):
                    lvl_vals = {str(v) for v in sub.columns.get_level_values(lvl)}
                    if ticker in lvl_vals:
                        try:
                            sub = sub.xs(ticker, axis=1, level=lvl)
                        except Exception:
                            pass
                        break
            sub = sub.dropna(how="all")
        except Exception:
            sub = None

    if sub is not None and not sub.empty:
        latest_row = sub.iloc[-1]
        ind_date   = str(sub.index[-1])[:10]

        # RSI
        rsi_col = next(
            (c for c in latest_row.index if "RSI" in str(c).upper()), None
        )
        if rsi_col and pd.notna(latest_row[rsi_col]):
            m["rsi"] = (float(latest_row[rsi_col]), ind_date, None)
        else:
            m["rsi"] = (None, None, "RSI not found in indicators")

        # MACD line (exclude Signal and Histogram columns)
        macd_col = next(
            (
                c for c in latest_row.index
                if "MACD" in str(c).upper()
                and "SIGNAL" not in str(c).upper()
                and "HIST" not in str(c).upper()
            ),
            None,
        )
        if macd_col and pd.notna(latest_row[macd_col]):
            m["macd"] = (float(latest_row[macd_col]), ind_date, None)
        else:
            m["macd"] = (None, None, "MACD not found in indicators")

        # Bollinger Band position: (price − lower) / (upper − lower) × 100
        bb_upper_col = next(
            (c for c in latest_row.index
             if "BOLLINGER" in str(c).upper() and "UPPER" in str(c).upper()),
            None,
        )
        bb_lower_col = next(
            (c for c in latest_row.index
             if "BOLLINGER" in str(c).upper() and "LOWER" in str(c).upper()),
            None,
        )
        if (
            bb_upper_col and bb_lower_col
            and pd.notna(latest_row[bb_upper_col])
            and pd.notna(latest_row[bb_lower_col])
            and close is not None and not close.empty
        ):
            u = float(latest_row[bb_upper_col])
            lo = float(latest_row[bb_lower_col])
            band_width = u - lo
            if band_width != 0:
                bb_pos = (float(close.iloc[-1]) - lo) / band_width * 100
                m["bb_position"] = (bb_pos, ind_date, None)
            else:
                m["bb_position"] = (None, None, "Bollinger Band width is zero")
        else:
            m["bb_position"] = (None, None, "Bollinger Bands not found in indicators")
    else:
        for k in ("rsi", "macd", "bb_position"):
            m[k] = (None, None, "no indicator data available")

    return m


def _classify_financial_metrics(derived: dict, price: dict, technical: dict) -> dict:
    """
    Produce deterministic plain-English classifications from computed metrics.
    All thresholds are hard-coded; no LLM inference is involved.
    """
    cls: dict = {}

    def _v(d, key):
        return d.get(key, (None,))[0]

    # profitability
    nm = _v(derived, "net_margin")
    if nm is not None:
        if nm > 15:
            cls["profitability_status"] = f"highly profitable (net margin {nm:.1f}%)"
        elif nm > 5:
            cls["profitability_status"] = f"moderately profitable (net margin {nm:.1f}%)"
        elif nm > 0:
            cls["profitability_status"] = f"marginally profitable (net margin {nm:.1f}%)"
        else:
            cls["profitability_status"] = f"unprofitable (net margin {nm:.1f}%)"
    else:
        cls["profitability_status"] = "unavailable — missing net margin"

    # liquidity
    cr = _v(derived, "current_ratio")
    if cr is not None:
        if cr >= 2.0:
            cls["liquidity_status"] = f"strong (current ratio {cr:.2f})"
        elif cr >= 1.0:
            cls["liquidity_status"] = f"adequate (current ratio {cr:.2f})"
        else:
            cls["liquidity_status"] = f"tight (current ratio {cr:.2f})"
    else:
        cls["liquidity_status"] = "unavailable — missing current ratio"

    # leverage
    de = _v(derived, "debt_to_equity")
    if de is not None:
        if de < 0.3:
            cls["leverage_status"] = f"low leverage (D/E {de:.2f})"
        elif de < 1.0:
            cls["leverage_status"] = f"moderate leverage (D/E {de:.2f})"
        elif de < 2.0:
            cls["leverage_status"] = f"elevated leverage (D/E {de:.2f})"
        else:
            cls["leverage_status"] = f"high leverage (D/E {de:.2f})"
    else:
        cls["leverage_status"] = "unavailable — missing D/E ratio"

    # cash flow
    ocf_v = _v(derived, "_ocf")
    fcf_v = _v(derived, "free_cash_flow")
    if ocf_v is not None and fcf_v is not None:
        if fcf_v > 0 and ocf_v > 0:
            cls["cash_flow_status"] = "FCF positive — self-funding"
        elif ocf_v > 0:
            cls["cash_flow_status"] = "OCF positive, FCF negative — heavy CapEx cycle"
        else:
            cls["cash_flow_status"] = "OCF negative — cash consuming"
    elif ocf_v is not None:
        cls["cash_flow_status"] = (
            "OCF positive (FCF unknown)" if ocf_v > 0
            else "OCF negative — cash consuming (FCF unknown)"
        )
    else:
        cls["cash_flow_status"] = "unavailable — missing OCF"

    # valuation
    ps = _v(derived, "price_to_sales")
    if ps is not None:
        if ps < 1:
            cls["valuation_status"] = f"low valuation (P/S {ps:.1f}x)"
        elif ps < 5:
            cls["valuation_status"] = f"moderate valuation (P/S {ps:.1f}x)"
        elif ps < 15:
            cls["valuation_status"] = f"elevated valuation (P/S {ps:.1f}x)"
        else:
            cls["valuation_status"] = f"high valuation (P/S {ps:.1f}x)"
    else:
        cls["valuation_status"] = "unavailable — missing P/S ratio"

    # short-term trend
    r5d   = _v(price,     "return_5d")
    vs_20 = _v(technical, "price_vs_sma_20")
    if r5d is not None and vs_20 is not None:
        if r5d > 2 and vs_20 > 0:
            cls["short_term_trend_status"] = "bullish (5D gain + above 20D SMA)"
        elif r5d < -2 and vs_20 < 0:
            cls["short_term_trend_status"] = "bearish (5D loss + below 20D SMA)"
        else:
            cls["short_term_trend_status"] = "neutral / mixed short-term signals"
    else:
        cls["short_term_trend_status"] = "unavailable — missing 5D return or 20D SMA"

    # medium-term trend
    vs_50  = _v(technical, "price_vs_sma_50")
    vs_200 = _v(technical, "price_vs_sma_200")
    if vs_50 is not None and vs_200 is not None:
        if vs_50 > 0 and vs_200 > 0:
            cls["medium_term_trend_status"] = "uptrend (above both 50D and 200D SMA)"
        elif vs_50 < 0 and vs_200 < 0:
            cls["medium_term_trend_status"] = "downtrend (below both 50D and 200D SMA)"
        else:
            cls["medium_term_trend_status"] = "mixed medium-term trend signals"
    else:
        cls["medium_term_trend_status"] = "unavailable — missing SMA data"

    # overall profile
    flags = []
    p_stat = cls["profitability_status"]
    if p_stat.startswith("highly") or p_stat.startswith("moderately"):
        flags.append("profitable")
    if cls["cash_flow_status"].startswith("FCF positive"):
        flags.append("FCF-generative")
    if cls["liquidity_status"].startswith("strong"):
        flags.append("strong liquidity")
    if cls["leverage_status"].startswith("low"):
        flags.append("low leverage")
    mt = cls["medium_term_trend_status"]
    if mt.startswith("uptrend"):
        flags.append("uptrend")
    elif mt.startswith("downtrend"):
        flags.append("downtrend")
    cls["overall_stock_profile"] = (
        "; ".join(flags) if flags else "insufficient data for overall profile"
    )

    return cls


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_financial_context(
    ticker: str,
    date_start: str | None,
    date_end: str | None,
    lookback_days: int = 365,
    include_technicals: bool = False,
) -> str:
    """
    Fetch financial data for *ticker* via FinanceToolkit and return an
    analysis-ready FINANCIAL DATA [F] block structured in 13 sections plus a quarterly snapshot:

      1. DATA AVAILABILITY       8. CASH FLOW
      2. COMPANY PROFILE         9. VALUATION
      3. FINANCIAL SNAPSHOT     10. STOCK PRICE TREND
      3A. QUARTERLY SNAPSHOT    11. TECHNICAL INDICATORS
      4. GROWTH
      5. PROFITABILITY          12. DERIVED CLASSIFICATIONS
      6. EXPENSE STRUCTURE      13. DATA LIMITATIONS
      7. LIQUIDITY & LEVERAGE

    Every metric includes its ticker, period/date, units, and an explicit
    "unavailable — <reason>" marker when the value cannot be computed.
    No LLM inference is used to fill gaps.

    Returns empty string when FMP_API_KEY is absent, FinanceToolkit is not
    installed, Toolkit init fails, or no usable data is returned for ticker.
    """
    api_key = os.getenv("FMP_API_KEY", "").strip()
    if not api_key:
        return ""

    try:
        from financetoolkit import Toolkit
    except ImportError:
        return ""

    from datetime import datetime, timezone, timedelta
    now      = datetime.now(timezone.utc)
    end_dt   = date_end   or now.strftime("%Y-%m-%d")
    start_dt = date_start or (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    statement_lookback_days = max(int(lookback_days or 365), 365 * 4)
    default_statement_start_dt = (now - timedelta(days=statement_lookback_days)).strftime("%Y-%m-%d")
    statement_start_dt = (
        date_start
        if date_start and date_start < default_statement_start_dt
        else default_statement_start_dt
    )
    quarterly_lookback_days = max(int(lookback_days or 365), 365 * 2)
    default_quarterly_start_dt = (now - timedelta(days=quarterly_lookback_days)).strftime("%Y-%m-%d")
    quarterly_start_dt = (
        date_start
        if date_start and date_start < default_quarterly_start_dt
        else default_quarterly_start_dt
    )

    def _init_toolkit(start_date: str, *, quarterly: bool | None = None):
        kwargs = {
            "api_key": api_key,
            "start_date": start_date,
            "end_date": end_dt,
        }
        if quarterly is not None:
            kwargs["quarterly"] = quarterly
        try:
            return Toolkit([ticker], **kwargs)
        except TypeError:
            # Older FinanceToolkit versions may not expose the quarterly kwarg.
            if quarterly is None:
                raise
            kwargs.pop("quarterly", None)
            return Toolkit([ticker], **kwargs)

    try:
        toolkit = _init_toolkit(start_dt)
    except Exception as exc:
        print(f"  [financial data] Toolkit init failed for {ticker}: {exc}")
        return ""

    try:
        statement_toolkit = _init_toolkit(statement_start_dt, quarterly=False)
        statement_period_request = "annual (quarterly=False)"
    except Exception as exc:
        print(f"  [financial data] annual Toolkit init failed for {ticker}: {exc}; falling back to price Toolkit")
        statement_toolkit = toolkit
        statement_period_request = "default (annual Toolkit unavailable)"

    try:
        quarterly_toolkit = _init_toolkit(quarterly_start_dt, quarterly=True)
        quarterly_period_request = "quarterly (quarterly=True)"
    except Exception as exc:
        print(f"  [financial data] quarterly Toolkit init failed for {ticker}: {exc}; quarterly statements unavailable")
        quarterly_toolkit = None
        quarterly_period_request = "unavailable"

    # ------------------------------------------------------------------ #
    # 1. Raw data fetches — each individually guarded                     #
    # ------------------------------------------------------------------ #
    fetch_status: dict[str, str] = {}

    profile = None
    try:
        profile = toolkit.get_profile()
        fetch_status["profile"] = "ok" if (profile is not None and not profile.empty) else "empty"
    except Exception as e:
        fetch_status["profile"] = f"error: {e}"

    hist_raw = None
    hist_df: "object | None" = None
    try:
        hist_raw = toolkit.get_historical_data()
        if hist_raw is not None and not hist_raw.empty:
            if hasattr(hist_raw.columns, "levels"):
                try:
                    hist_df = hist_raw.xs(ticker, axis=1, level=1)
                except KeyError:
                    hist_df = hist_raw
            else:
                hist_df = hist_raw
            fetch_status["historical"] = "ok"
        else:
            fetch_status["historical"] = "empty"
    except Exception as e:
        fetch_status["historical"] = f"error: {e}"

    income_raw = None
    try:
        income_raw = statement_toolkit.get_income_statement()
        fetch_status["income"] = "ok" if (income_raw is not None and not income_raw.empty) else "empty"
    except Exception as e:
        fetch_status["income"] = f"error: {e}"

    balance_raw = None
    try:
        balance_raw = statement_toolkit.get_balance_sheet_statement()
        fetch_status["balance"] = "ok" if (balance_raw is not None and not balance_raw.empty) else "empty"
    except Exception as e:
        fetch_status["balance"] = f"error: {e}"

    cashflow_raw = None
    try:
        cashflow_raw = statement_toolkit.get_cash_flow_statement()
        fetch_status["cashflow"] = "ok" if (cashflow_raw is not None and not cashflow_raw.empty) else "empty"
    except Exception as e:
        fetch_status["cashflow"] = f"error: {e}"

    quarterly_income_raw = None
    quarterly_balance_raw = None
    quarterly_cashflow_raw = None
    if quarterly_toolkit is not None:
        try:
            quarterly_income_raw = quarterly_toolkit.get_income_statement()
            fetch_status["income_quarterly"] = (
                "ok" if (quarterly_income_raw is not None and not quarterly_income_raw.empty) else "empty"
            )
        except Exception as e:
            fetch_status["income_quarterly"] = f"error: {e}"

        try:
            quarterly_balance_raw = quarterly_toolkit.get_balance_sheet_statement()
            fetch_status["balance_quarterly"] = (
                "ok" if (quarterly_balance_raw is not None and not quarterly_balance_raw.empty) else "empty"
            )
        except Exception as e:
            fetch_status["balance_quarterly"] = f"error: {e}"

        try:
            quarterly_cashflow_raw = quarterly_toolkit.get_cash_flow_statement()
            fetch_status["cashflow_quarterly"] = (
                "ok" if (quarterly_cashflow_raw is not None and not quarterly_cashflow_raw.empty) else "empty"
            )
        except Exception as e:
            fetch_status["cashflow_quarterly"] = f"error: {e}"
    else:
        fetch_status["income_quarterly"] = "skipped (quarterly Toolkit unavailable)"
        fetch_status["balance_quarterly"] = "skipped (quarterly Toolkit unavailable)"
        fetch_status["cashflow_quarterly"] = "skipped (quarterly Toolkit unavailable)"

    indicators_df = None
    if include_technicals:
        try:
            indicators_df = toolkit.technicals.collect_all_indicators()
            fetch_status["indicators"] = (
                "ok" if (indicators_df is not None and not indicators_df.empty) else "empty"
            )
        except Exception as e:
            fetch_status["indicators"] = f"error: {e}"
    else:
        fetch_status["indicators"] = "skipped (include_technicals=False)"

    # Bail early if nothing at all came back
    useful = [s for s in fetch_status.values() if s == "ok"]
    if not useful:
        print(
            f"  [financial data] no usable FMP data for {ticker} "
            "(ticker may be unsupported or premium-gated)"
        )
        return ""

    # ------------------------------------------------------------------ #
    # 2. Normalise DataFrames                                             #
    # ------------------------------------------------------------------ #
    income   = _normalize_fin_df(income_raw,   ticker)
    balance  = _normalize_fin_df(balance_raw,  ticker)
    cashflow = _normalize_fin_df(cashflow_raw, ticker)
    quarterly_income = _normalize_fin_df(quarterly_income_raw, ticker)
    quarterly_balance = _normalize_fin_df(quarterly_balance_raw, ticker)
    quarterly_cashflow = _normalize_fin_df(quarterly_cashflow_raw, ticker)

    # Detect period type for financial statements
    period_type = "annual"
    if income is not None and not income.empty:
        cols = list(income.columns)
        if len(cols) >= 2:
            try:
                import pandas as pd
                dt0 = pd.to_datetime(str(cols[-2]))
                dt1 = pd.to_datetime(str(cols[-1]))
                if abs((dt1 - dt0).days) < 120:
                    period_type = "quarterly"
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # 3. Compute analytical layers                                        #
    # ------------------------------------------------------------------ #
    derived   = _compute_derived_metrics(income, balance, cashflow, profile, ticker)
    quarterly_derived = _compute_derived_metrics(
        quarterly_income,
        quarterly_balance,
        quarterly_cashflow,
        profile,
        ticker,
    )
    price_m   = _compute_price_trend(hist_df)
    tech_m    = _compute_technical_trend(hist_df, indicators_df, ticker)
    cls       = _classify_financial_metrics(derived, price_m, tech_m)

    # ------------------------------------------------------------------ #
    # 4. Format [F] block                                                 #
    # ------------------------------------------------------------------ #
    lines: list[str] = []
    generated_at = now.strftime("%Y-%m-%d")

    def _ln(text: str = "") -> None:
        lines.append(text)

    def _row(label, val, period=None, unit="", note=None):
        """Append one metric row with consistent formatting."""
        period_str = f" [{period}]" if period else ""
        unit_str   = f" {unit}"    if unit   else ""
        note_str   = f"  ({note})" if note   else ""
        lines.append(f"  {label}{period_str}: {val}{unit_str}{note_str}")

    def _m(key, label, *, d=None, pct=False, scale="auto", unit="", decimals=2):
        """Render a single derived/price/tech metric row."""
        src = d if d is not None else derived
        entry = src.get(key, (None, None, "key not found"))
        val, period, reason = entry[0], entry[1], entry[2]
        if val is None:
            _row(label, f"unavailable — {reason}", period)
        else:
            if isinstance(val, str):
                _row(label, val, period)
            else:
                fmt = _fmt_val(val, pct=pct, scale=scale, decimals=decimals)
                _row(label, fmt, period, unit=unit)

    def _latest_periods(df, field: str, limit: int = 4) -> list[tuple[str, object]]:
        if df is None or field not in df.index:
            return []
        try:
            import pandas as pd
        except Exception:
            pd = None
        periods: list[tuple[str, object]] = []
        for col in reversed(list(df.columns)):
            try:
                val = df.loc[field, col]
                is_valid = pd.notna(val) if pd is not None else val is not None
                if is_valid:
                    periods.append((str(col)[:10], val))
            except Exception:
                continue
            if len(periods) >= limit:
                break
        return periods

    def _growth_between_latest_periods(df, field: str, periods_back: int = 1):
        periods = _latest_periods(df, field, limit=periods_back + 1)
        if len(periods) <= periods_back:
            return None, None, "insufficient periods"
        latest_period, latest_val = periods[0]
        _, prior_val = periods[periods_back]
        try:
            latest_num = float(latest_val)
            prior_num = float(prior_val)
            if prior_num == 0:
                return None, latest_period, "prior-period value is zero"
            return (latest_num / abs(prior_num) - 1) * 100, latest_period, None
        except Exception as exc:
            return None, latest_period, f"calculation error: {exc}"

    # ---- Header -----------------------------------------------------------
    _ln("FINANCIAL DATA [F]")
    _ln("Cite any fact from this block as [F], NOT as [Sx] or [M].")
    _ln(f"Source: FinancialModelingPrep via FinanceToolkit | Ticker: {ticker} | Generated: {generated_at}")

    # ---- Section 1: DATA AVAILABILITY ------------------------------------
    _ln()
    _ln("1. DATA AVAILABILITY")
    for src, status in fetch_status.items():
        _ln(f"  {src}: {status}")
    _ln(f"  statement_period_request: {statement_period_request}")
    _ln(f"  statement_start_date: {statement_start_dt}")
    _ln(f"  quarterly_period_request: {quarterly_period_request}")
    _ln(f"  quarterly_start_date: {quarterly_start_dt}")
    _ln(f"  period_type_detected: {period_type}")

    # ---- Section 2: COMPANY PROFILE ---------------------------------------
    _ln()
    _ln("2. COMPANY PROFILE")
    _PROFILE_FIELDS = [
        ("Company Name",        "Company Name"),
        ("Sector",              "Sector"),
        ("Industry",            "Industry"),
        ("CEO",                 "CEO"),
        ("Country",             "Country"),
        ("Exchange Full Name",  "Exchange"),
        ("Market Capitalization","Market Cap (profile)"),
        ("Beta",                "Beta"),
        ("Full Time Employees", "Employees"),
        ("IPO Date",            "IPO Date"),
        ("Website",             "Website"),
        ("Description",         "Description"),
    ]
    if profile is not None and not profile.empty:
        try:
            col = ticker if ticker in profile.columns else profile.columns[0]
            for src_key, label in _PROFILE_FIELDS:
                if src_key in profile.index:
                    val = profile.loc[src_key, col]
                    if val and str(val).strip() not in ("", "nan", "None"):
                        val_str = str(val).strip()
                        _ln(f"  {label}: {val_str}")
                    else:
                        _ln(f"  {label}: unavailable")
                else:
                    _ln(f"  {label}: unavailable")
        except Exception as e:
            _ln(f"  (profile parse error: {e})")
    else:
        _ln(f"  unavailable — {fetch_status.get('profile', 'not fetched')}")

    # ---- Section 3: FINANCIAL SNAPSHOT ------------------------------------
    _ln()
    _ln("3. FINANCIAL SNAPSHOT")
    _ln(f"  Ticker: {ticker}")
    _ln(f"  Period Type: {period_type}")
    revenue, rev_p = _safe_metric(income, "Revenue")
    _row("Revenue", _fmt_val(revenue), rev_p)
    net_inc, ni_p = _safe_metric(income, "Net Income")
    _row("Net Income", _fmt_val(net_inc), ni_p)
    ta, ta_p = _safe_metric(balance, "Total Assets")
    _row("Total Assets", _fmt_val(ta), ta_p)
    te, te_p = _safe_metric(balance, "Total Stockholders Equity")
    _row("Total Equity", _fmt_val(te), te_p)

    mc_entry = derived.get("market_cap", (None, None, "missing"))
    mc_val   = mc_entry[0]
    _row("Market Cap", _fmt_val(mc_val) if mc_val is not None else f"unavailable - {mc_entry[2]}", mc_entry[1])

    # ---- Section 3A: QUARTERLY SNAPSHOT -----------------------------------
    _ln()
    _ln("3A. QUARTERLY SNAPSHOT")
    if quarterly_income is not None or quarterly_balance is not None or quarterly_cashflow is not None:
        _ln("  Period Type: quarterly")
        q_revenue, q_rev_p = _safe_metric(quarterly_income, "Revenue")
        _row("Latest Quarter Revenue", _fmt_val(q_revenue), q_rev_p)
        q_net_inc, q_ni_p = _safe_metric(quarterly_income, "Net Income")
        _row("Latest Quarter Net Income", _fmt_val(q_net_inc), q_ni_p)
        q_ocf, q_ocf_p = _safe_metric(quarterly_cashflow, "Operating Cash Flow")
        _row("Latest Quarter Operating Cash Flow", _fmt_val(q_ocf), q_ocf_p)
        q_fcf_entry = quarterly_derived.get("free_cash_flow", (None, None, "missing OCF or CapEx"))
        q_fcf_val, q_fcf_p, q_fcf_reason = q_fcf_entry
        _row(
            "Latest Quarter Free Cash Flow",
            _fmt_val(q_fcf_val) if q_fcf_val is not None else f"unavailable - {q_fcf_reason}",
            q_fcf_p,
        )
        q_cur_ratio = quarterly_derived.get("current_ratio", (None, None, "missing Current Assets or Liabilities"))
        _row(
            "Latest Quarter Current Ratio",
            f"{q_cur_ratio[0]:.2f}" if q_cur_ratio[0] is not None else f"unavailable - {q_cur_ratio[2]}",
            q_cur_ratio[1],
        )
        q_rev_qoq, q_rev_growth_p, q_rev_qoq_reason = _growth_between_latest_periods(quarterly_income, "Revenue", 1)
        _row(
            "Revenue QoQ Growth",
            f"{q_rev_qoq:+.1f}%" if q_rev_qoq is not None else f"unavailable - {q_rev_qoq_reason}",
            q_rev_growth_p,
        )
        q_rev_yoy, q_rev_yoy_p, q_rev_yoy_reason = _growth_between_latest_periods(quarterly_income, "Revenue", 4)
        _row(
            "Revenue YoY Growth (quarter)",
            f"{q_rev_yoy:+.1f}%" if q_rev_yoy is not None else f"unavailable - {q_rev_yoy_reason}",
            q_rev_yoy_p,
        )
        revenue_periods = _latest_periods(quarterly_income, "Revenue", 4)
        if revenue_periods:
            _ln(
                "  Recent Quarterly Revenue: "
                + "; ".join(f"{period}: {_fmt_val(value)}" for period, value in revenue_periods)
            )
        net_income_periods = _latest_periods(quarterly_income, "Net Income", 4)
        if net_income_periods:
            _ln(
                "  Recent Quarterly Net Income: "
                + "; ".join(f"{period}: {_fmt_val(value)}" for period, value in net_income_periods)
            )
    else:
        _ln(f"  unavailable - {fetch_status.get('income_quarterly', 'quarterly statements not fetched')}")

    # ---- Section 4: GROWTH ------------------------------------------------
    _ln()
    _ln("4. GROWTH")
    _m("revenue_yoy_growth", "Revenue YoY Growth",  pct=True,  scale="raw", decimals=1)
    _m("revenue_3y_cagr",    "Revenue 3Y CAGR",     pct=True,  scale="raw", decimals=1)

    # ---- Section 5: PROFITABILITY -----------------------------------------
    _ln()
    _ln("5. PROFITABILITY")
    _m("gross_margin",    "Gross Margin",     pct=True, scale="raw", decimals=1)
    _m("operating_margin","Operating Margin", pct=True, scale="raw", decimals=1)
    _m("net_margin",      "Net Margin",       pct=True, scale="raw", decimals=1)
    _m("ocf_margin",      "OCF Margin",       pct=True, scale="raw", decimals=1)
    _m("fcf_margin",      "FCF Margin",       pct=True, scale="raw", decimals=1)

    # ---- Section 6: EXPENSE STRUCTURE ------------------------------------
    _ln()
    _ln("6. EXPENSE STRUCTURE")
    _m("sga_pct_revenue", "SG&A % of Revenue", pct=True, scale="raw", decimals=1)
    _m("rnd_pct_revenue", "R&D % of Revenue",  pct=True, scale="raw", decimals=1)
    sga_v, sga_p = _safe_metric(income, "Selling, General and Administrative Expenses")
    _row("SG&A (absolute)", _fmt_val(sga_v), sga_p)
    rnd_v, rnd_p = _safe_metric(income, "Research and Development Expenses")
    _row("R&D (absolute)",  _fmt_val(rnd_v), rnd_p)

    # ---- Section 7: LIQUIDITY & LEVERAGE ----------------------------------
    _ln()
    _ln("7. LIQUIDITY & LEVERAGE")
    _m("current_ratio",    "Current Ratio",    scale="raw", decimals=2)
    _m("cash_ratio",       "Cash Ratio",       scale="raw", decimals=2)
    _m("debt_to_equity",   "Debt-to-Equity",   scale="raw", decimals=2)
    _m("net_cash_position","Net Cash Position")
    ca_v, ca_p = _safe_metric(balance, "Cash and Cash Equivalents")
    _row("Cash & Equivalents", _fmt_val(ca_v), ca_p)
    td_v, td_p = _safe_metric(balance, "Total Debt")
    _row("Total Debt",         _fmt_val(td_v) if td_v is not None else "unavailable — not in balance sheet", td_p)

    # ---- Section 8: CASH FLOW ---------------------------------------------
    _ln()
    _ln("8. CASH FLOW")
    _m("operating_cash_flow","Operating Cash Flow")
    _m("free_cash_flow",     "Free Cash Flow")
    _m("ocf_margin",         "OCF Margin",   pct=True, scale="raw", decimals=1)
    _m("fcf_margin",         "FCF Margin",   pct=True, scale="raw", decimals=1)
    capex_v, capex_p = _safe_metric(cashflow, "Capital Expenditure")
    _row("CapEx", _fmt_val(capex_v), capex_p)

    burn_entry = derived.get("cash_burn_status", (None, None, "missing OCF"))
    burn_val, burn_p, burn_reason = burn_entry
    if burn_val is None:
        _row("Cash Burn Status", f"unavailable — {burn_reason}", burn_p)
    else:
        _row("Cash Burn Status", str(burn_val), burn_p)

    # ---- Section 9: VALUATION ---------------------------------------------
    _ln()
    _ln("9. VALUATION")
    _m("market_cap",        "Market Cap")
    _m("enterprise_value",  "Enterprise Value")
    _m("price_to_sales",    "Price/Sales",    scale="raw", decimals=2, unit="x")
    _m("ev_to_sales",       "EV/Sales",       scale="raw", decimals=2, unit="x")

    # ---- Section 10: STOCK PRICE TREND ------------------------------------
    _ln()
    _ln("10. STOCK PRICE TREND")
    lc_entry = price_m.get("latest_close", (None, None, "no historical data"))
    lc_val, lc_date, lc_reason = lc_entry
    _row("Latest Close", f"${lc_val:.2f}" if lc_val is not None else f"unavailable — {lc_reason}", lc_date)

    for key, label in [
        ("return_1d",  "1D Return"),
        ("return_5d",  "5D Return"),
        ("return_1m",  "1M Return"),
        ("return_3m",  "3M Return"),
        ("return_6m",  "6M Return"),
        ("return_ytd", "YTD Return"),
        ("return_1y",  "1Y Return"),
    ]:
        entry = price_m.get(key, (None, None, "no data"))
        v, p, r = entry
        if v is not None:
            _row(label, f"{v:+.2f}%", p)
        else:
            _row(label, f"unavailable — {r}", p)

    for key, label in [
        ("high_52w",      "52-Week High"),
        ("low_52w",       "52-Week Low"),
        ("dist_52w_high", "Distance from 52W High"),
        ("dist_52w_low",  "Distance from 52W Low"),
    ]:
        entry = price_m.get(key, (None, None, "no data"))
        v, p, r = entry
        if v is not None:
            if "dist" in key:
                _row(label, f"{v:+.2f}%", p)
            else:
                _row(label, f"${v:.2f}", p)
        else:
            _row(label, f"unavailable — {r}", p)

    for key, label in [
        ("latest_volume",  "Latest Volume"),
        ("avg_volume_20d", "20D Avg Volume"),
        ("volume_ratio",   "Volume Ratio (vs 20D avg)"),
    ]:
        entry = price_m.get(key, (None, None, "no data"))
        v, p, r = entry
        if v is not None:
            if key == "volume_ratio":
                _row(label, f"{v:.2f}x", p)
            else:
                _row(label, f"{v:,.0f}", p)
        else:
            _row(label, f"unavailable — {r}", p)

    # ---- Section 11: TECHNICAL INDICATORS ---------------------------------
    _ln()
    _ln("11. TECHNICAL INDICATORS")
    for key, label, is_pct in [
        ("sma_20",         "20D SMA",              False),
        ("sma_50",         "50D SMA",              False),
        ("sma_200",        "200D SMA",             False),
        ("price_vs_sma_20", "Price vs 20D SMA",   True),
        ("price_vs_sma_50", "Price vs 50D SMA",   True),
        ("price_vs_sma_200","Price vs 200D SMA",  True),
    ]:
        entry = tech_m.get(key, (None, None, "no data"))
        v, p, r = entry
        if v is not None:
            if is_pct:
                _row(label, f"{v:+.2f}%", p)
            else:
                _row(label, f"${v:.2f}", p)
        else:
            _row(label, f"unavailable — {r}", p)

    for key, label, fmt_fn in [
        ("rsi",         "RSI",                    lambda v: f"{v:.1f}"),
        ("macd",        "MACD",                   lambda v: f"{v:.4f}"),
        ("bb_position", "Bollinger Band Position", lambda v: f"{v:.1f}% of band"),
    ]:
        entry = tech_m.get(key, (None, None, "no data"))
        v, p, r = entry
        if v is not None:
            _row(label, fmt_fn(v), p)
        else:
            _row(label, f"unavailable — {r}", p)

    # ---- Section 12: DERIVED CLASSIFICATIONS ------------------------------
    _ln()
    _ln("12. DERIVED CLASSIFICATIONS")
    for key, label in [
        ("profitability_status",    "Profitability"),
        ("liquidity_status",        "Liquidity"),
        ("leverage_status",         "Leverage"),
        ("cash_flow_status",        "Cash Flow"),
        ("valuation_status",        "Valuation"),
        ("short_term_trend_status", "Short-Term Trend"),
        ("medium_term_trend_status","Medium-Term Trend"),
        ("overall_stock_profile",   "Overall Profile"),
    ]:
        _ln(f"  {label}: {cls.get(key, 'unavailable')}")

    # ---- Section 13: DATA LIMITATIONS -------------------------------------
    _ln()
    _ln("13. DATA LIMITATIONS")
    limitations = []
    for src, status in fetch_status.items():
        if status != "ok" and not status.startswith("skipped"):
            limitations.append(f"  - {src}: {status}")
    if period_type == "quarterly":
        limitations.append("  - Statements are quarterly; YoY growth and CAGR use sequential quarters, not fiscal years.")
    if not include_technicals:
        limitations.append("  - Technical indicator data was not requested (include_technicals=False); RSI/MACD/BB unavailable.")

    if limitations:
        for lim in limitations:
            _ln(lim)
    else:
        _ln("  None — all requested data sources returned successfully.")

    return "\n".join(lines)


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

    _TODAY_RE = re.compile(r"\btoday(?:'?s)?\b", re.IGNORECASE)
    _YESTERDAY_RE = re.compile(r"\byesterday(?:'?s)?\b", re.IGNORECASE)

    if (
        _TODAY_RE.search(query)
        and _YESTERDAY_RE.search(query)
        and not MULTI_TIME_WORDS.search(query)
    ):
        ys, ye = _resolve_time_phrase("yesterday")
        ts, te = _resolve_time_phrase("today")
        return [
            {"query": query, "time_start": ys, "time_end": ye},
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
    best_candidate: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    ambiguity_score: float = 0.0
    resolution_mode: str = "unresolved"
    needs_disambiguation: bool = False


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


def _lookup_entity_candidates_in_sqlite(conn: sqlite3.Connection, q_norm: str, limit: int = 8) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            canonical_entity_id,
            MAX(NULLIF(display_name, '')) AS display_name,
            MAX(NULLIF(entity_type, '')) AS entity_type,
            MAX(NULLIF(ticker, '')) AS ticker,
            MAX(COALESCE(confidence, 0)) AS confidence
        FROM entity_mentions
        WHERE canonical_entity_id IS NOT NULL
          AND canonical_entity_id <> ''
          AND (
            canonical_entity_id = ?
            OR lower(display_name) = lower(?)
            OR instr(lower(display_name), lower(?)) > 0
            OR instr(?, canonical_entity_id) > 0
          )
        GROUP BY canonical_entity_id
        ORDER BY confidence DESC, canonical_entity_id
        LIMIT ?
        """,
        (q_norm, q_norm, q_norm, q_norm, max(limit, 1)),
    ).fetchall()
    return [dict(row) for row in rows]


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


def _build_resolution_candidate(
    *,
    canonical_name: str,
    display_name: str,
    ticker: str | None,
    entity_type: str | None,
    confidence: float,
    match_source: str,
    matched_alias: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    return {
        "canonical_name": canonical_name,
        "display_name": display_name,
        "ticker": ticker,
        "entity_type": entity_type,
        "confidence": _clamp01(confidence),
        "match_source": match_source,
        "matched_alias": matched_alias,
        "category": category,
    }


def _dedupe_resolution_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    merged: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        canonical = str(cand.get("canonical_name") or "").strip()
        if not canonical:
            continue
        prev = merged.get(canonical)
        if prev is None or _safe_float(cand.get("confidence")) > _safe_float(prev.get("confidence")):
            merged[canonical] = cand
    sorted_cands = sorted(
        merged.values(),
        key=lambda c: (
            _safe_float(c.get("confidence")),
            len(str(c.get("matched_alias") or "")),
            len(str(c.get("display_name") or "")),
        ),
        reverse=True,
    )
    return sorted_cands


def _compute_ambiguity_score(candidates: list[dict[str, Any]], query: str) -> float:
    if not candidates:
        return 0.0
    if len(candidates) == 1:
        only = _safe_float(candidates[0].get("confidence"))
        return _clamp01(max(0.0, 1.0 - only) * 0.35)

    top_1 = _safe_float(candidates[0].get("confidence"))
    top_2 = _safe_float(candidates[1].get("confidence"))
    margin = max(0.0, top_1 - top_2)
    base_ambiguity = _clamp01(1.0 - margin)
    compare_boost = 0.12 if COMPARE_WORDS.search(query or "") else 0.0
    many_candidates_boost = 0.10 if len(candidates) >= 4 else 0.0
    return _clamp01(base_ambiguity + compare_boost + many_candidates_boost)


def _domain_alias_candidates(q_norm: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for alias, spec in DOMAIN_CANONICAL_ALIASES.items():
        alias_norm = _canonicalize(alias)
        if not alias_norm:
            continue
        if q_norm == alias_norm:
            conf = 0.94
        elif re.search(r"\b" + re.escape(alias_norm) + r"\b", q_norm):
            conf = min(0.90, 0.72 + 0.02 * len(alias_norm.split()))
        else:
            continue
        hits.append(
            _build_resolution_candidate(
                canonical_name=str(spec["canonical_name"]),
                display_name=str(spec.get("display_name") or spec["canonical_name"]),
                ticker=spec.get("ticker"),
                entity_type=spec.get("entity_type"),
                confidence=conf,
                match_source="domain_map",
                matched_alias=alias_norm,
                category=spec.get("category"),
            )
        )
    return hits


def _has_unmapped_ticker_context(query: str) -> bool:
    q = (query or "").strip().lower()
    return any(hint in q for hint in UNMAPPED_TICKER_CONTEXT_HINTS)


def _is_probable_unmapped_ticker_token(token: str, query: str) -> bool:
    token_norm = (token or "").strip().upper()
    if len(token_norm) < 2 or len(token_norm) > 5:
        return False
    if token_norm in UNMAPPED_TICKER_BLOCKLIST:
        return False
    return _has_unmapped_ticker_context(query)


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
    collected: list[dict[str, Any]] = []

    # Tier 1a: compound ticker patterns (e.g. BRK.A, BRK-B, DX-Y.NYB, CL=F).
    # The plain `\b([A-Z]{1,5})\b` regex below treats separators as boundaries
    # and can split one ticker into many partial tokens (e.g. `DX-Y.NYB` ->
    # `DX`, `Y`, `NYB`). Capture full compound forms first, then suppress
    # sub-token matches in Tier 1b.
    compound_pattern = r"(?<![A-Z0-9^.=\-])((?:\^)?[A-Z0-9]{1,6}(?:[.=\-][A-Z0-9]{1,6})+)(?![A-Z0-9.=\-])"
    compound_ticker_tokens = re.findall(compound_pattern, query)
    compound_parts: set[str] = set()
    for token in compound_ticker_tokens:
        core_token = token.lstrip("^")
        compound_parts.update(part for part in re.split(r"[.=\-]", core_token) if part)
        if token in ticker_to_canonical:
            collected.append(
                _build_resolution_candidate(
                    canonical_name=token,
                    display_name=ticker_to_canonical[token],
                    ticker=token,
                    entity_type="ORG",
                    confidence=0.99,
                    match_source="ticker_token_compound",
                    matched_alias=token.lower(),
                    category="equities",
                )
            )
        else:
            if not _has_unmapped_ticker_context(query):
                continue
            collected.append(
                _build_resolution_candidate(
                    canonical_name=token,
                    display_name=token,
                    ticker=token,
                    entity_type="ORG",
                    confidence=0.85,
                    match_source="ticker_token_compound_unmapped",
                    matched_alias=token.lower(),
                    category="equities",
                )
            )

    # Tier 1b: explicit ticker tokens (1-5 uppercase letters).
    ticker_tokens = re.findall(r"\b([A-Z]{1,5})\b", query)
    for token in ticker_tokens:
        if token in compound_parts:
            continue  # already absorbed into a compound ticker above
        if token in ticker_to_canonical:
            collected.append(
                _build_resolution_candidate(
                    canonical_name=token,
                    display_name=ticker_to_canonical[token],
                    ticker=token,
                    entity_type="ORG",
                    confidence=0.99,
                    match_source="ticker_token",
                    matched_alias=token.lower(),
                    category="etfs" if token in {"QQQ", "SPY", "IWM", "DIA"} else "equities",
                )
            )
        elif _is_probable_unmapped_ticker_token(token, query):
            collected.append(
                _build_resolution_candidate(
                    canonical_name=token,
                    display_name=token,
                    ticker=token,
                    entity_type="ORG",
                    confidence=0.84,
                    match_source="ticker_token_unmapped",
                    matched_alias=token.lower(),
                    category="equities",
                )
            )

    # Tier 2: direct alias map exact match.
    if q_norm in alias_to_ticker:
        ticker = alias_to_ticker[q_norm]
        collected.append(
            _build_resolution_candidate(
                canonical_name=ticker,
                display_name=ticker_to_canonical.get(ticker, ticker),
                ticker=ticker,
                entity_type="ORG",
                confidence=0.98,
                match_source="ticker_alias_exact",
                matched_alias=q_norm,
                category="equities",
            )
        )

    # Tier 2b: deterministic domain alias map (countries/central banks/etc).
    collected.extend(_domain_alias_candidates(q_norm))

    # Tier 3: longest ticker alias phrase in query.
    phrase_hits: list[tuple[str, str]] = []
    for alias, ticker in alias_to_ticker.items():
        if alias and re.search(r"\b" + re.escape(alias) + r"\b", q_norm):
            phrase_hits.append((alias, ticker))
    phrase_hits.sort(key=lambda item: len(item[0]), reverse=True)
    for alias, ticker in phrase_hits[:3]:
        collected.append(
            _build_resolution_candidate(
                canonical_name=ticker,
                display_name=ticker_to_canonical.get(ticker, alias),
                ticker=ticker,
                entity_type="ORG",
                confidence=min(0.96, 0.86 + 0.02 * len(alias.split())),
                match_source="ticker_alias_phrase",
                matched_alias=alias,
                category="equities",
            )
        )

    # Tier 4: financial entity map for central banks/macro institutions.
    if alias_to_fin_entity:
        for alias, entity in alias_to_fin_entity.items():
            if not alias:
                continue
            if q_norm == alias:
                confidence = 0.97
            elif re.search(r"\b" + re.escape(alias) + r"\b", q_norm):
                confidence = min(0.95, 0.80 + 0.02 * len(alias.split()))
            else:
                continue
            canonical_name = str(entity.get("canonical_name") or alias)
            display_name = str(entity.get("display_name") or canonical_name)
            collected.append(
                _build_resolution_candidate(
                    canonical_name=canonical_name,
                    display_name=display_name,
                    ticker=entity.get("ticker"),
                    entity_type=entity.get("entity_type") or "ORG",
                    confidence=confidence,
                    match_source="financial_entity_map",
                    matched_alias=alias,
                    category="central_banks" if "bank" in display_name.lower() else "macro_concepts",
                )
            )

    # Tier 5: SQLite fallback candidates.
    if sqlite_conn is not None:
        sqlite_candidates = _lookup_entity_candidates_in_sqlite(sqlite_conn, q_norm, limit=8)
        for row in sqlite_candidates:
            collected.append(
                _build_resolution_candidate(
                    canonical_name=str(row["canonical_entity_id"]),
                    display_name=str(row.get("display_name") or row["canonical_entity_id"]),
                    ticker=row.get("ticker"),
                    entity_type=row.get("entity_type"),
                    confidence=max(0.75, _safe_float(row.get("confidence"), 0.0)),
                    match_source="sqlite_mentions",
                    matched_alias=q_norm,
                    category="entities",
                )
            )

    deduped = _dedupe_resolution_candidates(collected)
    if not deduped:
        return QueryTarget(
            query_type=QUERY_TYPE_GENERAL,
            canonical_name=None,
            display_name=None,
            ticker=None,
            entity_type=None,
            confidence=0.0,
            best_candidate=None,
            candidates=[],
            ambiguity_score=0.0,
            resolution_mode="unresolved",
            needs_disambiguation=False,
        )

    best = deduped[0]
    ambiguity_score = _compute_ambiguity_score(deduped, query)
    confidence = _clamp01(_safe_float(best.get("confidence"), 0.0) * (1.0 - 0.25 * ambiguity_score))

    return QueryTarget(
        query_type=QUERY_TYPE_SINGLE,
        canonical_name=str(best.get("canonical_name") or ""),
        display_name=str(best.get("display_name") or best.get("canonical_name") or ""),
        ticker=best.get("ticker"),
        entity_type=best.get("entity_type"),
        confidence=confidence,
        best_candidate=best,
        candidates=deduped,
        ambiguity_score=ambiguity_score,
        resolution_mode=str(best.get("match_source") or "unknown"),
        needs_disambiguation=bool(ambiguity_score >= RESOLUTION_DISAMBIGUATION_THRESHOLD and len(deduped) > 1),
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
            a.source_trust_tier,
            a.content_class,
            a.article_quality_score,
            a.quality_flags_json,
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
            a.source_trust_tier,
            a.content_class,
            a.article_quality_score,
            a.quality_flags_json,
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
            a.source,
            a.source_trust_tier,
            a.content_class,
            a.article_quality_score,
            a.quality_flags_json
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
    if SQLITE_SEMANTIC_CANDIDATE_LIMIT > 0:
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
        semantic_score = cosine_sim(qvec, np.array(emb, dtype=np.float32))
        lexical_score = _keyword_overlap_score(query, item.get("text", ""))
        item["semantic_score"] = semantic_score
        item["keyword_score"] = lexical_score
        item["candidate_score"] = 0.85 * semantic_score + 0.15 * lexical_score
        item["retrieval_kind"] = "sqlite_semantic"
        scored.append(item)

    scored.sort(key=lambda item: item["candidate_score"], reverse=True)
    return scored[:top_k]


def retrieve_summary_chunks(
    sqlite_conn: sqlite3.Connection,
    embed_model: SentenceTransformer,
    query: str,
    period_keys: list[str] | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    source_filter: str | None = None,
    top_k: int = SUMMARY_TOP_K,
    expanded_k: int = SUMMARY_EXPANDED_K,
    recency_half_life_days: float = SUMMARY_RECENCY_HALF_LIFE_DAYS,
    reranker=None,
    candidate_limit: int = SUMMARY_CANDIDATE_LIMIT,
) -> list[dict]:
    """
    Summary-mode retrieval: broad SQLite recency + semantic pass.

    Deliberately avoids MacroEvent-heavy Neo4j paths that collapse results
    onto a few macro-style chunks.  Instead it:

      1. Fetches up to ``candidate_limit`` recent chunks with embeddings
         from the requested date window (all sources unless filtered).
      2. Scores each chunk as a weighted combination of semantic similarity
         to the query and exponential recency decay.
      3. Runs a per-source diversity sweep so that at least one chunk from
         every ingested source appears near the top of the ranking, labelled
         ``summary_diverse_source``.
      4. Returns ``expanded_k`` candidates to the caller; the global
         summary post-processing pipeline (article dedup, near-dup filter,
         per-source cap, min-unique-sources guard) is applied downstream.

    retrieval_kind values produced:
      - ``summary_recency_semantic``  – scored by recency + semantic sim
      - ``summary_diverse_source``    – first chunk per distinct source
        (same scoring, distinguished for debug logs only)
    """
    # ------------------------------------------------------------------
    # 1. Broad recency pass from SQLite
    #    Date filtering exclusively uses c.published_date (ISO format).
    #    articles is joined only for metadata (source, title, url).
    # ------------------------------------------------------------------
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
            a.source_trust_tier,
            a.content_class,
            a.article_quality_score,
            a.quality_flags_json
        FROM chunks c
        JOIN articles a ON a.article_id = c.article_id
        WHERE 1=1
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
    params.append(max(candidate_limit, expanded_k))

    print(
        f"  [retrieve_summary_chunks] SQL date field=c.published_date "
        f"date_start={date_start}, date_end={date_end}, "
        f"period_keys={period_keys}, source_filter={source_filter}"
    )

    rows = sqlite_conn.execute(sql, params).fetchall()
    if not rows:
        print(
            f"  [retrieve_summary_chunks] no rows returned "
            f"(date_start={date_start}, date_end={date_end}, source_filter={source_filter})"
        )
        return []

    # ------------------------------------------------------------------
    # 2. Score: semantic similarity + exponential recency decay.
    #    Chunks without embeddings fall back to recency-only scoring
    #    (sem_score=0.0) so that recently ingested but not-yet-embedded
    #    articles still surface in summary results.
    # ------------------------------------------------------------------
    qvec = embed_model.encode([query], normalize_embeddings=True)[0]
    now_ts = int(datetime.now(timezone.utc).timestamp())
    half_life_s = max(recency_half_life_days, 0.1) * 86400.0

    scored: list[dict] = []
    no_emb_count = 0
    for row in rows:
        item = _sqlite_row_to_chunk_dict(row)
        emb = item.get("embedding")
        if emb is None:
            no_emb_count += 1
            sem_score = 0.0
        else:
            sem_score = cosine_sim(qvec, np.array(emb, dtype=np.float32))
        pub_date = item.get("published_date")
        pub_ts = _published_date_to_ts(pub_date) if pub_date else None
        if pub_ts is not None:
            age_s = max(0.0, float(now_ts - pub_ts))
            rec_score = float(np.exp(-age_s / half_life_s))
        else:
            rec_score = 0.0
        item["semantic_score"] = sem_score
        item["recency_score"] = rec_score
        item["candidate_score"] = 0.70 * sem_score + 0.30 * rec_score
        item["retrieval_kind"] = "summary_recency_semantic"
        scored.append(item)

    if no_emb_count:
        print(
            f"  [retrieve_summary_chunks] {no_emb_count}/{len(rows)} chunks "
            f"have no embedding — scored by recency only"
        )

    scored.sort(key=lambda x: x["candidate_score"], reverse=True)

    # ------------------------------------------------------------------
    # 3. Per-source diversity sweep
    #    Mark the first occurrence of each source so the downstream
    #    dedup/cap pipeline can see that diversity was considered.
    # ------------------------------------------------------------------
    seen_sources: set[str] = set()
    source_debug: list[str] = []
    for item in scored:
        src = (item.get("source") or "unknown").strip().lower()
        if src not in seen_sources:
            seen_sources.add(src)
            item["retrieval_kind"] = "summary_diverse_source"
            source_debug.append(src)

    print(
        f"  [retrieve_summary_chunks] "
        f"candidates={len(scored)}, "
        f"unique_sources={sorted(seen_sources)}, "
        f"returning={min(expanded_k, len(scored))}"
    )

    return scored[:expanded_k]


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
    *,
    query: str,
    rows: list[dict],
    query_vec: np.ndarray,
    target: QueryTarget | None,
    route_type: str,
    recency_half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> list[dict]:
    profile = _resolve_route_profile(route_type)
    allowed_classes = profile.get("allowed_content_classes") or ()
    filtered = [
        row
        for row in rows
        if _route_allows_content_class(profile, row.get("content_class"))
    ]
    scored = _apply_unified_scoring(
        query=query,
        rows=filtered,
        query_vec=query_vec,
        target=target,
        route_type=route_type,
        recency_half_life_days=recency_half_life_days,
    )
    return scored[: max(int(profile.get("candidate_cap", len(scored))), 1)]


def rank_candidates(
    *,
    query: str,
    rows: list[dict],
    query_vec: np.ndarray,
    target: QueryTarget | None,
    route_type: str,
    recency_half_life_days: float,
    reranker=None,
) -> list[dict]:
    baseline = score_and_rank(
        query=query,
        rows=rows,
        query_vec=query_vec,
        target=target,
        route_type=route_type,
        recency_half_life_days=recency_half_life_days,
    )
    reranked = apply_reranker(query, baseline, reranker=reranker)
    return _apply_unified_scoring(
        query=query,
        rows=reranked,
        query_vec=query_vec,
        target=target,
        route_type=route_type,
        recency_half_life_days=recency_half_life_days,
    )


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
    route_type: str | None = None,
    reranker=None,
) -> tuple[list[dict], QueryTarget, dict[str, Any]]:
    """
    Hybrid retrieval for one sub-query.
    Returns (ranked_chunks, target, retrieval_trace).
    """
    target = resolve_query_target(
        query,
        alias_to_ticker,
        ticker_to_canonical,
        driver,
        sqlite_conn=sqlite_conn,
        alias_to_fin_entity=alias_to_fin_entity,
    )
    resolved_route = route_type if route_type in ROUTE_TYPES else classify_query_route(
        query=query,
        summary_mode=False,
        causal_intent=False,
        market_data_intent=False,
        target=target,
    )
    if target.needs_disambiguation:
        resolved_route = "ambiguous"
    route_profile = _resolve_route_profile(resolved_route)
    top_k = max(top_k, int(route_profile["top_k"]))
    expanded_k = max(expanded_k, int(route_profile["expanded_k"]))
    recency_half_life_days = float(route_profile["recency_half_life_days"])
    period_keys = _date_range_to_period_keys(date_start, date_end) or None

    query_vec = embed_model.encode([query], normalize_embeddings=True)[0]
    all_rows: list[dict] = []

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
        ranked = rank_candidates(
            query=query,
            rows=all_rows,
            query_vec=query_vec,
            target=target,
            route_type=resolved_route,
            recency_half_life_days=recency_half_life_days,
            reranker=reranker,
        )

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
            ranked = rank_candidates(
                query=query,
                rows=all_rows,
                query_vec=query_vec,
                target=target,
                route_type=resolved_route,
                recency_half_life_days=recency_half_life_days,
                reranker=reranker,
            )

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
            ranked = rank_candidates(
                query=query,
                rows=all_rows,
                query_vec=query_vec,
                target=target,
                route_type=resolved_route,
                recency_half_life_days=recency_half_life_days,
                reranker=reranker,
            )

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
        all_rows = asset_rows + macro_rows + sem_rows
        ranked = rank_candidates(
            query=query,
            rows=all_rows,
            query_vec=query_vec,
            target=target,
            route_type=resolved_route,
            recency_half_life_days=recency_half_life_days,
            reranker=reranker,
        )

    for row in ranked:
        row["route_type"] = resolved_route

    retrieval_trace = {
        "route_type": resolved_route,
        "candidate_count": len(all_rows),
        "ranked_count": len(ranked),
        "expanded_k": expanded_k,
        "source_filter": source_filter,
        "date_start": date_start,
        "date_end": date_end,
        "target": {
            "canonical_name": target.canonical_name,
            "display_name": target.display_name,
            "ticker": target.ticker,
            "ambiguity_score": target.ambiguity_score,
            "needs_disambiguation": target.needs_disambiguation,
            "resolution_mode": target.resolution_mode,
        },
        "ranked_candidates": ranked,
    }
    return ranked[:expanded_k], target, retrieval_trace


# ---------------------------------------------------------------------------
# Signal discovery retrieval
# ---------------------------------------------------------------------------

def _build_signal_discovery_answer(
    *,
    query: str,
    selected_signals: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
) -> str:
    if not selected_signals:
        return (
            "Answer: I could not find sufficiently supported top signals in the current store.\n"
            "Evidence: The retrieved signal tables did not yield enough corroborated cluster evidence.\n"
            "Theory: Any interpretation would be speculative without corroborated signal evidence."
        )
    citation_map = build_citation_map(evidence_rows)
    evidence_by_signal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        if row.get("signal_id"):
            evidence_by_signal[str(row["signal_id"])].append(row)

    answer_lines: list[str] = []
    evidence_lines: list[str] = []
    theory_lines: list[str] = []
    for index, signal in enumerate(selected_signals, start=1):
        rows = evidence_by_signal.get(str(signal["signal_id"]), [])
        citations = "".join(
            citation_map.get(row.get("chunk_uid"), "")
            and f"[{citation_map[row['chunk_uid']]}]"
            for row in rows[:2]
            if row.get("chunk_uid") in citation_map
        )
        components = signal.get("score_components") or {}
        answer_lines.append(
            f"{index}. {signal.get('headline') or signal.get('summary') or signal.get('text')} "
            f"(signal_score={_safe_float(signal.get('signal_score'), 0.0):.2f}; "
            f"recency={_safe_float(components.get('recency_score'), 0.0):.2f}; "
            f"source_quality={_safe_float(components.get('source_quality_score'), 0.0):.2f}){citations}"
        )
        if rows:
            supporting_sources = sorted({str(row.get("source") or "unknown") for row in rows})
            support = _safe_float(signal.get("event_support_score"), 0.0)
            evidence_citations = "".join(
                f"[{citation_map[row['chunk_uid']]}]"
                for row in rows[:2]
                if row.get("chunk_uid") in citation_map
            )
            evidence_lines.append(
                f"- {signal.get('headline') or signal.get('summary')}: "
                f"sources={', '.join(supporting_sources[:3])}; "
                f"event_support={support:.2f}; "
                f"evidence={evidence_citations}"
            )
            if len(supporting_sources) <= 1:
                theory_lines.append(
                    f"- {signal.get('headline') or signal.get('summary')}: source concentration is high, so treat the ranking as provisional."
                )
            else:
                theory_lines.append(
                    f"- {signal.get('headline') or signal.get('summary')}: corroboration exists across {len(supporting_sources)} sources, but follow-through still needs confirmation."
                )
        else:
            evidence_lines.append(
                f"- {signal.get('headline') or signal.get('summary')} is ranked from clustered signal metadata."
            )
            theory_lines.append(
                f"- {signal.get('headline') or signal.get('summary')}: theory is limited because no chunk-level evidence was materialized."
            )

    answer_lines = [line for line in answer_lines if line.strip()]
    return (
        "Answer:\n"
        + "\n".join(answer_lines)
        + "\nEvidence:\n"
        + "\n".join(evidence_lines[: max(3, len(selected_signals))])
        + "\nTheory:\n"
        + "\n".join(theory_lines[: max(3, len(selected_signals))])
    )


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(
    query: str,
    target: QueryTarget,
    chunks: list[dict],
) -> str:
    citation_map = build_citation_map(chunks)
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
        chunk_uid = ch.get("chunk_uid")
        if chunk_uid:
            lines.append(f"[CITATION: {citation_map.get(chunk_uid, '?')}]")
            lines.append(f"[CHUNK ID: {chunk_uid}]")
        lines.append(f"[SOURCE: {ch.get('source', '?')}]")
        lines.append(f"[TITLE: {ch.get('title', '?')}]")
        lines.append(f"[URL: {ch.get('url', '?')}]")
        lines.append(f"[PERIOD: {ch.get('period_key', '?')}]")
        if ch.get("score") is not None:
            lines.append(f"[RANK SCORE: {float(ch['score']):.4f}]")
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
        if ch.get("evidence_text"):
            lines.append(f"[EVIDENCE SPAN: {ch['evidence_text']}]")
        lines.append(ch.get("text", ""))
        lines.append("")

    return "\n".join(lines)


def build_financial_only_context(
    query: str,
    target: QueryTarget,
    financial_context: str,
) -> str:
    lines = []
    lines.append(f"QUERY: {query}")
    lines.append(f"TARGET ENTITY : {target.display_name or 'unknown'}")
    lines.append(f"TARGET CANONICAL: {target.canonical_name or 'unknown'}")
    lines.append(f"TARGET TICKER : {target.ticker or 'N/A'}")
    lines.append(f"ENTITY TYPE   : {target.entity_type or 'unknown'}")
    lines.append(f"CONFIDENCE    : {target.confidence:.2f}")
    lines.append("")
    lines.append("Use only the FINANCIAL DATA [F] block below.")
    lines.append("")
    lines.append(financial_context.strip())
    return "\n".join(lines)


def build_signal_context(
    query: str,
    target: QueryTarget,
    signals: list[dict[str, Any]],
) -> str:
    lines = []
    lines.append(f"QUERY: {query}")
    lines.append(f"TARGET ENTITY : {target.display_name or 'general'}")
    lines.append(f"TARGET CANONICAL: {target.canonical_name or 'general'}")
    lines.append("")

    if not signals:
        lines.append("No ranked signals found.")
        return "\n".join(lines)

    for index, signal in enumerate(signals, start=1):
        lines.append(f"SIGNAL {index}: {signal.get('headline') or signal.get('summary') or 'Untitled signal'}")
        lines.append(f"[SIGNAL ID: {signal.get('signal_id', '?')}]")
        lines.append(f"[CLUSTER ID: {signal.get('cluster_id', '?')}]")
        lines.append(f"[SIGNAL SCORE: {_safe_float(signal.get('signal_score'), 0.0):.2f}]")
        lines.append(f"[RANK SCORE: {_safe_float(signal.get('final_score'), _safe_float(signal.get('score'), 0.0)):.4f}]")
        if signal.get("event_type"):
            lines.append(f"[EVENT TYPE: {signal['event_type']}]")
        if signal.get("region"):
            lines.append(f"[REGION: {signal['region']}]")
        if signal.get("novelty_hint") or signal.get("urgency") or signal.get("market_surprise"):
            lines.append(
                f"[QUALIFIERS: novelty={signal.get('novelty_hint') or '?'}, "
                f"urgency={signal.get('urgency') or '?'}, surprise={signal.get('market_surprise') or '?'}]"
            )
        if signal.get("summary"):
            lines.append(str(signal["summary"]))
        for evidence in signal.get("evidence_chunks") or []:
            snippet = re.sub(r"\s+", " ", str(evidence.get("text") or "")).strip()[:220]
            lines.append(f"- Evidence: {evidence.get('source', '?')} | {evidence.get('title', '?')}")
            lines.append(f"  {snippet}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

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


def _append_non_overlapping_text(base: str, extra: str, max_overlap: int = 240) -> str:
    if not base:
        return extra
    if not extra:
        return base

    b = base.strip()
    e = extra.strip()
    overlap_cap = min(len(b), len(e), max_overlap)
    for size in range(overlap_cap, 15, -1):
        if b[-size:].lower() == e[:size].lower():
            return (b + e[size:]).strip()
    return (b + "\n" + e).strip()


def _clamp_outlook_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _infer_outlook_and_confidence(text: str) -> tuple[str, float]:
    lower = (text or "").lower()

    bullish_hits = sum(
        lower.count(term)
        for term in (
            "bullish",
            "uptrend",
            "trading above",
            "positive return",
            "strong liquidity",
            "improving margin",
            "momentum",
            "rally",
        )
    )
    bearish_hits = sum(
        lower.count(term)
        for term in (
            "bearish",
            "downtrend",
            "trading below",
            "negative return",
            "net loss",
            "decline",
            "weakness",
            "deteriorat",
        )
    )

    if bullish_hits > bearish_hits:
        delta = bullish_hits - bearish_hits
        return "Bullish", _clamp_outlook_confidence(0.60 + 0.05 * min(delta, 5))
    if bearish_hits > bullish_hits:
        delta = bearish_hits - bullish_hits
        return "Bearish", _clamp_outlook_confidence(0.40 - 0.05 * min(delta, 5))
    return "Neutral", 0.50


def _enforce_single_ticker_outlook_sections(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return (
            "Answer: Insufficient financial data [F] is available to answer this ticker query.\n"
            "Outlook: Neutral.\n"
            "Outlook Confidence (0-1): 0.50"
        )

    outlook_match = re.search(
        r"(?im)^\s*Outlook:\s*(Bullish|Bearish|Neutral)\.?\s*$",
        text,
    )
    confidence_match = re.search(
        r"(?im)^\s*Outlook Confidence \(0-1\):\s*([01](?:\.\d+)?)\s*$",
        text,
    )

    if outlook_match:
        outlook = outlook_match.group(1).capitalize()
    else:
        outlook, _ = _infer_outlook_and_confidence(text)

    if confidence_match:
        confidence = _clamp_outlook_confidence(float(confidence_match.group(1)))
    else:
        _, confidence = _infer_outlook_and_confidence(text)

    # Keep confidence directionally consistent with outlook label.
    if outlook == "Bullish" and confidence < 0.50:
        confidence = 0.60
    elif outlook == "Bearish" and confidence > 0.50:
        confidence = 0.40
    elif outlook == "Neutral":
        confidence = 0.50 if abs(confidence - 0.50) <= 0.15 else confidence

    outlook_line = f"Outlook: {outlook}."
    confidence_line = f"Outlook Confidence (0-1): {confidence:.2f}"

    lines = text.splitlines()
    has_outlook = any(re.match(r"(?i)^\s*Outlook:\s*", ln) for ln in lines)
    has_conf = any(re.match(r"(?i)^\s*Outlook Confidence \(0-1\):\s*", ln) for ln in lines)

    if has_outlook or has_conf:
        normalized: list[str] = []
        seen_outlook = False
        seen_conf = False
        for ln in lines:
            if re.match(r"(?i)^\s*Outlook:\s*", ln):
                if not seen_outlook:
                    normalized.append(outlook_line)
                    seen_outlook = True
                continue
            if re.match(r"(?i)^\s*Outlook Confidence \(0-1\):\s*", ln):
                if not seen_conf:
                    normalized.append(confidence_line)
                    seen_conf = True
                continue
            normalized.append(ln)
        if not seen_outlook:
            normalized.append(outlook_line)
        if not seen_conf:
            normalized.append(confidence_line)
        return "\n".join(normalized).strip()

    theory_idx = None
    for idx, ln in enumerate(lines):
        if re.match(r"(?i)^\s*Theory:\s*", ln):
            theory_idx = idx
            break

    if theory_idx is None:
        lines.extend([outlook_line, confidence_line])
    else:
        lines[theory_idx:theory_idx] = [outlook_line, confidence_line]
    return "\n".join(lines).strip()


def generate_answer(
    query: str,
    context: str,
    gen_client: Any,
    system_prompt: str,
    *,
    max_tokens: int | None = None,
) -> str:
    budget = max(128, int(max_tokens or GEN_MAX_TOKENS))
    messages = [
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                "Use the context below to answer.\n"
                "Prefer directly relevant evidence over tangential mentions.\n"
                "If evidence is weak or indirect, say so clearly.\n\n"
                f"Context:\n{context}"
            ),
        }
    ]

    out = gen_client.messages.create(
        model=GEN_MODEL_NAME,
        max_tokens=budget,
        temperature=0,
        system=system_prompt,
        messages=messages,
    )
    raw = anthropic_text(out)
    assembled = strip_think_tags(raw).strip()

    continuation_rounds = max(0, GEN_CONTINUATION_MAX_ROUNDS)
    stop_reason = str(getattr(out, "stop_reason", "") or "").lower()
    while continuation_rounds > 0 and stop_reason in {"max_tokens", "length"}:
        continuation_rounds -= 1
        continuation_prompt = (
            "Continue the previous answer from exactly where it stopped.\n"
            "Do not repeat any text already written.\n"
            "Return only the continuation text.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer so far:\n"
            f"{assembled}"
        )
        out = gen_client.messages.create(
            model=GEN_MODEL_NAME,
            max_tokens=budget,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": continuation_prompt}],
        )
        continuation = strip_think_tags(anthropic_text(out)).strip()
        if not continuation:
            break
        assembled = _append_non_overlapping_text(assembled, continuation)
        stop_reason = str(getattr(out, "stop_reason", "") or "").lower()

    return assembled or "I could not generate a grounded answer from the retrieved context."


def run_query_once(
    *,
    query: str,
    embed_model: SentenceTransformer,
    reranker,
    gen_client: Any,
    driver,
    sqlite_conn: sqlite3.Connection,
    alias_to_ticker: dict[str, str],
    ticker_to_canonical: dict[str, str],
    alias_to_fin_entity: dict[str, dict[str, Any]],
    base_system_prompt: str,
    base_causal_system_prompt: str,
    base_daily_summary_prompt: str,
    base_single_ticker_financial_prompt: str,
    memory: ConversationMemory,
    skip_generation: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    run_id = str(uuid.uuid4())
    query, was_rewritten = resolve_coreference(query, memory)
    market_data_intent = is_market_data_intent(query)
    causal_intent = is_causal_analysis_intent(query)
    summary_mode = is_summary_query(query)
    financial_intent = is_single_ticker_financial_intent(query)
    explicit_latest_news_intent = is_explicit_latest_news_query(query)
    forced_summary_start, forced_summary_end = infer_summary_date_range(query)
    source_filter = extract_source_filter(query)
    early_target = resolve_query_target(
        query,
        alias_to_ticker,
        ticker_to_canonical,
        driver,
        sqlite_conn=sqlite_conn,
        alias_to_fin_entity=alias_to_fin_entity,
    )
    route_seed = classify_query_route(
        query=query,
        summary_mode=summary_mode,
        causal_intent=causal_intent,
        market_data_intent=market_data_intent,
        target=early_target,
        financial_intent=financial_intent,
        explicit_latest_news_intent=explicit_latest_news_intent,
    )
    sub_queries = (
        [_extract_single_time_range(query)]
        if route_seed == "single_ticker_financial"
        else resolve_temporal_carryover(decompose_query(query, gen_client), memory)
    )

    all_contexts: list[str] = []
    all_urls: list[str] = []
    all_chunks: list[dict[str, Any]] = []
    selected_signals: list[dict[str, Any]] = []
    summary_raw_candidates: list[dict[str, Any]] = []
    retrieval_traces: list[dict[str, Any]] = []
    observability_candidates: list[dict[str, Any]] = []
    summary_used_date_ranges: list[tuple[str | None, str | None]] = []
    market_ctx = ""
    fin_ctx = ""
    logs: list[str] = []
    primary_target: QueryTarget | None = None
    primary_date_start: str | None = None
    primary_date_end: str | None = None

    if was_rewritten:
        logs.append(f'  [memory] Coreference resolved -> "{query}"')
    if source_filter:
        logs.append(f"  [source filter: {source_filter}]")
    if len(sub_queries) > 1:
        logs.append(f"  [decomposed into {len(sub_queries)} sub-queries]")
    if route_seed == "single_ticker_financial":
        logs.append("  [route_type=single_ticker_financial]")

    def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidate_id": candidate.get("candidate_id"),
            "chunk_uid": candidate.get("chunk_uid"),
            "article_id": candidate.get("article_id"),
            "macro_event_id": candidate.get("macro_event_id"),
            "cluster_id": candidate.get("cluster_id"),
            "signal_id": candidate.get("signal_id"),
            "retrieval_kind": candidate.get("retrieval_kind"),
            "source": candidate.get("source"),
            "title": candidate.get("title"),
            "headline": candidate.get("headline"),
            "content_class": candidate.get("content_class"),
            "source_trust_tier": candidate.get("source_trust_tier"),
            "score": candidate.get("final_score", candidate.get("score")),
            "signal_score": candidate.get("signal_score"),
            "semantic_score": candidate.get("semantic_score"),
            "cross_encoder_score": candidate.get("cross_encoder_score"),
            "keyword_overlap_score": candidate.get("keyword_overlap_score"),
            "target_match_score": candidate.get("target_match_score"),
            "source_quality_score": candidate.get("source_quality_score"),
            "recency_score": candidate.get("recency_score"),
            "graph_relevance_score": candidate.get("graph_relevance_score"),
            "event_support_score": candidate.get("event_support_score"),
            "duplicate_penalty": candidate.get("duplicate_penalty"),
            "ambiguity_penalty": candidate.get("ambiguity_penalty"),
        }

    def _compact_signal(signal: dict[str, Any]) -> dict[str, Any]:
        return {
            "signal_id": signal.get("signal_id"),
            "cluster_id": signal.get("cluster_id"),
            "headline": signal.get("headline"),
            "summary": signal.get("summary"),
            "event_type": signal.get("event_type"),
            "region": signal.get("region"),
            "signal_score": signal.get("signal_score"),
            "rank_score": signal.get("final_score", signal.get("score")),
            "novelty_hint": signal.get("novelty_hint"),
            "urgency": signal.get("urgency"),
            "market_surprise": signal.get("market_surprise"),
            "score_components": signal.get("score_components") or {},
        }

    def _empty_result(message: str, route_type: str) -> dict[str, Any]:
        resolved_target = {
            "canonical_name": primary_target.canonical_name if primary_target else None,
            "display_name": primary_target.display_name if primary_target else None,
            "ticker": primary_target.ticker if primary_target else None,
            "query_type": primary_target.query_type if primary_target else QUERY_TYPE_GENERAL,
            "entity_type": primary_target.entity_type if primary_target else None,
            "best_candidate": primary_target.best_candidate if primary_target else None,
            "candidates": primary_target.candidates if primary_target else [],
            "ambiguity_score": primary_target.ambiguity_score if primary_target else 0.0,
            "resolution_mode": primary_target.resolution_mode if primary_target else "unresolved",
            "needs_disambiguation": primary_target.needs_disambiguation if primary_target else False,
        }
        return {
            "run_id": run_id,
            "query": query,
            "answer": ensure_structured_answer(message, []),
            "chunks": [],
            "urls": [],
            "logs": logs,
            "citation_map": {},
            "provenance": "Why this answer: no retrieved evidence.",
            "target": primary_target,
            "resolved_target": resolved_target,
            "resolved_target_json": resolved_target,
            "route_type": route_type,
            "retrieval_trace": {
                "route_type": route_type,
                "route_profile": _resolve_route_profile(route_type),
                "scoring_weights": _resolve_scoring_weights(route_type),
                "selected_chunk_count": 0,
                "finance_context_present": any(
                    bool(trace.get("finance_context_present")) for trace in retrieval_traces
                ),
                "sub_trace_count": len(retrieval_traces),
                "sub_traces": retrieval_traces,
            },
            "answer_confidence": 0.0,
            "decision": "abstain",
            "answer_meta": {
                "answer_confidence": 0.0,
                "decision": "abstain",
                "signals": {
                    "relevant_chunks": 0.0,
                    "source_diversity": 0.0,
                    "retrieval_margin": 0.0,
                    "verifier_support": 0.0,
                    "recency_coverage": 0.0,
                    "ambiguity_score": resolved_target["ambiguity_score"] or 0.0,
                    "contradiction_signals": 0.0,
                },
            },
            "selected_macro_events": [],
            "selected_signals": [],
            "contradiction_signals": False,
            "date_start": primary_date_start,
            "date_end": primary_date_end,
        }

    if route_seed == "single_ticker_financial":
        financial_sub_query = sub_queries[0] if sub_queries else {"query": query, "time_start": None, "time_end": None}
        primary_date_start = financial_sub_query.get("time_start")
        primary_date_end = financial_sub_query.get("time_end")
        if primary_date_start or primary_date_end:
            logs.append(f"  [time filter: {primary_date_start} -> {primary_date_end}]")

        if primary_target is None:
            primary_target = early_target
        if primary_target is None:
            primary_target = QueryTarget(
                query_type=QUERY_TYPE_GENERAL,
                canonical_name=None,
                display_name="general news",
                ticker=None,
                entity_type=None,
                confidence=0.0,
            )

        if not primary_target.ticker:
            retrieval_traces.append(
                {
                    "sub_query": query,
                    "date_start": primary_date_start,
                    "date_end": primary_date_end,
                    "route_type": "single_ticker_financial",
                    "candidate_count": 0,
                    "ranked_count": 0,
                    "ranked_candidates": [],
                    "finance_context_present": False,
                }
            )
            return _empty_result(
                (
                    "Answer: Insufficient financial data [F] is available to answer this ticker query.\n"
                    "Evidence: The FINANCIAL DATA [F] block is missing or unusable.\n"
                    "Theory: None."
                ),
                "single_ticker_financial",
            )

        fin_ctx = fetch_financial_context(
            ticker=primary_target.ticker,
            date_start=primary_date_start,
            date_end=primary_date_end,
            include_technicals=True,
        )
        finance_context_present = bool((fin_ctx or "").strip())
        retrieval_traces.append(
            {
                "sub_query": query,
                "date_start": primary_date_start,
                "date_end": primary_date_end,
                "route_type": "single_ticker_financial",
                "candidate_count": 0,
                "ranked_count": 0,
                "ranked_candidates": [],
                "finance_context_present": finance_context_present,
            }
        )
        if not finance_context_present:
            logs.append(f"  [financial data] no usable [F] block for {primary_target.ticker}")
            return _empty_result(
                (
                    "Answer: Insufficient financial data [F] is available to answer this ticker query.\n"
                    "Evidence: The FINANCIAL DATA [F] block is missing or unusable.\n"
                    "Theory: None."
                ),
                "single_ticker_financial",
            )
        logs.append(f"  [financial data] fetched [F] block for {primary_target.ticker}")
        all_contexts.append(build_financial_only_context(query, primary_target, fin_ctx))
        sub_queries = []

    for sq in sub_queries:
        sub_query = sq["query"]
        date_start = sq.get("time_start")
        date_end = sq.get("time_end")
        if summary_mode and forced_summary_start and forced_summary_end:
            date_start = forced_summary_start
            date_end = forced_summary_end
        if summary_mode:
            summary_used_date_ranges.append((date_start, date_end))
        if date_start or date_end:
            logs.append(f"  [time filter: {date_start} -> {date_end}]")

        chunks: list[dict[str, Any]] = []
        trace: dict[str, Any] = {
            "sub_query": sub_query,
            "date_start": date_start,
            "date_end": date_end,
        }

        if route_seed == "signal_discovery":
            sub_selected_signals, target, retrieve_trace = retrieve_top_signals(
                query=sub_query,
                embed_model=embed_model,
                sqlite_conn=sqlite_conn,
                driver=driver,
                alias_to_ticker=alias_to_ticker,
                ticker_to_canonical=ticker_to_canonical,
                alias_to_fin_entity=alias_to_fin_entity,
                top_k=max(TOP_K, 5),
                expanded_k=max(EXPANDED_K, 12),
                recency_half_life_days=RECENCY_HALF_LIFE_DAYS,
                source_filter=source_filter,
                date_start=date_start,
                date_end=date_end,
            )
            chunks = []
            seen_chunk_ids: set[str] = set()
            for signal in sub_selected_signals:
                for evidence in signal.get("evidence_chunks") or []:
                    chunk_uid = evidence.get("chunk_uid")
                    if not chunk_uid or chunk_uid in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(chunk_uid)
                    chunks.append({**evidence, "signal_id": signal.get("signal_id"), "cluster_id": signal.get("cluster_id")})
            trace.update(retrieve_trace)
            trace["sub_query"] = sub_query
            selected_signals.extend(sub_selected_signals)
        elif causal_intent and not market_data_intent:
            hops = decompose_causal_chain(sub_query, gen_client)
            if hops:
                logs.append(f"  [causal chain: {' -> '.join(hops)}]")
                chunks, target = retrieve_causal_chain(
                    query=sub_query,
                    hops=hops,
                    embed_model=embed_model,
                    reranker=reranker,
                    driver=driver,
                    sqlite_conn=sqlite_conn,
                    alias_to_ticker=alias_to_ticker,
                    ticker_to_canonical=ticker_to_canonical,
                    alias_to_fin_entity=alias_to_fin_entity,
                    source_filter=source_filter,
                    date_start=date_start,
                    date_end=date_end,
                )
                trace.update(
                    {
                        "route_type": "macro_causal",
                        "causal_hops": hops,
                        "candidate_count": len(chunks),
                        "ranked_count": len(chunks),
                        "ranked_candidates": chunks,
                    }
                )
                market_ctx = fetch_market_context(
                    hops=hops,
                    date_start=date_start,
                    date_end=date_end,
                    query=sub_query,
                )
            else:
                logs.append("  [causal chain] decomposition empty, falling back to standard retrieve")
                chunks, target, retrieve_trace = retrieve(
                    query=sub_query,
                    embed_model=embed_model,
                    reranker=reranker,
                    driver=driver,
                    sqlite_conn=sqlite_conn,
                    alias_to_ticker=alias_to_ticker,
                    ticker_to_canonical=ticker_to_canonical,
                    alias_to_fin_entity=alias_to_fin_entity,
                    top_k=SUMMARY_TOP_K if summary_mode else TOP_K,
                    expanded_k=SUMMARY_EXPANDED_K if summary_mode else EXPANDED_K,
                    recency_half_life_days=(
                        SUMMARY_RECENCY_HALF_LIFE_DAYS if summary_mode else RECENCY_HALF_LIFE_DAYS
                    ),
                    source_filter=source_filter,
                    date_start=date_start,
                    date_end=date_end,
                    route_type=route_seed,
                )
                trace.update(retrieve_trace)
                trace["sub_query"] = sub_query
        elif summary_mode:
            sum_period_keys = _date_range_to_period_keys(date_start, date_end) or None
            chunks = retrieve_summary_chunks(
                sqlite_conn=sqlite_conn,
                embed_model=embed_model,
                query=sub_query,
                period_keys=sum_period_keys,
                date_start=date_start,
                date_end=date_end,
                source_filter=source_filter,
                top_k=SUMMARY_TOP_K,
                expanded_k=SUMMARY_EXPANDED_K,
                recency_half_life_days=SUMMARY_RECENCY_HALF_LIFE_DAYS,
                reranker=reranker,
            )
            target = QueryTarget(
                query_type=QUERY_TYPE_GENERAL,
                canonical_name=None,
                display_name="general news",
                ticker=None,
                entity_type=None,
                confidence=0.0,
                resolution_mode="summary_general",
            )
            trace.update(
                {
                    "route_type": "daily_summary",
                    "candidate_count": len(chunks),
                    "ranked_count": len(chunks),
                    "ranked_candidates": chunks,
                }
            )
            sum_kinds = sorted({ch.get("retrieval_kind", "?") for ch in chunks})
            sum_sources = sorted({(ch.get("source") or "?").lower() for ch in chunks})
            logs.append(
                f"  [summary-retrieve] sub-query chunks={len(chunks)}, "
                f"sources={sum_sources}, kinds={sum_kinds}"
            )
        else:
            chunks, target, retrieve_trace = retrieve(
                query=sub_query,
                embed_model=embed_model,
                reranker=reranker,
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
                route_type=route_seed,
            )
            trace.update(retrieve_trace)
            trace["sub_query"] = sub_query

        logs.append(
            f"  [entity: {target.display_name or 'general'} | canonical: {target.canonical_name or '-'} | "
            f"confidence: {target.confidence:.2f} | chunks: {len(chunks)}]"
        )

        if primary_target is None:
            primary_target = target
            primary_date_start = date_start
            primary_date_end = date_end
        elif target.canonical_name and (not primary_target.canonical_name or target.confidence > primary_target.confidence):
            primary_target = target
            primary_date_start = date_start
            primary_date_end = date_end

        retrieval_traces.append(trace)
        if summary_mode:
            summary_raw_candidates.extend(chunks)
        else:
            all_chunks.extend(chunks)
            observability_candidates.extend(trace.get("ranked_candidates") or chunks)
            period_label = f"[{date_start} -> {date_end}] " if (date_start or date_end) else ""
            if route_seed == "signal_discovery":
                ctx = build_signal_context(sub_query, target, sub_selected_signals)
            else:
                ctx = build_context(sub_query, target, chunks)
            if market_ctx:
                ctx = ctx + "\n\n" + market_ctx
            all_contexts.append(period_label + ctx)
            for ch in chunks:
                url = ch.get("url")
                if url and url not in all_urls:
                    all_urls.append(url)

    if primary_target is None:
        primary_target = QueryTarget(
            query_type=QUERY_TYPE_GENERAL,
            canonical_name=None,
            display_name="general news",
            ticker=None,
            entity_type=None,
            confidence=0.0,
        )

    # Fetch financial context for stock-specific queries with a resolved ticker
    if (
        primary_target.ticker
        and not summary_mode
        and route_seed != "signal_discovery"
        and route_seed != "single_ticker_financial"
    ):
        fin_ctx = fetch_financial_context(
            ticker=primary_target.ticker,
            date_start=primary_date_start,
            date_end=primary_date_end,
        )
        if fin_ctx:
            logs.append(f"  [financial data] fetched [F] block for {primary_target.ticker}")
            all_contexts.append(fin_ctx)

    if selected_signals:
        deduped_signals: list[dict[str, Any]] = []
        seen_signal_ids: set[str] = set()
        for signal in selected_signals:
            signal_id = str(signal.get("signal_id") or "")
            if not signal_id or signal_id in seen_signal_ids:
                continue
            seen_signal_ids.add(signal_id)
            deduped_signals.append(signal)
        selected_signals = deduped_signals

    if summary_mode:
        if not summary_raw_candidates:
            return _empty_result("No relevant chunks found.", "daily_summary")

        logs.append(
            f"  [summary] merged candidate pool: {len(summary_raw_candidates)} chunks "
            f"from {len(sub_queries)} sub-query/ies"
        )

        query_vec = embed_model.encode([query], normalize_embeddings=True)[0]
        globally_ranked = rank_candidates(
            query=query,
            rows=summary_raw_candidates,
            query_vec=query_vec,
            target=primary_target,
            route_type="daily_summary",
            recency_half_life_days=SUMMARY_RECENCY_HALF_LIFE_DAYS,
            reranker=reranker,
        )
        observability_candidates.extend(globally_ranked)

        deduped = dedupe_chunks_for_summary(globally_ranked, max_per_article=2)
        all_chunks = _filter_summary_chunks(deduped)
        all_chunks = _cluster_summary_themes(all_chunks)

        unique_summary_sources = {
            (ch.get("source") or "unknown").strip().lower()
            for ch in all_chunks
        }
        print(
            f"  [summary] unique sources after filtering "
            f"({len(unique_summary_sources)}): {sorted(unique_summary_sources)}"
        )
        if len(unique_summary_sources) < SUMMARY_MIN_UNIQUE_SOURCES:
            msg = (
                f"Not enough distinct news sources to generate a reliable daily summary "
                f"(found {len(unique_summary_sources)}, minimum required: {SUMMARY_MIN_UNIQUE_SOURCES}). "
                "Try a broader date range or check that more sources have been ingested."
            )
            logs.append(
                f"  [summary] skipped generation - only {len(unique_summary_sources)} "
                "unique source(s) found"
            )
            return _empty_result(msg, "daily_summary")

        for ch in all_chunks:
            url = ch.get("url")
            if url and url not in all_urls:
                all_urls.append(url)

        citation_map = build_citation_map(all_chunks)
        theme_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for chunk in all_chunks:
            key = (
                str(chunk.get("theme_id") or "general"),
                str(chunk.get("theme_label") or "General"),
            )
            theme_groups[key].append(chunk)

        theme_lines = ["SUMMARY THEMES:"]
        ordered_theme_groups = sorted(theme_groups.items(), key=lambda item: len(item[1]), reverse=True)
        for idx, ((theme_id, theme_label), theme_chunks) in enumerate(ordered_theme_groups, start=1):
            avg_conf = sum(_safe_float(ch.get("theme_confidence"), 0.0) for ch in theme_chunks) / max(len(theme_chunks), 1)
            theme_lines.append(f"Theme {idx}: {theme_label} ({theme_id}) | confidence={avg_conf:.2f} | chunks={len(theme_chunks)}")
            for chunk in theme_chunks[:3]:
                chunk_uid = chunk.get("chunk_uid")
                citation = citation_map.get(chunk_uid, "?")
                source = (chunk.get("source") or "unknown").strip()
                title = (chunk.get("title") or "untitled").strip()
                theme_lines.append(f"  [{citation}] {source} | {title}")

        date_resolution_block = _build_summary_date_resolution_block(
            query,
            forced_summary_start,
            forced_summary_end,
            summary_used_date_ranges,
        )
        theme_context = "\n".join(theme_lines)
        ctx = date_resolution_block + "\n\n" + theme_context + "\n\n" + build_context(query, primary_target, all_chunks)
        if market_ctx:
            ctx = ctx + "\n\n" + market_ctx
        if fin_ctx:
            ctx = ctx + "\n\n" + fin_ctx
        all_contexts = [ctx]

    if not all_contexts and route_seed != "signal_discovery":
        return _empty_result("No relevant chunks found.", route_seed)

    final_route = (
        "single_ticker_financial"
        if route_seed == "single_ticker_financial"
        else classify_query_route(
            query=query,
            summary_mode=summary_mode,
            causal_intent=causal_intent,
            market_data_intent=market_data_intent,
            target=primary_target,
            financial_intent=financial_intent,
            explicit_latest_news_intent=explicit_latest_news_intent,
        )
    )
    if primary_target.needs_disambiguation:
        final_route = "ambiguous"
    route_profile = _resolve_route_profile(final_route)

    active_base_system_prompt = base_daily_summary_prompt if summary_mode else base_system_prompt
    system_prompt = build_system_prompt(active_base_system_prompt, memory)
    causal_system_prompt = (
        build_system_prompt(base_causal_system_prompt, memory)
        if causal_intent and not market_data_intent and not summary_mode
        else None
    )
    # Keep single-ticker instructions isolated from chat memory to reduce
    # section-label drift (e.g., reverting to Answer/Evidence/Theory format).
    single_ticker_financial_system_prompt = (
        base_single_ticker_financial_prompt
        if final_route == "single_ticker_financial"
        else None
    )
    merged_context = "\n\n---\n\n".join(all_contexts)

    # Dump the full context fed to the model on every query so it can be inspected.
    try:
        import datetime as _dt
        _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        _safe_q = re.sub(r"[^\w\s-]", "", query)[:60].strip().replace(" ", "_")
        _dump_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"query_fin_context.txt",
        )
        with open(_dump_path, "w", encoding="utf-8") as _f:
            _f.write(f"QUERY      : {query}\n")
            _f.write(f"ROUTE      : {final_route}\n")
            _f.write(f"TIMESTAMP  : {_ts}\n")
            _f.write(f"FIN_CTX    : {'present' if (fin_ctx or '').strip() else 'absent'}\n")
            _f.write("=" * 80 + "\n\n")
            _f.write(merged_context)
        print(f"  [context dump] {_dump_path}")
    except Exception as _dump_exc:
        print(f"  [context dump] failed: {_dump_exc}")

    final = ""
    market_data_note: str | None = None

    if final_route == "signal_discovery":
        final = _build_signal_discovery_answer(
            query=query,
            selected_signals=selected_signals,
            evidence_rows=all_chunks,
        )
    elif skip_generation or DEBUG_SKIP_GENERATION:
        final = "Generation skipped because DEBUG_SKIP_GENERATION is enabled."
    else:
        if single_ticker_financial_system_prompt:
            active_prompt = single_ticker_financial_system_prompt
        else:
            active_prompt = causal_system_prompt if causal_system_prompt else system_prompt
        gen_max_tokens = (
            SINGLE_TICKER_GEN_MAX_TOKENS
            if final_route == "single_ticker_financial"
            else GEN_MAX_TOKENS
        )
        final = generate_answer(
            query,
            merged_context,
            gen_client,
            active_prompt,
            max_tokens=gen_max_tokens,
        )
        if final_route == "single_ticker_financial":
            enforced_final = _enforce_single_ticker_outlook_sections(final)
            if enforced_final != final:
                logs.append(
                    "  [single_ticker_outlook_enforcement] "
                    f"applied ({SINGLE_TICKER_OUTLOOK_ENFORCEMENT_VERSION})"
                )
            else:
                logs.append(
                    "  [single_ticker_outlook_enforcement] "
                    f"already_satisfied ({SINGLE_TICKER_OUTLOOK_ENFORCEMENT_VERSION})"
                )
            final = enforced_final

    if market_data_note:
        final = f"{final}\n\nNote: {market_data_note}"

    if final_route == "single_ticker_financial":
        finance_context_present = bool((fin_ctx or "").strip())
        answer_confidence = 78.0 if finance_context_present else 0.0
        decision = "answer" if finance_context_present else "abstain"
        confidence_signals = {
            "relevant_chunks": 0.0,
            "source_diversity": 0.0,
            "retrieval_margin": 0.0,
            "verifier_support": 0.0,
            "recency_coverage": 0.0,
            "ambiguity_score": _clamp01(primary_target.ambiguity_score if primary_target else 0.0),
            "contradiction_signals": 0.0,
            "stream_brief_count": 0,
            "stream_brief_share": 0.0,
            "finance_context_present": finance_context_present,
            "route_type": final_route,
        }
    else:
        answer_confidence, confidence_signals = _compute_answer_confidence(
            all_chunks,
            primary_target,
            decision_hint="ambiguous" if final_route == "ambiguous" else None,
            route_type=final_route,
        )
        decision = _decide_answer_mode(answer_confidence)
        stream_brief_share = _safe_float(confidence_signals.get("stream_brief_share"), 0.0)
        if stream_brief_share >= STREAM_BRIEF_DOMINANT_THRESHOLD:
            decision = "abstain"
        elif stream_brief_share > STREAM_BRIEF_CONFIDENCE_PENALTY_THRESHOLD and decision == "answer":
            decision = "cautious_answer"

    if decision == "abstain" and not (skip_generation or DEBUG_SKIP_GENERATION):
        final = (
            "Answer: I cannot answer this confidently from the retrieved evidence.\n"
            "Evidence: Retrieved support is limited, ambiguous, contradictory, or too weak for a reliable conclusion.\n"
            "Theory: The query likely needs tighter target disambiguation and/or stronger corroborated sources."
        )
    if final_route != "single_ticker_financial":
        final = ensure_structured_answer(final, all_chunks)

    resolved_target = {
        "canonical_name": primary_target.canonical_name,
        "display_name": primary_target.display_name,
        "ticker": primary_target.ticker,
        "query_type": primary_target.query_type,
        "entity_type": primary_target.entity_type,
        "best_candidate": primary_target.best_candidate,
        "candidates": primary_target.candidates,
        "ambiguity_score": primary_target.ambiguity_score,
        "resolution_mode": primary_target.resolution_mode,
        "needs_disambiguation": primary_target.needs_disambiguation,
    }

    selected_macro_events: list[dict[str, Any]] = []
    seen_macro_ids: set[str] = set()
    for chunk in all_chunks:
        macro_event_id = chunk.get("macro_event_id")
        if not macro_event_id or macro_event_id in seen_macro_ids:
            continue
        seen_macro_ids.add(macro_event_id)
        selected_macro_events.append(
            {
                "macro_event_id": macro_event_id,
                "verification_status": chunk.get("verification_status") or "unknown",
                "support_score": chunk.get("support_score"),
                "confidence_calibrated": chunk.get("confidence_calibrated"),
            }
        )

    sub_trace_payloads: list[dict[str, Any]] = []
    for trace in retrieval_traces:
        ranked_candidates = trace.get("ranked_candidates") or []
        sub_trace_payloads.append(
            {
                "sub_query": trace.get("sub_query"),
                "route_type": trace.get("route_type"),
                "candidate_count": trace.get("candidate_count"),
                "ranked_count": trace.get("ranked_count"),
                "date_start": trace.get("date_start"),
                "date_end": trace.get("date_end"),
                "signal_ids": trace.get("signal_ids"),
                "finance_context_present": bool(trace.get("finance_context_present")),
                "top_candidates": [_compact_candidate(candidate) for candidate in ranked_candidates[:10]],
            }
        )
    retrieval_trace_payload = {
        "route_type": final_route,
        "route_profile": route_profile,
        "scoring_weights": _resolve_scoring_weights(final_route),
        "selected_chunk_count": len(all_chunks),
        "finance_context_present": bool((fin_ctx or "").strip()) if final_route == "single_ticker_financial" else False,
        "sub_trace_count": len(sub_trace_payloads),
        "sub_traces": sub_trace_payloads,
    }

    deduped_obs_candidates: list[dict[str, Any]] = []
    seen_obs_keys: set[str] = set()
    for candidate in observability_candidates or all_chunks:
        key = str(
            candidate.get("signal_id")
            or candidate.get("cluster_id")
            or candidate.get("chunk_uid")
            or candidate.get("macro_event_id")
            or candidate.get("article_id")
            or uuid.uuid4()
        )
        if key in seen_obs_keys:
            continue
        seen_obs_keys.add(key)
        deduped_obs_candidates.append(candidate)

    latency_ms = (time.perf_counter() - started_at) * 1000.0
    try:
        _log_observability(
            conn=sqlite_conn,
            run_id=run_id,
            query=query,
            route_type=final_route,
            target=primary_target,
            candidates=deduped_obs_candidates,
            selected_chunks=all_chunks,
            selected_signals=selected_signals,
            answer_confidence=answer_confidence,
            decision=decision,
            latency_ms=latency_ms,
            retrieval_trace=retrieval_trace_payload,
            route_reason={
                "route_seed": route_seed,
                "final_route": final_route,
                "triggered_by": "signal_discovery_keywords" if final_route == "signal_discovery" else None,
            },
            answer_meta={
                "answer_confidence": answer_confidence,
                "decision": decision,
                "route_type": final_route,
                "selected_signal_count": len(selected_signals),
            },
        )
    except Exception as exc:
        logs.append(f"  [observability] logging skipped: {exc}")

    return {
        "run_id": run_id,
        "query": query,
        "answer": final,
        "chunks": all_chunks,
        "urls": all_urls,
        "logs": logs,
        "citation_map": build_citation_map(all_chunks),
        "provenance": (
            "Why this answer: FINANCIAL DATA [F] only."
            if final_route == "single_ticker_financial"
            else format_provenance(all_chunks)
        ),
        "target": primary_target,
        "resolved_target": resolved_target,
        "resolved_target_json": resolved_target,
        "route_type": final_route,
        "retrieval_trace": retrieval_trace_payload,
        "answer_confidence": answer_confidence,
        "decision": decision,
        "answer_meta": {
            "answer_confidence": answer_confidence,
            "decision": decision,
            "route_type": final_route,
            "signals": confidence_signals,
            "contradiction_signals": bool(confidence_signals.get("contradiction_signals")),
        },
        "selected_macro_events": selected_macro_events,
        "selected_signals": selected_signals,
        "contradiction_signals": bool(confidence_signals.get("contradiction_signals")),
        "date_start": primary_date_start,
        "date_end": primary_date_end,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = load_reranker()

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
    base_daily_summary_prompt = DAILY_SUMMARY_PROMPT_TEMPLATE.format(date_min=date_min, date_max=date_max)
    base_single_ticker_financial_prompt = SINGLE_TICKER_FINANCIAL_PROMPT_TEMPLATE.format(
        date_min=date_min,
        date_max=date_max,
    )

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
        result = run_query_once(
            query=query,
            embed_model=embed_model,
            reranker=reranker,
            gen_client=gen_client,
            driver=driver,
            sqlite_conn=sqlite_conn,
            alias_to_ticker=alias_to_ticker,
            ticker_to_canonical=ticker_to_canonical,
            alias_to_fin_entity=alias_to_fin_entity,
            base_system_prompt=base_system_prompt,
            base_causal_system_prompt=base_causal_system_prompt,
            base_daily_summary_prompt=base_daily_summary_prompt,
            base_single_ticker_financial_prompt=base_single_ticker_financial_prompt,
            memory=memory,
        )
        for log_line in result["logs"]:
            print(log_line)

        elapsed = time.perf_counter() - t0
        print(f"\nAssistant: {result['answer']}")
        if SHOW_PROVENANCE:
            print()
            print(result["provenance"])
        print(f"  [{elapsed:.1f}s]\n")

        # ── Hook 5: Record turn to memory ─────────────────────────────────────
        memory.record_turn(
            query=result["query"],
            target=result["target"],
            date_start=result["date_start"],
            date_end=result["date_end"],
            answer=result["answer"],
            chunks=result["chunks"],
            source_urls=result["urls"],
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
