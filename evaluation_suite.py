from __future__ import annotations

import argparse
import importlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

import chatter
from convo_memory import ConversationMemory
from create_sql_db import create_database
from graph_schema import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from tgrag_setup import EMBED_MODEL_NAME, load_financial_entity_map, load_ticker_company_map


DEFAULT_GOLD_PATH = Path("gold_eval_cases.json")
DEFAULT_RELEASE_GATE_THRESHOLDS: dict[str, float] = {
    "case_pass_rate_min": 0.8,
    "target_resolution_accuracy_min": 0.8,
    "abstention_correctness_min": 0.8,
    "source_trust_compliance_min": 0.95,
    "verifier_precision_min": 0.7,
    "contradiction_rate_max": 0.15,
    "unsupported_mechanism_rate_max": 0.2,
}

_ABSTAIN_PATTERNS: list[str] = [
    r"\binsufficient\s+(evidence|information|context)\b",
    r"\bnot\s+enough\s+(evidence|information|context)\b",
    r"\bcannot\s+determine\b",
    r"\bunable\s+to\s+determine\b",
    r"\bno\s+relevant\s+chunks\s+found\b",
]

_RECENCY_SENSITIVE_TERMS: tuple[str, ...] = (
    "latest",
    "recent",
    "today",
    "yesterday",
    "daily",
    "now",
    "newest",
)

LOW_CONTEXT_CONTENT_CLASSES: frozenset[str] = frozenset({"stream_brief"})

# ---------------------------------------------------------------------------
# Answer-quality parsing helpers (deterministic, no LLM judge)
# ---------------------------------------------------------------------------

# Phrases the model is instructed to use (from CAUSAL_SYSTEM_PROMPT_TEMPLATE).
_DIRECTION_PATTERNS: list[tuple[str, str]] = [
    # (regex pattern, canonical label)
    (r"\b(strongly\s+)?positive\b", "positive"),
    (r"\b(strongly\s+)?negative\b", "negative"),
    (r"\bmixed\b", "mixed"),
    (r"\bunclear\b", "unclear"),
    (r"\bnet\s+positive\b", "positive"),
    (r"\bnet\s+negative\b", "negative"),
    (r"\boffsets?\b.*\bgains?\b|\bgains?\b.*\boffsets?\b", "mixed"),
    (r"\binsufficient\s+(evidence|article|context)\b", "unclear"),
]

# Patterns that signal hedged / mixed-evidence language
_MIXED_LANGUAGE_PATTERNS: list[str] = [
    r"\bmixed\b",
    r"\boffset(ting|s)?\b",
    r"\bcounterbalance[sd]?\b",
    r"\bpartly\s+offset\b",
    r"\bon\s+(the\s+)?one\s+hand\b",
    r"\bon\s+(the\s+)?other\s+hand\b",
    r"\bwhile\s+(also|simultaneously)\b",
    r"\bbut\s+(also|simultaneously)\b",
    r"\bdepends\s+on\b",
    r"\buncertain\b",
    r"\bconflict(ing|s)?\b.*\b(evidence|signal|data)\b",
    r"\b(evidence|signal|data)\b.*\bconflict(ing|s)?\b",
]

# Overclaim patterns: asserting one direction cleanly despite hedge language
_OVERCLAIM_PATTERNS: list[str] = [
    r"\bwill\s+(certainly|definitely|surely)\b",
    r"\bclearly\s+(bullish|bearish|positive|negative)\b",
    r"\bno\s+doubt\b",
    r"\bguaranteed\s+to\b",
    r"\bwithout\s+question\b",
    r"\bonly\s+(positive|negative|bullish|bearish)\b",
    r"\bpurely\s+(positive|negative|bullish|bearish)\b",
    r"\bstraightforward(ly)?\b",
    # asserting a direction even though offsetting channels exist
    r"\boverall(ly)?\s+(positive|negative|bullish|bearish)\b",
    r"\bnet\s+effect\s+is\s+(clearly|definitely|certainly)\b",
]

# Confidence extraction: looks for "confidence: 75%" or "75% confident" etc.
_CONFIDENCE_RE = re.compile(
    r"confidence[:\s]+(\d{1,3})\s*%|(\d{1,3})\s*%\s+confident",
    re.IGNORECASE,
)


def _extract_direction(text: str) -> str:
    """
    Extract dominant direction from Answer section.
    Prioritizes explicit final calls.
    """
    lower = text.lower()

    # Strong explicit phrases first
    if re.search(r"\bnegative\s+for\b|\bbearish\b", lower):
        return "negative"
    if re.search(r"\bpositive\s+for\b|\bbullish\b", lower):
        return "positive"

    if re.search(r"\bmixed\b", lower):
        return "mixed"
    if re.search(r"\bunclear\b|\binsufficient\b", lower):
        return "unclear"

    # fallback to patterns
    for pattern, label in _DIRECTION_PATTERNS:
        if re.search(pattern, lower):
            return label

    return "unclear"


def _extract_confidence(text: str) -> int | None:
    """Return confidence as 0-100 integer, or None if not found."""
    m = _CONFIDENCE_RE.search(text)
    if m:
        raw = int(m.group(1) or m.group(2))
        return max(0, min(100, raw))
    return None


def _extract_mechanisms(text: str, known_mechanisms: list[str]) -> list[str]:
    found = []
    lower = text.lower()

    for mech in known_mechanisms:
        tokens = mech.lower().replace("_", " ").split()

        # require partial match instead of exact phrase
        if any(token in lower for token in tokens):
            found.append(mech)

    return found


def _detect_mixed_language(text: str) -> list[str]:
    """Return which mixed/hedge patterns fire in the text."""
    lower = text.lower()
    return [p for p in _MIXED_LANGUAGE_PATTERNS if re.search(p, lower)]


def _detect_overclaims(text: str) -> list[str]:
    """Return which overclaim patterns fire in the text."""
    lower = text.lower()
    return [p for p in _OVERCLAIM_PATTERNS if re.search(p, lower)]

def _split_sections(answer: str) -> dict[str, str]:
    """
    Split answer into Answer / Evidence / Theory sections.
    """
    sections = {"answer": "", "evidence": "", "theory": ""}
    current = None

    for line in (answer or "").splitlines():
        lower = line.lower().strip()

        if lower.startswith("answer"):
            current = "answer"
            continue
        elif lower.startswith("evidence"):
            current = "evidence"
            continue
        elif lower.startswith("theory"):
            current = "theory"
            continue

        if current:
            sections[current] += line + "\n"

    return sections

def parse_answer_meta(answer: str, all_mechanisms: list[str]) -> dict[str, Any]:
    counterarg_re = re.compile(
        r"\b(however|nevertheless|that said|on the other hand|offset|counteract|"
        r"but also|though|mitigat(es?|ing)|headwind|tailwind|risk|caveat)\b",
        re.IGNORECASE,
    )
    sections = _split_sections(answer)
    answer_section = sections["answer"]
    full_text = answer

    return {
        "direction": _extract_direction(answer_section),  # ✅ only Answer
        "confidence": _extract_confidence(answer_section),
        "mechanisms_found": _extract_mechanisms(full_text, all_mechanisms),
        "has_counterargs": bool(counterarg_re.search(full_text)),
        "mixed_lang_hits": _detect_mixed_language(full_text),
        "overclaim_hits": _detect_overclaims(answer_section),  # ✅ only Answer
        "_raw_answer": answer,
        "_sections": sections,
    }

def _has_dominance_language(text: str) -> bool:
    return bool(re.search(
        r"\b(dominates?|outweighs?|more\s+than\s+offsets?)\b",
        (text or "").lower()
    ))


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _target_attr(target: Any, key: str) -> Any:
    if target is None:
        return None
    if isinstance(target, dict):
        return target.get(key)
    return getattr(target, key, None)


def _extract_route_type(case: dict[str, Any], result: dict[str, Any]) -> str:
    explicit = (
        result.get("route_type")
        or result.get("route")
        or result.get("query_route")
        or (result.get("routing") or {}).get("route_type")
        or case.get("expected_route_type")
    )
    if explicit:
        return str(explicit)

    query = str(case.get("query", "")).lower()
    if "summary" in query or "recap" in query:
        return "daily_summary"
    if any(term in query for term in ("latest", "recent", "today", "yesterday")):
        return "latest_news"

    target = result.get("target")
    query_type = _target_attr(target, "query_type")
    if query_type == "single_entity":
        return "entity_profile"
    if query_type == "general":
        return "broad_exploration"
    return "unknown"


def _extract_answer_confidence(result: dict[str, Any], parsed: dict[str, Any] | None) -> float | None:
    explicit = (
        result.get("answer_confidence")
        or result.get("confidence")
        or (result.get("answer_meta") or {}).get("confidence")
    )
    score = _to_float(explicit)
    if score is None and parsed is not None:
        score = _to_float(parsed.get("confidence"))
    if score is None:
        return None
    # Normalize either 0-1 or 0-100 into a 0-100 scale.
    if 0.0 <= score <= 1.0:
        score = score * 100.0
    return max(0.0, min(100.0, score))


def _extract_decision(result: dict[str, Any], answer: str, answer_confidence: float | None) -> str:
    explicit = (
        result.get("decision")
        or result.get("answer_decision")
        or (result.get("answer_meta") or {}).get("decision")
    )
    norm = _normalized_text(explicit)
    if norm in {"abstain", "cautious_answer", "answer"}:
        return norm

    lower = (answer or "").lower()
    if any(re.search(pattern, lower) for pattern in _ABSTAIN_PATTERNS):
        return "abstain"
    if answer_confidence is not None and answer_confidence < 35.0:
        return "abstain"
    if answer_confidence is not None and answer_confidence <= 60.0:
        return "cautious_answer"
    return "answer"


def _extract_resolved_target(result: dict[str, Any]) -> dict[str, Any]:
    explicit = result.get("resolved_target_json") or result.get("resolved_target")
    if isinstance(explicit, dict):
        canonical = explicit.get("canonical_name") or explicit.get("best_candidate")
        candidates = explicit.get("candidates") or []
        ambiguity_score = _to_float(explicit.get("ambiguity_score"))
        needs_disambiguation = explicit.get("needs_disambiguation")
        if needs_disambiguation is None and ambiguity_score is not None:
            needs_disambiguation = ambiguity_score >= 0.66
        return {
            "canonical_name": canonical,
            "ticker": explicit.get("ticker"),
            "query_type": explicit.get("query_type"),
            "candidates": candidates if isinstance(candidates, list) else [],
            "ambiguity_score": ambiguity_score,
            "resolution_mode": explicit.get("resolution_mode") or "explicit",
            "needs_disambiguation": bool(needs_disambiguation),
        }

    target = result.get("target")
    if target is None:
        return {
            "canonical_name": None,
            "ticker": None,
            "query_type": None,
            "candidates": [],
            "ambiguity_score": None,
            "resolution_mode": "missing",
            "needs_disambiguation": False,
        }

    candidates = _target_attr(target, "candidates") or []
    confidence = _to_float(_target_attr(target, "confidence"))
    ambiguity_score = None
    if len(candidates) > 1 and confidence is not None:
        ambiguity_score = max(0.0, min(1.0, 1.0 - confidence))
    elif len(candidates) > 1:
        ambiguity_score = min(1.0, (len(candidates) - 1) / 5.0)
    elif confidence is not None:
        ambiguity_score = max(0.0, min(1.0, 1.0 - confidence))

    needs_disambiguation = bool(ambiguity_score is not None and ambiguity_score >= 0.66)
    resolution_mode = "candidate_list" if len(candidates) > 1 else "direct"
    return {
        "canonical_name": _target_attr(target, "canonical_name"),
        "ticker": _target_attr(target, "ticker"),
        "query_type": _target_attr(target, "query_type"),
        "candidates": candidates if isinstance(candidates, list) else [],
        "ambiguity_score": ambiguity_score,
        "resolution_mode": resolution_mode,
        "needs_disambiguation": needs_disambiguation,
    }


def _resolve_expected_target(case: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected_target")
    if isinstance(expected, str):
        return {"canonical_name": expected}
    if not isinstance(expected, dict):
        expected = {}
    if "canonical_name" not in expected and case.get("expected_target_canonical"):
        expected["canonical_name"] = case.get("expected_target_canonical")
    if "ticker" not in expected and case.get("expected_target_ticker"):
        expected["ticker"] = case.get("expected_target_ticker")
    if "query_type" not in expected and case.get("expected_query_type"):
        expected["query_type"] = case.get("expected_query_type")
    return expected


def _evaluate_target_resolution(
    case: dict[str, Any],
    resolved_target: dict[str, Any],
) -> dict[str, Any]:
    expected = _resolve_expected_target(case)
    if not expected:
        return {"available": False, "skipped": True, "passed": True}

    checks: list[tuple[str, bool, Any, Any]] = []
    for key in ("canonical_name", "ticker", "query_type"):
        exp_val = expected.get(key)
        if exp_val is None:
            continue
        obs_val = resolved_target.get(key)
        passed = _normalized_text(exp_val) == _normalized_text(obs_val)
        checks.append((key, passed, exp_val, obs_val))

    if not checks:
        return {"available": False, "skipped": True, "passed": True}

    passed = all(item[1] for item in checks)
    return {
        "available": True,
        "passed": passed,
        "checks": [
            {"field": field, "passed": ok, "expected": exp, "observed": obs}
            for field, ok, exp, obs in checks
        ],
    }


def _is_abstain_decision(decision: str, answer: str) -> bool:
    if _normalized_text(decision) == "abstain":
        return True
    lower = (answer or "").lower()
    return any(re.search(pattern, lower) for pattern in _ABSTAIN_PATTERNS)


def _evaluate_abstention(
    case: dict[str, Any],
    decision: str,
    answer: str,
) -> dict[str, Any]:
    expected_abstain = case.get("expected_abstain")
    if expected_abstain is None:
        return {"available": False, "skipped": True, "passed": True}
    observed_abstain = _is_abstain_decision(decision, answer)
    passed = bool(observed_abstain == bool(expected_abstain))
    return {
        "available": True,
        "passed": passed,
        "expected_abstain": bool(expected_abstain),
        "observed_abstain": observed_abstain,
    }


def _table_columns(sqlite_conn, table_name: str) -> set[str]:
    rows = sqlite_conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    columns = set()
    for row in rows:
        if isinstance(row, sqlite3.Row):
            columns.add(str(row["name"]))
        else:
            columns.add(str(row[1]))
    return columns


def _fetch_chunk_article_meta(sqlite_conn, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not chunk_ids:
        return {}

    article_cols = _table_columns(sqlite_conn, "articles")
    select_fields = [
        "c.chunk_id AS chunk_id",
        "a.source AS source",
    ]
    if "source_trust_tier" in article_cols:
        select_fields.append("a.source_trust_tier AS source_trust_tier")
    if "content_class" in article_cols:
        select_fields.append("a.content_class AS content_class")
    if "article_quality_score" in article_cols:
        select_fields.append("a.article_quality_score AS article_quality_score")

    placeholders = ",".join("?" for _ in chunk_ids)
    sql = f"""
        SELECT {", ".join(select_fields)}
        FROM chunks c
        JOIN articles a ON a.article_id = c.article_id
        WHERE c.chunk_id IN ({placeholders})
    """
    rows = sqlite_conn.execute(sql, chunk_ids).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        out[str(row["chunk_id"])] = {
            "source": row["source"],
            "source_trust_tier": row["source_trust_tier"] if "source_trust_tier" in row.keys() else None,
            "content_class": row["content_class"] if "content_class" in row.keys() else None,
            "article_quality_score": (
                float(row["article_quality_score"])
                if "article_quality_score" in row.keys() and row["article_quality_score"] is not None
                else None
            ),
        }
    return out


def _evaluate_source_trust(
    case: dict[str, Any],
    source_meta_by_chunk: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    policy = case.get("source_trust_policy", {}) or {}
    disallow_tiers = set(
        policy.get("disallow_tiers")
        or case.get("disallow_source_trust_tiers")
        or ["blocked"]
    )
    allow_tiers = set(policy.get("allow_tiers") or case.get("allow_source_trust_tiers") or [])
    allowed_classes = set(
        policy.get("allowed_content_classes")
        or case.get("allowed_content_classes")
        or []
    )
    min_quality = _to_float(
        policy.get("min_article_quality_score")
        if "min_article_quality_score" in policy
        else case.get("min_article_quality_score")
    )

    tiers = [
        str(item.get("source_trust_tier"))
        for item in source_meta_by_chunk.values()
        if item.get("source_trust_tier") not in (None, "")
    ]
    classes = [
        str(item.get("content_class"))
        for item in source_meta_by_chunk.values()
        if item.get("content_class") not in (None, "")
    ]
    qualities = [
        float(item["article_quality_score"])
        for item in source_meta_by_chunk.values()
        if item.get("article_quality_score") is not None
    ]

    has_signals = bool(tiers or classes or qualities)
    if not has_signals:
        return {"available": False, "skipped": True, "passed": True}

    violations: list[str] = []
    if disallow_tiers:
        blocked_hits = [tier for tier in tiers if tier in disallow_tiers]
        if blocked_hits:
            violations.append(f"disallowed tiers present: {sorted(set(blocked_hits))}")
    if allow_tiers:
        outside = [tier for tier in tiers if tier not in allow_tiers]
        if outside:
            violations.append(f"tiers outside allow-list: {sorted(set(outside))}")
    if allowed_classes:
        bad_classes = [cls for cls in classes if cls not in allowed_classes]
        if bad_classes:
            violations.append(f"content classes outside allow-list: {sorted(set(bad_classes))}")
    if min_quality is not None:
        below = [score for score in qualities if score < min_quality]
        if below:
            violations.append(
                f"article quality below min {min_quality}: count={len(below)}"
            )

    return {
        "available": True,
        "passed": not violations,
        "violations": violations,
        "tier_counts": {
            tier: tiers.count(tier)
            for tier in sorted(set(tiers))
        },
        "content_class_counts": {
            cls: classes.count(cls)
            for cls in sorted(set(classes))
        },
        "quality_count": len(qualities),
    }


def _extract_verifier_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw = (
        result.get("selected_macro_events")
        or result.get("macro_events")
        or result.get("verifier_events")
        or []
    )
    if not isinstance(raw, list):
        return []
    events: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        event_id = (
            item.get("macro_event_id")
            or item.get("event_id")
            or item.get("id")
        )
        if not event_id:
            continue
        events.append(
            {
                "macro_event_id": str(event_id),
                "verification_status": _normalized_text(item.get("verification_status")) or "unknown",
                "support_score": _to_float(item.get("support_score")),
                "confidence_calibrated": _to_float(item.get("confidence_calibrated")),
            }
        )
    return events


def _evaluate_verifier_precision(
    case: dict[str, Any],
    verifier_events: list[dict[str, Any]],
) -> dict[str, Any]:
    verified_ids = {
        event["macro_event_id"]
        for event in verifier_events
        if event.get("verification_status") == "verified"
    }
    if not verified_ids:
        return {
            "available": False,
            "skipped": True,
            "passed": True,
            "hits": 0,
            "predicted_verified": 0,
            "precision": None,
        }

    expected = set(case.get("expected_macro_event_ids", []))
    hits = sorted(verified_ids & expected)
    precision = _safe_ratio(len(hits), len(verified_ids))
    return {
        "available": True,
        "passed": True,
        "hits": len(hits),
        "predicted_verified": len(verified_ids),
        "precision": precision,
        "expected_macro_event_ids": sorted(expected),
        "verified_macro_event_ids": sorted(verified_ids),
    }


def _detect_direction_contradiction(answer: str) -> bool:
    sections = _split_sections(answer)
    answer_section = (sections.get("answer") or answer or "").lower()
    has_positive = bool(re.search(r"\b(positive|bullish|net\s+positive)\b", answer_section))
    has_negative = bool(re.search(r"\b(negative|bearish|net\s+negative)\b", answer_section))
    has_mixed = bool(re.search(r"\bmixed\b", answer_section))
    return has_positive and has_negative and not has_mixed


def _evaluate_contradiction(result: dict[str, Any]) -> dict[str, Any]:
    explicit = (
        result.get("contradiction_signals")
        or (result.get("answer_meta") or {}).get("contradiction_signals")
    )
    if isinstance(explicit, bool):
        has_contradiction = explicit
        source = "explicit_bool"
    elif isinstance(explicit, (list, tuple, set)):
        has_contradiction = len(explicit) > 0
        source = "explicit_list"
    elif isinstance(explicit, dict):
        has_contradiction = bool(explicit.get("has_contradiction"))
        source = "explicit_dict"
    else:
        has_contradiction = _detect_direction_contradiction(result.get("answer", ""))
        source = "deterministic_answer_parse"
    return {
        "available": True,
        "has_contradiction": bool(has_contradiction),
        "source": source,
    }


def _evaluate_unsupported_mechanisms(
    case: dict[str, Any],
    result: dict[str, Any],
    parsed_answer: dict[str, Any] | None,
) -> dict[str, Any]:
    explicit = result.get("unsupported_mechanisms")
    if isinstance(explicit, list):
        unsupported = [str(item) for item in explicit if str(item).strip()]
        return {
            "available": True,
            "unsupported_count": len(unsupported),
            "mechanism_count": len(unsupported),
            "unsupported_mechanisms": unsupported,
        }

    expected_quality = case.get("expected_answer_quality", {}) or {}
    allowed = set(expected_quality.get("required_mechanisms", [])) | set(
        expected_quality.get("optional_mechanisms", [])
    ) | set(case.get("allowed_mechanisms", []))
    if parsed_answer is None or not allowed:
        return {
            "available": False,
            "skipped": True,
            "unsupported_count": 0,
            "mechanism_count": 0,
            "unsupported_mechanisms": [],
        }

    observed = set(parsed_answer.get("mechanisms_found", []))
    unsupported = sorted(observed - allowed)
    return {
        "available": True,
        "unsupported_count": len(unsupported),
        "mechanism_count": len(observed),
        "unsupported_mechanisms": unsupported,
        "observed_mechanisms": sorted(observed),
        "allowed_mechanisms": sorted(allowed),
    }


def _ambiguity_slice(resolved_target: dict[str, Any]) -> str:
    if resolved_target.get("needs_disambiguation"):
        return "high"
    score = _to_float(resolved_target.get("ambiguity_score"))
    if score is None:
        return "unknown"
    if score >= 0.66:
        return "high"
    if score >= 0.33:
        return "medium"
    return "low"


def _recency_sensitivity_slice(case: dict[str, Any]) -> str:
    explicit = case.get("recency_sensitivity") or case.get("slice_recency_sensitivity")
    if explicit:
        return str(explicit)
    query = str(case.get("query", "")).lower()
    if any(term in query for term in _RECENCY_SENSITIVE_TERMS):
        return "high"
    return "low"


def _source_quality_slice(source_meta_by_chunk: dict[str, dict[str, Any]], case: dict[str, Any]) -> str:
    explicit = case.get("source_quality_slice")
    if explicit:
        return str(explicit)

    tiers = [
        str(item.get("source_trust_tier"))
        for item in source_meta_by_chunk.values()
        if item.get("source_trust_tier")
    ]
    classes = [
        str(item.get("content_class")).strip().lower()
        for item in source_meta_by_chunk.values()
        if item.get("content_class")
    ]
    qualities = [
        float(item["article_quality_score"])
        for item in source_meta_by_chunk.values()
        if item.get("article_quality_score") is not None
    ]

    if "blocked" in tiers or "tier_3" in tiers:
        return "low"
    if classes and all(cls in LOW_CONTEXT_CONTENT_CLASSES for cls in classes):
        # stream_brief is a legitimate source class, but should be evaluated as low-context.
        if qualities and (sum(qualities) / len(qualities)) >= 0.70:
            return "medium"
        return "low"
    if qualities:
        avg = sum(qualities) / len(qualities)
        if avg >= 0.75:
            return "high"
        if avg >= 0.45:
            return "medium"
        return "low"
    if "tier_1" in tiers and "tier_3" not in tiers:
        return "high" if "tier_2" not in tiers else "medium"
    if tiers:
        return "medium"
    return "unknown"


def _aggregate_boolean_metric(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    total = 0
    hits = 0
    case_ids: list[str] = []
    for result in results:
        metric = ((result.get("v2") or {}).get(key) or {})
        if not metric.get("available"):
            continue
        total += 1
        if metric.get("passed"):
            hits += 1
        case_ids.append(result.get("id", "unknown"))
    return {
        "value": _safe_ratio(hits, total),
        "hits": hits,
        "total": total,
        "cases": case_ids,
    }


def _aggregate_verifier_precision(results: list[dict[str, Any]]) -> dict[str, Any]:
    hits = 0
    predicted = 0
    for result in results:
        metric = ((result.get("v2") or {}).get("verifier_precision_eval") or {})
        if not metric.get("available"):
            continue
        hits += int(metric.get("hits", 0))
        predicted += int(metric.get("predicted_verified", 0))
    return {
        "value": _safe_ratio(hits, predicted),
        "hits": hits,
        "predicted_verified": predicted,
    }


def _aggregate_contradiction_rate(results: list[dict[str, Any]]) -> dict[str, Any]:
    contradictions = 0
    total = 0
    for result in results:
        metric = ((result.get("v2") or {}).get("contradiction_eval") or {})
        if not metric.get("available", False):
            continue
        total += 1
        if metric.get("has_contradiction"):
            contradictions += 1
    return {
        "value": _safe_ratio(contradictions, total),
        "contradictions": contradictions,
        "total": total,
    }


def _aggregate_unsupported_mechanism_rate(results: list[dict[str, Any]]) -> dict[str, Any]:
    unsupported = 0
    total = 0
    for result in results:
        metric = ((result.get("v2") or {}).get("unsupported_mechanism_eval") or {})
        if not metric.get("available"):
            continue
        unsupported += int(metric.get("unsupported_count", 0))
        total += int(metric.get("mechanism_count", 0))
    return {
        "value": _safe_ratio(unsupported, total),
        "unsupported": unsupported,
        "mechanisms_considered": total,
    }


def _compute_v2_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "target_resolution_accuracy": _aggregate_boolean_metric(results, "target_resolution_eval"),
        "abstention_correctness": _aggregate_boolean_metric(results, "abstention_eval"),
        "source_trust_compliance": _aggregate_boolean_metric(results, "source_trust_eval"),
        "verifier_precision": _aggregate_verifier_precision(results),
        "contradiction_rate": _aggregate_contradiction_rate(results),
        "unsupported_mechanism_rate": _aggregate_unsupported_mechanism_rate(results),
    }


def _bucketize(results: list[dict[str, Any]], slice_key: str) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        label = (((result.get("v2") or {}).get("slices") or {}).get(slice_key) or "unknown")
        buckets.setdefault(str(label), []).append(result)
    return buckets


def _slice_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for result in results if result.get("passed"))
    chunk_recalls = [float(result["chunk_score"]["recall"]) for result in results if result.get("chunk_score")]
    return {
        "cases": len(results),
        "pass_rate": _safe_ratio(passed, len(results)),
        "avg_chunk_recall": (
            round(sum(chunk_recalls) / len(chunk_recalls), 4) if chunk_recalls else None
        ),
        "metrics": _compute_v2_metrics(results),
    }


def _build_slice_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for slice_key in ("query_type_route", "ambiguity", "recency_sensitivity", "source_quality"):
        bucketed = _bucketize(results, slice_key)
        report[slice_key] = {
            label: _slice_summary(bucket_results)
            for label, bucket_results in sorted(bucketed.items(), key=lambda item: item[0])
        }
    return report


def _evaluate_release_gate(
    *,
    metrics: dict[str, Any],
    case_passed: int,
    case_total: int,
    thresholds: dict[str, float],
    require_all_metrics: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    missing: list[str] = []

    case_pass_rate = _safe_ratio(case_passed, case_total) if case_total else 0.0
    case_threshold = thresholds["case_pass_rate_min"]
    case_ok = (case_pass_rate or 0.0) >= case_threshold
    checks.append(
        {
            "metric": "case_pass_rate",
            "operator": ">=",
            "threshold": case_threshold,
            "observed": case_pass_rate,
            "available": True,
            "passed": case_ok,
        }
    )
    if not case_ok:
        failures.append("case_pass_rate")

    metric_specs = [
        ("target_resolution_accuracy", ">=", thresholds["target_resolution_accuracy_min"]),
        ("abstention_correctness", ">=", thresholds["abstention_correctness_min"]),
        ("source_trust_compliance", ">=", thresholds["source_trust_compliance_min"]),
        ("verifier_precision", ">=", thresholds["verifier_precision_min"]),
        ("contradiction_rate", "<=", thresholds["contradiction_rate_max"]),
        ("unsupported_mechanism_rate", "<=", thresholds["unsupported_mechanism_rate_max"]),
    ]

    for metric_name, operator, threshold in metric_specs:
        observed = _to_float((metrics.get(metric_name) or {}).get("value"))
        available = observed is not None
        if not available:
            passed = not require_all_metrics
            missing.append(metric_name)
        elif operator == ">=":
            passed = observed >= threshold
        else:
            passed = observed <= threshold
        checks.append(
            {
                "metric": metric_name,
                "operator": operator,
                "threshold": threshold,
                "observed": observed,
                "available": available,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(metric_name)

    if failures:
        status = "fail"
    elif missing:
        status = "pass_with_gaps"
    else:
        status = "pass"

    return {
        "status": status,
        "checks": checks,
        "thresholds": thresholds,
        "require_all_metrics": require_all_metrics,
        "missing_metrics": missing,
        "failed_metrics": failures,
    }

# ---------------------------------------------------------------------------
# Macro-answer evaluation against gold expectations
# ---------------------------------------------------------------------------


def evaluate_macro_answer(
    parsed: dict[str, Any],
    gold: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare parsed answer metadata against gold-case answer-quality fields.

    Gold fields (all optional):
      expected_direction        – exact expected label (positive/negative/mixed/unclear)
      allowed_directions        – list of acceptable labels (overrides expected_direction check)
      required_mechanisms       – mechanisms that MUST appear in the answer
      optional_mechanisms       – mechanisms that may appear (informational only)
      required_counterarguments – set True / list of strings that must appear
      forbidden_overclaims      – list of overclaim patterns that must NOT fire
      max_confidence_if_mixed   – upper bound on confidence when direction is mixed/unclear

    Returns a dict with per-check pass/fail booleans and an overall passed flag.
    """
    failures: list[str] = []
    details: dict[str, Any] = {}

    # --- direction check ---
    direction = parsed["direction"]
    allowed = gold.get("allowed_directions") or (
        [gold["expected_direction"]] if gold.get("expected_direction") else None
    )
    if allowed is not None:
        direction_ok = direction in allowed
        details["direction_check"] = {
            "passed": direction_ok,
            "expected_one_of": allowed,
            "observed": direction,
        }
        if not direction_ok:
            failures.append(f"direction '{direction}' not in {allowed}")
    else:
        details["direction_check"] = {"passed": True, "skipped": True}

    # --- confidence-cap check for mixed/unclear ---
    confidence = parsed["confidence"]
    max_conf = gold.get("max_confidence_if_mixed")
    if max_conf is not None and direction in ("mixed", "unclear"):
        cap_ok = (confidence is None) or (confidence <= max_conf)
        details["confidence_cap_check"] = {
            "passed": cap_ok,
            "max_allowed": max_conf,
            "observed": confidence,
            "direction": direction,
        }
        if not cap_ok:
            failures.append(
                f"confidence {confidence}% exceeds cap {max_conf}% for direction '{direction}'"
            )
    else:
        details["confidence_cap_check"] = {"passed": True, "skipped": True}

    # --- required mechanisms ---
    req_mechs = gold.get("required_mechanisms", [])
    if req_mechs:
        found_mechs = set(parsed["mechanisms_found"])
        missing = [m for m in req_mechs if m not in found_mechs]
        mech_ok = not missing
        details["required_mechanisms_check"] = {
            "passed": mech_ok,
            "required": req_mechs,
            "found": sorted(found_mechs),
            "missing": missing,
        }
        if not mech_ok:
            failures.append(f"required mechanisms not mentioned: {missing}")
    else:
        details["required_mechanisms_check"] = {"passed": True, "skipped": True}

    # --- required counterarguments ---
    req_counter = gold.get("required_counterarguments", False)
    if req_counter is True or (isinstance(req_counter, list) and req_counter):
        if isinstance(req_counter, list):
            # Specific phrases expected
            lower_answer = (parsed.get("_raw_answer") or "").lower()
            missing_ca = [c for c in req_counter if c.lower() not in lower_answer]
            ca_ok = not missing_ca
            details["counterarg_check"] = {
                "passed": ca_ok,
                "required": req_counter,
                "missing": missing_ca,
            }
            if not ca_ok:
                failures.append(f"required counterargument phrases missing: {missing_ca}")
        else:
            # Just requires that some counterargument language exists
            ca_ok = parsed["has_counterargs"]
            details["counterarg_check"] = {
                "passed": ca_ok,
                "required": True,
                "has_counterargs": ca_ok,
            }
            if not ca_ok:
                failures.append("answer must acknowledge counterarguments / offsetting channels")
    else:
        details["counterarg_check"] = {"passed": True, "skipped": True}

    # --- forbidden overclaims ---
    forbidden_oc = gold.get("forbidden_overclaims", [])
    answer_section = (
        parsed.get("_sections", {}).get("answer")
        if isinstance(parsed.get("_sections"), dict)
        else parsed.get("_raw_answer", "")
    ) or ""
    raw_answer = parsed.get("_raw_answer") or ""
    has_counter = parsed["has_counterargs"]
    confidence = parsed["confidence"]

    if forbidden_oc:
        fired = [
            p for p in forbidden_oc
            if re.search(p, raw_answer, re.IGNORECASE)
        ]
        oc_ok = not fired
        details["overclaim_check"] = {
            "passed": oc_ok,
            "forbidden_patterns": forbidden_oc,
            "fired": fired,
        }
        if not oc_ok:
            failures.append(f"forbidden overclaim patterns fired: {fired}")

    elif has_counter and direction in ("positive", "negative"):
        if not _has_dominance_language(answer_section):
            if confidence is None or confidence > 50:
                oc_ok = False
                details["overclaim_check"] = {
                    "passed": False,
                    "auto_penalty": True,
                    "has_counterargs": has_counter,
                    "direction": direction,
                    "confidence": confidence,
                    "dominance_language": False,
                    "reason": (
                        "Directional call made despite counterarguments without "
                        "dominance justification and with too much confidence."
                    ),
                }
                failures.append(
                    "overclaim: directional call made despite counterarguments "
                    "without dominance justification"
                )
            else:
                oc_ok = True
                details["overclaim_check"] = {
                    "passed": True,
                    "softened_by_low_confidence": True,
                    "has_counterargs": has_counter,
                    "direction": direction,
                    "confidence": confidence,
                }
        else:
            oc_ok = True
            details["overclaim_check"] = {
                "passed": True,
                "dominance_language": True,
                "has_counterargs": has_counter,
                "direction": direction,
                "confidence": confidence,
            }

    elif parsed["overclaim_hits"] and parsed.get("mixed_lang_hits"):
        oc_ok = False
        details["overclaim_check"] = {
            "passed": False,
            "auto_penalty": True,
            "overclaim_hits": parsed["overclaim_hits"],
            "mixed_lang_hits": parsed["mixed_lang_hits"],
            "reason": (
                "Model used strong directional language alongside mixed/hedge language."
            ),
        }
        failures.append("overclaim: strong directional call conflicts with hedge language")

    else:
        oc_ok = True
        details["overclaim_check"] = {"passed": True, "skipped": True}

    passed = not failures
    return {
        "passed": passed,
        "failures": failures,
        "direction": direction,
        "confidence": confidence,
        "mechanisms_found": parsed["mechanisms_found"],
        "has_counterargs": parsed["has_counterargs"],
        "mixed_lang_hits": parsed["mixed_lang_hits"],
        "overclaim_hits": parsed["overclaim_hits"],
        **details,
    }


def load_gold_cases(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_gold_cases(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_runtime(*, enable_reranker: bool, need_generation: bool) -> dict[str, Any]:
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = chatter.load_reranker() if enable_reranker else None
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    create_database(chatter.SQLITE_DB)
    sqlite_conn = chatter.connect_sqlite(chatter.SQLITE_DB)
    alias_to_ticker, ticker_to_canonical = load_ticker_company_map(chatter.TICKER_MAP_PATH)
    alias_to_fin_entity = load_financial_entity_map(chatter.FIN_ENTITY_MAP_PATH, chatter.TICKER_MAP_PATH)
    date_min, date_max = chatter._get_sqlite_date_range(sqlite_conn)
    base_system_prompt = chatter.SYSTEM_PROMPT_TEMPLATE.format(date_min=date_min, date_max=date_max)
    base_causal_system_prompt = chatter.CAUSAL_SYSTEM_PROMPT_TEMPLATE.format(date_min=date_min, date_max=date_max)
    base_daily_summary_prompt = chatter.DAILY_SUMMARY_PROMPT_TEMPLATE.format(
        date_min=date_min,
        date_max=date_max,
    )

    gen_client = None
    if need_generation:
        if not chatter.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is required unless --skip-generation is used.")
        anthropic_module = importlib.import_module("anthropic")
        Anthropic = getattr(anthropic_module, "Anthropic")
        gen_client = Anthropic(api_key=chatter.ANTHROPIC_API_KEY)

    return {
        "embed_model": embed_model,
        "reranker": reranker,
        "driver": driver,
        "sqlite_conn": sqlite_conn,
        "alias_to_ticker": alias_to_ticker,
        "ticker_to_canonical": ticker_to_canonical,
        "alias_to_fin_entity": alias_to_fin_entity,
        "base_system_prompt": base_system_prompt,
        "base_causal_system_prompt": base_causal_system_prompt,
        "base_daily_summary_prompt": base_daily_summary_prompt,
        "gen_client": gen_client,
    }


def close_runtime(runtime: dict[str, Any]) -> None:
    runtime["sqlite_conn"].close()
    runtime["driver"].close()


def _chunk_entities(sqlite_conn, chunk_ids: list[str]) -> list[str]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = sqlite_conn.execute(
        f"""
        SELECT DISTINCT canonical_entity_id
        FROM entity_mentions
        WHERE chunk_id IN ({placeholders})
          AND canonical_entity_id IS NOT NULL
          AND canonical_entity_id <> ''
        ORDER BY canonical_entity_id
        """,
        chunk_ids,
    ).fetchall()
    return [row["canonical_entity_id"] for row in rows]


def _chunk_macro_events(sqlite_conn, chunk_ids: list[str]) -> tuple[list[str], list[str]]:
    if not chunk_ids:
        return [], []
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = sqlite_conn.execute(
        f"""
        SELECT DISTINCT macro_event_id, event_type
        FROM macro_events
        WHERE chunk_id IN ({placeholders})
        ORDER BY macro_event_id
        """,
        chunk_ids,
    ).fetchall()
    return [row["macro_event_id"] for row in rows], [row["event_type"] for row in rows if row["event_type"]]


def _score_set(expected: list[str], observed: list[str]) -> dict[str, Any]:
    expected_set = set(expected)
    observed_set = set(observed)
    hits = sorted(expected_set & observed_set)
    recall = (len(hits) / len(expected_set)) if expected_set else 1.0
    precision = (len(hits) / len(observed_set)) if observed_set else 1.0
    return {
        "expected": sorted(expected_set),
        "observed": sorted(observed_set),
        "hits": hits,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
    }


def _answer_grounding(answer: str, citation_map: dict[str, str]) -> dict[str, Any]:
    labels = set(citation_map.values())
    used = re.findall(r"\[(S\d+)\]", answer or "")
    used_unique = sorted({label for label in used if label in labels})
    normalized = (answer or "").lower()
    return {
        "citation_count": len(used_unique),
        "used_citations": used_unique,
        "has_inline_citations": bool(used_unique),
        "has_evidence_section": "evidence" in normalized,
        "has_theory_section": "theory" in normalized,
        "has_answer_section": "answer" in normalized,
    }


def evaluate_case(case: dict[str, Any], runtime: dict[str, Any], *, skip_generation: bool) -> dict[str, Any]:
    result = chatter.run_query_once(
        query=case["query"],
        embed_model=runtime["embed_model"],
        reranker=runtime["reranker"],
        gen_client=runtime["gen_client"],
        driver=runtime["driver"],
        sqlite_conn=runtime["sqlite_conn"],
        alias_to_ticker=runtime["alias_to_ticker"],
        ticker_to_canonical=runtime["ticker_to_canonical"],
        alias_to_fin_entity=runtime["alias_to_fin_entity"],
        base_system_prompt=runtime["base_system_prompt"],
        base_causal_system_prompt=runtime["base_causal_system_prompt"],
        base_daily_summary_prompt=runtime["base_daily_summary_prompt"],
        memory=ConversationMemory(),
        skip_generation=skip_generation,
    )
    chunk_ids = [chunk["chunk_uid"] for chunk in result["chunks"] if chunk.get("chunk_uid")]
    entities = _chunk_entities(runtime["sqlite_conn"], chunk_ids)
    macro_event_ids, macro_event_types = _chunk_macro_events(runtime["sqlite_conn"], chunk_ids)
    source_meta_by_chunk = _fetch_chunk_article_meta(runtime["sqlite_conn"], chunk_ids)
    grounding = _answer_grounding(result["answer"], result["citation_map"])
    required_grounding = case.get("expected_answer_grounding", {})

    chunk_score = _score_set(case.get("expected_chunks", []), chunk_ids)
    entity_score = _score_set(
        case.get("expected_entities", []),
        entities + (
            [result["target"].canonical_name]
            if result.get("target") and result["target"].canonical_name
            else []
        ),
    )
    macro_id_score = _score_set(case.get("expected_macro_event_ids", []), macro_event_ids)
    macro_type_score = _score_set(case.get("expected_macro_event_types", []), macro_event_types)

    # --- existing retrieval + grounding passes ---
    passes = [
        chunk_score["recall"] >= case.get("min_chunk_recall", 0.5),
        entity_score["recall"] >= case.get("min_entity_recall", 1.0),
        macro_type_score["recall"] >= case.get("min_macro_type_recall", 0.5),
        (
            grounding["has_inline_citations"]
            if (required_grounding.get("require_inline_citations", True) and not skip_generation)
            else True
        ),
        (
            grounding["has_evidence_section"]
            if (required_grounding.get("require_evidence_section", True) and not skip_generation)
            else True
        ),
        (
            grounding["has_theory_section"]
            if (required_grounding.get("require_theory_section", True) and not skip_generation)
            else True
        ),
        (
            grounding["citation_count"] >= int(required_grounding.get("min_cited_sources", 1))
            if not skip_generation
            else True
        ),
    ]

    # --- macro answer quality evaluation (new) ---
    macro_answer_eval: dict[str, Any] = {"passed": True, "skipped": True}
    macro_answer_passed = True
    parsed_answer_meta: dict[str, Any] | None = None

    gold_answer_quality = case.get("expected_answer_quality", {})
    if gold_answer_quality and not skip_generation:
        # Gather all mechanism strings the evaluator should look for
        all_mechs: list[str] = list(
            set(gold_answer_quality.get("required_mechanisms", []))
            | set(gold_answer_quality.get("optional_mechanisms", []))
        )
        parsed = parse_answer_meta(result["answer"], all_mechs)
        # Attach raw answer so evaluate_macro_answer can do phrase matching
        parsed["_raw_answer"] = result["answer"]
        parsed_answer_meta = parsed

        macro_answer_eval = evaluate_macro_answer(parsed, gold_answer_quality)
        macro_answer_passed = macro_answer_eval["passed"]

    passes.append(macro_answer_passed)

    route_type = _extract_route_type(case, result)
    resolved_target = _extract_resolved_target(result)
    answer_confidence = _extract_answer_confidence(result, parsed_answer_meta)
    decision = _extract_decision(result, result["answer"], answer_confidence)
    verifier_events = _extract_verifier_events(result)

    target_resolution_eval = _evaluate_target_resolution(case, resolved_target)
    abstention_eval = _evaluate_abstention(case, decision, result["answer"])
    source_trust_eval = _evaluate_source_trust(case, source_meta_by_chunk)
    verifier_precision_eval = _evaluate_verifier_precision(case, verifier_events)
    contradiction_eval = _evaluate_contradiction(result)
    unsupported_mechanism_eval = _evaluate_unsupported_mechanisms(
        case,
        result,
        parsed_answer_meta,
    )
    slices = {
        "query_type_route": route_type,
        "ambiguity": _ambiguity_slice(resolved_target),
        "recency_sensitivity": _recency_sensitivity_slice(case),
        "source_quality": _source_quality_slice(source_meta_by_chunk, case),
    }

    return {
        "id": case["id"],
        "query": case["query"],
        "chunk_score": chunk_score,
        "entity_score": entity_score,
        "macro_event_id_score": macro_id_score,
        "macro_event_type_score": macro_type_score,
        "grounding": grounding,
        "macro_answer_eval": macro_answer_eval,
        "macro_answer_passed": macro_answer_passed,
        "answer": result["answer"],
        "provenance": result["provenance"],
        "v2": {
            "route_type": route_type,
            "answer_confidence": answer_confidence,
            "decision": decision,
            "resolved_target": resolved_target,
            "target_resolution_eval": target_resolution_eval,
            "abstention_eval": abstention_eval,
            "source_trust_eval": source_trust_eval,
            "verifier_events": verifier_events,
            "verifier_precision_eval": verifier_precision_eval,
            "contradiction_eval": contradiction_eval,
            "unsupported_mechanism_eval": unsupported_mechanism_eval,
            "slices": slices,
        },
        "passed": all(passes),
    }


def run_suite(gold_path: Path, *, skip_generation: bool, disable_reranker: bool) -> dict[str, Any]:
    payload = load_gold_cases(gold_path)
    runtime = build_runtime(enable_reranker=not disable_reranker, need_generation=not skip_generation)
    try:
        results = [
            evaluate_case(case, runtime, skip_generation=skip_generation)
            for case in payload.get("cases", [])
        ]
    finally:
        close_runtime(runtime)
    passed = sum(1 for result in results if result["passed"])
    v2_metrics = _compute_v2_metrics(results)
    slice_report = _build_slice_report(results)
    release_gate_thresholds = {
        **DEFAULT_RELEASE_GATE_THRESHOLDS,
        **(payload.get("release_gate_thresholds") or {}),
    }
    release_gate = _evaluate_release_gate(
        metrics=v2_metrics,
        case_passed=passed,
        case_total=len(results),
        thresholds=release_gate_thresholds,
        require_all_metrics=bool(payload.get("release_gate_require_all_metrics", False)),
    )
    return {
        "suite": str(gold_path),
        "cases": results,
        "passed": passed,
        "total": len(results),
        "v2_metrics": v2_metrics,
        "slice_report": slice_report,
        "release_gate": release_gate,
    }


def compare_retrieval(gold_path: Path, *, skip_generation: bool) -> dict[str, Any]:
    payload = load_gold_cases(gold_path)
    baseline_runtime = build_runtime(enable_reranker=False, need_generation=not skip_generation)
    reranked_runtime = build_runtime(enable_reranker=True, need_generation=not skip_generation)
    try:
        comparisons = []
        for case in payload.get("cases", []):
            baseline = evaluate_case(case, baseline_runtime, skip_generation=skip_generation)
            reranked = evaluate_case(case, reranked_runtime, skip_generation=skip_generation)
            comparisons.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "baseline_chunk_recall": baseline["chunk_score"]["recall"],
                    "reranked_chunk_recall": reranked["chunk_score"]["recall"],
                    "baseline_macro_recall": baseline["macro_event_type_score"]["recall"],
                    "reranked_macro_recall": reranked["macro_event_type_score"]["recall"],
                }
            )
    finally:
        close_runtime(baseline_runtime)
        close_runtime(reranked_runtime)
    return {"suite": str(gold_path), "comparisons": comparisons}


def bootstrap_case(gold_path: Path, case_id: str, query: str, *, skip_generation: bool) -> None:
    payload = load_gold_cases(gold_path) if gold_path.exists() else {"schema_version": 1, "cases": []}
    runtime = build_runtime(enable_reranker=True, need_generation=not skip_generation)
    try:
        result = chatter.run_query_once(
            query=query,
            embed_model=runtime["embed_model"],
            reranker=runtime["reranker"],
            gen_client=runtime["gen_client"],
            driver=runtime["driver"],
            sqlite_conn=runtime["sqlite_conn"],
            alias_to_ticker=runtime["alias_to_ticker"],
            ticker_to_canonical=runtime["ticker_to_canonical"],
            alias_to_fin_entity=runtime["alias_to_fin_entity"],
            base_system_prompt=runtime["base_system_prompt"],
            base_causal_system_prompt=runtime["base_causal_system_prompt"],
            base_daily_summary_prompt=runtime["base_daily_summary_prompt"],
            memory=ConversationMemory(),
            skip_generation=skip_generation,
        )
        chunk_ids = [chunk["chunk_uid"] for chunk in result["chunks"] if chunk.get("chunk_uid")]
        entities = _chunk_entities(runtime["sqlite_conn"], chunk_ids)
        macro_event_ids, macro_event_types = _chunk_macro_events(runtime["sqlite_conn"], chunk_ids)
    finally:
        close_runtime(runtime)

    case = {
        "id": case_id,
        "query": query,
        "expected_chunks": chunk_ids[:4],
        "expected_entities": entities[:6],
        "expected_macro_event_ids": macro_event_ids[:6],
        "expected_macro_event_types": macro_event_types[:6],
        "expected_answer_grounding": {
            "require_inline_citations": True,
            "require_evidence_section": True,
            "require_theory_section": True,
            "min_cited_sources": 1,
        },
        # Stub for macro answer quality — edit after bootstrapping
        "expected_answer_quality": {
            "expected_direction": None,
            "allowed_directions": [],
            "required_mechanisms": [],
            "optional_mechanisms": [],
            "required_counterarguments": False,
            "forbidden_overclaims": [],
            "max_confidence_if_mixed": None,
        },
        # Optional v2 deterministic evaluation hooks.
        "expected_route_type": None,
        "expected_target": {},
        "expected_abstain": None,
        "source_trust_policy": {},
        "min_chunk_recall": 0.5,
        "min_entity_recall": 1.0,
        "min_macro_type_recall": 0.5,
    }
    payload["cases"] = [existing for existing in payload.get("cases", []) if existing.get("id") != case_id] + [case]
    save_gold_cases(gold_path, payload)
    print(f"[eval] bootstrapped case {case_id} into {gold_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end evaluation suite for the RAG pipeline")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD_PATH), help="Path to gold evaluation JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the gold evaluation suite")
    run_parser.add_argument("--skip-generation", action="store_true", help="Evaluate retrieval/grounding without calling the generation model")
    run_parser.add_argument("--disable-reranker", action="store_true", help="Run without the local cross-encoder reranker")

    compare_parser = subparsers.add_parser("compare", help="Compare baseline retrieval versus reranked retrieval")
    compare_parser.add_argument("--skip-generation", action="store_true", help="Skip answer generation during comparison")

    bootstrap_parser = subparsers.add_parser("bootstrap", help="Bootstrap one gold case from current retrieval outputs")
    bootstrap_parser.add_argument("--id", required=True, help="Case id")
    bootstrap_parser.add_argument("--query", required=True, help="Gold question to snapshot")
    bootstrap_parser.add_argument("--skip-generation", action="store_true", help="Skip generation while bootstrapping")

    args = parser.parse_args()
    gold_path = Path(args.gold)
    if args.command == "run":
        print(json.dumps(run_suite(gold_path, skip_generation=args.skip_generation, disable_reranker=args.disable_reranker), indent=2))
    elif args.command == "compare":
        print(json.dumps(compare_retrieval(gold_path, skip_generation=args.skip_generation), indent=2))
    elif args.command == "bootstrap":
        bootstrap_case(gold_path, args.id, args.query, skip_generation=args.skip_generation)


if __name__ == "__main__":
    main()
