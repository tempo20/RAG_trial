from __future__ import annotations

import argparse
import importlib
import json
import re
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

        macro_answer_eval = evaluate_macro_answer(parsed, gold_answer_quality)
        macro_answer_passed = macro_answer_eval["passed"]

    passes.append(macro_answer_passed)

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
    return {
        "suite": str(gold_path),
        "cases": results,
        "passed": passed,
        "total": len(results),
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
