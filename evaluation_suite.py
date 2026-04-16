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
        memory=ConversationMemory(),
        skip_generation=skip_generation,
    )
    chunk_ids = [chunk["chunk_uid"] for chunk in result["chunks"] if chunk.get("chunk_uid")]
    entities = _chunk_entities(runtime["sqlite_conn"], chunk_ids)
    macro_event_ids, macro_event_types = _chunk_macro_events(runtime["sqlite_conn"], chunk_ids)
    grounding = _answer_grounding(result["answer"], result["citation_map"])
    required_grounding = case.get("expected_answer_grounding", {})

    chunk_score = _score_set(case.get("expected_chunks", []), chunk_ids)
    entity_score = _score_set(case.get("expected_entities", []), entities + ([result["target"].canonical_name] if result.get("target") and result["target"].canonical_name else []))
    macro_id_score = _score_set(case.get("expected_macro_event_ids", []), macro_event_ids)
    macro_type_score = _score_set(case.get("expected_macro_event_types", []), macro_event_types)

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

    return {
        "id": case["id"],
        "query": case["query"],
        "chunk_score": chunk_score,
        "entity_score": entity_score,
        "macro_event_id_score": macro_id_score,
        "macro_event_type_score": macro_type_score,
        "grounding": grounding,
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
