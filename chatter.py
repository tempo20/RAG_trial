"""
Temporal-Graph RAG Chatbot

Connects to an already-populated Neo4j graph (built by tgrag_setup.py)
and runs interactive retrieval + generation in a terminal loop.

Features:
  - Temporal Query Decomposition (TQD): splits multi-temporal queries
  - Three-layer retrieval: temporal filter -> entity match -> semantic ranking
  - Sub-answer aggregation for comparative / multi-period questions

Usage:
    python chatter.py
"""

import json
import os
import re
import warnings
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=FutureWarning)

import dateparser
import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from transformers import pipeline

from tgrag_setup import EMBED_MODEL_NAME, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

GEN_MODEL_NAME = "Qwen/Qwen3-0.6B"

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


# ── Temporal Query Decomposition ────────────────────────────────────

DECOMPOSE_PROMPT = """\
Analyze this question and identify any time references. \
If the question compares multiple time periods, decompose it into separate sub-queries.

Rules:
- Today is {today}.
- If no time reference exists, return the original query with null dates.
- Use ISO format YYYY-MM-DD for dates.
- "last week" = the 7 days before today.
- "yesterday" = one day before today.
- "this week" = from the most recent Monday to today.
- "this month" = from the 1st of the current month to today.

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


def decompose_query(query: str, pipe) -> list[dict]:
    """Split a query into temporal sub-queries."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = DECOMPOSE_PROMPT.format(today=today, query=query)
    messages = [{"role": "user", "content": prompt}]
    out = pipe(messages, max_new_tokens=512, do_sample=False)
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

    return [{"query": query, "time_start": None, "time_end": None}]


# ── Three-Layer Retrieval ───────────────────────────────────────────

def three_layer_retrieve(
    query: str,
    embed_model: SentenceTransformer,
    driver,
    top_k: int = 3,
    expanded_k: int = 6,
    recency_half_life_days: float = 7.0,
    source_filter: str | None = None,
    time_start: int | None = None,
    time_end: int | None = None,
):
    qvec = embed_model.encode([query], normalize_embeddings=True)[0]
    now_ts = int(datetime.now(timezone.utc).timestamp())
    half_life_seconds = recency_half_life_days * 86400.0

    with driver.session() as session:
        # ── Layer 1: Temporal subgraph filter ──
        time_clauses = ""
        params: dict = {"source_filter": source_filter}
        if time_start is not None:
            time_clauses += " AND c.published_ts >= $ts_start"
            params["ts_start"] = time_start
        if time_end is not None:
            time_clauses += " AND c.published_ts <= $ts_end"
            params["ts_end"] = time_end

        rows = session.run(
            f"""
            MATCH (c:Chunk)<-[:HAS_CHUNK]-(a:Article)
            WHERE ($source_filter IS NULL OR c.source = $source_filter)
            {time_clauses}
            RETURN c.chunk_uid AS chunk_uid,
                   c.text AS text,
                   c.embedding AS embedding,
                   c.published_ts AS published_ts,
                   c.source AS source,
                   a.title AS title,
                   a.url AS url,
                   c.article_id AS article_id,
                   c.chunk_id AS chunk_id
            """,
            params,
        ).data()

        # ── Layer 2: Entity-based coarse retrieval ──
        query_lower = query.lower()
        entity_rows = session.run(
            "MATCH (e:Entity) RETURN e.canonical_name AS cname, e.name AS name"
        ).data()

        matched_entities: list[str] = []
        for e in entity_rows:
            cname = e["cname"]
            name_lower = e["name"].lower()
            if cname in query_lower or name_lower in query_lower:
                matched_entities.append(cname)
                continue
            for word in cname.split():
                if len(word) > 3 and re.search(
                    r"\b" + re.escape(word) + r"\b", query_lower
                ):
                    matched_entities.append(cname)
                    break

        if matched_entities:
            related = session.run(
                """
                MATCH (e:Entity)-[:RELATED_TO]-(e2:Entity)
                WHERE e.canonical_name IN $enames
                RETURN DISTINCT e2.canonical_name AS cname
                """,
                {"enames": matched_entities},
            ).data()
            for r in related:
                if r["cname"] not in matched_entities:
                    matched_entities.append(r["cname"])

        entity_chunk_uids: set[str] = set()
        if matched_entities:
            ent_params: dict = {
                "enames": matched_entities,
                "source_filter": source_filter,
            }
            ent_time = ""
            if time_start is not None:
                ent_time += " AND c.published_ts >= $ts_start"
                ent_params["ts_start"] = time_start
            if time_end is not None:
                ent_time += " AND c.published_ts <= $ts_end"
                ent_params["ts_end"] = time_end

            ent_chunks = session.run(
                f"""
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE e.canonical_name IN $enames
                  AND ($source_filter IS NULL OR c.source = $source_filter)
                  {ent_time}
                RETURN DISTINCT c.chunk_uid AS chunk_uid
                """,
                ent_params,
            ).data()
            entity_chunk_uids = {r["chunk_uid"] for r in ent_chunks}

    # ── Layer 3: Semantic fine-grained retrieval ──
    scored = []
    for r in rows:
        emb = np.array(r["embedding"], dtype=np.float32)
        sim = cosine_sim(qvec, emb)

        ts = r.get("published_ts")
        if ts is None:
            recency_weight = 0.85
        else:
            age = max(0, now_ts - int(ts))
            recency_weight = float(
                np.exp(-np.log(2) * age / half_life_seconds)
            )

        entity_boost = 1.0 if r["chunk_uid"] in entity_chunk_uids else 0.0
        score = (0.70 * sim) + (0.20 * recency_weight) + (0.10 * entity_boost)

        r["semantic_sim"] = sim
        r["recency_weight"] = recency_weight
        r["entity_match"] = bool(entity_boost)
        r["score"] = score
        scored.append(r)

    scored.sort(key=lambda x: x["score"], reverse=True)
    seeds = scored[:top_k]

    # Graph expansion via NEXT_CHUNK neighbours
    expanded = {s["chunk_uid"]: s for s in seeds}
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
                {"chunk_uid": s["chunk_uid"], "source_filter": source_filter},
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
                        }

    expanded_list = sorted(
        expanded.values(), key=lambda x: x.get("score", 0), reverse=True
    )
    return expanded_list[:expanded_k]


# ── Generation ──────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Answer the user's question using only the provided context. "
    "If there is not enough evidence, say so clearly."
)

AGGREGATE_PROMPT = """\
You were asked: {query}

Here are answers from different time periods:

{sub_answers_text}

Synthesize these into a single coherent answer. \
Highlight any temporal differences or changes."""


def generate_answer(query: str, context: str, pipe) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    out = pipe(messages, max_new_tokens=1024, do_sample=False)
    raw = out[0]["generated_text"][-1]["content"]
    return strip_think_tags(raw)


def aggregate_answers(
    query: str, sub_answers: list[dict], pipe
) -> str:
    """Merge multiple sub-answers into one response."""
    if len(sub_answers) == 1:
        return sub_answers[0]["answer"]

    parts = []
    for i, sa in enumerate(sub_answers, 1):
        label = ""
        if sa.get("time_start") or sa.get("time_end"):
            label = f" ({sa.get('time_start', '?')} to {sa.get('time_end', '?')})"
        parts.append(f"{i}.{label}\n{sa['answer']}")

    sub_text = "\n\n".join(parts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": AGGREGATE_PROMPT.format(
                query=query, sub_answers_text=sub_text
            ),
        },
    ]
    out = pipe(messages, max_new_tokens=1024, do_sample=False)
    raw = out[0]["generated_text"][-1]["content"]
    return strip_think_tags(raw)


# ── Main loop ───────────────────────────────────────────────────────

def main():
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Loading generation model...")
    pipe = pipeline("text-generation", model=GEN_MODEL_NAME)

    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    print("\n--- TG-RAG Chatbot ready ---")
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

        source_filter = extract_source_filter(query)
        if source_filter:
            print(f"  [source filter: {source_filter}]")

        # ── Step 1: Temporal Query Decomposition ──
        sub_queries = decompose_query(query, pipe)
        if len(sub_queries) > 1:
            print(f"  [decomposed into {len(sub_queries)} sub-queries]")

        # ── Step 2: Three-layer retrieval + generation per sub-query ──
        sub_answers: list[dict] = []
        for sq in sub_queries:
            ts_start = resolve_date(sq["time_start"])
            ts_end = resolve_date(sq["time_end"])

            if sq["time_start"] or sq["time_end"]:
                print(
                    f"  [time filter: "
                    f"{sq.get('time_start', '?')} -> "
                    f"{sq.get('time_end', '?')}]"
                )

            retrieved = three_layer_retrieve(
                query=sq["query"],
                embed_model=embed_model,
                driver=driver,
                top_k=3,
                expanded_k=6,
                recency_half_life_days=5,
                source_filter=source_filter,
                time_start=ts_start,
                time_end=ts_end,
            )

            if not retrieved:
                sub_answers.append({
                    "query": sq["query"],
                    "answer": "No relevant information found for this time period.",
                    "time_start": sq.get("time_start"),
                    "time_end": sq.get("time_end"),
                })
                continue

            context = "\n\n".join(
                x["text"] for x in retrieved if x.get("text")
            )
            answer = generate_answer(sq["query"], context, pipe)
            sub_answers.append({
                "query": sq["query"],
                "answer": answer,
                "time_start": sq.get("time_start"),
                "time_end": sq.get("time_end"),
            })

        # ── Step 3: Aggregate sub-answers ──
        if not sub_answers:
            print("Assistant: No relevant chunks found.\n")
            continue

        final = aggregate_answers(query, sub_answers, pipe)
        print(f"\nAssistant: {final}\n")

    driver.close()


if __name__ == "__main__":
    main()
