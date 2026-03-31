"""
Temporal-Graph RAG Chatbot

Connects to an already-populated Neo4j graph (built by tgrag_text.ipynb)
and runs interactive retrieval + generation in a terminal loop.

Usage:
    python chatter.py
"""

import os
import re
import warnings
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=FutureWarning)

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

# GRAPH RETRIEVAL

def temporal_graph_retrieve(
    query: str,
    embed_model: SentenceTransformer,
    driver,
    top_k: int = 3,
    expanded_k: int = 6,
    recency_half_life_days: float = 7.0,
    source_filter: str | None = None,
):
    qvec = embed_model.encode([query], normalize_embeddings=True)[0]
    now_ts = int(datetime.now(timezone.utc).timestamp())
    half_life_seconds = recency_half_life_days * 86400.0

    with driver.session() as session:
        rows = session.run(
            """
            MATCH (c:Chunk)<-[:HAS_CHUNK]-(a:Article)
            WHERE ($source_filter IS NULL OR c.source = $source_filter)
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
            {"source_filter": source_filter},
        ).data()

    scored = []
    for r in rows:
        emb = np.array(r["embedding"], dtype=np.float32)
        sim = cosine_sim(qvec, emb)

        ts = r.get("published_ts")
        if ts is None:
            recency_weight = 0.85
        else:
            age = max(0, now_ts - int(ts))
            recency_weight = float(np.exp(-np.log(2) * age / half_life_seconds))

        score = (0.8 * sim) + (0.2 * recency_weight)
        r["semantic_sim"] = sim
        r["recency_weight"] = recency_weight
        r["score"] = score
        scored.append(r)

    scored.sort(key=lambda x: x["score"], reverse=True)
    seeds = scored[:top_k]

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


# GENERATION

SYSTEM_PROMPT = (
    "Answer the user's question using only the provided context. "
    "If there is not enough evidence, say so clearly."
)


def strip_think_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def generate_answer(query: str, context: str, pipe) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    out = pipe(messages, max_new_tokens=1024, do_sample=False)
    raw = out[0]["generated_text"][-1]["content"]
    return strip_think_tags(raw)


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

        retrieved = temporal_graph_retrieve(
            query=query,
            embed_model=embed_model,
            driver=driver,
            top_k=3,
            expanded_k=6,
            recency_half_life_days=5,
            source_filter=source_filter,
        )

        if not retrieved:
            print("Assistant: No relevant chunks found.\n")
            continue

        context = "\n\n".join(x["text"] for x in retrieved if x.get("text"))
        answer = generate_answer(query, context, pipe)

        print(f"\nAssistant: {answer}\n")

    driver.close()


if __name__ == "__main__":
    main()
