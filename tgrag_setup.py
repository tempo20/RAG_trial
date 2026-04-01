"""
Temporal-Graph RAG Setup

Reads cnbc_articles.json, chunks the articles, embeds them,
extracts entities (via Qwen2.5-7B-Instruct), and populates
the Neo4j temporal graph.

Usage:
    python tgrag_setup.py                    # incremental update
    python tgrag_setup.py --reset            # wipe graph and rebuild
    python tgrag_setup.py --skip-entities    # skip entity extraction
"""

import argparse
import gc
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse
from datetime import timezone

from dotenv import load_dotenv
load_dotenv()

import tiktoken
from dateutil import parser as dtparser
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase

EMBED_MODEL_NAME = "BAAI/bge-m3"
EXTRACTION_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = Path("cnbc_articles.json")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

enc = tiktoken.get_encoding("cl100k_base")


def token_len(text: str) -> int:
    return len(enc.encode(text))


def safe_parse_datetime(value: str):
    if not value:
        return None
    try:
        dt = dtparser.parse(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def canonicalize(name: str) -> str:
    """Normalize an entity name for deduplication."""
    return re.sub(r"\s+", " ", name.strip()).lower()


# Entity extraction 

EXTRACTION_PROMPT = """\
You are an entity and relationship extractor for news articles.

Extract all named entities and relationships from the article below.

Entity types: PERSON, ORG, STOCK, EVENT, LOCATION, CONCEPT

Output ONLY valid JSON (no markdown, no explanation):
{{"entities": [{{"name": "...", "type": "..."}}], "relationships": [{{"source": "...", "target": "...", "type": "..."}}]}}

Article:
{text}"""


def load_extraction_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as _pipeline
    print(f"Loading extraction model ({EXTRACTION_MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(
        EXTRACTION_MODEL_NAME, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        EXTRACTION_MODEL_NAME, local_files_only=True
    )
    return _pipeline("text-generation", model=model, tokenizer=tokenizer)


def _parse_extraction_json(raw: str) -> dict:
    """Best-effort parse of model output into {entities, relationships}."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            entities = [
                e for e in data.get("entities", [])
                if isinstance(e, dict) and "name" in e and "type" in e
            ]
            relationships = [
                r for r in data.get("relationships", [])
                if isinstance(r, dict)
                and "source" in r and "target" in r and "type" in r
            ]
            return {"entities": entities, "relationships": relationships}
        except (json.JSONDecodeError, TypeError):
            pass
    return {"entities": [], "relationships": []}


def extract_entities(article_text: str, pipe) -> dict:
    prompt = EXTRACTION_PROMPT.format(text=article_text[:4000])
    messages = [{"role": "user", "content": prompt}]
    out = pipe(messages, max_new_tokens=1024, do_sample=False)
    raw = out[0]["generated_text"][-1]["content"]
    return _parse_extraction_json(raw)


def extract_all_entities(
    chunks: list[dict],
    pipe,
) -> dict[str, dict]:
    """Group chunks by article, extract entities per article."""
    articles: dict[str, str] = {}
    for c in chunks:
        aid = c["article_id"]
        if aid not in articles:
            articles[aid] = ""
        articles[aid] += c["text"] + "\n"

    entity_data: dict[str, dict] = {}
    total = len(articles)
    for idx, (aid, text) in enumerate(articles.items(), 1):
        print(f"  Extracting entities [{idx}/{total}] {aid[:60]}...")
        result = extract_entities(text, pipe)
        for e in result["entities"]:
            e["canonical_name"] = canonicalize(e["name"])
        for r in result["relationships"]:
            r["source_canonical"] = canonicalize(r["source"])
            r["target_canonical"] = canonicalize(r["target"])
        entity_data[aid] = result

    ent_count = sum(len(d["entities"]) for d in entity_data.values())
    rel_count = sum(len(d["relationships"]) for d in entity_data.values())
    print(
        f"Extracted {ent_count} entities and {rel_count} relationships "
        f"from {total} articles"
    )
    return entity_data


# Existing-article check 

def get_existing_article_ids(driver) -> set[str]:
    with driver.session() as session:
        rows = session.run(
            "MATCH (a:Article) RETURN a.article_id AS aid"
        ).data()
    return {r["aid"] for r in rows}


# Load and chunk 

def load_and_chunk(skip_ids: set[str] | None = None) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    articles = data["articles"]

    skip_ids = skip_ids or set()
    chunks: list[dict] = []
    skipped = 0

    for i, a in enumerate(articles):
        text = (a.get("text") or "").strip()
        if a.get("status") != "ok" or not text:
            continue

        url_path = urlparse(a.get("url", "")).path.strip("/")
        article_id = url_path.replace("/", "_") if url_path else f"article_{i}"

        if article_id in skip_ids:
            skipped += 1
            continue

        published_dt = safe_parse_datetime(a.get("published"))

        for chunk_id, chunk in enumerate(splitter.split_text(text)):
            chunks.append(
                {
                    "article_id": article_id,
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "tokens": token_len(chunk),
                    "url": a.get("url"),
                    "title": a.get("title"),
                    "published": a.get("published"),
                    "source": a.get("source"),
                    "published_ts": (
                        int(published_dt.timestamp()) if published_dt else None
                    ),
                }
            )

    print(
        f"Prepared {len(chunks)} new chunks "
        f"({skipped} existing articles skipped)"
    )
    return chunks


# Embedding 

def embed_chunks(chunks: list[dict]) -> list[dict]:
    if not chunks:
        print("No new chunks to embed")
        return chunks

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True)

    for i, vec in enumerate(vectors):
        chunks[i]["embedding"] = vec.tolist()

    print("Embeddings ready")
    return chunks


# Populate Neo4j 

def populate_neo4j(
    chunks: list[dict],
    reset: bool = False,
    entity_data: dict[str, dict] | None = None,
) -> None:
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    with driver.session() as session:
        if reset:
            print("Resetting graph...")
            session.run("MATCH (n) DETACH DELETE n")

        session.run(
            """
            CREATE CONSTRAINT article_id_unique IF NOT EXISTS
            FOR (a:Article) REQUIRE a.article_id IS UNIQUE
            """
        )
        session.run(
            """
            CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
            FOR (c:Chunk) REQUIRE c.chunk_uid IS UNIQUE
            """
        )
        session.run(
            """
            CREATE CONSTRAINT entity_canonical_unique IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.canonical_name IS UNIQUE
            """
        )

        if not chunks:
            print("No new chunks to load")
            driver.close()
            return

        new_article_ids = list({c["article_id"] for c in chunks})

        # Upsert chunks 
        print(f"Upserting {len(chunks)} chunks from {len(new_article_ids)} new articles...")
        for c in chunks:
            chunk_uid = f"{c['article_id']}_chunk_{c['chunk_id']}"
            session.run(
                """
                MERGE (a:Article {article_id: $article_id})
                SET a.title     = $title,
                    a.url       = $url,
                    a.published = $published,
                    a.source    = $source,
                    a.published_ts = $published_ts

                MERGE (ch:Chunk {chunk_uid: $chunk_uid})
                SET ch.article_id   = $article_id,
                    ch.chunk_id     = $chunk_id,
                    ch.text         = $text,
                    ch.tokens       = $tokens,
                    ch.embedding    = $embedding,
                    ch.source       = $source,
                    ch.published_ts = $published_ts

                MERGE (a)-[:HAS_CHUNK]->(ch)
                """,
                {
                    "article_id": c["article_id"],
                    "title": c["title"],
                    "url": c["url"],
                    "published": c["published"],
                    "source": c["source"],
                    "published_ts": c["published_ts"],
                    "chunk_uid": chunk_uid,
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "tokens": c["tokens"],
                    "embedding": c["embedding"],
                },
            )

        # NEXT_CHUNK edges
        print("Adding NEXT_CHUNK edges for new articles...")
        session.run(
            """
            MATCH (c:Chunk)
            WHERE c.article_id IN $new_ids
            WITH c ORDER BY c.article_id, c.chunk_id
            WITH c.article_id AS aid, collect(c) AS cs
            UNWIND range(0, size(cs) - 2) AS i
            WITH cs[i] AS c1, cs[i + 1] AS c2
            MERGE (c1)-[:NEXT_CHUNK]->(c2)
            """,
            {"new_ids": new_article_ids},
        )

        # NEAR_IN_TIME edges
        print("Adding NEAR_IN_TIME edges for new articles...")
        session.run(
            """
            MATCH (a1:Article), (a2:Article)
            WHERE a1.article_id IN $new_ids
              AND a1.article_id <> a2.article_id
              AND a1.published_ts IS NOT NULL
              AND a2.published_ts IS NOT NULL
              AND abs(a1.published_ts - a2.published_ts) <= 86400
            MERGE (a1)-[:NEAR_IN_TIME]->(a2)
            """,
            {"new_ids": new_article_ids},
        )

        # Entity nodes + MENTIONS + RELATED_TO
        if entity_data:
            print("Creating Entity nodes and MENTIONS edges...")
            for aid in new_article_ids:
                art_ents = entity_data.get(aid)
                if not art_ents:
                    continue

                for ent in art_ents.get("entities", []):
                    session.run(
                        """
                        MERGE (e:Entity {canonical_name: $canonical_name})
                        SET e.name = $name,
                            e.type = $type
                        """,
                        {
                            "canonical_name": ent["canonical_name"],
                            "name": ent["name"],
                            "type": ent.get("type", "CONCEPT"),
                        },
                    )

                article_chunks = [c for c in chunks if c["article_id"] == aid]
                for c in article_chunks:
                    chunk_uid = f"{c['article_id']}_chunk_{c['chunk_id']}"
                    chunk_lower = c["text"].lower()
                    for ent in art_ents.get("entities", []):
                        if ent["canonical_name"] in chunk_lower:
                            session.run(
                                """
                                MATCH (ch:Chunk {chunk_uid: $chunk_uid})
                                MATCH (e:Entity {canonical_name: $cname})
                                MERGE (ch)-[:MENTIONS]->(e)
                                """,
                                {
                                    "chunk_uid": chunk_uid,
                                    "cname": ent["canonical_name"],
                                },
                            )

                for rel in art_ents.get("relationships", []):
                    session.run(
                        """
                        MATCH (e1:Entity {canonical_name: $src})
                        MATCH (e2:Entity {canonical_name: $tgt})
                        MERGE (e1)-[r:RELATED_TO]->(e2)
                        SET r.relation_type = $rel_type
                        """,
                        {
                            "src": rel["source_canonical"],
                            "tgt": rel["target_canonical"],
                            "rel_type": rel["type"],
                        },
                    )

            ent_nodes = session.run(
                "MATCH (e:Entity) RETURN count(e) AS cnt"
            ).single()["cnt"]
            print(f"Graph now has {ent_nodes} Entity nodes")

    driver.close()
    print("Neo4j temporal graph updated successfully")


def main():
    parser = argparse.ArgumentParser(description="TG-RAG setup")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the entire graph and rebuild from scratch",
    )
    parser.add_argument(
        "--skip-entities",
        action="store_true",
        help="Skip LLM-based entity extraction (faster setup)",
    )
    args = parser.parse_args()

    skip_ids: set[str] = set()
    if not args.reset:
        print("Checking existing articles in Neo4j...")
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        skip_ids = get_existing_article_ids(driver)
        driver.close()
        print(f"Found {len(skip_ids)} articles already in graph")

    chunks = load_and_chunk(skip_ids=skip_ids)

    # Entity extraction with larger model (freed before embedding)
    entity_data = None
    if not args.skip_entities and chunks:
        pipe = load_extraction_model()
        entity_data = extract_all_entities(chunks, pipe)
        del pipe
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    chunks = embed_chunks(chunks)
    populate_neo4j(chunks, reset=args.reset, entity_data=entity_data)
    print("\nSetup complete. You can now run:  python chatter.py")


if __name__ == "__main__":
    main()
