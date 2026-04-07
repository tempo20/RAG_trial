"""
Temporal-Graph RAG Setup

Reads cnbc_articles.json, chunks the articles, embeds them,
extracts entities (via GLiNER), and populates
the Neo4j temporal graph.

Usage:
    python tgrag_setup.py                    # incremental update
    python tgrag_setup.py --reset            # wipe graph and rebuild
    python tgrag_setup.py --skip-entities    # skip NER extraction
    python tgrag_setup.py --cooccur-mode article
"""

import argparse
import gc
import json
import os
import csv
import re
from collections import defaultdict
from itertools import combinations
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
EXTRACTION_MODEL_NAME = "urchade/gliner_medium-v2.1"
COOCCUR_DEFAULT_MODE = "chunk"
NER_THRESHOLD = 0.35
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
    """Normalize an entity name (for dedup + exact mapping).

    Rule: lowercase, trim, strip punctuation, collapse repeated spaces.
    """
    if name is None:
        return ""
    s = str(name).strip().lower()
    # Strip punctuation but keep alphanumerics/underscore and spaces.
    s = re.sub(r"[^\w\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_ticker_company_map(csv_path: Path) -> dict[str, str]:
    """Load ticker mapping for exact normalized company/alias -> ticker.

    Expected CSV columns:
      - ticker (e.g. NVDA)
      - company_name (e.g. NVIDIA)
      - aliases (optional; separated by ';' or ','; can be empty)
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Ticker map CSV not found: {csv_path.resolve()}")

    lookup: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"ticker", "company_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Ticker map CSV missing required columns: {sorted(missing)}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            company_name = (row.get("company_name") or "").strip()
            aliases_raw = row.get("aliases") or ""

            if not ticker or not company_name:
                continue

            keys: list[str] = [company_name]
            # aliases can be separated by ';' or ','; normalize each token.
            aliases_parts = re.split(r"\s*[;,]\s*", aliases_raw.strip())
            keys.extend([p for p in aliases_parts if p])

            for k in keys:
                nk = canonicalize(k)
                if not nk:
                    continue
                if nk in lookup and lookup[nk] != ticker:
                    # Deterministic behavior: keep first, warn.
                    # (Ambiguous mapping is worse than missing it for your exact-match approach.)
                    print(f"[ticker-map] WARNING: {nk!r} maps to multiple tickers ({lookup[nk]} vs {ticker}); keeping {lookup[nk]}")
                    continue
                lookup[nk] = ticker

    return lookup


# NER extraction
GLINER_LABELS = [
    "person",
    "organization",
    "company",
    "stock ticker",
    "stock index",
    "etf",
    "commodity",
    "currency",
    "cryptocurrency",
    "location",
    "country",
    "city",
    "event",
    "product",
    "sector",
    "industry",
    "law",
    "government agency",
    "economic indicator",
    "technology",
    "concept",
]

def in_text(canonical: str, text: str) -> bool:
    pattern = r'\b' + re.escape(canonical) + r'\b'
    return bool(re.search(pattern, text))

def _normalize_label(label: str) -> str:
    label = (label or "").strip().lower()
    return re.sub(r"[\s\-]+", "_", label)


def _map_raw_label_to_type(raw_label: str) -> str:
    lbl = _normalize_label(raw_label)

    person_tokens = {
        "person", "individual", "executive", "analyst", "investor",
        "founder", "politician", "official",
    }
    org_tokens = {
        "org", "organization", "company", "corporation", "bank",
        "institution", "agency", "government_agency", "brand",
    }
    stock_tokens = {
        "stock", "stock_ticker", "ticker", "symbol", "equity",
        "index", "stock_index", "etf", "fund", "bond", "commodity",
        "currency", "cryptocurrency", "crypto", "token",
    }
    event_tokens = {
        "event", "earnings", "ipo", "merger", "acquisition",
        "lawsuit", "conference", "meeting", "election", "launch",
    }
    location_tokens = {
        "location", "loc", "city", "country", "region",
        "state", "continent", "address",
    }

    if any(tok in lbl for tok in person_tokens):
        return "PERSON"
    if any(tok in lbl for tok in org_tokens):
        return "ORG"
    if any(tok in lbl for tok in stock_tokens):
        return "STOCK"
    if any(tok in lbl for tok in event_tokens):
        return "EVENT"
    if any(tok in lbl for tok in location_tokens):
        return "LOCATION"
    return "CONCEPT"


def load_extraction_model():
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeError(
            "GLiNER is not installed. Install it with: pip install gliner"
        ) from exc

    print(f"Loading extraction model ({EXTRACTION_MODEL_NAME})...")
    try:
        return GLiNER.from_pretrained(EXTRACTION_MODEL_NAME, local_files_only=True)
    except TypeError:
        return GLiNER.from_pretrained(EXTRACTION_MODEL_NAME)
    except Exception:
        return GLiNER.from_pretrained(EXTRACTION_MODEL_NAME)


def extract_entities(article_text: str, pipe) -> dict:
    text = article_text[:8000]
    try:
        preds = pipe.predict_entities(
            text, labels=GLINER_LABELS, threshold=NER_THRESHOLD
        )
    except TypeError:
        try:
            preds = pipe.predict_entities(text, GLINER_LABELS, NER_THRESHOLD)
        except TypeError:
            preds = pipe.predict_entities(text, GLINER_LABELS)

    entities = []
    for pred in preds:
        if not isinstance(pred, dict):
            continue
        name = (pred.get("text") or "").strip()
        if not name:
            continue
        raw_label = str(pred.get("label") or "concept").strip()
        ent_type = _map_raw_label_to_type(raw_label)
        score = float(pred.get("score", 0.0) or 0.0)
        entities.append(
            {
                "name": name,
                "type": ent_type,
                "raw_label": raw_label,
                "score": score,
            }
        )

    return {"entities": entities, "relationships": []}


MIN_CANONICAL_LEN = 4
ALLOWED_REL_TYPES = {"ORG", "PERSON", "STOCK", "EVENT", "PRODUCT"}
NOISE_PATTERN = re.compile(r'^\$?[\d,\.]+[bm]?$', re.IGNORECASE)


def _in_text(canonical: str, text: str) -> bool:
    return bool(re.search(r'\b' + re.escape(canonical) + r'\b', text))


def _build_cooccurrence_relationships(
    entities: list[dict],
    chunk_texts: list[str],
    mode: str,
) -> list[dict]:
    rel_type = "CO_OCCURS_ARTICLE" if mode == "article" else "CO_OCCURS_CHUNK"

    canonicals = sorted({
        e["canonical_name"]
        for e in entities
        if e.get("canonical_name")
        and len(e["canonical_name"]) >= MIN_CANONICAL_LEN
        and e.get("type") in ALLOWED_REL_TYPES
        and not NOISE_PATTERN.match(e["canonical_name"])
    })

    if len(canonicals) < 2:
        return []

    pairs: set[tuple[str, str]] = set()
    if mode == "article":
        for src, tgt in combinations(canonicals, 2):
            pairs.add((src, tgt))
    else:
        for text in chunk_texts:
            chunk_norm = canonicalize(text)
            mentioned = sorted([c for c in canonicals if _in_text(c, chunk_norm)])
            if len(mentioned) < 2:
                continue
            for src, tgt in combinations(mentioned, 2):
                pairs.add((src, tgt))

    relationships = []
    for src, tgt in sorted(pairs):
        relationships.append(
            {
                "source": src,
                "target": tgt,
                "source_canonical": src,
                "target_canonical": tgt,
                "type": rel_type,
            }
        )
    return relationships


def extract_all_entities(
    chunks: list[dict],
    pipe,
    cooccur_mode: str = COOCCUR_DEFAULT_MODE,
    ticker_lookup: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Group chunks by article, extract entities per article."""
    cooccur_mode = (cooccur_mode or COOCCUR_DEFAULT_MODE).lower()
    if cooccur_mode not in {"chunk", "article"}:
        raise ValueError("cooccur_mode must be 'chunk' or 'article'")

    article_texts: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        aid = c["article_id"]
        article_texts[aid].append(c["text"])

    entity_data: dict[str, dict] = {}
    total = len(article_texts)
    print(f"Using co-occurrence mode: {cooccur_mode}")

    for idx, (aid, texts) in enumerate(article_texts.items(), 1):
        print(f"  Extracting entities [{idx}/{total}] {aid[:60]}...")
        text = "\n".join(texts)
        result = extract_entities(text, pipe)

        deduped: dict[str, dict] = {}
        for ent in result["entities"]:
            cname = canonicalize(ent["name"])
            if not cname:
                continue
            candidate = {
                "name": ent["name"],
                "type": ent.get("type", "CONCEPT"),
                "raw_label": ent.get("raw_label", "concept"),
                "canonical_name": cname,
                "_score": float(ent.get("score", 0.0) or 0.0),
            }
            if ticker_lookup and cname in ticker_lookup:
                candidate["mapped_ticker"] = ticker_lookup[cname]
            prev = deduped.get(cname)
            if prev is None or candidate["_score"] > prev["_score"]:
                deduped[cname] = candidate

        entities = []
        for e in deduped.values():
            e.pop("_score", None)
            entities.append(e)

        relationships = _build_cooccurrence_relationships(
            entities,
            texts,
            cooccur_mode,
        )

        entity_data[aid] = {
            "entities": entities,
            "relationships": relationships,
        }

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
                        ON CREATE SET e.name = $name,
                                      e.type = $type,
                                      e.raw_label = $raw_label,
                                      e.mapped_ticker = $mapped_ticker
                        ON MATCH SET e.name = $name,
                                     e.type = $type,
                                     e.mapped_ticker = CASE WHEN $mapped_ticker IS NULL THEN e.mapped_ticker ELSE $mapped_ticker END
                        """,
                        {
                            "canonical_name": ent["canonical_name"],
                            "name": ent["name"],
                            "type": ent.get("type", "CONCEPT"),
                            "raw_label": ent.get("raw_label", "concept"),
                            "mapped_ticker": ent.get("mapped_ticker"),
                        },
                    )

                article_chunks = [c for c in chunks if c["article_id"] == aid]
                for c in article_chunks:
                    chunk_uid = f"{c['article_id']}_chunk_{c['chunk_id']}"
                    chunk_norm = canonicalize(c["text"])
                    for ent in art_ents.get("entities", []):
                        cname = ent["canonical_name"]
                        if cname and _in_text(cname, chunk_norm):
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
                        ON CREATE SET r.relation_type = $rel_type, r.weight = 1
                        ON MATCH SET r.weight = r.weight + 1
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
        help="Skip NER-based entity extraction (faster setup)",
    )
    parser.add_argument(
        "--cooccur-mode",
        choices=["chunk", "article"],
        default=COOCCUR_DEFAULT_MODE,
        help=(
            "How to build RELATED_TO edges: "
            "'chunk' (default, lower noise) or 'article' (denser recall)"
        ),
    )
    parser.add_argument(
        "--ticker-map",
        type=Path,
        default=Path("ticker_company_map.csv"),
        help="CSV mapping of company_name/aliases to stock ticker (exact match).",
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

    # NER extraction (freed before embedding)
    entity_data = None
    if not args.skip_entities and chunks:
        pipe = load_extraction_model()

        # Deterministic exact mapping (no fuzzy matching) from company/alias -> ticker.
        # If the file is missing, we fail loudly to avoid building an unlinked graph.
        ticker_lookup = load_ticker_company_map(args.ticker_map)

        entity_data = extract_all_entities(
            chunks,
            pipe,
            cooccur_mode=args.cooccur_mode,
            ticker_lookup=ticker_lookup,
        )
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
