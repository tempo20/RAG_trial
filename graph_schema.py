"""
graph_schema.py — T-GRAG Temporal Knowledge Graph Schema

Node types and their roles in the three-layer retrieval:

  (:Article)          — raw scraped article, anchors temporal period
  (:Chunk)            — fixed-size text block (processing unit d^{t_i}_j)
  (:KnowledgeNode)    — temporal entity snapshot e^{T_i}_i  (ONE per entity PER period)
  (:Knowledge)        — atomic knowledge unit k^{t_i}  (fact extracted from a chunk)
  (:Period)           — discrete time bucket (e.g. "2024-W12", "2024-Q1", "2024-11")

Edge types:
  (:Article)-[:IN_PERIOD]->(:Period)
  (:Article)-[:HAS_CHUNK]->(:Chunk)
  (:Chunk)-[:IN_PERIOD]->(:Period)           # denormalized for fast R_time
  (:Chunk)-[:SOURCED_FROM]->(:KnowledgeNode) # chunk -> which entity node it fed
  (:KnowledgeNode)-[:IN_PERIOD]->(:Period)
  (:KnowledgeNode)-[:HAS_KNOWLEDGE]->(:Knowledge)
  (:Knowledge)-[:SOURCED_FROM_CHUNK]->(:Chunk)  # back-link for source text extractor
  (:KnowledgeNode)-[:RELATED_TO {type, period_key}]->(:KnowledgeNode)  # typed relations
  (:KnowledgeNode)-[:SAME_ENTITY_AS]->(:KnowledgeNode)  # cross-period identity chain

Retrieval layers:
  R_time      : MATCH (p:Period {key:$period_key})<-[:IN_PERIOD]-(kn:KnowledgeNode)
  R_node      : vector index on KnowledgeNode.embedding  (coarse)
  R_knowledge : vector index on Knowledge.embedding      (fine-grained)

Period granularity choices (set PERIOD_GRANULARITY in env or config):
  "week"    -> "2024-W12"    high temporal resolution, more nodes
  "month"   -> "2024-11"     good default for daily news
  "quarter" -> "2024-Q4"     good for slower-moving domains
"""

from __future__ import annotations

import os
from datetime import datetime, date
from typing import Literal

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

PeriodGranularity = Literal["week", "month", "quarter"]
PERIOD_GRANULARITY: PeriodGranularity = os.getenv("PERIOD_GRANULARITY", "month")  # type: ignore


# ---------------------------------------------------------------------------
# Period key helpers
# ---------------------------------------------------------------------------

def period_key_for(dt: datetime | date, granularity: PeriodGranularity = PERIOD_GRANULARITY) -> str:
    """
    Convert a datetime to the canonical period key string.

    Examples:
        period_key_for(datetime(2024, 11, 15), "month")   -> "2024-11"
        period_key_for(datetime(2024, 11, 15), "week")    -> "2024-W46"
        period_key_for(datetime(2024, 11, 15), "quarter") -> "2024-Q4"
    """
    if isinstance(dt, datetime):
        d = dt.date()
    else:
        d = dt

    if granularity == "month":
        return d.strftime("%Y-%m")
    elif granularity == "week":
        iso = d.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    elif granularity == "quarter":
        q = (d.month - 1) // 3 + 1
        return f"{d.year}-Q{q}"
    else:
        raise ValueError(f"Unknown granularity: {granularity}")


def knowledge_node_uid(canonical_name: str, period_key: str) -> str:
    """
    Stable unique ID for a KnowledgeNode (entity × period).
    e.g. "apple_inc::2024-11"
    """
    safe = canonical_name.lower().strip().replace(" ", "_")
    return f"{safe}::{period_key}"


# ---------------------------------------------------------------------------
# Schema bootstrap — constraints and indexes
# ---------------------------------------------------------------------------

CONSTRAINTS = [
    # Uniqueness
    ("period_key_unique",
     "CREATE CONSTRAINT period_key_unique IF NOT EXISTS "
     "FOR (p:Period) REQUIRE p.key IS UNIQUE"),

    ("article_id_unique",
     "CREATE CONSTRAINT article_id_unique IF NOT EXISTS "
     "FOR (a:Article) REQUIRE a.article_id IS UNIQUE"),

    ("chunk_uid_unique",
     "CREATE CONSTRAINT chunk_uid_unique IF NOT EXISTS "
     "FOR (c:Chunk) REQUIRE c.chunk_uid IS UNIQUE"),

    ("knowledge_node_uid_unique",
     "CREATE CONSTRAINT knowledge_node_uid_unique IF NOT EXISTS "
     "FOR (kn:KnowledgeNode) REQUIRE kn.kn_uid IS UNIQUE"),

    ("knowledge_uid_unique",
     "CREATE CONSTRAINT knowledge_uid_unique IF NOT EXISTS "
     "FOR (k:Knowledge) REQUIRE k.knowledge_uid IS UNIQUE"),
]

INDEXES = [
    # Fast R_time lookups: given period_key, pull all KnowledgeNodes / Chunks
    ("idx_knowledge_node_period",
     "CREATE INDEX idx_knowledge_node_period IF NOT EXISTS "
     "FOR (kn:KnowledgeNode) ON (kn.period_key)"),

    ("idx_chunk_period",
     "CREATE INDEX idx_chunk_period IF NOT EXISTS "
     "FOR (c:Chunk) ON (c.period_key)"),

    ("idx_article_period",
     "CREATE INDEX idx_article_period IF NOT EXISTS "
     "FOR (a:Article) ON (a.period_key)"),

    # Entity name lookup (for cross-period SAME_ENTITY_AS chaining)
    ("idx_knowledge_node_canonical",
     "CREATE INDEX idx_knowledge_node_canonical IF NOT EXISTS "
     "FOR (kn:KnowledgeNode) ON (kn.canonical_name)"),

    # Published date range scans
    ("idx_article_published",
     "CREATE INDEX idx_article_published IF NOT EXISTS "
     "FOR (a:Article) ON (a.published_date)"),

    ("idx_chunk_published",
     "CREATE INDEX idx_chunk_published IF NOT EXISTS "
     "FOR (c:Chunk) ON (c.published_date)"),
]

VECTOR_INDEXES = [
    # R_node — coarse retrieval over KnowledgeNode summaries
    # Embedding dim must match your model (e.g. 1536 for text-embedding-3-small,
    # 768 for all-mpnet-base-v2, 1024 for stella-en-1.5B-v5)
    ("vector_knowledge_node",
     """
     CREATE VECTOR INDEX vector_knowledge_node IF NOT EXISTS
     FOR (kn:KnowledgeNode) ON kn.embedding
     OPTIONS {indexConfig: {
       `vector.dimensions`: $dim,
       `vector.similarity_function`: 'cosine'
     }}
     """),

    # R_knowledge — fine-grained retrieval over atomic Knowledge facts
    ("vector_knowledge",
     """
     CREATE VECTOR INDEX vector_knowledge IF NOT EXISTS
     FOR (k:Knowledge) ON k.embedding
     OPTIONS {indexConfig: {
       `vector.dimensions`: $dim,
       `vector.similarity_function`: 'cosine'
     }}
     """),

    # Chunk-level vector (used by source text extractor)
    ("vector_chunk",
     """
     CREATE VECTOR INDEX vector_chunk IF NOT EXISTS
     FOR (c:Chunk) ON c.embedding
     OPTIONS {indexConfig: {
       `vector.dimensions`: $dim,
       `vector.similarity_function`: 'cosine'
     }}
     """),
]


def ensure_schema(session, embedding_dim: int = 768) -> None:
    """
    Idempotently create all constraints, indexes, and vector indexes.
    Safe to call on every startup.
    """
    for name, cypher in CONSTRAINTS:
        session.run(cypher)

    for name, cypher in INDEXES:
        session.run(cypher)

    for name, cypher in VECTOR_INDEXES:
        session.run(cypher, {"dim": embedding_dim})

    print(f"[schema] Constraints and indexes ensured (embedding_dim={embedding_dim})")


# ---------------------------------------------------------------------------
# Graph wipe helpers
# ---------------------------------------------------------------------------

TGRAG_LABELS = [
    "Article",
    "Chunk",
    "KnowledgeNode",
    "Knowledge",
    "Period"
]

def wipe_tgrag_graph(session) -> None:
    """
    Delete all T-GRAG nodes and relationships.
    Uses batched deletes to avoid OOM on large graphs.
    Does NOT drop constraints or indexes.
    """
    for label in TGRAG_LABELS:
        while True:
            result = session.run(
                f"MATCH (n:{label}) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS cnt"
            ).single()
            cnt = result["cnt"] if result else 0
            if cnt == 0:
                break
            print(f"[wipe] Deleted batch of {cnt} :{label} nodes")
    print("[wipe] T-GRAG graph cleared")


# ---------------------------------------------------------------------------
# Cypher upsert templates
# ---------------------------------------------------------------------------

UPSERT_PERIOD = """
MERGE (p:Period {key: $key})
ON CREATE SET p.granularity = $granularity,
              p.label = $label
RETURN p.key AS key
"""

UPSERT_ARTICLE = """
MERGE (a:Article {article_id: $article_id})
ON CREATE SET
    a.url          = $url,
    a.title        = $title,
    a.source       = $source,
    a.published_date = date($published_date),
    a.scraped_at   = datetime($scraped_at),
    a.period_key   = $period_key,
    a.content_hash = $content_hash
WITH a
MATCH (p:Period {key: $period_key})
MERGE (a)-[:IN_PERIOD]->(p)
RETURN a.article_id AS article_id
"""

UPSERT_CHUNK = """
MERGE (c:Chunk {chunk_uid: $chunk_uid})
ON CREATE SET
    c.text          = $text,
    c.chunk_index   = $chunk_index,
    c.article_id    = $article_id,
    c.published_date = date($published_date),
    c.period_key    = $period_key,
    c.token_count   = $token_count
WITH c
MATCH (a:Article {article_id: $article_id})
MERGE (a)-[:HAS_CHUNK]->(c)
WITH c
MATCH (p:Period {key: $period_key})
MERGE (c)-[:IN_PERIOD]->(p)
RETURN c.chunk_uid AS chunk_uid
"""

UPSERT_CHUNK_EMBEDDING = """
MATCH (c:Chunk {chunk_uid: $chunk_uid})
SET c.embedding = $embedding
"""

UPSERT_KNOWLEDGE_NODE = """
MERGE (kn:KnowledgeNode {kn_uid: $kn_uid})
ON CREATE SET
    kn.canonical_name = $canonical_name,
    kn.entity_type    = $entity_type,
    kn.period_key     = $period_key,
    kn.description    = $description
ON MATCH SET
    kn.description    = $description
WITH kn
MATCH (p:Period {key: $period_key})
MERGE (kn)-[:IN_PERIOD]->(p)
RETURN kn.kn_uid AS kn_uid
"""

UPSERT_KNOWLEDGE_NODE_EMBEDDING = """
MATCH (kn:KnowledgeNode {kn_uid: $kn_uid})
SET kn.embedding = $embedding
"""

UPSERT_KNOWLEDGE = """
MERGE (k:Knowledge {knowledge_uid: $knowledge_uid})
ON CREATE SET
    k.text        = $text,
    k.kn_uid      = $kn_uid,
    k.period_key  = $period_key,
    k.fact_type   = $fact_type
ON MATCH SET
    k.text        = $text
WITH k
MATCH (kn:KnowledgeNode {kn_uid: $kn_uid})
MERGE (kn)-[:HAS_KNOWLEDGE]->(k)
WITH k
MATCH (c:Chunk {chunk_uid: $chunk_uid})
MERGE (k)-[:SOURCED_FROM_CHUNK]->(c)
RETURN k.knowledge_uid AS knowledge_uid
"""

UPSERT_KNOWLEDGE_EMBEDDING = """
MATCH (k:Knowledge {knowledge_uid: $knowledge_uid})
SET k.embedding = $embedding
"""

UPSERT_CHUNK_TO_KN_EDGE = """
MATCH (c:Chunk {chunk_uid: $chunk_uid})
MATCH (kn:KnowledgeNode {kn_uid: $kn_uid})
MERGE (c)-[:SOURCED_FROM]->(kn)
"""

UPSERT_RELATION = """
MATCH (src:KnowledgeNode {kn_uid: $src_uid})
MATCH (tgt:KnowledgeNode {kn_uid: $tgt_uid})
MERGE (src)-[r:RELATED_TO {type: $rel_type, period_key: $period_key}]->(tgt)
ON CREATE SET r.description = $description
"""

LINK_SAME_ENTITY = """
// Connect KnowledgeNodes of the same entity across adjacent periods.
// Called once after a batch of new nodes is written.
MATCH (older:KnowledgeNode), (newer:KnowledgeNode)
WHERE older.canonical_name = newer.canonical_name
  AND older.period_key < newer.period_key
  AND NOT (older)-[:SAME_ENTITY_AS]->(newer)
  // Only link if no intermediate period node exists (direct adjacency)
  AND NOT EXISTS {
    MATCH (mid:KnowledgeNode)
    WHERE mid.canonical_name = older.canonical_name
      AND mid.period_key > older.period_key
      AND mid.period_key < newer.period_key
  }
MERGE (older)-[:SAME_ENTITY_AS]->(newer)
RETURN count(*) AS linked
"""


# ---------------------------------------------------------------------------
# CLI: run schema bootstrap standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrap T-GRAG schema in Neo4j")
    parser.add_argument("--dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--wipe", action="store_true", help="Wipe existing T-GRAG graph first")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    with driver.session() as session:
        if args.wipe:
            wipe_tgrag_graph(session)
        ensure_schema(session, embedding_dim=args.dim)

    driver.close()
    print("Schema bootstrap complete.")