"""
graph_schema.py — Lean Neo4j schema for hybrid macro QA.

This graph intentionally stores only retrieval-ready structured objects:

  (:Period)
  (:Entity)
  (:MacroEvent)
  (:Asset)
  (:Channel)   # optional, but supported

Raw article and chunk text remain in SQLite. Neo4j stores only foreign keys
such as chunk_id / article_id on MacroEvent nodes so the chatbot can fetch
evidence text from SQLite when needed.
"""

from __future__ import annotations

import os
from datetime import datetime, date
from typing import Literal

from dotenv import load_dotenv
from neo4j import GraphDatabase

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
# Lean graph helpers
# ---------------------------------------------------------------------------

def asset_key(target_type: str, target_id: str) -> str:
    """Stable key for an Asset node."""
    return f"{(target_type or '').strip().lower()}:{(target_id or '').strip()}"


def channel_key(channel_name: str) -> str:
    """Stable key for a Channel node."""
    return (channel_name or "").strip().lower()


# ---------------------------------------------------------------------------
# Schema bootstrap — constraints and indexes
# ---------------------------------------------------------------------------

CONSTRAINTS = [
    ("period_key_unique",
     "CREATE CONSTRAINT period_key_unique IF NOT EXISTS "
     "FOR (p:Period) REQUIRE p.key IS UNIQUE"),
    ("entity_id_unique",
     "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
     "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"),
    ("macro_event_id_unique",
     "CREATE CONSTRAINT macro_event_id_unique IF NOT EXISTS "
     "FOR (m:MacroEvent) REQUIRE m.macro_event_id IS UNIQUE"),
    ("asset_key_unique",
     "CREATE CONSTRAINT asset_key_unique IF NOT EXISTS "
     "FOR (a:Asset) REQUIRE a.asset_key IS UNIQUE"),
    ("channel_name_unique",
     "CREATE CONSTRAINT channel_name_unique IF NOT EXISTS "
     "FOR (c:Channel) REQUIRE c.name IS UNIQUE"),
]

INDEXES = [
    ("idx_entity_display_name",
     "CREATE INDEX idx_entity_display_name IF NOT EXISTS "
     "FOR (e:Entity) ON (e.display_name)"),
    ("idx_entity_ticker",
     "CREATE INDEX idx_entity_ticker IF NOT EXISTS "
     "FOR (e:Entity) ON (e.ticker)"),
    ("idx_macro_event_period",
     "CREATE INDEX idx_macro_event_period IF NOT EXISTS "
     "FOR (m:MacroEvent) ON (m.period_key)"),
    ("idx_macro_event_event_type",
     "CREATE INDEX idx_macro_event_event_type IF NOT EXISTS "
     "FOR (m:MacroEvent) ON (m.event_type)"),
    ("idx_macro_event_chunk_id",
     "CREATE INDEX idx_macro_event_chunk_id IF NOT EXISTS "
     "FOR (m:MacroEvent) ON (m.chunk_id)"),
    ("idx_asset_target",
     "CREATE INDEX idx_asset_target IF NOT EXISTS "
     "FOR (a:Asset) ON (a.target_type, a.target_id)"),
]


def ensure_schema(session, embedding_dim: int = 768) -> None:
    """
    Idempotently create all lean-schema constraints and indexes.
    embedding_dim is accepted for backwards compatibility with older callers.
    """
    for _, cypher in CONSTRAINTS:
        session.run(cypher)

    for _, cypher in INDEXES:
        session.run(cypher)

    print("[schema] Lean graph constraints and indexes ensured")


# ---------------------------------------------------------------------------
# Graph wipe helpers
# ---------------------------------------------------------------------------

LEAN_LABELS = [
    "MacroEvent",
    "Entity",
    "Asset",
    "Channel",
    "Period",
]


def wipe_lean_graph(session) -> None:
    """
    Delete all lean-graph nodes and relationships.
    Uses batched deletes to avoid OOM on large graphs.
    Does NOT drop constraints or indexes.
    """
    for label in LEAN_LABELS:
        while True:
            result = session.run(
                f"MATCH (n:{label}) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS cnt"
            ).single()
            cnt = result["cnt"] if result else 0
            if cnt == 0:
                break
            print(f"[wipe] Deleted batch of {cnt} :{label} nodes")
    print("[wipe] Lean macro graph cleared")


# ---------------------------------------------------------------------------
# Cypher upsert templates
# ---------------------------------------------------------------------------

UPSERT_PERIOD = """
MERGE (p:Period {key: $key})
ON CREATE SET p.granularity = $granularity,
              p.label = $label
RETURN p.key AS key
"""

UPSERT_ENTITY = """
MERGE (e:Entity {entity_id: $entity_id})
ON CREATE SET
    e.display_name = coalesce($display_name, $entity_id),
    e.entity_type  = $entity_type,
    e.ticker       = $ticker
ON MATCH SET
    e.display_name = coalesce($display_name, e.display_name),
    e.entity_type  = coalesce($entity_type, e.entity_type),
    e.ticker       = coalesce($ticker, e.ticker)
RETURN e.entity_id AS entity_id
"""

UPSERT_MACRO_EVENT = """
MERGE (m:MacroEvent {macro_event_id: $macro_event_id})
ON CREATE SET
    m.run_id           = $run_id,
    m.article_id       = $article_id,
    m.chunk_id         = $chunk_id,
    m.period_key       = $period_key,
    m.published_date   = $published_date,
    m.event_type       = $event_type,
    m.summary          = $summary,
    m.region           = $region,
    m.time_horizon     = $time_horizon,
    m.confidence       = $confidence,
    m.shock_types      = $shock_types,
    m.evidence_id      = $evidence_id,
    m.evidence_text    = $evidence_text
ON MATCH SET
    m.article_id       = $article_id,
    m.chunk_id         = $chunk_id,
    m.period_key       = $period_key,
    m.published_date   = $published_date,
    m.event_type       = $event_type,
    m.summary          = $summary,
    m.region           = $region,
    m.time_horizon     = $time_horizon,
    m.confidence       = $confidence,
    m.shock_types      = $shock_types,
    m.evidence_id      = $evidence_id,
    m.evidence_text    = $evidence_text
WITH m
MATCH (p:Period {key: $period_key})
MERGE (m)-[:IN_PERIOD]->(p)
RETURN m.macro_event_id AS macro_event_id
"""

UPSERT_ASSET = """
MERGE (a:Asset {asset_key: $asset_key})
ON CREATE SET
    a.target_type  = $target_type,
    a.target_id    = $target_id,
    a.display_name = coalesce($display_name, $target_id)
ON MATCH SET
    a.target_type  = $target_type,
    a.target_id    = $target_id,
    a.display_name = coalesce($display_name, a.display_name)
RETURN a.asset_key AS asset_key
"""

UPSERT_CHANNEL = """
MERGE (c:Channel {name: $name})
RETURN c.name AS name
"""

UPSERT_IMPACTS_EDGE = """
MATCH (m:MacroEvent {macro_event_id: $macro_event_id})
MATCH (a:Asset {asset_key: $asset_key})
MERGE (m)-[r:IMPACTS {impact_id: $impact_id}]->(a)
SET r.direction = $direction,
    r.strength  = $strength,
    r.horizon   = $horizon,
    r.confidence = $confidence,
    r.rationale = $rationale
RETURN r.impact_id AS impact_id
"""

UPSERT_INVOLVES_EDGE = """
MATCH (m:MacroEvent {macro_event_id: $macro_event_id})
MATCH (e:Entity {entity_id: $entity_id})
MERGE (m)-[:INVOLVES]->(e)
"""

UPSERT_CHANNEL_EDGE = """
MATCH (m:MacroEvent {macro_event_id: $macro_event_id})
MATCH (c:Channel {name: $channel_name})
MERGE (m)-[r:TRANSMITS_VIA {channel_name: $channel_name}]->(c)
SET r.direction = $direction,
    r.strength = $strength,
    r.confidence = $confidence
RETURN r.channel_name AS channel_name
"""


# ---------------------------------------------------------------------------
# CLI: run schema bootstrap standalone
# ---------------------------------------------------------------------------

# Backwards-compatible alias for older callers that still refer to the old name.
wipe_tgrag_graph = wipe_lean_graph


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrap lean macro schema in Neo4j")
    parser.add_argument("--dim", type=int, default=768, help="Unused compatibility argument")
    parser.add_argument("--wipe", action="store_true", help="Wipe existing lean graph first")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    with driver.session() as session:
        if args.wipe:
            wipe_lean_graph(session)
        ensure_schema(session, embedding_dim=args.dim)

    driver.close()
    print("Schema bootstrap complete.")