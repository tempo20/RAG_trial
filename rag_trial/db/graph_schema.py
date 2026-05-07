"""
Lean Neo4j schema for hybrid macro QA.

This graph intentionally stores only retrieval-ready structured objects:

  (:Period)
  (:Entity)
  (:MacroEvent)
  (:EventCluster)
  (:Signal)
  (:Asset)
  (:Channel)

Raw article and chunk text remain in SQLite. Neo4j stores only foreign keys
and compact structured metadata so retrieval can hydrate evidence from SQLite.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Literal

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

PeriodGranularity = Literal["week", "month", "quarter"]
PERIOD_GRANULARITY: PeriodGranularity = os.getenv("PERIOD_GRANULARITY", "month")  # type: ignore[assignment]


def period_key_for(dt: datetime | date, granularity: PeriodGranularity = PERIOD_GRANULARITY) -> str:
    """Convert a date-like object to the canonical period key string."""
    d = dt.date() if isinstance(dt, datetime) else dt
    if granularity == "month":
        return d.strftime("%Y-%m")
    if granularity == "week":
        iso = d.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if granularity == "quarter":
        quarter = (d.month - 1) // 3 + 1
        return f"{d.year}-Q{quarter}"
    raise ValueError(f"Unknown granularity: {granularity}")


def knowledge_node_uid(canonical_name: str, period_key: str) -> str:
    """Stable unique ID for an entity-period knowledge node."""
    safe = canonical_name.lower().strip().replace(" ", "_")
    return f"{safe}::{period_key}"


def asset_key(target_type: str, target_id: str) -> str:
    """Stable key for an Asset node."""
    return f"{(target_type or '').strip().lower()}:{(target_id or '').strip()}"


def channel_key(channel_name: str) -> str:
    """Stable key for a Channel node."""
    return (channel_name or "").strip().lower()


CONSTRAINTS = [
    (
        "period_key_unique",
        "CREATE CONSTRAINT period_key_unique IF NOT EXISTS "
        "FOR (p:Period) REQUIRE p.key IS UNIQUE",
    ),
    (
        "entity_id_unique",
        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    ),
    (
        "macro_event_id_unique",
        "CREATE CONSTRAINT macro_event_id_unique IF NOT EXISTS "
        "FOR (m:MacroEvent) REQUIRE m.macro_event_id IS UNIQUE",
    ),
    (
        "event_cluster_id_unique",
        "CREATE CONSTRAINT event_cluster_id_unique IF NOT EXISTS "
        "FOR (c:EventCluster) REQUIRE c.cluster_id IS UNIQUE",
    ),
    (
        "signal_id_unique",
        "CREATE CONSTRAINT signal_id_unique IF NOT EXISTS "
        "FOR (s:Signal) REQUIRE s.signal_id IS UNIQUE",
    ),
    (
        "asset_key_unique",
        "CREATE CONSTRAINT asset_key_unique IF NOT EXISTS "
        "FOR (a:Asset) REQUIRE a.asset_key IS UNIQUE",
    ),
    (
        "channel_name_unique",
        "CREATE CONSTRAINT channel_name_unique IF NOT EXISTS "
        "FOR (c:Channel) REQUIRE c.name IS UNIQUE",
    ),
]

INDEXES = [
    (
        "idx_entity_display_name",
        "CREATE INDEX idx_entity_display_name IF NOT EXISTS "
        "FOR (e:Entity) ON (e.display_name)",
    ),
    (
        "idx_entity_ticker",
        "CREATE INDEX idx_entity_ticker IF NOT EXISTS "
        "FOR (e:Entity) ON (e.ticker)",
    ),
    (
        "idx_macro_event_period",
        "CREATE INDEX idx_macro_event_period IF NOT EXISTS "
        "FOR (m:MacroEvent) ON (m.period_key)",
    ),
    (
        "idx_macro_event_event_type",
        "CREATE INDEX idx_macro_event_event_type IF NOT EXISTS "
        "FOR (m:MacroEvent) ON (m.event_type)",
    ),
    (
        "idx_macro_event_chunk_id",
        "CREATE INDEX idx_macro_event_chunk_id IF NOT EXISTS "
        "FOR (m:MacroEvent) ON (m.chunk_id)",
    ),
    (
        "idx_event_cluster_event_type",
        "CREATE INDEX idx_event_cluster_event_type IF NOT EXISTS "
        "FOR (c:EventCluster) ON (c.event_type)",
    ),
    (
        "idx_event_cluster_region",
        "CREATE INDEX idx_event_cluster_region IF NOT EXISTS "
        "FOR (c:EventCluster) ON (c.region)",
    ),
    (
        "idx_signal_date",
        "CREATE INDEX idx_signal_date IF NOT EXISTS "
        "FOR (s:Signal) ON (s.signal_date)",
    ),
    (
        "idx_signal_score",
        "CREATE INDEX idx_signal_score IF NOT EXISTS "
        "FOR (s:Signal) ON (s.signal_score)",
    ),
    (
        "idx_asset_target",
        "CREATE INDEX idx_asset_target IF NOT EXISTS "
        "FOR (a:Asset) ON (a.target_type, a.target_id)",
    ),
]


def ensure_schema(session, embedding_dim: int = 768) -> None:
    """Idempotently create all lean-schema constraints and indexes."""
    _ = embedding_dim
    for _, cypher in CONSTRAINTS:
        session.run(cypher)
    for _, cypher in INDEXES:
        session.run(cypher)
    print("[schema] Lean graph constraints and indexes ensured")


LEAN_LABELS = [
    "MacroEvent",
    "EventCluster",
    "Signal",
    "Entity",
    "Asset",
    "Channel",
    "Period",
]


def wipe_lean_graph(session) -> None:
    """Delete all lean-graph nodes and relationships without dropping schema objects."""
    for label in LEAN_LABELS:
        while True:
            result = session.run(
                f"MATCH (n:{label}) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS cnt"
            ).single()
            count = result["cnt"] if result else 0
            if count == 0:
                break
            print(f"[wipe] Deleted batch of {count} :{label} nodes")
    print("[wipe] Lean macro graph cleared")


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
    m.run_id              = $run_id,
    m.article_id          = $article_id,
    m.chunk_id            = $chunk_id,
    m.period_key          = $period_key,
    m.published_date      = $published_date,
    m.event_type          = $event_type,
    m.summary             = $summary,
    m.region              = $region,
    m.time_horizon        = $time_horizon,
    m.confidence          = $confidence,
    m.verification_status = $verification_status,
    m.support_score       = $support_score,
    m.shock_types         = $shock_types
ON MATCH SET
    m.article_id          = $article_id,
    m.chunk_id            = $chunk_id,
    m.period_key          = $period_key,
    m.published_date      = $published_date,
    m.event_type          = $event_type,
    m.summary             = $summary,
    m.region              = $region,
    m.time_horizon        = $time_horizon,
    m.confidence          = $confidence,
    m.verification_status = $verification_status,
    m.support_score       = $support_score,
    m.shock_types         = $shock_types
WITH m
MATCH (p:Period {key: $period_key})
MERGE (m)-[:IN_PERIOD]->(p)
RETURN m.macro_event_id AS macro_event_id
"""

UPSERT_EVENT_CLUSTER = """
MERGE (c:EventCluster {cluster_id: $cluster_id})
ON CREATE SET
    c.event_type          = $event_type,
    c.primary_shock_type  = $primary_shock_type,
    c.region              = $region,
    c.canonical_summary   = $canonical_summary,
    c.first_event_time    = $first_event_time,
    c.last_event_time     = $last_event_time,
    c.member_count        = $member_count,
    c.unique_source_count = $unique_source_count
ON MATCH SET
    c.event_type          = $event_type,
    c.primary_shock_type  = $primary_shock_type,
    c.region              = $region,
    c.canonical_summary   = $canonical_summary,
    c.first_event_time    = $first_event_time,
    c.last_event_time     = $last_event_time,
    c.member_count        = $member_count,
    c.unique_source_count = $unique_source_count
RETURN c.cluster_id AS cluster_id
"""

UPSERT_SIGNAL = """
MERGE (s:Signal {signal_id: $signal_id})
ON CREATE SET
    s.cluster_id    = $cluster_id,
    s.signal_date   = $signal_date,
    s.rank          = $rank,
    s.signal_score  = $signal_score,
    s.headline      = $headline,
    s.summary       = $summary,
    s.novelty_hint  = $novelty_hint,
    s.urgency       = $urgency,
    s.market_surprise = $market_surprise,
    s.status        = $status
ON MATCH SET
    s.cluster_id    = $cluster_id,
    s.signal_date   = $signal_date,
    s.rank          = $rank,
    s.signal_score  = $signal_score,
    s.headline      = $headline,
    s.summary       = $summary,
    s.novelty_hint  = $novelty_hint,
    s.urgency       = $urgency,
    s.market_surprise = $market_surprise,
    s.status        = $status
RETURN s.signal_id AS signal_id
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

UPSERT_CLUSTER_IMPACTS_EDGE = """
MATCH (c:EventCluster {cluster_id: $cluster_id})
MATCH (a:Asset {asset_key: $asset_key})
MERGE (c)-[r:IMPACTS {cluster_id: $cluster_id, asset_key: $asset_key}]->(a)
SET r.direction = $direction,
    r.signal_score = $signal_score
RETURN r.asset_key AS asset_key
"""

UPSERT_INVOLVES_EDGE = """
MATCH (m:MacroEvent {macro_event_id: $macro_event_id})
MATCH (e:Entity {entity_id: $entity_id})
MERGE (m)-[:INVOLVES]->(e)
"""

UPSERT_MEMBER_OF_EDGE = """
MATCH (m:MacroEvent {macro_event_id: $macro_event_id})
MATCH (c:EventCluster {cluster_id: $cluster_id})
MERGE (m)-[:MEMBER_OF]->(c)
"""

UPSERT_SIGNAL_BASED_ON_EDGE = """
MATCH (s:Signal {signal_id: $signal_id})
MATCH (c:EventCluster {cluster_id: $cluster_id})
MERGE (s)-[:BASED_ON]->(c)
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


# Backwards-compatible alias for older callers.
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
