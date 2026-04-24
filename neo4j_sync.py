"""
neo4j_sync.py - Rebuild the lean Neo4j macro graph from SQLite.

Graph shape:
  (:Period)
  (:Entity)
  (:MacroEvent)
  (:EventCluster)
  (:Signal)
  (:Asset)
  (:Channel)

SQLite remains the source of truth for raw article/chunk evidence. Neo4j stores
only retrieval-ready structured objects plus foreign keys back to SQLite rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter, defaultdict
from typing import Iterable

from dotenv import load_dotenv
from neo4j import GraphDatabase

from graph_schema import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    PERIOD_GRANULARITY,
    UPSERT_ASSET,
    UPSERT_CHANNEL,
    UPSERT_CHANNEL_EDGE,
    UPSERT_CLUSTER_IMPACTS_EDGE,
    UPSERT_ENTITY,
    UPSERT_EVENT_CLUSTER,
    UPSERT_IMPACTS_EDGE,
    UPSERT_INVOLVES_EDGE,
    UPSERT_MACRO_EVENT,
    UPSERT_MEMBER_OF_EDGE,
    UPSERT_PERIOD,
    UPSERT_SIGNAL,
    UPSERT_SIGNAL_BASED_ON_EDGE,
    asset_key,
    channel_key,
    ensure_schema,
    wipe_lean_graph,
)

load_dotenv()

SQLITE_DB = os.getenv("SQLITE_DB", "my_database.db")
BATCH_SIZE = int(os.getenv("NEO4J_SYNC_BATCH_SIZE", "500"))


def connect_sqlite(db_path: str = SQLITE_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _batched(items: list[dict], batch_size: int = BATCH_SIZE) -> Iterable[list[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _run_row_upserts(session, cypher: str, rows: list[dict], label: str) -> None:
    if not rows:
        print(f"[neo4j_sync] {label}: 0")
        return
    written = 0
    for batch in _batched(rows):
        for row in batch:
            session.run(cypher, row)
        written += len(batch)
        print(f"[neo4j_sync] {label}: {written}/{len(rows)}")


def load_period_rows(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT DISTINCT period_key
        FROM chunks
        WHERE period_key IS NOT NULL
          AND period_key <> ''
        ORDER BY period_key
        """
    ).fetchall()
    return [
        {
            "key": row["period_key"],
            "granularity": PERIOD_GRANULARITY,
            "label": row["period_key"],
        }
        for row in rows
    ]


def load_entity_rows(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            canonical_entity_id AS entity_id,
            MAX(NULLIF(display_name, '')) AS display_name,
            MAX(NULLIF(entity_type, '')) AS entity_type,
            MAX(NULLIF(ticker, '')) AS ticker
        FROM entity_mentions
        WHERE canonical_entity_id IS NOT NULL
          AND canonical_entity_id <> ''
        GROUP BY canonical_entity_id
        ORDER BY canonical_entity_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _load_shock_type_lookup(conn: sqlite3.Connection) -> dict[str, list[str]]:
    rows = conn.execute(
        """
        SELECT macro_event_id, shock_type
        FROM macro_event_shock_types
        ORDER BY macro_event_id, shock_type
        """
    ).fetchall()
    lookup: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        lookup[row["macro_event_id"]].append(row["shock_type"])
    return dict(lookup)


def load_macro_event_rows(conn: sqlite3.Connection) -> list[dict]:
    shock_types = _load_shock_type_lookup(conn)
    rows = conn.execute(
        """
        SELECT
            m.macro_event_id,
            m.run_id,
            m.article_id,
            m.chunk_id,
            m.event_type,
            m.summary,
            m.region,
            m.time_horizon,
            m.confidence,
            m.verification_status,
            m.support_score,
            c.period_key,
            c.published_date
        FROM macro_events m
        LEFT JOIN chunks c ON c.chunk_id = m.chunk_id
        ORDER BY m.macro_event_id
        """
    ).fetchall()
    out: list[dict] = []
    skipped = 0
    for row in rows:
        if not row["period_key"]:
            skipped += 1
            continue
        out.append(
            {
                "macro_event_id": row["macro_event_id"],
                "run_id": row["run_id"],
                "article_id": row["article_id"],
                "chunk_id": row["chunk_id"],
                "period_key": row["period_key"],
                "published_date": row["published_date"],
                "event_type": row["event_type"],
                "summary": row["summary"],
                "region": row["region"],
                "time_horizon": row["time_horizon"],
                "confidence": row["confidence"],
                "verification_status": row["verification_status"],
                "support_score": row["support_score"],
                "shock_types": shock_types.get(row["macro_event_id"], []),
            }
        )
    if skipped:
        print(f"[neo4j_sync] skipped {skipped} macro event(s) with no period_key")
    return out


def load_event_cluster_rows(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            cluster_id,
            event_type,
            primary_shock_type,
            region,
            canonical_summary,
            first_event_time,
            last_event_time,
            member_count,
            unique_source_count
        FROM event_clusters
        WHERE cluster_status = 'active'
        ORDER BY last_event_time DESC, cluster_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def load_signal_rows(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            signal_id,
            cluster_id,
            signal_date,
            rank,
            signal_score,
            headline,
            summary,
            novelty_hint,
            urgency,
            market_surprise,
            status
        FROM signal_alerts
        WHERE status = 'active'
        ORDER BY signal_score DESC, signal_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def load_asset_rows(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT DISTINCT target_type, target_id
        FROM asset_impacts
        WHERE target_id IS NOT NULL
          AND target_id <> ''
        ORDER BY target_type, target_id
        """
    ).fetchall()
    return [
        {
            "asset_key": asset_key(row["target_type"], row["target_id"]),
            "target_type": row["target_type"],
            "target_id": row["target_id"],
            "display_name": row["target_id"],
        }
        for row in rows
    ]


def load_impact_rows(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            impact_id,
            macro_event_id,
            target_type,
            target_id,
            direction,
            strength,
            horizon,
            confidence,
            rationale
        FROM asset_impacts
        ORDER BY macro_event_id, impact_id
        """
    ).fetchall()
    return [
        {
            "impact_id": row["impact_id"],
            "macro_event_id": row["macro_event_id"],
            "asset_key": asset_key(row["target_type"], row["target_id"]),
            "direction": row["direction"],
            "strength": row["strength"],
            "horizon": row["horizon"],
            "confidence": row["confidence"],
            "rationale": row["rationale"],
        }
        for row in rows
    ]


def load_cluster_impact_rows(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            cm.cluster_id,
            ai.target_type,
            ai.target_id,
            ai.direction,
            sa.signal_score
        FROM cluster_members cm
        JOIN asset_impacts ai ON ai.macro_event_id = cm.macro_event_id
        LEFT JOIN signal_alerts sa ON sa.cluster_id = cm.cluster_id AND sa.status = 'active'
        ORDER BY cm.cluster_id, ai.target_type, ai.target_id
        """
    ).fetchall()
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["cluster_id"], asset_key(row["target_type"], row["target_id"]))].append(dict(row))
    out: list[dict] = []
    for (cluster_id, resolved_asset_key), items in grouped.items():
        direction_counts = Counter(str(item.get("direction") or "mixed") for item in items)
        out.append(
            {
                "cluster_id": cluster_id,
                "asset_key": resolved_asset_key,
                "direction": direction_counts.most_common(1)[0][0] if direction_counts else "mixed",
                "signal_score": max(float(item.get("signal_score") or 0.0) for item in items),
            }
        )
    return out


def load_member_of_rows(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT cluster_id, macro_event_id
        FROM cluster_members
        ORDER BY cluster_id, macro_event_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def load_signal_based_on_rows(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT signal_id, cluster_id
        FROM signal_alerts
        WHERE status = 'active'
        ORDER BY signal_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def load_involves_rows(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT DISTINCT
            m.macro_event_id,
            em.canonical_entity_id AS entity_id
        FROM macro_events m
        JOIN entity_mentions em ON em.chunk_id = m.chunk_id
        WHERE em.canonical_entity_id IS NOT NULL
          AND em.canonical_entity_id <> ''
        ORDER BY m.macro_event_id, em.canonical_entity_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def load_channel_rows(conn: sqlite3.Connection) -> tuple[list[dict], list[dict]]:
    node_rows = conn.execute(
        """
        SELECT DISTINCT channel_name
        FROM macro_channels
        WHERE channel_name IS NOT NULL
          AND channel_name <> ''
        ORDER BY channel_name
        """
    ).fetchall()
    edge_rows = conn.execute(
        """
        SELECT
            macro_event_id,
            channel_name,
            direction,
            strength,
            confidence
        FROM macro_channels
        ORDER BY macro_event_id, channel_name
        """
    ).fetchall()
    return (
        [{"name": channel_key(row["channel_name"])} for row in node_rows],
        [
            {
                "macro_event_id": row["macro_event_id"],
                "channel_name": channel_key(row["channel_name"]),
                "direction": row["direction"],
                "strength": row["strength"],
                "confidence": row["confidence"],
            }
            for row in edge_rows
        ],
    )


def get_graph_counts(driver) -> dict[str, int]:
    queries = {
        "periods": "MATCH (n:Period) RETURN count(n) AS n",
        "entities": "MATCH (n:Entity) RETURN count(n) AS n",
        "macro_events": "MATCH (n:MacroEvent) RETURN count(n) AS n",
        "event_clusters": "MATCH (n:EventCluster) RETURN count(n) AS n",
        "signals": "MATCH (n:Signal) RETURN count(n) AS n",
        "assets": "MATCH (n:Asset) RETURN count(n) AS n",
        "channels": "MATCH (n:Channel) RETURN count(n) AS n",
        "impacts": "MATCH ()-[r:IMPACTS]->() RETURN count(r) AS n",
        "involves": "MATCH ()-[r:INVOLVES]->() RETURN count(r) AS n",
        "member_of": "MATCH ()-[r:MEMBER_OF]->() RETURN count(r) AS n",
        "based_on": "MATCH ()-[r:BASED_ON]->() RETURN count(r) AS n",
        "transmits_via": "MATCH ()-[r:TRANSMITS_VIA]->() RETURN count(r) AS n",
    }
    counts: dict[str, int] = {}
    with driver.session() as session:
        for key, cypher in queries.items():
            row = session.run(cypher).single()
            counts[key] = int(row["n"]) if row else 0
    return counts


def sync_sqlite_to_neo4j(
    db_path: str = SQLITE_DB,
    wipe: bool = False,
    include_channels: bool = True,
) -> dict[str, int]:
    conn = connect_sqlite(db_path)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    period_rows = load_period_rows(conn)
    entity_rows = load_entity_rows(conn)
    macro_event_rows = load_macro_event_rows(conn)
    event_cluster_rows = load_event_cluster_rows(conn)
    signal_rows = load_signal_rows(conn)
    asset_rows = load_asset_rows(conn)
    impact_rows = load_impact_rows(conn)
    cluster_impact_rows = load_cluster_impact_rows(conn)
    involves_rows = load_involves_rows(conn)
    member_of_rows = load_member_of_rows(conn)
    signal_based_on_rows = load_signal_based_on_rows(conn)
    channel_rows, channel_edge_rows = load_channel_rows(conn) if include_channels else ([], [])

    with driver.session() as session:
        if wipe:
            print("[neo4j_sync] wiping existing lean graph ...")
            wipe_lean_graph(session)
        ensure_schema(session)
        _run_row_upserts(session, UPSERT_PERIOD, period_rows, "periods")
        _run_row_upserts(session, UPSERT_ENTITY, entity_rows, "entities")
        _run_row_upserts(session, UPSERT_MACRO_EVENT, macro_event_rows, "macro_events")
        _run_row_upserts(session, UPSERT_EVENT_CLUSTER, event_cluster_rows, "event_clusters")
        _run_row_upserts(session, UPSERT_SIGNAL, signal_rows, "signals")
        _run_row_upserts(session, UPSERT_ASSET, asset_rows, "assets")
        _run_row_upserts(session, UPSERT_IMPACTS_EDGE, impact_rows, "impact_edges")
        _run_row_upserts(session, UPSERT_CLUSTER_IMPACTS_EDGE, cluster_impact_rows, "cluster_impact_edges")
        _run_row_upserts(session, UPSERT_INVOLVES_EDGE, involves_rows, "involves_edges")
        _run_row_upserts(session, UPSERT_MEMBER_OF_EDGE, member_of_rows, "member_of_edges")
        _run_row_upserts(session, UPSERT_SIGNAL_BASED_ON_EDGE, signal_based_on_rows, "based_on_edges")
        if include_channels:
            _run_row_upserts(session, UPSERT_CHANNEL, channel_rows, "channels")
            _run_row_upserts(session, UPSERT_CHANNEL_EDGE, channel_edge_rows, "channel_edges")

    counts = get_graph_counts(driver)
    driver.close()
    conn.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild lean Neo4j graph from SQLite")
    parser.add_argument("--db", default=SQLITE_DB, help="Path to SQLite DB")
    parser.add_argument("--wipe", action="store_true", help="Wipe lean graph before syncing")
    parser.add_argument(
        "--no-channels",
        action="store_true",
        help="Skip Channel nodes and TRANSMITS_VIA edges",
    )
    args = parser.parse_args()
    counts = sync_sqlite_to_neo4j(
        db_path=args.db,
        wipe=args.wipe,
        include_channels=not args.no_channels,
    )
    print("[neo4j_sync] complete")
    for key, value in counts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
