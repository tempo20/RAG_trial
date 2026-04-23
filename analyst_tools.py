from __future__ import annotations

import argparse
import json
import sqlite3

from create_sql_db import connect_sqlite, create_database


def _connect(db_path: str) -> sqlite3.Connection:
    create_database(db_path)
    conn = connect_sqlite(db_path, fk=False)
    conn.row_factory = sqlite3.Row
    return conn


def show_latest_macro_events(db_path: str, limit: int, min_confidence: float | None) -> None:
    conn = _connect(db_path)
    sql = """
        SELECT
            m.macro_event_id,
            m.event_type,
            m.summary,
            m.confidence,
            m.region,
            m.time_horizon,
            m.chunk_id,
            c.published_date,
            a.source,
            a.title
        FROM macro_events m
        LEFT JOIN chunks c ON c.chunk_id = m.chunk_id
        LEFT JOIN articles a ON a.article_id = m.article_id
    """
    params: list[object] = []
    if min_confidence is not None:
        sql += " WHERE coalesce(m.confidence, 0) >= ?"
        params.append(min_confidence)
    sql += " ORDER BY coalesce(m.confidence, 0) DESC, c.published_date DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    for row in rows:
        print(
            f"- {row['macro_event_id']} confidence={row['confidence']} type={row['event_type']} "
            f"date={row['published_date']} source={row['source']}"
        )
        print(f"  title: {row['title']}")
        print(f"  summary: {row['summary']}")
        print(f"  region={row['region']} horizon={row['time_horizon']} chunk={row['chunk_id']}")
    conn.close()


def show_event_evidence(db_path: str, event_id: str) -> None:
    conn = _connect(db_path)
    row = conn.execute(
        """
        SELECT
            m.macro_event_id,
            m.event_type,
            m.summary,
            m.confidence,
            m.chunk_id,
            c.text AS chunk_text,
            c.published_date,
            a.title,
            a.source,
            a.url
        FROM macro_events m
        LEFT JOIN chunks c ON c.chunk_id = m.chunk_id
        LEFT JOIN articles a ON a.article_id = m.article_id
        WHERE m.macro_event_id = ?
        """,
        (event_id,),
    ).fetchone()
    if not row:
        print("[analyst] macro event not found")
        conn.close()
        return

    evidence_rows = conn.execute(
        """
        SELECT evidence_text
        FROM evidence_spans
        WHERE parent_kind = 'macro_event'
          AND parent_id = ?
        ORDER BY evidence_id
        """,
        (event_id,),
    ).fetchall()
    impact_rows = conn.execute(
        """
        SELECT target_type, target_id, direction, strength, horizon, rationale
        FROM asset_impacts
        WHERE macro_event_id = ?
        ORDER BY target_type, target_id
        """,
        (event_id,),
    ).fetchall()
    print(
        f"{row['macro_event_id']} type={row['event_type']} confidence={row['confidence']} "
        f"date={row['published_date']} source={row['source']}"
    )
    print(f"title: {row['title']}")
    print(f"url: {row['url']}")
    print(f"summary: {row['summary']}")
    print("evidence:")
    for evidence in evidence_rows:
        print(f"- {evidence['evidence_text']}")
    print("impacts:")
    for impact in impact_rows:
        print(
            f"- {impact['target_type']}::{impact['target_id']} direction={impact['direction']} "
            f"strength={impact['strength']} horizon={impact['horizon']} rationale={impact['rationale']}"
        )
    print("chunk:")
    print(row["chunk_text"] or "")
    conn.close()


def show_entities_with_most_impact_links(db_path: str, limit: int) -> None:
    conn = _connect(db_path)
    rows = conn.execute(
        """
        SELECT
            em.canonical_entity_id,
            MAX(NULLIF(em.display_name, '')) AS display_name,
            COUNT(DISTINCT ai.impact_id) AS impact_links,
            COUNT(DISTINCT m.macro_event_id) AS event_count
        FROM entity_mentions em
        JOIN macro_events m ON m.chunk_id = em.chunk_id
        JOIN asset_impacts ai ON ai.macro_event_id = m.macro_event_id
        GROUP BY em.canonical_entity_id
        ORDER BY impact_links DESC, event_count DESC, em.canonical_entity_id
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    for row in rows:
        print(
            f"- {row['display_name'] or row['canonical_entity_id']} "
            f"({row['canonical_entity_id']}): impacts={row['impact_links']} events={row['event_count']}"
        )
    conn.close()


def show_questionable_events(db_path: str, limit: int) -> None:
    conn = _connect(db_path)
    rows = conn.execute(
        """
        SELECT
            m.macro_event_id,
            m.event_type,
            m.confidence,
            m.summary,
            a.queue_name,
            a.review_reasons_json,
            a.status,
            a.failure_reason,
            c.published_date,
            art.source,
            art.title
        FROM macro_processing_audit a
        LEFT JOIN macro_events m ON m.run_id = a.run_id
        LEFT JOIN chunks c ON c.chunk_id = a.chunk_id
        LEFT JOIN articles art ON art.article_id = a.article_id
        WHERE a.suspicious = 1
           OR a.queue_name IS NOT NULL
           OR a.status IN ('empty_success', 'parse_failed', 'api_failed')
        ORDER BY a.created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    for row in rows:
        reasons = json.loads(row["review_reasons_json"] or "[]")
        print(
            f"- event={row['macro_event_id'] or '—'} status={row['status']} queue={row['queue_name'] or '—'} "
            f"confidence={row['confidence']} date={row['published_date']} source={row['source']}"
        )
        print(f"  title: {row['title']}")
        print(f"  summary: {row['summary']}")
        print(f"  reasons: {', '.join(reasons) or row['failure_reason'] or '—'}")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyst-facing SQLite inspection tools")
    parser.add_argument("--db", default="my_database.db", help="Path to SQLite DB")
    subparsers = parser.add_subparsers(dest="command", required=True)

    latest_parser = subparsers.add_parser("latest-events", help="Show latest macro events by confidence")
    latest_parser.add_argument("--limit", type=int, default=10, help="Rows to print")
    latest_parser.add_argument("--min-confidence", type=float, default=None, help="Optional confidence floor")

    evidence_parser = subparsers.add_parser("event-evidence", help="Show event -> evidence -> chunk")
    evidence_parser.add_argument("--event-id", required=True, help="Macro event id")

    impact_parser = subparsers.add_parser("impact-entities", help="Show entities with most impact links")
    impact_parser.add_argument("--limit", type=int, default=10, help="Rows to print")

    questionable_parser = subparsers.add_parser("questionable-events", help="Show suspicious or questionable events")
    questionable_parser.add_argument("--limit", type=int, default=20, help="Rows to print")

    args = parser.parse_args()
    if args.command == "latest-events":
        show_latest_macro_events(args.db, args.limit, args.min_confidence)
    elif args.command == "event-evidence":
        show_event_evidence(args.db, args.event_id)
    elif args.command == "impact-entities":
        show_entities_with_most_impact_links(args.db, args.limit)
    elif args.command == "questionable-events":
        show_questionable_events(args.db, args.limit)


if __name__ == "__main__":
    main()
