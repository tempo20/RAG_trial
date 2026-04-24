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
    for row in conn.execute(sql, params).fetchall():
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


def show_latest_stream_briefs(db_path: str, limit: int = 20) -> None:
    conn = _connect(db_path)
    rows = conn.execute(
        """
        SELECT
            source,
            title,
            substr(raw_text, 1, 280) AS summary,
            content_class,
            article_quality_score,
            quality_flags_json,
            published_at,
            scraped_at_utc
        FROM articles
        WHERE source = 'TradingEconomics'
          AND source_provider = 'stream'
        ORDER BY coalesce(published_at, scraped_at_utc) DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    if not rows:
        print("[analyst] no TradingEconomics stream briefs found")
        conn.close()
        return
    for row in rows:
        flags = []
        try:
            payload = json.loads(row["quality_flags_json"] or "{}")
            if isinstance(payload, dict):
                flags = payload.get("flags") or []
            elif isinstance(payload, list):
                flags = payload
        except json.JSONDecodeError:
            flags = []
        print(
            f"- source={row['source']} class={row['content_class']} "
            f"quality={row['article_quality_score']} date={row['published_at'] or row['scraped_at_utc']}"
        )
        print(f"  title: {row['title']}")
        print(f"  summary: {row['summary']}")
        print(f"  flags: {', '.join(str(flag) for flag in flags) or '-'}")
    conn.close()


def show_top_signals(db_path: str, limit: int) -> None:
    conn = _connect(db_path)
    rows = conn.execute(
        """
        SELECT
            sa.signal_id,
            sa.cluster_id,
            sa.signal_date,
            sa.rank,
            sa.signal_score,
            sa.headline,
            sa.novelty_hint,
            sa.urgency,
            sa.market_surprise
        FROM signal_alerts sa
        WHERE sa.status = 'active'
        ORDER BY sa.signal_score DESC, sa.signal_date DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    for row in rows:
        print(
            f"- signal={row['signal_id']} cluster={row['cluster_id']} score={row['signal_score']} "
            f"rank={row['rank']} date={row['signal_date']}"
        )
        print(
            f"  headline: {row['headline']} | novelty={row['novelty_hint'] or '-'} "
            f"urgency={row['urgency'] or '-'} surprise={row['market_surprise'] or '-'}"
        )
    conn.close()


def show_cluster(db_path: str, cluster_id: str) -> None:
    conn = _connect(db_path)
    cluster = conn.execute(
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
            unique_source_count,
            asset_targets_json
        FROM event_clusters
        WHERE cluster_id = ?
        """,
        (cluster_id,),
    ).fetchone()
    if not cluster:
        print("[analyst] cluster not found")
        conn.close()
        return
    print(
        f"cluster={cluster['cluster_id']} type={cluster['event_type']} region={cluster['region']} "
        f"members={cluster['member_count']} sources={cluster['unique_source_count']}"
    )
    print(f"summary: {cluster['canonical_summary']}")
    print(f"window: {cluster['first_event_time']} -> {cluster['last_event_time']}")
    assets = json.loads(cluster["asset_targets_json"] or "[]")
    print(f"assets: {', '.join(assets) or '-'}")
    members = conn.execute(
        """
        SELECT
            cm.macro_event_id,
            cm.similarity_score,
            m.summary,
            m.confidence,
            m.support_score,
            cm.source,
            cm.event_time
        FROM cluster_members cm
        JOIN macro_events m ON m.macro_event_id = cm.macro_event_id
        WHERE cm.cluster_id = ?
        ORDER BY cm.event_time DESC, cm.similarity_score DESC
        """,
        (cluster_id,),
    ).fetchall()
    print("members:")
    for row in members:
        print(
            f"- {row['macro_event_id']} sim={row['similarity_score']} confidence={row['confidence']} "
            f"support={row['support_score']} source={row['source']} time={row['event_time']}"
        )
        print(f"  {row['summary']}")
    conn.close()


def show_signal_evidence(db_path: str, signal_id: str) -> None:
    conn = _connect(db_path)
    signal = conn.execute(
        """
        SELECT signal_id, cluster_id, signal_score, signal_date, headline, summary
        FROM signal_alerts
        WHERE signal_id = ?
        """,
        (signal_id,),
    ).fetchone()
    if not signal:
        print("[analyst] signal not found")
        conn.close()
        return
    print(
        f"signal={signal['signal_id']} cluster={signal['cluster_id']} "
        f"score={signal['signal_score']} date={signal['signal_date']}"
    )
    print(f"headline: {signal['headline']}")
    print(f"summary: {signal['summary']}")
    rows = conn.execute(
        """
        SELECT
            m.macro_event_id,
            m.event_type,
            m.summary,
            m.confidence,
            a.source,
            art.title,
            e.evidence_text
        FROM cluster_members cm
        JOIN macro_events m ON m.macro_event_id = cm.macro_event_id
        LEFT JOIN articles art ON art.article_id = m.article_id
        LEFT JOIN cluster_members a ON a.cluster_id = cm.cluster_id AND a.macro_event_id = cm.macro_event_id
        LEFT JOIN evidence_spans e
          ON e.parent_kind = 'macro_event'
         AND e.parent_id = m.macro_event_id
        WHERE cm.cluster_id = ?
        ORDER BY m.confidence DESC, m.macro_event_id, e.evidence_id
        """,
        (signal["cluster_id"],),
    ).fetchall()
    print("evidence:")
    for row in rows:
        print(
            f"- event={row['macro_event_id']} type={row['event_type']} confidence={row['confidence']} "
            f"source={row['source']}"
        )
        print(f"  title: {row['title']}")
        print(f"  summary: {row['summary']}")
        print(f"  evidence: {row['evidence_text'] or '-'}")
    conn.close()


def show_missed_signals(db_path: str, limit: int) -> None:
    conn = _connect(db_path)
    rows = conn.execute(
        """
        SELECT
            mr.signal_id,
            mr.cluster_id,
            mr.target_type,
            mr.target_id,
            mr.predicted_direction,
            mr.return_1d,
            mr.return_3d,
            mr.return_5d,
            sa.signal_score,
            sa.headline
        FROM market_reactions mr
        LEFT JOIN signal_alerts sa ON sa.signal_id = mr.signal_id
        WHERE mr.outcome_label = 'miss'
        ORDER BY sa.signal_score DESC, mr.created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    for row in rows:
        print(
            f"- signal={row['signal_id']} target={row['target_type']}::{row['target_id']} "
            f"predicted={row['predicted_direction']} score={row['signal_score']}"
        )
        print(
            f"  returns: 1d={row['return_1d']} 3d={row['return_3d']} 5d={row['return_5d']}"
        )
        print(f"  headline: {row['headline']}")
    conn.close()


def show_best_performing_signals(db_path: str, limit: int) -> None:
    conn = _connect(db_path)
    rows = conn.execute(
        """
        SELECT
            sa.signal_id,
            sa.cluster_id,
            sa.signal_score,
            sa.headline,
            COUNT(*) AS reaction_count,
            AVG(coalesce(mr.return_5d, mr.return_3d, mr.return_1d)) AS avg_forward_return
        FROM signal_alerts sa
        JOIN market_reactions mr ON mr.signal_id = sa.signal_id
        WHERE mr.outcome_label = 'hit'
        GROUP BY sa.signal_id, sa.cluster_id, sa.signal_score, sa.headline
        ORDER BY avg_forward_return DESC, sa.signal_score DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    for row in rows:
        print(
            f"- signal={row['signal_id']} score={row['signal_score']} hits={row['reaction_count']} "
            f"avg_forward_return={row['avg_forward_return']}"
        )
        print(f"  headline: {row['headline']}")
    conn.close()


def show_source_quality(db_path: str, limit: int) -> None:
    conn = _connect(db_path)
    rows = conn.execute(
        """
        SELECT
            source_name,
            source_trust_tier,
            quality_score,
            reliability_label,
            article_count,
            last_article_at
        FROM source_quality
        ORDER BY quality_score DESC, article_count DESC, source_name
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    for row in rows:
        print(
            f"- source={row['source_name']} tier={row['source_trust_tier']} quality={row['quality_score']} "
            f"label={row['reliability_label']} articles={row['article_count']} last={row['last_article_at']}"
        )
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

    stream_parser = subparsers.add_parser("latest-stream-briefs", help="Show latest TradingEconomics stream briefs")
    stream_parser.add_argument("--limit", type=int, default=20, help="Rows to print")

    top_signals_parser = subparsers.add_parser("top-signals", help="Show top ranked signals")
    top_signals_parser.add_argument("--limit", type=int, default=10, help="Rows to print")

    cluster_parser = subparsers.add_parser("cluster", help="Inspect one event cluster")
    cluster_parser.add_argument("--cluster-id", required=True, help="Event cluster id")

    signal_evidence_parser = subparsers.add_parser("signal-evidence", help="Show signal -> cluster -> evidence")
    signal_evidence_parser.add_argument("--signal-id", required=True, help="Signal id")

    missed_parser = subparsers.add_parser("missed-signals", help="Show missed signal outcomes")
    missed_parser.add_argument("--limit", type=int, default=10, help="Rows to print")

    best_parser = subparsers.add_parser("best-performing-signals", help="Show best-performing signals")
    best_parser.add_argument("--limit", type=int, default=10, help="Rows to print")

    source_quality_parser = subparsers.add_parser("source-quality", help="Show source quality scores")
    source_quality_parser.add_argument("--limit", type=int, default=20, help="Rows to print")

    args = parser.parse_args()
    if args.command == "latest-events":
        show_latest_macro_events(args.db, args.limit, args.min_confidence)
    elif args.command == "event-evidence":
        show_event_evidence(args.db, args.event_id)
    elif args.command == "impact-entities":
        show_entities_with_most_impact_links(args.db, args.limit)
    elif args.command == "questionable-events":
        show_questionable_events(args.db, args.limit)
    elif args.command == "latest-stream-briefs":
        show_latest_stream_briefs(args.db, args.limit)
    elif args.command == "top-signals":
        show_top_signals(args.db, args.limit)
    elif args.command == "cluster":
        show_cluster(args.db, args.cluster_id)
    elif args.command == "signal-evidence":
        show_signal_evidence(args.db, args.signal_id)
    elif args.command == "missed-signals":
        show_missed_signals(args.db, args.limit)
    elif args.command == "best-performing-signals":
        show_best_performing_signals(args.db, args.limit)
    elif args.command == "source-quality":
        show_source_quality(args.db, args.limit)


if __name__ == "__main__":
    main()
