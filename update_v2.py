"""
update_v2.py - Full SQLite + lean Neo4j pipeline orchestrator.

Pipeline:
    Step 1  Scrape RSS feeds -> articles saved to SQLite
    Step 2  SQLite pass      -> chunk articles + extract entity mentions -> SQLite
    Step 3  Macro extraction -> verified macro events -> SQLite
    Step 4  Event clustering -> event_clusters / cluster_members -> SQLite
    Step 5  Signal scoring   -> event_cluster_scores / signal_alerts -> SQLite
    Step 6  Market feedback  -> market_reactions -> SQLite
    Step 7  Neo4j sync       -> rebuild lean graph from SQLite
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="SQLite pipeline orchestrator (v2)")
    parser.add_argument(
        "--no-scrape",
        action="store_true",
        help="Skip RSS scraping; process articles already in SQLite",
    )
    parser.add_argument(
        "--skip-entities",
        action="store_true",
        help="Skip NER entity extraction in the SQLite pass",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip chunk embedding generation/backfill in the SQLite pass",
    )
    parser.add_argument(
        "--no-macro",
        action="store_true",
        help="Skip macro extraction (Step 3)",
    )
    parser.add_argument(
        "--no-cluster",
        action="store_true",
        help="Skip event clustering (Step 4)",
    )
    parser.add_argument(
        "--no-signal-score",
        action="store_true",
        help="Skip signal scoring (Step 5)",
    )
    parser.add_argument(
        "--no-market-feedback",
        action="store_true",
        help="Skip market feedback (Step 6)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reprocess all articles from scratch (ignores existing chunks)",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Wipe and recreate the entire SQLite database before running the pipeline",
    )
    parser.add_argument(
        "--macro-limit",
        type=int,
        default=None,
        metavar="N",
        help="Max number of chunks to process in the macro extraction pass",
    )
    parser.add_argument(
        "--cluster-window-days",
        type=int,
        default=7,
        metavar="N",
        help="Max day gap for event clustering comparisons (default: 7)",
    )
    parser.add_argument(
        "--signal-limit",
        type=int,
        default=None,
        metavar="N",
        help="Optional max number of ranked signals to persist",
    )
    parser.add_argument(
        "--keep-per-singletons",
        action="store_true",
        help="Keep PER entities that appear in only one chunk (disables singleton filter)",
    )
    parser.add_argument(
        "--no-neo4j-sync",
        action="store_true",
        help="Skip Neo4j sync (Step 7)",
    )
    parser.add_argument(
        "--wipe-neo4j",
        action="store_true",
        help="Wipe the lean Neo4j graph before syncing from SQLite",
    )
    args = parser.parse_args()

    if args.reset_db:
        from create_sql_db import create_database

        db_path = os.getenv("SQLITE_DB", "my_database.db")
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Removed existing database: {db_path}")
        create_database(db_path)
        print(f"Fresh database created: {db_path}")

    print("=" * 60)
    if args.no_scrape:
        print("Step 1: Skipping scrape (--no-scrape)")
    else:
        print("Step 1: Scraping RSS feeds -> SQLite")
        print("=" * 60)
        from simple_scraper_v2 import main as scrape

        scrape()

    print()
    print("=" * 60)
    print("Step 2: SQLite pass (chunking + entity extraction)")
    print("=" * 60)
    from tgrag_setup import run_sqlite_pass

    run_sqlite_pass(
        reset=args.reset,
        skip_entities=args.skip_entities,
        keep_per_singletons=args.keep_per_singletons,
        skip_embeddings=args.skip_embeddings,
    )

    print()
    print("=" * 60)
    if args.no_macro:
        print("Step 3: Skipping macro extraction (--no-macro)")
    else:
        print("Step 3: Macro extraction -> SQLite")
        print("=" * 60)
        from macro_extract import run_extraction

        run_extraction(limit=args.macro_limit)

    print()
    print("=" * 60)
    if args.no_cluster:
        print("Step 4: Skipping event clustering (--no-cluster)")
    else:
        print("Step 4: Event clustering -> SQLite")
        print("=" * 60)
        from event_cluster import run_event_clustering

        cluster_summary = run_event_clustering(window_days=args.cluster_window_days)
        print("[event_cluster] summary:")
        for key, value in cluster_summary.items():
            print(f"  {key}: {value}")

    print()
    print("=" * 60)
    if args.no_signal_score:
        print("Step 5: Skipping signal scoring (--no-signal-score)")
    else:
        print("Step 5: Signal scoring -> SQLite")
        print("=" * 60)
        from signal_scoring import run_signal_scoring

        scoring_summary = run_signal_scoring(limit=args.signal_limit)
        print("[signal_scoring] summary:")
        for key, value in scoring_summary.items():
            print(f"  {key}: {value}")

    print()
    print("=" * 60)
    if args.no_market_feedback:
        print("Step 6: Skipping market feedback (--no-market-feedback)")
    else:
        print("Step 6: Market feedback -> SQLite")
        print("=" * 60)
        from market_feedback import run_market_feedback

        feedback_summary = run_market_feedback()
        print("[market_feedback] summary:")
        for key, value in feedback_summary.items():
            print(f"  {key}: {value}")

    print()
    print("=" * 60)
    if args.no_neo4j_sync:
        print("Step 7: Skipping Neo4j sync (--no-neo4j-sync)")
    else:
        print("Step 7: Neo4j sync <- SQLite")
        print("=" * 60)
        from neo4j_sync import sync_sqlite_to_neo4j

        counts = sync_sqlite_to_neo4j(wipe=args.wipe_neo4j)
        print("[neo4j_sync] counts:")
        for key, value in counts.items():
            print(f"  {key}: {value}")

    print()
    print("=" * 60)
    print("Pipeline complete.")
    print("  Inspect DB:  python -c \"import sqlite3; ...\"")
    print("  Chat:        python chatter.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
