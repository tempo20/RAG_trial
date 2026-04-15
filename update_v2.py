"""
update_v2.py — Full SQLite + lean Neo4j pipeline orchestrator.

Pipeline:
    Step 1  Scrape RSS feeds → articles saved to SQLite
    Step 2  SQLite pass      → chunk articles + extract entity mentions → SQLite
    Step 3  Macro extraction → Claude extracts macro events → SQLite
    Step 4  Neo4j sync       → rebuild lean MacroEvent/Entity/Asset graph

Usage:
    python update_v2.py                   # full pipeline
    python update_v2.py --no-scrape       # skip scraping, use existing DB articles
    python update_v2.py --no-macro        # skip macro extraction
    python update_v2.py --skip-entities   # skip NER extraction in SQLite pass
    python update_v2.py --reset           # reprocess all articles (ignore existing chunks)
    python update_v2.py --reset-db        # wipe entire DB and rebuild from scratch
    python update_v2.py --macro-limit 50  # process at most N chunks in macro pass
    python update_v2.py --wipe-neo4j      # rebuild Neo4j from a clean graph
"""

import argparse


def main():
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
        "--no-macro",
        action="store_true",
        help="Skip macro extraction (Step 3)",
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
        "--keep-per-singletons",
        action="store_true",
        help="Keep PER entities that appear in only one chunk (disables singleton filter)",
    )
    parser.add_argument(
        "--no-neo4j-sync",
        action="store_true",
        help="Skip Neo4j sync (Step 4)",
    )
    parser.add_argument(
        "--wipe-neo4j",
        action="store_true",
        help="Wipe the lean Neo4j graph before syncing from SQLite",
    )
    args = parser.parse_args()

    # ── DB reset (before everything else) ────────────────────────────────────
    if args.reset_db:
        import os
        from create_sql_db import create_database
        db_path = os.getenv("SQLITE_DB", "my_database.db")
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Removed existing database: {db_path}")
        create_database(db_path)
        print(f"Fresh database created: {db_path}")

    # ── Step 1: Scrape ───────────────────────────────────────────────────────
    print("=" * 60)
    if args.no_scrape:
        print("Step 1: Skipping scrape (--no-scrape)")
    else:
        print("Step 1: Scraping RSS feeds → SQLite")
        print("=" * 60)
        from simple_scraper_v2 import main as scrape
        scrape()

    # ── Step 2: SQLite pass (chunk + entity mentions) ────────────────────────
    print()
    print("=" * 60)
    print("Step 2: SQLite pass (chunking + entity extraction)")
    print("=" * 60)
    from tgrag_setup import run_sqlite_pass
    run_sqlite_pass(
        reset=args.reset,
        skip_entities=args.skip_entities,
        keep_per_singletons=args.keep_per_singletons,
    )

    # ── Step 3: Macro extraction ─────────────────────────────────────────────
    print()
    print("=" * 60)
    if args.no_macro:
        print("Step 3: Skipping macro extraction (--no-macro)")
    else:
        print("Step 3: Macro extraction → SQLite")
        print("=" * 60)
        from macro_extract import run_extraction
        run_extraction(limit=args.macro_limit)

    # ── Step 4: Neo4j sync ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    if args.no_neo4j_sync:
        print("Step 4: Skipping Neo4j sync (--no-neo4j-sync)")
    else:
        print("Step 4: Neo4j sync ← SQLite")
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
