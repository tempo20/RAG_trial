"""
update.py — Scrape + T-GRAG Graph Update

Scrapes RSS feeds, then runs the T-GRAG temporal knowledge graph
construction pipeline (chunk → extract → embed → write to Neo4j).

hist_to_db.py is retired. MarketBar/Instrument nodes are no longer used.

Usage:
    python update.py                  # scrape + incremental graph update
    python update.py --reset          # scrape + wipe graph and rebuild
    python update.py --skip-entities  # skip NER extraction (chunks + embeddings only)
    python update.py --no-scrape      # skip scraping, use existing cnbc_articles.json
"""

import argparse

from simple_scraper import main as scrape
from tgrag_setup import run_setup


def main():
    parser = argparse.ArgumentParser(description="Scrape + T-GRAG update")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the entire graph and rebuild from scratch",
    )
    parser.add_argument(
        "--skip-entities",
        action="store_true",
        help="Skip NER-based entity extraction (chunks + embeddings only)",
    )
    parser.add_argument(
        "--no-scrape",
        action="store_true",
        help="Skip RSS scraping, use existing cnbc_articles.json",
    )
    args = parser.parse_args()

    if not args.no_scrape:
        print("=" * 50)
        print("Step 1: Scraping RSS feeds")
        print("=" * 50)
        scrape()
    else:
        print("Step 1: Skipping scrape (--no-scrape)")

    print()
    print("=" * 50)
    print("Step 2: T-GRAG temporal graph construction")
    print("=" * 50)

    run_setup(
        reset=args.reset,
        skip_entities=args.skip_entities,
    )

    print()
    print("Update complete. Run:  python chatter.py")


if __name__ == "__main__":
    main()