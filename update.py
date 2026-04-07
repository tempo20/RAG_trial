"""
Scrape + TG-RAG Setup (combined)

Scrapes RSS feeds, saves articles to JSON, then chunks, embeds,
extracts entities, and loads new articles into the Neo4j temporal graph.

Usage:
    python update.py                    # scrape + incremental graph update
    python update.py --reset            # scrape + wipe graph and rebuild
    python update.py --skip-entities    # skip NER extraction
    python update.py --cooccur-mode article
"""

import argparse
import gc

from simple_scraper import main as scrape
from tgrag_setup import (
    load_and_chunk, embed_chunks, populate_neo4j,
    get_existing_article_ids, load_extraction_model, load_ticker_company_map,
    extract_all_entities,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
)
from pathlib import Path
from neo4j import GraphDatabase


def main():
    parser = argparse.ArgumentParser(description="Scrape + TG-RAG update")
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
        default="chunk",
        help=(
            "How to build co-occurrence edges: "
            "'chunk' -> :CO_OCCURS_CHUNK (default, lower noise) or "
            "'article' -> :CO_OCCURS_ARTICLE (denser recall)"
        ),
    )
    args = parser.parse_args()

    # Step 1: Scrape RSS feeds -> cnbc_articles.json
    print("=" * 50)
    print("Step 1: Scraping RSS feeds")
    print("=" * 50)
    scrape()

    # Step 2: Chunk, embed, load into Neo4j
    print()
    print("=" * 50)
    print("Step 2: Updating Neo4j temporal graph")
    print("=" * 50)

    skip_ids: set[str] = set()
    if not args.reset:
        print("Checking existing articles in Neo4j...")
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        skip_ids = get_existing_article_ids(driver)
        driver.close()
        print(f"Found {len(skip_ids)} articles already in graph")

    chunks = load_and_chunk(skip_ids=skip_ids)
    entity_data = None
    TICKER_MAP_PATH = Path("ticker_company_map.csv")
    if not args.skip_entities and chunks:
        pipe = load_extraction_model()
        ticker_lookup = load_ticker_company_map(TICKER_MAP_PATH)
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
    populate_neo4j(chunks, reset=args.reset, entity_data=entity_data if not args.skip_entities else None)

    print()
    print("Update complete. You can now run:  python chatter.py")


if __name__ == "__main__":
    main()
