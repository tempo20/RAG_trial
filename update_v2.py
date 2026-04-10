"""
update.py — Scrape + T-GRAG Graph Update (v2)

Uses simple_scraper_v2 with async + trafilatura extraction.

Usage:
    python update.py                  # scrape v2 + incremental graph update
    python update.py --reset          # scrape v2 + wipe graph and rebuild
    python update.py --skip-entities  # skip NER extraction (chunks + embeddings only)
    python update.py --no-scrape      # skip scraping, use existing cnbc_articles.json
    python update.py --old-scraper    # use original simple_scraper.py (synchronous)
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Scrape + T-GRAG update (v2)")
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
    parser.add_argument(
        "--old-scraper",
        action="store_true",
        help="Use original simple_scraper.py (synchronous, BeautifulSoup)",
    )
    args = parser.parse_args()

    if not args.no_scrape:
        print("=" * 60)
        print("Step 1: Scraping RSS feeds")
        print("=" * 60)
        
        if args.old_scraper:
            print("Using simple_scraper.py (synchronous, BeautifulSoup)")
            from simple_scraper import main as scrape
        else:
            print("Using simple_scraper_v2.py (async, trafilatura)")
            from simple_scraper_v2 import main as scrape
        
        scrape()
    else:
        print("Step 1: Skipping scrape (--no-scrape)")

    print()
    print("=" * 60)
    print("Step 2: T-GRAG temporal graph construction")
    print("=" * 60)

    from tgrag_setup import run_setup
    
    run_setup(
        reset=args.reset,
        skip_entities=args.skip_entities,
    )

    print()
    print("=" * 60)
    print("Update complete. Run:  python chatter.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
