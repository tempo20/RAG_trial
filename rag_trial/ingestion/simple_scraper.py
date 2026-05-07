"""
Backward-compatible entrypoint for the news scraper.

All provider-specific implementation lives in `simple_scraper_v2.py`.
"""

from rag_trial.ingestion.simple_scraper_v2 import main


if __name__ == "__main__":
    main()
