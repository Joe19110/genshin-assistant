"""
scrape.py — root entry point for the scraper.

Thin wrapper around src.scraper.runner so the workflow command stays simple:

    py scrape.py              # scrape all playable characters
    py scrape.py --force      # re-scrape everything
    py scrape.py --dry-run    # print discovered URLs only
    py scrape.py --limit 5    # first 5 characters (testing)
"""

from src.scraper.runner import main

if __name__ == "__main__":
    main()
