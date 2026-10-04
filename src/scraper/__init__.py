"""
src.scraper
===========
Scrapes game8.co character build pages and writes structured JSON files
to character_data/.

Public API
----------
scrape_character_page(url)  -> dict   live HTTP request
scrape_from_local_file(path) -> dict  dev helper (no network)
"""

from .page_parser import scrape_character_page, scrape_from_local_file

__all__ = ["scrape_character_page", "scrape_from_local_file"]
